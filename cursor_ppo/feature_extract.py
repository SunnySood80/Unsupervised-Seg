import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as cuda_autocast
from torchvision import models
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50, ResNet50_Weights

###############################################################################
#  SmallTransformerEncoder (optional if you need it)
###############################################################################
class SmallTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=8, num_layers=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = torch.cuda.Stream(device=self.device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU(inplace=True)
        self.to(self.device)

    @torch.amp.autocast('cuda')
    def forward(self, x):
        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True)
            B, C, H, W = x.shape
            x_reduced = self.pool(x)
            x_flat = x_reduced.flatten(2).transpose(1, 2)

            if self.training:
                def transformer_chunk(z):
                    return self.transformer(z)
                x_transformed = checkpoint(transformer_chunk, x_flat)
            else:
                x_transformed = self.transformer(x_flat)

            x_restored = x_transformed.transpose(1, 2).view(B, C, H // 2, W // 2)
            x_upsampled = self.upsample(x_restored)
        return x_upsampled

###############################################################################
#  FPNDecoder (optional if you need it)
###############################################################################
class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list=(256, 512, 1024), out_channels=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = torch.cuda.Stream(device=self.device)

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_c in in_channels_list
        ])

        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            for _ in range(2)
        ])

        self.refine_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=8, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(2)
        ])

        self.final_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)
        self.to(self.device)

    @torch.amp.autocast('cuda')
    def forward(self, x1, x2, x3):
        with torch.cuda.stream(self.stream):
            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            x3 = x3.to(self.device, non_blocking=True)

            p3 = self.lateral_convs[2](x3)
            p2 = self.lateral_convs[1](x2)
            p1 = self.lateral_convs[0](x1)

            p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
            p2 = self.relu(p2 + p3_up)
            p2 = p2 * self.channel_attention[0](p2)
            p2 = p2 + self.refine_blocks[0](p2)

            p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
            p1 = self.relu(p1 + p2_up)
            p1 = p1 * self.channel_attention[1](p1)
            p1 = p1 + self.refine_blocks[1](p1)

            p1 = p1 + self.final_enhance(p1)
        return p1

###############################################################################
#  HybridResNet50FPN
###############################################################################
class HybridResNet50FPN(nn.Module):
    """
    A ResNet50-based feature extractor with a simple FPN top-down pathway.
    """
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # Create stem (first layer)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # Get ResNet stages
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # FPN layers
        self.lateral4 = nn.Conv2d(2048, out_channels, 1)
        self.lateral3 = nn.Conv2d(1024, out_channels, 1)
        self.lateral2 = nn.Conv2d(512, out_channels, 1)
        self.lateral1 = nn.Conv2d(256, out_channels, 1)

        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    @torch.amp.autocast('cuda')
    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lateral4(c5)
        p4 = self._upsample_add(p5, self.lateral3(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))
        p2 = self._upsample_add(p3, self.lateral1(c2))

        # Smooth
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p2 = self.smooth1(p2)

        return p2

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode='nearest') + y

###############################################################################
#  FilterWeightingSegmenter
###############################################################################
class FilterWeightingSegmenter(nn.Module):
    """
    High-level module that outputs a 256-channel feature map from a ResNet50-FPN.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load model without classifier
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        del self.backbone.fc  # Remove classifier
        
        # Cache for features
        self.feature_cache = {}
        self.max_cache_size = 1000
        
    @torch.no_grad()  # Add no_grad decorator
    def forward(self, x):
        # Check cache first
        key = hash(x.cpu().numpy().tobytes())
        if key in self.feature_cache:
            return self.feature_cache[key]
            
        features = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Cache result
        if len(self.feature_cache) < self.max_cache_size:
            self.feature_cache[key] = x
            
        return x

###############################################################################
#  DDPFeatureExtractor
###############################################################################
class DDPFeatureExtractor(nn.Module):
    """
    Wrapper class for FilterWeightingSegmenter to support DDP training
    """
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        self.feature_extractor = FilterWeightingSegmenter(
            pretrained=pretrained,
            out_channels=out_channels
        )

    @torch.amp.autocast('cuda')
    def forward(self, x):
        return self.feature_extractor(x)
