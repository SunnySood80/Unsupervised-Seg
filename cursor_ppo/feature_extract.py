import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import models
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from functools import lru_cache

# Enable faster training configurations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

###############################################################################
#  SmallTransformerEncoder (optional if you need it)
###############################################################################
class SmallTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=8, num_layers=4, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        self.device = torch.device(f'cuda:{self.rank}')
        
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

    @torch.amp.autocast('cuda')
    def forward(self, x):
        B, C, H, W = x.shape
        x_reduced = self.pool(x)
        x_flat = x_reduced.flatten(2).transpose(1, 2)
        
        if self.training:
            def transformer_chunk(x):
                return self.transformer(x)
            x_transformed = checkpoint(transformer_chunk, x_flat, use_reentrant=False)
        else:
            x_transformed = self.transformer(x_flat)
        
        x_restored = x_transformed.transpose(1, 2).view(B, C, H // 2, W // 2)
        x_upsampled = self.upsample(x_restored)
        return x_upsampled

###############################################################################
#  FPNDecoder (optional if you need it)
###############################################################################
class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list=(256, 512, 1024), out_channels=256, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        
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

    @torch.amp.autocast('cuda')
    def forward(self, x1, x2, x3):
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
    def __init__(self, pretrained=True, out_channels=256, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        
        # Initialize ResNet50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = models.resnet50(weights=weights)
        base_resnet = base_resnet.to(memory_format=torch.channels_last)
        
        # Freeze backbone parameters
        for param in base_resnet.parameters():
            param.requires_grad = False
        
        # Extract layers
        self.stem = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool
        )
        self.layer1 = base_resnet.layer1  # 256 channels
        self.layer2 = base_resnet.layer2  # 512 channels
        self.layer3 = base_resnet.layer3  # 1024 channels

        # Add transformer and FPN
        self.transformer = SmallTransformerEncoder(embed_dim=1024, rank=self.rank)
        self.fpn = FPNDecoder(
            in_channels_list=(256, 512, 1024),
            out_channels=out_channels,
            rank=self.rank
        )
        
        # Feature caching
        self.feature_cache = {}
        self.cache_size_limit = 1000

    @torch.amp.autocast('cuda')
    def forward(self, x):
        # Check cache
        cache_key = (x.data_ptr(), x.shape)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract features
        x = self.stem(x)
        x1 = self.layer1(x)    # 256 channels
        x2 = self.layer2(x1)   # 512 channels
        x3 = self.layer3(x2)   # 1024 channels

        # Apply transformer to high-level features
        x3 = self.transformer(x3)
        
        # FPN decoder
        p1 = self.fpn(x1, x2, x3)

        # Cache result
        if len(self.feature_cache) >= self.cache_size_limit:
            self.feature_cache.clear()
        self.feature_cache[cache_key] = p1

        return p1

###############################################################################
#  FilterWeightingSegmenter
###############################################################################
class FilterWeightingSegmenter(nn.Module):
    """
    High-level module that outputs a 256-channel feature map from a ResNet50-FPN.
    """
    def __init__(self, pretrained=True, rank=None, out_channels=256):
        super().__init__()
        self.rank = rank if rank is not None else 0
        
        self.feature_extractor = HybridResNet50FPN(
            pretrained=pretrained, 
            out_channels=out_channels,
            rank=self.rank
        )

    @torch.amp.autocast('cuda')
    def forward(self, x):
        return self.feature_extractor(x)

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
