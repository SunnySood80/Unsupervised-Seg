# ===========================
# Cell 4: DDP-Enabled ResNet50 + FPN
# ===========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as cuda_autocast
from torchvision import models
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import socket
import os
from functools import lru_cache
import datetime
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Enable faster training configurations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class SingleProcessDDP:
    """DDP wrapper for single process multi-GPU setup"""
    def __init__(self, model, device_ids, output_device=None):
        self.model = model
        self.device_ids = device_ids
        self.output_device = output_device if output_device is not None else device_ids[0]
        self.model = self.model.to(f'cuda:{self.device_ids[0]}')
        self.stream = torch.cuda.Stream(device=self.device_ids[0])

    def __call__(self, *args, **kwargs):
        with torch.cuda.stream(self.stream):
            args = [arg.to(f'cuda:{self.device_ids[0]}', non_blocking=True) if torch.is_tensor(arg) else arg for arg in args]
            kwargs = {k: v.to(f'cuda:{self.device_ids[0]}', non_blocking=True) if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return self.model(*args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state['stream'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stream = torch.cuda.Stream(device=self.device_ids[0])

class SmallTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=8, num_layers=4, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        self.device = torch.device(f'cuda:{self.rank}')
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

    @cuda_autocast('cuda')
    def forward(self, x):
        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True)
            B, C, H, W = x.shape
            x_reduced = self.pool(x)
            x_flat = x_reduced.flatten(2).transpose(1, 2)
            
            if self.training:
                def transformer_chunk(x):
                    return self.transformer(x)
                x_transformed = checkpoint(transformer_chunk, x_flat)
            else:
                x_transformed = self.transformer(x_flat)
            
            x_restored = x_transformed.transpose(1, 2).view(B, C, H // 2, W // 2)
            x_upsampled = self.upsample(x_restored)
            return x_upsampled

    def __getstate__(self):
        state = self.__dict__.copy()
        state['stream'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stream = torch.cuda.Stream(device=self.device)

class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list=(256, 512, 1024), out_channels=256, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        self.device = torch.device(f'cuda:{self.rank}')
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

    @cuda_autocast('cuda')
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state['stream'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stream = torch.cuda.Stream(device=self.device)

class HybridResNet50FPN(nn.Module):
    def __init__(self, pretrained=True, out_channels=256, rank=0):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Create stem (first layer)
        self.stem = nn.Sequential(
            resnet.conv1,  # This becomes stem.0
            resnet.bn1,    # This becomes stem.1
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

    @cuda_autocast('cuda')
    def forward(self, x):
        # ResNet stages
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN top-down pathway
        p5 = self.lateral4(c5)
        p4 = self._upsample_add(p5, self.lateral3(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))
        p2 = self._upsample_add(p3, self.lateral1(c2))
        
        # Smooth
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p2 = self.smooth1(p2)
        
        return p2  # Return only the finest resolution feature map

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode='nearest') + y

class FilterWeightingSegmenter(nn.Module):
    def __init__(self, pretrained=True, rank=None, out_channels=256):
        super().__init__()
        self.rank = rank if rank is not None else 0
        self.device = torch.device(f'cuda:{self.rank}')
        self.stream = torch.cuda.Stream(device=self.device)
        
        # Create feature extractor
        self.feature_extractor = HybridResNet50FPN(
            pretrained=pretrained,
            out_channels=out_channels
        )
        
        self.to(self.device)

    @cuda_autocast('cuda')
    def forward(self, x):
        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True)
            return self.feature_extractor(x)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['stream'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stream = torch.cuda.Stream(device=self.device)

class DDPFeatureExtractor(nn.Module):
    def __init__(self, world_size=3, start_gpu=1):
        super().__init__()
        self.world_size = world_size
        self.start_gpu = start_gpu
        self.model = None
        
        # Set up multiprocessing spawn
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(self._setup_process, nprocs=world_size, 
                args=(world_size,), join=True)

    def _setup_process(self, rank, world_size):
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank + self.start_gpu
        )

        # Create model and move to GPU
        local_model = FilterWeightingSegmenter(
            pretrained=True,
            rank=rank + self.start_gpu,
            out_channels=256
        )
        local_model = local_model.to(rank + self.start_gpu)

        # Wrap model with DDP
        self.model = DDP(
            local_model,
            device_ids=[rank + self.start_gpu],
            output_device=rank + self.start_gpu
        )

    @torch.no_grad()
    def forward(self, x):
        if self.model is None:
            raise RuntimeError("Model not initialized! DDP setup failed.")
        
        # Split input across GPUs
        chunks = x.chunk(self.world_size)
        outputs = []
        
        for i, chunk in enumerate(chunks):
            device = f'cuda:{i + self.start_gpu}'
            chunk = chunk.to(device)
            out = self.model(chunk)
            outputs.append(out.to('cuda:0'))
            
        return torch.cat(outputs, dim=0)

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        if self.model is not None:
            self.model.to('cpu')
            self.model = None
        torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()