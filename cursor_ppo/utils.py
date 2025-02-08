# ===========================
# Cell 2
# ===========================
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import csv
import os
import copy
from feature_extract import DDPFeatureExtractor

# Define Sobel filters as constants
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

class MomentumEncoder:
    def __init__(self, model: nn.Module, momentum: float = 0.999):
        self.momentum = momentum

        # Special handling for our feature extractor
        if isinstance(model, DDPFeatureExtractor):
            print("Creating momentum encoder for DDP model...")
            self.ema_model = DDPFeatureExtractor(
                world_size=model.world_size,
                start_gpu=model.start_gpu
            )
            # Copy state dict instead of deep copying
            self.ema_model.load_state_dict(model.state_dict())
        else:
            print("Creating momentum encoder for standard model...")
            self.ema_model = type(model)()
            self.ema_model.load_state_dict(model.state_dict())
        
        self.ema_model.eval()
        
        # Disable gradients for momentum encoder
        for param in self.ema_model.parameters():
            param.requires_grad = False
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update momentum encoder weights"""
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data = self.momentum * ema_param.data + (1 - self.momentum) * param.data
            
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ema_model(x)

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)

def compute_contrastive_loss(
    query_feats: torch.Tensor,
    key_feats: torch.Tensor,
    temperature: float = 0.07
) -> float:
    """Optimized InfoNCE contrastive loss computation"""
    # Pre-normalize to avoid repeated normalization
    query = F.normalize(query_feats.flatten(2), dim=1)  # (B, C, H*W)
    key = F.normalize(key_feats.flatten(2), dim=1)      # (B, C, H*W)
    
    # Use batch matrix multiplication instead of einsum
    sim_matrix = torch.bmm(query.transpose(1, 2), key) / temperature
    
    # Use arange once and expand
    B, N, _ = sim_matrix.shape
    labels = torch.arange(N, device=query.device).expand(B, N)
    
    return float(F.cross_entropy(sim_matrix, labels).item())

def compute_cluster_separation_fast(features: torch.Tensor, eps: float = 0.5) -> Tuple[float, torch.Tensor]:
    """Optimized DBSCAN-based cluster separation"""
    feat_flat = features.reshape(-1, features.size(-1))
    feat_norm = F.normalize(feat_flat, dim=1)
    
    # Use chunked distance computation to save memory
    chunk_size = 1024
    N = feat_norm.size(0)
    core_points = torch.zeros(N, dtype=torch.bool, device=features.device)
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk_dists = torch.cdist(feat_norm[i:end], feat_norm)
        core_points[i:end] = (chunk_dists <= eps).sum(1) >= 4
    
    if not core_points.any():
        return 0.0, torch.zeros(2, feat_flat.size(1), device=features.device)
    
    # Fast cluster assignment using connected components
    labels = -torch.ones(N, device=features.device)
    current_label = 0
    
    for i in torch.where(core_points)[0]:
        if labels[i] >= 0:
            continue
            
        # Find cluster in one shot using matrix operations
        cluster_mask = torch.cdist(feat_norm[i:i+1], feat_norm) <= eps
        labels[cluster_mask[0]] = current_label
        current_label += 1
        
        if current_label >= 2:  # Early exit if we have enough clusters
            break
    
    if current_label < 2:
        return 0.0, torch.zeros(2, feat_flat.size(1), device=features.device)
    
    # Fast center computation using matrix operations
    centers = []
    for i in range(2):
        mask = labels == i
        if mask.any():
            center = feat_norm[mask].mean(0, keepdim=True)
            centers.append(F.normalize(center, dim=1))
        else:
            centers.append(torch.zeros(1, feat_flat.size(1), device=features.device))
    
    centers = torch.cat(centers, dim=0)
    separation = torch.cdist(centers, centers)[0, 1]
    
    return float(separation.item()), centers

def compute_feature_diversity(features: torch.Tensor, batch_size: int = 256) -> float:
    """Optimized feature diversity computation"""
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    # Flatten and normalize once
    features_flat = F.normalize(features.reshape(features.size(0), -1), p=2, dim=1)
    
    # Use matrix multiplication with chunking
    total_similarity = 0.0
    count = 0
    B = features_flat.size(0)
    
    for i in range(0, B, batch_size):
        i_end = min(i + batch_size, B)
        current = features_flat[i:i_end]
        
        # Compute similarities for this chunk
        sim = torch.mm(current, features_flat.t())
        
        # Zero out self-similarities
        if i == 0:
            sim.fill_diagonal_(0)
            
        total_similarity += sim.sum().item()
        count += sim.numel() - (i_end - i)  # Subtract diagonal count
    
    return 1 - (total_similarity / max(count, 1))

def compute_feature_statistics(features):
    """
    Compute basic statistics of features in a memory-efficient way
    """
    # Move to CPU for statistics
    features = features.cpu()
    
    mean = features.mean().item()
    std = features.std().item()
    
    return {
        'mean': mean,
        'std': std
    }

# Add a memory tracking decorator for debugging if needed
def track_memory(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before {func.__name__}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            print(f"GPU memory after {func.__name__}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        return result
    return wrapper

def compute_consistency_reward(original_feats: torch.Tensor, aug_feats_list: List[torch.Tensor] = None) -> float:
    """
    Compute consistency reward between original features and its augmentations.
    If no augmentations provided, return a neutral reward.
    """
    if aug_feats_list is None or len(aug_feats_list) == 0:
        return 0.5  # Neutral reward when no augmentations
        
    orig_flat = original_feats.view(original_feats.size(0), original_feats.size(1), -1)
    
    similarities = []
    for aug_feats in aug_feats_list:
        aug_flat = aug_feats.view(aug_feats.size(0), aug_feats.size(1), -1)
        
        orig_norm = F.normalize(orig_flat, dim=2)
        aug_norm = F.normalize(aug_flat, dim=2)
        
        similarity = torch.bmm(orig_norm, aug_norm.transpose(1, 2)).mean(dim=(1, 2))
        similarities.append(similarity)
    
    avg_similarity = torch.stack(similarities).mean()
    return float(avg_similarity.item())

def compute_boundary_strength(features):
    """
    Compute boundary strength from feature maps using gradients.
    Ensures consistent dimensions by proper padding.
    """
    # Move Sobel filters to the same device as features and expand to match input channels
    n_channels = features.size(1)
    
    # Expand Sobel filters to match input channels
    sobel_x = SOBEL_X.to(features.device)
    sobel_y = SOBEL_Y.to(features.device)
    
    # Expand to [n_channels, 1, 3, 3]
    sobel_x = sobel_x.repeat(n_channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(n_channels, 1, 1, 1)
    
    # Calculate gradients in x and y directions using same padding
    grad_x = F.conv2d(features, sobel_x, padding=1, groups=n_channels)
    grad_y = F.conv2d(features, sobel_y, padding=1, groups=n_channels)
    
    # Ensure grad_x and grad_y have same dimensions
    if grad_x.size() != grad_y.size():
        # Use the smaller size for both
        min_size = min(grad_x.size(2), grad_y.size(2))
        grad_x = grad_x[:, :, :min_size, :min_size]
        grad_y = grad_y[:, :, :min_size, :min_size]
    
    # Calculate gradient magnitude
    gradient_magnitude = torch.sqrt(grad_x.pow(2).sum(1) + grad_y.pow(2).sum(1))
    
    # Normalize gradient magnitude
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / \
                        (gradient_magnitude.max() - gradient_magnitude.min() + 1e-6)
    
    return gradient_magnitude.mean()

def compute_local_coherence(features: torch.Tensor, kernel_size: int = 3) -> float:
    """Optimized local coherence computation"""
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    # Use unfold for efficient neighborhood computation
    padding = kernel_size // 2
    padded = F.pad(features, (padding, padding, padding, padding))
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    local_mean = patches.mean(dim=(-2, -1))
    
    return float(-F.mse_loss(features, local_mean).item())

def compute_segmentation_map(w_feats: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute binary segmentation map from weighted features using simple clustering.
    Returns a binary (black/white) segmentation map.
    """
    # Resize binary mask to match feature size
    mask_resized = F.interpolate(
        binary_mask,
        size=(w_feats.shape[2], w_feats.shape[3]),  # Match feature spatial dimensions
        mode='nearest'
    )
    
    # Apply binary mask to features first
    w_feats = w_feats * mask_resized  # Now dimensions match
    
    # Flatten features for clustering
    feat_flat = w_feats.squeeze(0).permute(1,2,0).reshape(-1, w_feats.size(1))
    feat_flat = F.normalize(feat_flat, dim=1)
    
    # Select two random points as initial centroids
    N, D = feat_flat.size()
    idx = torch.randperm(N, device=w_feats.device)[:2]
    centroids = feat_flat[idx].clone()
    
    # Simple 2-means clustering
    dists = torch.cdist(feat_flat, centroids)
    clusters = dists.argmin(dim=1)
    
    # Reshape back to spatial dimensions - keep binary
    seg_map = clusters.reshape(w_feats.shape[2], w_feats.shape[3]).float()
    
    # Apply binary mask again to ensure masked regions are 0
    seg_map = seg_map * mask_resized.squeeze()
    
    return seg_map

def visualize_map_with_augs(image_tensors: list, 
                           heatmaps: list,
                           ground_truth: np.ndarray,
                           binary_mask: torch.Tensor,
                           reward: float,
                           save_path: str = None):
    """Creates a grid showing all augmentations and their binary maps."""
    n_images = len(image_tensors)
    n_cols = 4
    n_rows = n_images
    
    plt.figure(figsize=(20, 5*n_rows))
    
    # Prepare binary mask
    mask_resized = F.interpolate(
        binary_mask,
        size=(256, 256),
        mode='nearest'
    ).squeeze().cpu().numpy()
    
    for idx, (img, hmap) in enumerate(zip(image_tensors, heatmaps)):
        if img.dim() == 4:
            img = img.squeeze(0)
            
        # Resize image
        img_resized = F.interpolate(
            img.unsqueeze(0), 
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Resize heatmap and apply mask
        hmap_resized = F.interpolate(
            hmap.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode='nearest'
        ).squeeze().detach().cpu().numpy()
        hmap_resized = hmap_resized * mask_resized  # Apply binary mask
        
        img_np = img_resized.detach().cpu().permute(1,2,0).numpy()
        
        # Original Image
        plt.subplot(n_rows, n_cols, idx*n_cols + 1)
        plt.imshow(img_np)
        plt.title(f"{'Original' if idx==0 else f'Aug {idx}'}")
        plt.axis("off")
        
        # Ground Truth
        plt.subplot(n_rows, n_cols, idx*n_cols + 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")
        
        # Binary Map (black/white)
        plt.subplot(n_rows, n_cols, idx*n_cols + 3)
        plt.imshow(hmap_resized, cmap='gray')
        plt.title(f"Binary Map (R={reward:.3f})")
        plt.axis("off")
        
        # Overlay - using resized versions of both image and heatmap
        plt.subplot(n_rows, n_cols, idx*n_cols + 4)
        plt.imshow(img_np)
        plt.imshow(hmap_resized, cmap='gray', alpha=0.6)
        plt.title("Overlay")
        plt.axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()