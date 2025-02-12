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

class MomentumEncoder:
    def __init__(self, model: nn.Module, momentum: float = 0.999):
        self.momentum = momentum
        self.model = model  # Store original model
        
        # Create EMA model with same architecture
        if isinstance(model, DDPFeatureExtractor):
            print("Creating momentum encoder for DDP model...")
            self.ema_model = DDPFeatureExtractor(
                world_size=model.world_size,
                start_gpu=model.start_gpu
            )
        else:
            print("Creating momentum encoder for standard model...")
            # Create new instance with same parameters as original model
            if hasattr(model, 'module'):
                # Handle DDP wrapped model
                base_model = model.module
            else:
                base_model = model
                
            # Create new instance with same constructor arguments
            self.ema_model = type(base_model)(
                pretrained=True  # Assuming FilterWeightingSegmenter takes this arg
            )
            
        # Copy weights
        if hasattr(model, 'module'):
            # If DDP model, copy from module state dict
            self.ema_model.load_state_dict(model.module.state_dict())
        else:
            self.ema_model.load_state_dict(model.state_dict())
            
        self.ema_model.eval()
        
        # Disable gradients for momentum encoder
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update momentum encoder weights"""
        source_model = model.module if hasattr(model, 'module') else model
        for ema_param, param in zip(self.ema_model.parameters(), source_model.parameters()):
            ema_param.data = self.momentum * ema_param.data + (1 - self.momentum) * param.data
            
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features - x should already be feature maps"""
        return x  # Just return features since they're already extracted

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


def compute_feature_diversity(features: torch.Tensor, batch_size: int = 64) -> float:
    """
    Compute diversity score for features in a memory-efficient way
    
    Args:
        features: Tensor of shape [B, C, H, W] or [C, H, W]
        batch_size: Size of batches to process at once
    """
    # Ensure features are batched
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    # Get dimensions
    B, C, H, W = features.shape
    
    # Reshape features to [B, C*H*W]
    features_flat = features.reshape(B, -1)
    
    # Normalize features
    features_flat = F.normalize(features_flat, p=2, dim=1)
    
    total_similarity = 0
    count = 0
    
    # Process in larger batches
    for i in range(0, B, batch_size):
        batch_end = min(i + batch_size, B)
        current_batch = features_flat[i:batch_end]
        
        # Use larger batches for more stable diversity computation
        for j in range(0, B, batch_size):
            j_end = min(j + batch_size, B)
            other_batch = features_flat[j:j_end]
            
            # Compute batch similarity
            batch_similarity = torch.mm(current_batch, other_batch.t())
            
            # Don't count self-similarities
            if i == j:
                batch_similarity.fill_diagonal_(0)
            
            total_similarity += batch_similarity.sum().item()
            count += (batch_end - i) * (j_end - j)
            if i == j:
                count -= (batch_end - i)  # Subtract diagonal count
    
    # Compute average similarity
    avg_similarity = total_similarity / max(count, 1)
    
    # Convert similarity to diversity (higher is better)
    diversity = 1 - avg_similarity
    
    return diversity

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

def compute_boundary_strength(features: torch.Tensor) -> float:
    """
    Compute boundary strength based on feature gradients.
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
        
    B, C, H, W = features.shape
    
    grad_x = torch.zeros_like(features)
    grad_y = torch.zeros_like(features)
    
    grad_x[..., :-1] = features[..., 1:] - features[..., :-1]
    grad_y[..., :-1, :] = features[..., 1:, :] - features[..., :-1, :]
    
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = gradient_magnitude.contiguous().view(B, -1)
    
    if gradient_magnitude.shape[1] > 0:
        gradient_magnitude = F.normalize(gradient_magnitude, dim=1)
        boundary_score = gradient_magnitude.mean().item()
    else:
        boundary_score = 0.0
    
    return float(boundary_score)

def compute_local_coherence(features: torch.Tensor, kernel_size: int = 3) -> float:
    """
    Compute local feature coherence using average pooling.
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    padding = kernel_size // 2
    local_mean = F.avg_pool2d(features, kernel_size=kernel_size, 
                             stride=1, padding=padding)
    
    coherence = -F.mse_loss(features, local_mean)
    return float(coherence.item())

def compute_segmentation_map(w_feats: torch.Tensor, binary_mask: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """
    Compute binary segmentation map using improved k-means clustering with:
    1. Better centroid initialization
    2. Multiple refinement iterations
    3. Spatial regularization
    4. Confidence thresholding
    
    Args:
        w_feats: Weighted feature tensor [B, C, H, W]
        binary_mask: Binary mask tensor [1, 1, H, W]
        n_iters: Number of k-means iterations
    Returns:
        Binary segmentation map tensor [H, W]
    """
    # Resize binary mask to match feature size
    mask_resized = F.interpolate(
        binary_mask,
        size=(w_feats.shape[2], w_feats.shape[3]),
        mode='nearest'
    )
    
    # Apply binary mask and normalize features
    w_feats = w_feats * mask_resized
    feat_flat = w_feats.squeeze(0).permute(1, 2, 0)  # [H, W, C]
    feat_flat = F.normalize(feat_flat, dim=-1)
    
    # Add spatial coordinates for regularization
    H, W = w_feats.shape[2:]
    y_coords = torch.linspace(-1, 1, H, device=w_feats.device)
    x_coords = torch.linspace(-1, 1, W, device=w_feats.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    spatial_coords = torch.stack([yy, xx], dim=-1) * 0.1  # Scale factor for spatial influence
    
    # Combine features with spatial coordinates
    feat_spatial = torch.cat([
        feat_flat,
        spatial_coords
    ], dim=-1)
    
    # Flatten for clustering
    N = H * W
    D = feat_spatial.shape[-1]
    feat_spatial = feat_spatial.reshape(N, D)
    
    # Initialize centroids using k-means++
    centroids = torch.zeros((2, D), device=w_feats.device)
    
    # First centroid: maximum feature response
    feat_norms = torch.norm(feat_flat.reshape(N, -1), dim=1)
    first_idx = torch.argmax(feat_norms)
    centroids[0] = feat_spatial[first_idx]
    
    # Second centroid: furthest point from first
    dists = torch.norm(feat_spatial - centroids[0], dim=1)
    second_idx = torch.argmax(dists)
    centroids[1] = feat_spatial[second_idx]
    
    # K-means iterations
    clusters = None
    for _ in range(n_iters):
        # Compute distances and assign clusters
        dists = torch.cdist(feat_spatial, centroids)
        new_clusters = dists.argmin(dim=1)
        
        if clusters is not None and torch.all(new_clusters == clusters):
            break
            
        clusters = new_clusters
        
        # Update centroids
        for i in range(2):
            mask = (clusters == i)
            if mask.any():
                centroids[i] = feat_spatial[mask].mean(dim=0)
    
    # Reshape clusters to spatial dimensions
    seg_map = clusters.reshape(H, W).float()
    
    # Compute confidence scores
    dists = torch.cdist(feat_spatial, centroids)
    confidence = torch.abs(dists[:, 0] - dists[:, 1])
    confidence = confidence.reshape(H, W)
    
    # Apply confidence thresholding
    conf_threshold = torch.quantile(confidence[mask_resized.squeeze().bool()], 0.2)
    uncertain_mask = confidence < conf_threshold
    
    # Refine uncertain regions using spatial consistency
    kernel_size = 3
    padding = kernel_size // 2
    for _ in range(2):  # Number of refinement iterations
        # Compute local majority vote
        seg_map_padded = F.pad(seg_map.unsqueeze(0).unsqueeze(0), 
                              (padding, padding, padding, padding), 
                              mode='reflect')
        neighbors = F.unfold(seg_map_padded, kernel_size=kernel_size)
        neighbors = neighbors.reshape(1, kernel_size*kernel_size, H, W)
        local_sum = neighbors.sum(dim=1).squeeze()
        local_vote = (local_sum > (kernel_size*kernel_size/2)).float()
        
        # Update uncertain regions
        seg_map[uncertain_mask] = local_vote[uncertain_mask]
    
    # Final mask application
    seg_map = seg_map * mask_resized.squeeze()
    
    return seg_map

def visualize_map_with_augs(
    image_tensors,
    heatmaps,
    ground_truth,
    binary_mask,
    reward,
    save_path
):
    # The error is happening here:
    mask_resized = F.interpolate(
        binary_mask,  # binary_mask needs to be properly shaped
        size=(256, 256),
        mode='nearest'
    )
    
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