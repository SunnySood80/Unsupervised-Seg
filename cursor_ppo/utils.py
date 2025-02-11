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

def compute_contrastive_loss(
    query_feats: torch.Tensor,
    key_feats: torch.Tensor,
    temperature: float = 0.07
) -> float:
    """
    Compute InfoNCE contrastive loss between query and key features.
    """
    # Flatten and normalize features
    query = F.normalize(query_feats.flatten(2), dim=1)  # (B, C, H*W)
    key = F.normalize(key_feats.flatten(2), dim=1)      # (B, C, H*W)
    
    # Compute similarity matrix
    sim_matrix = torch.einsum('bci,bcj->bij', query, key) / temperature
    
    # For each position, treat it as positive example for itself
    # and all other positions as negatives
    B, _, N = query.shape
    labels = torch.arange(N, device=query.device)
    labels = labels.unsqueeze(0).expand(B, N)  # (B, N)
    
    # Compute InfoNCE loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return float(loss.item())

def compute_cluster_separation_fast(features: torch.Tensor, eps: float = 0.5) -> Tuple[float, torch.Tensor]:
    """
    DBSCAN-based cluster separation. Memory efficient implementation.
    Returns cluster separation score and centers.
    """
    feat_flat = features.reshape(-1, features.size(-1))
    N, D = feat_flat.size()
    
    # Normalize features
    feat_norm = F.normalize(feat_flat, dim=1)
    
    # Compute pairwise distances efficiently
    dists = torch.cdist(feat_norm, feat_norm)
    
    # DBSCAN core point finding
    core_points = (dists <= eps).sum(1) >= 4  # MinPts = 4
    
    if not core_points.any():
        return 0.0, torch.zeros(2, D, device=features.device)  # Return dummy centers
    
    # Cluster assignment using core points
    labels = -torch.ones(N, device=features.device)
    current_label = 0
    
    for i in range(N):
        if labels[i] >= 0 or not core_points[i]:
            continue
            
        # Find points in eps neighborhood
        neighbors = dists[i] <= eps
        if core_points[i]:
            labels[neighbors] = current_label
            
            # Expand cluster
            to_check = neighbors.clone()
            while to_check.any():
                new_points = torch.zeros_like(to_check)
                for idx in torch.where(to_check)[0]:
                    if core_points[idx]:
                        curr_neighbors = dists[idx] <= eps
                        new_points = new_points | (curr_neighbors & (labels < 0))
                        labels[curr_neighbors] = current_label
                to_check = new_points
                
            current_label += 1
    
    # Ensure we have at least 2 clusters
    if current_label < 2:
        return 0.0, torch.zeros(2, D, device=features.device)
        
    # Calculate cluster centers
    centers = []
    for i in range(min(2, current_label)):  # Take at most 2 clusters
        mask = labels == i
        if mask.any():
            center = feat_norm[mask].mean(0)
            centers.append(F.normalize(center.unsqueeze(0), dim=1))
    
    # Pad to exactly 2 centers if needed
    while len(centers) < 2:
        centers.append(torch.zeros(1, D, device=features.device))
    
    centers = torch.cat(centers, dim=0)
    
    # Compute separation score
    separation = torch.cdist(centers, centers)[0, 1]  # Distance between first two centers
    
    return float(separation.item()), centers


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