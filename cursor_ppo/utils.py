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


def compute_feature_diversity(features: torch.Tensor) -> float:
    """
    Compute how diverse the features are: 1 - mean cosine similarity.
    """
    features_flat = F.normalize(features.view(-1, features.size(-1)), dim=1)
    similarity = torch.mm(features_flat, features_flat.t())
    similarity.fill_diagonal_(0)
    return float(1.0 - similarity.mean().item())

def compute_consistency_reward(original_feats: torch.Tensor, aug_feats_list: List[torch.Tensor]) -> float:
    """
    Compute consistency reward between original features and its augmentations.
    """
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

def compute_segmentation_map(w_feats: torch.Tensor) -> torch.Tensor:
    """
    Compute binary segmentation map from weighted features.
    """
    feat_flat = w_feats.squeeze(0).permute(1,2,0).reshape(-1, w_feats.size(1))
    feat_flat = F.normalize(feat_flat, dim=1)
    
    N, D = feat_flat.size()
    idx = torch.randperm(N, device=w_feats.device)[:2]
    centroids = feat_flat[idx].clone()
    
    dists = torch.cdist(feat_flat, centroids)
    clusters = dists.argmin(dim=1)
    
    seg_map = clusters.reshape(w_feats.shape[2], w_feats.shape[3]).float()
    seg_map = seg_map - seg_map.min()
    seg_map = seg_map / (seg_map.max() + 1e-8)
    
    return seg_map