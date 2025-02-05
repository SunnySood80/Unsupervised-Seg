import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure no inline display if you want to avoid crashes
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import h5py
import gym
from gym import spaces
from tqdm.auto import tqdm

# CUDA Optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def load_processed_samples(h5_path: str = '/home/sks6nv/Projects/PPO/cursor_ppo/processed_samples_color_only.h5') -> List[List[np.ndarray]]:
    """
    Load all preprocessed samples (mask + original + augmentations) from an HDF5 file.
    Returns:
        List of samples, where each sample = [gt_mask, original_image, aug1, aug2, ...]
    """
    print(f"Loading processed samples from {h5_path}")
    samples = []
    
    with h5py.File(h5_path, 'r') as f:
        sample_keys = [k for k in f.keys() if k.startswith('sample_')]
        print(f"Found {len(sample_keys)} samples")
        
        for sample_key in tqdm(sample_keys, desc="Loading samples"):
            sample_group = f[sample_key]
            sample_data = []
            
            # Load ground truth mask
            mask = sample_group['mask'][:].astype(np.float32)
            sample_data.append(mask)
            
            # Load original image
            original = sample_group['original'][:].astype(np.float32) / 255.0
            sample_data.append(original)
            
            # Load augmentations
            aug_idx = 0
            while f'aug_{aug_idx}' in sample_group:
                aug = sample_group[f'aug_{aug_idx}'][:].astype(np.float32) / 255.0
                sample_data.append(aug)
                aug_idx += 1
            
            samples.append(sample_data)
    
    print(f"Loaded {len(samples)} samples with {len(samples[0]) - 1} augmentations each")
    return samples

def visualize_map_with_augs(image_tensors: List[torch.Tensor],
                           heatmaps: List[torch.Tensor],
                           ground_truth: np.ndarray,
                           binary_mask: torch.Tensor,
                           reward: float,
                           save_path: str = None):
    """
    Creates a grid showing all augmentations and their maps.
    """
    n_images = len(image_tensors)
    n_cols = 4  # [Image | GT | Map | Overlay]
    n_rows = n_images
    
    plt.figure(figsize=(20, 5*n_rows))
    
    # Prepare binary mask
    mask = F.interpolate(
        binary_mask,
        size=heatmaps[0].shape,
        mode='nearest'
    ).squeeze().cpu()
    
    for idx, (img, hmap) in enumerate(zip(image_tensors, heatmaps)):
        # Convert to numpy - handle both 3D and 4D inputs
        if img.dim() == 4:  # (B,C,H,W)
            img = img.squeeze(0)  # Remove batch dimension
            
        # Ensure image is same size as heatmap
        H, W = hmap.shape[-2:]
        img_resized = F.interpolate(img.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        
        img_np = img_resized.detach().cpu().permute(1,2,0).numpy()
        hmap_np = hmap.detach().cpu().numpy()
        
        # Use binary mask
        hmap_np[mask == 0] = 0  # Make masked areas black
        
        # Original Image
        plt.subplot(n_rows, n_cols, idx*n_cols + 1)
        plt.imshow(img_np)
        plt.title(f"{'Original' if idx==0 else f'Augmentation {idx}'}")
        plt.axis("off")
        
        # Ground Truth
        plt.subplot(n_rows, n_cols, idx*n_cols + 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")
        
        # Heatmap
        plt.subplot(n_rows, n_cols, idx*n_cols + 3)
        plt.imshow(hmap_np, cmap='hot')
        plt.title(f"Segmentation Map (R={reward:.3f})")
        plt.axis("off")
        
        # Overlay
        plt.subplot(n_rows, n_cols, idx*n_cols + 4)
        plt.imshow(img_np)  # Using resized image
        plt.imshow(hmap_np, cmap='hot', alpha=0.6)
        plt.title("Overlay")
        plt.axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

print("Loading preprocessed samples...")
processed_samples = load_processed_samples()