import torch
import torch.nn.functional as F
import numpy as np
import os
import gc
import gym
from gym import spaces
import csv
import time
from typing import List, Dict, Optional, Tuple
from torchvision.models import resnet50
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add these imports
from load_data import load_processed_samples, visualize_map_with_augs
from utils import (MomentumEncoder, ProjectionHead, compute_contrastive_loss,
                  compute_cluster_separation_fast, compute_feature_diversity,
                  compute_consistency_reward, compute_boundary_strength,
                  compute_local_coherence, compute_segmentation_map)
from feature_extract import DDPFeatureExtractor
from custom_ppo import PPO

device = 'cuda:0'

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup_ddp():
    dist.destroy_process_group()

class FeatureWeightingEnv(gym.Env):
    def __init__(self, segmenter_model, processed_samples, device, batch_size=32, enable_render=False, render_patience=128, history_length=3):
        super().__init__()
        self.segmenter = segmenter_model
        self.processed_samples = processed_samples
        self.device = device
        self.batch_size = batch_size
        self.enable_render = enable_render
        self.render_patience = render_patience
        self.history_length = history_length
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(256,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256 * history_length,), dtype=np.float32)
        
        # Initialize state
        self.current_weights = None
        self.current_features = None
        self.weight_history = []
        self.steps_since_render = 0
        self.total_steps = 0
        
        # Metrics history
        self.reward_history = []
        self.diversity_history = []
        self.consistency_history = []
        self.boundary_history = []
        self.coherence_history = []
        
    def reset(self):
        # Initialize weights uniformly
        self.current_weights = torch.ones(256, device=self.device) / 256
        self.weight_history = [self.current_weights.clone() for _ in range(self.history_length)]
        
        # Get initial features
        sample_indices = torch.randint(0, len(self.processed_samples), (self.batch_size,))
        self.current_features = self._extract_features(sample_indices)
        
        # Reset counters
        self.steps_since_render = 0
        
        # Return initial observation
        return self._get_observation()
    
    def _extract_features(self, sample_indices):
        # Extract features for the selected samples
        features = []
        for idx in sample_indices:
            img = self.processed_samples[idx]['image'].to(self.device)
            with torch.no_grad():
                feat = self.segmenter(img)
            features.append(feat)
        return torch.cat(features, dim=0)
    
    def _get_observation(self):
        # Concatenate weight history
        obs = torch.cat(self.weight_history, dim=0)
        return obs.cpu().numpy()
    
    def step(self, action):
        self.total_steps += 1
        self.steps_since_render += 1
        
        # Update weights (ensure they sum to 1)
        new_weights = torch.tensor(action, device=self.device)
        new_weights = F.softmax(new_weights, dim=0)
        self.current_weights = new_weights
        
        # Update weight history
        self.weight_history.pop(0)
        self.weight_history.append(self.current_weights.clone())
        
        # Sample new batch
        sample_indices = torch.randint(0, len(self.processed_samples), (self.batch_size,))
        self.current_features = self._extract_features(sample_indices)
        
        # Compute rewards
        diversity_reward = compute_feature_diversity(self.current_features)
        consistency_reward = compute_consistency_reward(self.current_features)
        boundary_reward = compute_boundary_strength(self.current_features)
        coherence_reward = compute_local_coherence(self.current_features)
        
        # Combine rewards
        reward = (
            0.4 * diversity_reward +
            0.3 * consistency_reward +
            0.2 * boundary_reward +
            0.1 * coherence_reward
        )
        
        # Update histories
        self.reward_history.append(reward.item())
        self.diversity_history.append(diversity_reward.item())
        self.consistency_history.append(consistency_reward.item())
        self.boundary_history.append(boundary_reward.item())
        self.coherence_history.append(coherence_reward.item())
        
        # Render if enabled
        if self.enable_render and self.steps_since_render >= self.render_patience:
            self._render()
            self.steps_since_render = 0
        
        # Get observation
        obs = self._get_observation()
        
        # Never done
        done = False
        
        return obs, reward.item(), done, {}
    
    def _render(self):
        # Plot training curves
        if len(self.reward_history) > 0:
            print(f"\nStep {self.total_steps}")
            print(f"Recent rewards: {np.mean(self.reward_history[-100:]):.3f}")
            print(f"Recent diversity: {np.mean(self.diversity_history[-100:]):.3f}")
            print(f"Recent consistency: {np.mean(self.consistency_history[-100:]):.3f}")
            print(f"Recent boundary: {np.mean(self.boundary_history[-100:]):.3f}")
            print(f"Recent coherence: {np.mean(self.coherence_history[-100:]):.3f}")

def train_ddp(rank, world_size, processed_samples):
    setup_ddp(rank, world_size)
    
    # Memory optimization settings
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Training parameters
    VERBOSE = True
    BATCH_SIZE = 2048
    TOTAL_TIMESTEPS = 1_000_000
    CLEANUP_INTERVAL = 3000

    print(f"\nInitializing components on rank {rank}...")
    
    # Initialize feature extractor with DDP
    feature_extractor = DDPFeatureExtractor(world_size=3, start_gpu=1)
    
    # Initialize environment
    env = FeatureWeightingEnv(
        segmenter_model=feature_extractor,
        processed_samples=processed_samples,
        device=torch.device(f'cuda:{rank}'),
        batch_size=BATCH_SIZE,
        enable_render=rank == 0,  # Only render on main process
        render_patience=512,
        history_length=3
    )

    # Initialize PPO agent
    agent = PPO(
        env=env,
        n_steps=BATCH_SIZE,
        batch_size=2048,
        policy_hidden_sizes=[768, 768, 512, 256],
        value_hidden_sizes=[512, 512, 256],
        gamma=0.85,
        clip_ratio=0.15,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=15,
        train_v_iters=15,
        lam=0.97,
        target_kl=0.01,
        default_device=torch.device('cpu'),
        processing_device=torch.device(f'cuda:{rank}')
    )

    # Wrap agent's policy and value function with DDP
    agent.policy = DDP(
        agent.policy,
        device_ids=[rank],
        output_device=rank
    )
    agent.value_function = DDP(
        agent.value_function,
        device_ids=[rank],
        output_device=rank
    )

    try:
        agent.learn(env, total_timesteps=TOTAL_TIMESTEPS)
    finally:
        cleanup_ddp()

def train_custom_ppo():
    print("Loading preprocessed samples...")
    processed_samples = load_processed_samples()
    
    world_size = 4  # Using all 4 GPUs
    mp.spawn(
        train_ddp,
        args=(world_size, processed_samples),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    train_custom_ppo()