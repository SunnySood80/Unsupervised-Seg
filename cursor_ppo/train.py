import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import gym
from gym import spaces
import time
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import socket
import math
import sys
from torch.utils.tensorboard import SummaryWriter

# Local imports
from load_data import load_processed_samples
from utils import (
    compute_feature_diversity,
    compute_consistency_reward,
    compute_boundary_strength,
    compute_local_coherence,
    compute_segmentation_map,
    visualize_map_with_augs
)
from feature_extract import FilterWeightingSegmenter
from custom_ppo import PPO  # Your PPO code

# At the top of train.py, after imports
binary_mask = torch.from_numpy(np.load('/home/sks6nv/Projects/PPO/binary_image.npy'))
binary_mask = F.interpolate(
    binary_mask.unsqueeze(0).unsqueeze(0).float(),
    size=(256, 256),
    mode='nearest'
).to('cuda')

def setup_ddp(rank, world_size):
    """Setup DDP with increased timeout and better error handling"""
    try:
        print(f"[train.py] setup_ddp called. rank={rank}, world_size={world_size}")
        
        # Use fixed port but different for each run
        base_port = 29500
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(base_port + rank)
        
        # Increase timeout and add NCCL configurations
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_SOCKET_TIMEOUT"] = "300"
        os.environ["NCCL_IB_TIMEOUT"] = "300"
        
        # Initialize process group with increased timeout
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{base_port}',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=5)  # Increased from 60 seconds to 5 minutes
        )
        
        # Set device and CUDA settings
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set NCCL to use the same device as PyTorch
        torch.cuda.set_device(rank)
        
        print(f"[train.py] init_process_group done for rank={rank}.")
        
    except Exception as e:
        print(f"[train.py] Error in setup_ddp for rank {rank}: {str(e)}")
        raise

def cleanup_ddp(rank):
    """Cleanup DDP process group"""
    try:
        if dist.is_initialized():
            print(f"[train.py] [Rank={rank}] cleanup_ddp: destroying process group...")
            dist.destroy_process_group()
            print(f"[train.py] [Rank={rank}] cleanup_ddp done.")
    except Exception as e:
        print(f"[train.py] [Rank={rank}] Error in cleanup_ddp: {str(e)}")

def compute_cross_set_consistency(features: torch.Tensor, batch_size: int) -> float:
    """
    Compute consistency using cross-attention between different images in the batch
    
    Args:
        features: Tensor of shape [B, C, H, W] - batch of feature maps
        batch_size: Size of image sets to compare
    """
    B, C, H, W = features.shape
    
    # Reshape features for attention
    flat_features = features.view(B, C, -1)  # [B, C, H*W]
    
    # Split batch into sets
    set_similarities = []
    for i in range(0, B, batch_size):
        set1 = flat_features[i:i+batch_size]  # First set of images
        
        # Compare with other sets
        for j in range(0, B, batch_size):
            if i == j:  # Skip self-comparison
                continue
                
            set2 = flat_features[j:j+batch_size]
            
            # Compute cross-attention between sets
            # Scale dot product attention
            attention = torch.bmm(set1.transpose(1, 2), set2) / math.sqrt(C)  # [batch_size, H*W, H*W]
            attention_weights = F.softmax(attention, dim=-1)
            
            # Apply attention to get attended features
            attended_features = torch.bmm(attention_weights, set2.transpose(1, 2))  # [batch_size, H*W, C]
            
            # Compute similarity between original and attended features
            similarity = F.cosine_similarity(
                set1.transpose(1, 2).reshape(-1, C),
                attended_features.reshape(-1, C)
            ).mean()
            
            set_similarities.append(similarity)
    
    return torch.stack(set_similarities).mean().item()

###############################################################################
#  FeatureWeightingEnv
###############################################################################
class FeatureWeightingEnv(gym.Env):
    def __init__(
        self,
        segmenter_model,
        processed_samples,
        device,
        batch_size=32,
        enable_render=False,
        render_patience=128,
        history_length=3,
        visualize=False,
        writer=None  # Add tensorboard writer parameter
    ):
        self.segmenter = segmenter_model
        self.processed_samples = processed_samples
        self.device = device
        self.batch_size = batch_size
        self.enable_render = enable_render
        self.render_patience = render_patience
        self.history_length = history_length
        self.visualize = visualize
        self.writer = writer  # Store tensorboard writer

        # Add these attributes for rendering
        self.current_sample_idx = 0
        self.cached_images = {}
        self._cache_images()  # Pre-cache images

        # Get feature dimension from a test forward pass
        with torch.no_grad():
            test_img = processed_samples[0][1]
            test_tensor = torch.from_numpy(test_img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            test_features = self.segmenter(test_tensor)
            self.feature_dim = test_features.shape[1]  # Get actual feature dimension

        # Update observation and action spaces with correct dimensions
        self.obs_shape = (self.feature_dim * self.history_length,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.obs_shape,
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.feature_dim,),
            dtype=np.float32
        )

        # Initialize weights with correct dimension
        self.current_weights = torch.ones(self.feature_dim, device=self.device) / self.feature_dim
        self.weight_history = [self.current_weights.clone() for _ in range(self.history_length)]
        
        # Initialize metric histories
        self.reward_history = []
        self.diversity_history = []
        self.consistency_history = []
        self.boundary_history = []
        self.coherence_history = []
        
        # Add step counter
        self.total_steps = 0

        # Pre-extract features
        print("Pre-extracting features...")
        self.all_features = {}
        for i in tqdm(range(len(processed_samples))):
            img_np = processed_samples[i][1]
            img_torch = torch.from_numpy(img_np).permute(2, 0, 1).float().to(device)
            with torch.no_grad():
                self.all_features[i] = self.segmenter(img_torch.unsqueeze(0))
        print("Feature extraction complete")

    def _cache_images(self):
        """Cache processed images for visualization"""
        for s_idx, sample_list in enumerate(self.processed_samples):
            self.cached_images[s_idx] = {}
            
            # Process original image
            original = sample_list[1]
            ten = torch.from_numpy(original).float()
            if ten.dim() == 3 and ten.shape[-1] == 3:
                ten = ten.permute(2,0,1)
            ten = F.interpolate(
                ten.unsqueeze(0),
                size=(256,256),
                mode='bilinear',
                align_corners=False
            ).to(self.device)
            self.cached_images[s_idx][0] = ten
            
            # Process augmented image if available
            if len(sample_list) > 2:
                aug = sample_list[2]
                ten = torch.from_numpy(aug).float()
                if ten.dim() == 3 and ten.shape[-1] == 3:
                    ten = ten.permute(2,0,1)
                ten = F.interpolate(
                    ten.unsqueeze(0),
                    size=(256,256),
                    mode='bilinear',
                    align_corners=False
                ).to(self.device)
                self.cached_images[s_idx][1] = ten

    def reset(self):
        """Reset environment state"""
        # Reset weights to uniform
        self.current_weights = torch.ones(self.feature_dim, device=self.device) / self.feature_dim
        self.weight_history = [self.current_weights.clone() for _ in range(self.history_length)]
        
        # Clear histories on reset
        self.reward_history = []
        self.diversity_history = []
        self.consistency_history = []
        self.boundary_history = []
        self.coherence_history = []
        
        # Reset step counter
        #self.total_steps = 0
        
        # Get initial observation
        obs = self._get_observation()
        return obs, {}

    def _extract_features(self, sample_indices):
        """Extract features with proper reshaping"""
        features = []
        for idx in sample_indices:
            feat = self.all_features[idx.item()]
            # Ensure features are properly shaped [B, C, H, W]
            if feat.dim() == 3:
                feat = feat.unsqueeze(0)
            features.append(feat)
        return torch.cat(features, dim=0)

    def _get_observation(self):
        """Get stacked weight history as observation"""
        obs = torch.cat(self.weight_history, dim=0)
        return obs.cpu().numpy()

    def step(self, action):
        """Execute environment step"""
        self.total_steps += 1
        
        # Update weights
        new_weights = torch.tensor(action, device=self.device)
        new_weights = F.softmax(new_weights, dim=0)
        self.current_weights = new_weights

        # Update history
        self.weight_history.pop(0)
        self.weight_history.append(self.current_weights.clone())

        # Get features for current batch
        with torch.amp.autocast('cuda'):
            sample_indices = torch.randint(0, len(self.processed_samples), (self.batch_size,))
            features = self._extract_features(sample_indices)
            
            # Compute individual rewards
            diversity_reward = compute_feature_diversity(features)
            boundary_reward = compute_boundary_strength(features)
            coherence_reward = compute_local_coherence(features)
            consistency_reward = compute_consistency_reward(features)

            # Combine rewards with weights
            reward = (
                0.4 * diversity_reward +
                0.3 * consistency_reward + # try .05
                0.2 * boundary_reward +
                0.1 * coherence_reward # try .015
            )

        # Store metrics
        self.reward_history.append(reward)
        self.diversity_history.append(diversity_reward)
        self.consistency_history.append(consistency_reward)
        self.boundary_history.append(boundary_reward)
        self.coherence_history.append(coherence_reward)

        # Render if needed
        if self.enable_render and (self.total_steps % self.render_patience == 0):
            self._render()

        # Log metrics to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('Rewards/diversity', diversity_reward, self.total_steps)
            self.writer.add_scalar('Rewards/consistency', consistency_reward, self.total_steps)
            self.writer.add_scalar('Rewards/boundary', boundary_reward, self.total_steps)
            self.writer.add_scalar('Rewards/coherence', coherence_reward, self.total_steps)
            self.writer.add_scalar('Rewards/total', reward, self.total_steps)
            
            # Log weight distribution
            self.writer.add_histogram('weights/distribution', self.current_weights, self.total_steps)

        # Get next observation
        obs = self._get_observation()
        done = False
        info = {}

        return obs, reward, done, info

    def _render(self):
        """Render current state visualization"""
        if not self.enable_render:
            return
            
        # Get current sample data
        sample_data = self.processed_samples[self.current_sample_idx]
        gt_mask = sample_data[0]
        
        # Get original and augmented images
        orig_img = self.cached_images[self.current_sample_idx][0]
        aug_img = self.cached_images[self.current_sample_idx].get(1, orig_img)
        aug_list = [aug_img]
        
        # Get feature maps
        with torch.no_grad():
            orig_feats = self.segmenter(orig_img)
            aug_feats_list = [self.segmenter(aug) for aug in aug_list]
        
        # Apply current weights
        alpha = self.current_weights.view(1, -1, 1, 1)
        weighted_orig_feats = orig_feats * alpha
        weighted_aug_feats = [feat * alpha for feat in aug_feats_list]
        
        # Compute segmentation maps with binary mask
        orig_map = compute_segmentation_map(weighted_orig_feats, binary_mask)
        aug_maps = [compute_segmentation_map(f, binary_mask) for f in weighted_aug_feats]
        
        # Use visualization function
        visualize_map_with_augs(
            image_tensors=[orig_img] + aug_list,
            heatmaps=[orig_map] + aug_maps,
            ground_truth=gt_mask,
            binary_mask=binary_mask,
            reward=self.reward_history[-1] if self.reward_history else 0,
            save_path=f'final_visuals/map_step_{self.total_steps}.png'
        )

        # Update sample index
        self.current_sample_idx = (self.current_sample_idx + 1) % len(self.processed_samples)

###############################################################################
#  train_ddp
###############################################################################
def train_ddp(rank, world_size, processed_samples):
    """Training function with better error handling"""
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        
        # Initialize tensorboard writer for rank 0
        writer = None
        if rank == 0:
            writer = SummaryWriter(f'runs/ppo_training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
            os.makedirs('final_visuals', exist_ok=True)
        
        # Reduce batch size and steps to lower memory usage
        per_gpu_batch_size = 1024  # Reduced from 256
        n_steps = 32 // world_size  # Reduced from 1024
        
        # Create environment
        feature_extractor = FilterWeightingSegmenter(pretrained=True).to(device)
        ddp_feature_extractor = DDP(
            feature_extractor,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
            static_graph=True
        )

        env = FeatureWeightingEnv(
            segmenter_model=ddp_feature_extractor,
            processed_samples=processed_samples,
            device=device,
            batch_size=per_gpu_batch_size,
            enable_render=(rank == 0),
            render_patience=10,
            visualize=True,
            writer=writer  # Pass writer to environment
        )

        # Create PPO agent with learning rate schedulers
        initial_pi_lr = 1e-5  # Keep current initial policy learning rate
        initial_vf_lr = 1e-3  # Keep current initial value function learning rate
        
        # Define learning rate scheduler parameters
        lr_decay_steps = 10_000  # Total steps after which learning rate reaches minimum
        min_lr = 1e-6  # Minimum learning rate
        
        def get_lr_lambda(initial_lr):
            def lr_lambda(step):
                # Linear decay from initial_lr to min_lr over lr_decay_steps
                decay_factor = 1.0 - (min(step, lr_decay_steps) / lr_decay_steps)
                return max(min_lr / initial_lr, decay_factor)
            return lr_lambda

        agent = PPO(
            env=env,
            n_steps=n_steps,
            batch_size=1024,
            policy_hidden_sizes=[2048, 1024, 512],  
            value_hidden_sizes=[512, 512, 256],    
            gamma=0.99,
            clip_ratio=0.15,
            pi_lr=initial_pi_lr,
            vf_lr=initial_vf_lr,
            train_pi_iters=15,
            train_v_iters=15,
            lam=0.95,
            target_kl=0.015,
            device=device
        )

        # Create schedulers after optimizer initialization
        pi_scheduler = torch.optim.lr_scheduler.LambdaLR(
            agent.pi_optimizer, 
            lr_lambda=get_lr_lambda(initial_pi_lr)
        )
        
        vf_scheduler = torch.optim.lr_scheduler.LambdaLR(
            agent.vf_optimizer,
            lr_lambda=get_lr_lambda(initial_vf_lr)
        )

        # Move networks to GPU and wrap in DDP
        agent.actor = DDP(agent.actor, device_ids=[rank], find_unused_parameters=False, static_graph=True)
        agent.critic = DDP(agent.critic, device_ids=[rank], find_unused_parameters=False, static_graph=True)

        if rank == 0:
            print(f"\nStarting training with {world_size} GPUs")
            print(f"Steps per GPU: {n_steps}")
            print(f"Batch size per GPU: {per_gpu_batch_size}\n")

        # Training loop with better synchronization
        start_time = time.time()
        total_frames = 0
        total_timesteps = 10_000
        episode_count = 0
        
        # Initialize metrics
        metrics = None
        
        while total_frames < total_timesteps:
            try:
                # Synchronize before training step
                torch.cuda.synchronize(device)
                dist.barrier()
                
                if rank == 0:
                    pbar = tqdm(total=n_steps, desc=f"Episode {episode_count}", leave=False)
                
                # Run training step
                metrics = agent.learn(total_timesteps=n_steps, pbar=pbar if rank == 0 else None)
                
                # Synchronize after training step
                torch.cuda.synchronize(device)
                dist.barrier()
                
                torch.cuda.empty_cache()
                
                if metrics is not None:
                    total_frames += n_steps
                    episode_count += 1
                    
                    # Get average reward and value loss for this episode
                    rewards = agent.buffer.rews
                    if isinstance(rewards, torch.Tensor):
                        rewards = rewards.cpu().numpy()
                    elif isinstance(rewards, list):
                        rewards = np.array(rewards)
                    
                    avg_reward = float(np.mean(rewards))
                    value_loss = float(metrics['v_loss'])
                    
                    # Update learning rates based on steps
                    pi_scheduler.step(total_frames)
                    vf_scheduler.step(total_frames)
                    
                    # Get current learning rates
                    pi_lr = agent.pi_optimizer.param_groups[0]['lr']
                    vf_lr = agent.vf_optimizer.param_groups[0]['lr']
                    
                    # Print metrics only from rank 0
                    if rank == 0:
                        elapsed_time = time.time() - start_time
                        fps = total_frames / elapsed_time
                        
                        # Print metrics
                        print("\n" + "="*50)
                        print(f"Episode {episode_count} Summary:")
                        print("="*50)
                        print(f"Steps this episode: {n_steps}")
                        print(f"Total steps: {total_frames}")
                        print(f"FPS: {fps:.2f}")
                        print(f"Time elapsed: {datetime.timedelta(seconds=int(elapsed_time))}")
                        print("\nTraining Metrics:")
                        print(f"Policy Loss: {metrics['pi_loss']:.4f}")
                        print(f"Value Loss: {metrics['v_loss']:.4f}")
                        print(f"Policy KL: {metrics['policy_kl']:.4f}")
                        print(f"Clip Fraction: {metrics['clip_frac']:.4f}")
                        print("\nReward Statistics:")
                        print(f"Average Reward: {avg_reward:.4f}")
                        print(f"Value Loss: {value_loss:.4f}")
                        print("\nLearning Rates:")
                        print(f"Policy LR: {pi_lr:.2e}")
                        print(f"Value LR: {vf_lr:.2e}")
                        print("="*50 + "\n")
                        
                        # Ensure printing
                        sys.stdout.flush()

                        # Log to tensorboard
                        writer.add_scalar('Training/policy_loss', metrics['pi_loss'], total_frames)
                        writer.add_scalar('Training/value_loss', metrics['v_loss'], total_frames)
                        
                        # Log reward statistics
                        rewards = agent.buffer.rews
                        if isinstance(rewards, torch.Tensor):
                            rewards = rewards.cpu().numpy()
                        
                        writer.add_scalar('Rewards/mean', avg_reward, total_frames)
                        writer.add_scalar('Rewards/value_loss', value_loss, total_frames)
                        
                        # Log training progress
                        writer.add_scalar('Progress/fps', fps, total_frames)
                        writer.add_scalar('Progress/episodes', episode_count, total_frames)
                        
                        # Log network gradients (summarized statistics only)
                        total_grad_norm_actor = 0
                        total_grad_norm_critic = 0
                        
                        for param in agent.actor.parameters():
                            if param.grad is not None:
                                total_grad_norm_actor += param.grad.norm().item()
                        
                        for param in agent.critic.parameters():
                            if param.grad is not None:
                                total_grad_norm_critic += param.grad.norm().item()
                                
                        writer.add_scalar('Gradients/actor_total_norm', total_grad_norm_actor, total_frames)
                        writer.add_scalar('Gradients/critic_total_norm', total_grad_norm_critic, total_frames)
                        
                        # Log memory usage
                        writer.add_scalar('System/gpu_memory_allocated', torch.cuda.memory_allocated(device), total_frames)
                        writer.add_scalar('System/gpu_memory_cached', torch.cuda.memory_reserved(device), total_frames)

                        # Add learning rate logging to tensorboard
                        writer.add_scalar('LearningRates/policy_lr', pi_lr, total_frames)
                        writer.add_scalar('LearningRates/value_lr', vf_lr, total_frames)

            except RuntimeError as e:
                print(f"\nError on rank {rank}: {str(e)}")
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            except Exception as e:
                print(f"\nUnexpected error on rank {rank}: {str(e)}")
                raise

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise
    finally:
        if rank == 0 and writer is not None:
            writer.close()
        cleanup_ddp(rank)
        torch.cuda.empty_cache()

def train_custom_ppo():
    """Main training function with better cleanup"""
    try:
        # Kill any existing process groups and clear environment
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Clear environment variables
        for key in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']:
            if key in os.environ:
                del os.environ[key]
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Create visualization directory
        os.makedirs('final_visuals', exist_ok=True)
        
        print("[main] Loading preprocessed samples...")
        processed_samples = load_processed_samples()
        print(f"[main] loaded {len(processed_samples)} samples")

        # Use all available GPUs
        world_size = torch.cuda.device_count()
        print(f"[main] Using {world_size} GPUs")
        
        # Start training processes
        mp.spawn(
            train_ddp,
            args=(world_size, processed_samples),
            nprocs=world_size,
            join=True
        )
    
    except Exception as e:
        print(f"Error in train_custom_ppo: {str(e)}")
        raise
    finally:
        # Final cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_custom_ppo()
