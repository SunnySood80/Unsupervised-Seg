import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch.distributions import Normal
import time
import gc

# For conditional autocast (if needed)
from contextlib import nullcontext

##############################################
# Device settings:
# - default_device: where networks, optimizers, and environment interaction run (CPU).
# - processing_device: used for heavy batch computations (GPU if available).
##############################################
default_device = torch.device('cpu')
processing_device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print(f"Using processing device: {processing_device} (networks default to CPU)")

##############################################
# PPOBuffer: Stores all experience data on CPU (with pinned memory for large tensors).
# When get_minibatch() is called, data is transferred to the processing_device (GPU) in small batches.
##############################################
@dataclass
class PPOBuffer:
    def __init__(self, n_steps: int, obs_dim: int, act_dim: int, gamma: float, lam: float,
                 device: torch.device):
        self.gamma = gamma
        self.lam = lam
        self.n_steps = n_steps
        self.obs_dim = obs_dim  # Store observation dimension
        self.act_dim = act_dim  # Store action dimension
        self.device = device
        self.reset()

    def reset(self):
        """Reset buffer counters and tensors"""
        self.ptr = 0
        self.path_start_idx = 0

        # Initialize tensors with proper dimensions
        self.obs = torch.zeros((self.n_steps, self.obs_dim), device=self.device)
        self.acts = torch.zeros((self.n_steps, self.act_dim), device=self.device)
        self.rews = torch.zeros(self.n_steps, device=self.device)
        self.dones = torch.zeros(self.n_steps, device=self.device)
        self.vals = torch.zeros(self.n_steps, device=self.device)
        self.logprobs = torch.zeros(self.n_steps, device=self.device)
        self.returns = torch.zeros(self.n_steps, device=self.device)
        self.advantages = torch.zeros(self.n_steps, device=self.device)
        self.vals_old = torch.zeros(self.n_steps, device=self.device)

    def store(self, obs: torch.Tensor, act: torch.Tensor, rew: float, done: bool, val: float, logprob: float):
        # Reset pointer if buffer is full
        if self.ptr >= self.n_steps:
            self.ptr = 0
            self.path_start_idx = 0
            
        # Store on the specified device
        self.obs[self.ptr] = obs.to(self.device)
        self.acts[self.ptr] = act.to(self.device)
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = done
        self.vals[self.ptr] = val
        self.vals_old[self.ptr] = val
        self.logprobs[self.ptr] = logprob
        self.ptr += 1

    def finish_path(self, last_val: float = 0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat([self.rews[path_slice], torch.tensor([last_val], device=self.device)])
        vals = torch.cat([self.vals[path_slice], torch.tensor([last_val], device=self.device)])
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute returns
        self.returns[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x: torch.Tensor, discount: float) -> torch.Tensor:
        disc_cumsum = torch.zeros_like(x)
        disc_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            disc_cumsum[t] = x[t] + discount * disc_cumsum[t+1]
        return disc_cumsum

    def prepare_buffer(self):
        """Normalize advantages."""
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    def get_minibatch(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return a minibatch of data on the processing device."""
        return {
            'obs': self.obs[indices].to(self.processing_device),
            'acts': self.acts[indices].to(self.processing_device),
            'returns': self.returns[indices].to(self.processing_device),
            'advantages': self.advantages[indices].to(self.processing_device),
            'logprobs': self.logprobs[indices].to(self.processing_device),
            'vals_old': self.vals_old[indices].to(self.processing_device)
        }

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data from buffer and prepare it for updates."""
        # Make sure we've used all data
        assert self.ptr == self.n_steps, "Buffer not full"
        
        # Normalize advantages
        self.prepare_buffer()
        
        # Return all data as a dictionary of tensors
        return {
            'obs': self.obs,
            'acts': self.acts,
            'returns': self.returns,
            'advantages': self.advantages,
            'logprobs': self.logprobs,
            'vals_old': self.vals_old
        }

##############################################
# Actor and Critic Networks (remain unchanged).
##############################################
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True)
            ])
            in_size = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(in_size, act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        self._init_weights()
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.0)
    def forward(self, obs: torch.Tensor) -> Normal:
        net_out = self.net(obs)
        mu = self.mu(net_out)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return Normal(mu, std)

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True)
            ])
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

##############################################
# PPO Class: Networks and optimizers default to CPU.
# Updates are performed by temporarily moving networks and mini-batches to GPU.
##############################################
class PPO:
    def __init__(self, env, n_steps=2048, batch_size=64, gamma=0.99, clip_ratio=0.2,
                 pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
                 lam=0.95, target_kl=0.01, policy_hidden_sizes=(64,64),
                 value_hidden_sizes=(64,64), device=None):
        
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.target_kl = target_kl
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get dimensions from environment
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Initialize networks directly on correct device
        self.actor = Actor(self.obs_dim, self.act_dim, policy_hidden_sizes).to(self.device)
        self.critic = Critic(self.obs_dim, value_hidden_sizes).to(self.device)
        
        # Initialize optimizers
        self.pi_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.vf_optimizer = torch.optim.Adam(self.critic.parameters(), lr=vf_lr)
        
        # Initialize buffer
        self.buffer = PPOBuffer(
            n_steps=n_steps,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            gamma=gamma,
            lam=lam,
            device=self.device
        )
        
        # Initialize CUDA grad scaler
        self.scaler = torch.amp.GradScaler()
        
        # Initialize metrics
        self.total_timesteps = 0
        self.metrics = {
            'rewards': [],
            'pi_loss': [],
            'v_loss': [],
            'policy_kl': [],
            'clip_frac': []
        }

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action using the policy network"""
        with torch.no_grad():
            # Move observation to processing device
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            
            # Get action distribution and value
            dist = self.actor(obs_tensor)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            value = self.critic(obs_tensor)
            
            # Move everything to CPU before converting to numpy
            action_cpu = action.cpu()
            value_cpu = value.cpu()
            logp_cpu = logp.cpu()
            
            return action_cpu.numpy(), value_cpu.item(), logp_cpu.item()

    def learn(self, total_timesteps: int, pbar=None):
        metrics = {
            'pi_loss': [],
            'v_loss': [],
            'policy_kl': [],
            'clip_frac': [],
            'rewards': []
        }
        
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:
            # Reset buffer at the start of each iteration
            self.buffer.reset()
            
            batch_rewards = []
            obs, _ = self.env.reset()
            episode_return = 0
            
            # Collect experience
            for t in range(self.n_steps):
                # Get action from policy
                action, value, logp = self.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                episode_return += reward
                
                # Store experience in buffer
                self.buffer.store(
                    torch.as_tensor(obs, device=self.device),
                    torch.as_tensor(action, device=self.device),
                    reward,
                    done,
                    value,
                    logp
                )
                
                obs = next_obs
                
                if done or (t == self.n_steps - 1):
                    last_val = 0 if done else self.select_action(obs)[1]
                    self.buffer.finish_path(last_val)
                    if done:
                        batch_rewards.append(episode_return)
                        obs, _ = self.env.reset()
                        episode_return = 0
            
            # Update policy and value function
            update_info = self.update()
            if update_info is not None:
                for k, v in update_info.items():
                    if k in metrics:
                        metrics[k].append(v)
            
            if pbar is not None:
                pbar.update(1)
            
            timesteps_so_far += 1
        
        if pbar is not None:
            pbar.close()
        
        # Return average metrics
        return {k: np.mean(v) for k, v in metrics.items() if len(v) > 0}

    def _compute_returns(self, rews: torch.Tensor, vals: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(rews)
        returns[-1] = vals[-1]
        for t in reversed(range(len(rews) - 1)):
            returns[t] = rews[t] + self.gamma * (1 - dones[t]) * returns[t + 1]
        return returns

    def _compute_advantages(self, returns: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        advantages = returns - vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _update_policy(self, obs: torch.Tensor, act: torch.Tensor, advantage: torch.Tensor, logp_old: torch.Tensor):
        dist = self.actor(obs)
        logp = dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantage
        loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()
        
        approx_kl = (logp_old - logp).mean().item()
        ent = dist.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        
        self.pi_optimizer.zero_grad()
        self.scaler.scale(loss_pi).backward()
        self.scaler.step(self.pi_optimizer)
        self.scaler.update()
        
        return {
            'pi_loss': loss_pi.item(),
            'policy_kl': approx_kl,
            'clip_frac': clipfrac
        }

    def save(self, save_path: str):
        # Move models to CPU to avoid GPU OOM during saving
        self.actor.to(self.device)
        self.critic.to(self.device)
        actor_state = {k: v.half() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.half() for k, v in self.critic.state_dict().items()}
        last_n = 1000
        metrics_summary = {k: v[-last_n:] if len(v) > last_n else v for k, v in self.metrics.items()}
        torch.save({
            'actor_state_dict': actor_state,
            'critic_state_dict': critic_state,
            'config': {
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'clip_ratio': self.clip_ratio,
                'train_pi_iters': self.train_pi_iters,
                'train_v_iters': self.train_v_iters,
                'lam': self.lam,
                'target_kl': self.target_kl
            },
            'total_timesteps': self.total_timesteps,
            'metrics_summary': metrics_summary,
            'final_performance': {
                'mean_reward': np.mean(self.metrics['rewards'][-100:]) if self.metrics.get('rewards') else 0,
                'mean_loss': np.mean(self.metrics['pi_loss'][-100:]) if self.metrics.get('pi_loss') else 0
            }
        }, save_path)

    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        actor_state = {k: v.float() for k, v in checkpoint['actor_state_dict'].items()}
        critic_state = {k: v.float() for k, v in checkpoint['critic_state_dict'].items()}
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)
        config = checkpoint['config']
        self.n_steps = config['n_steps']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.clip_ratio = config['clip_ratio']
        self.train_pi_iters = config['train_pi_iters']
        self.train_v_iters = config['train_v_iters']
        self.lam = config['lam']
        self.target_kl = config['target_kl']
        if 'metrics_summary' in checkpoint:
            self.metrics = checkpoint['metrics_summary']
        if 'total_timesteps' in checkpoint:
            self.total_timesteps = checkpoint['total_timesteps']
        return checkpoint.get('final_performance', {})

    def update(self):
        """Update policy and value function"""
        try:
            data = self.buffer.get()
            all_indices = torch.randperm(self.buffer.ptr, device=self.device)
            
            pi_info = dict(kl=0, ent=0, cf=0)
            v_loss_avg = 0
            loss_pi = None
            
            # Policy updates
            for i in range(self.train_pi_iters):
                for start in range(0, len(all_indices), self.batch_size):
                    idx = all_indices[start:start + self.batch_size]
                    batch = {k: v[idx] for k, v in data.items()}
                    
                    with torch.amp.autocast(device_type='cuda'):
                        # Compute policy loss
                        dist = self.actor(batch['obs'])
                        logp = dist.log_prob(batch['acts']).sum(-1)
                        ratio = torch.exp(logp - batch['logprobs'])
                        
                        # Clipped surrogate objective
                        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * batch['advantages']
                        loss_pi = -(torch.min(ratio * batch['advantages'], clip_adv)).mean()
                        
                        # Info for logging
                        approx_kl = (batch['logprobs'] - logp).mean().item()
                        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
                        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                        
                        # Update policy
                        self.pi_optimizer.zero_grad()
                        self.scaler.scale(loss_pi).backward()
                        self.scaler.step(self.pi_optimizer)
                        self.scaler.update()
                        
                        pi_info['kl'] += approx_kl
                        pi_info['cf'] += clipfrac
                    
                    if approx_kl > 1.5 * self.target_kl:
                        break
                
                if approx_kl > 1.5 * self.target_kl:
                    break
            
            # Value function updates
            for _ in range(self.train_v_iters):
                for start in range(0, len(all_indices), self.batch_size):
                    idx = all_indices[start:start + self.batch_size]
                    batch = {k: v[idx] for k, v in data.items()}
                    
                    with torch.amp.autocast(device_type='cuda'):
                        # Compute value loss
                        v = self.critic(batch['obs'])
                        v_loss = F.mse_loss(v, batch['returns'])
                        
                        # Update value function
                        self.vf_optimizer.zero_grad()
                        self.scaler.scale(v_loss).backward()
                        self.scaler.step(self.vf_optimizer)
                        self.scaler.update()
                        
                        v_loss_avg += v_loss.item()
            
            # Compute average metrics
            num_policy_updates = (i + 1) * ((len(all_indices) + self.batch_size - 1) // self.batch_size)
            num_value_updates = self.train_v_iters * ((len(all_indices) + self.batch_size - 1) // self.batch_size)
            
            pi_info['kl'] /= max(1, num_policy_updates)
            pi_info['cf'] /= max(1, num_policy_updates)
            v_loss_avg /= max(1, num_value_updates)
            
            return {
                'pi_loss': loss_pi.item() if loss_pi is not None else 0,
                'v_loss': v_loss_avg,
                'policy_kl': pi_info['kl'],
                'clip_frac': pi_info['cf']
            }
        
        except Exception as e:
            print(f"Error in update: {str(e)}")
            return None
        
        finally:
            torch.cuda.empty_cache()