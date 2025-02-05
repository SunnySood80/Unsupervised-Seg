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
                 device: torch.device, processing_device: torch.device):
        self.gamma = gamma
        self.lam = lam
        self.n_steps = n_steps
        self.device = device
        self.processing_device = processing_device
        self.ptr = 0
        self.path_start_idx = 0

        # Initialize tensors
        self.obs = torch.zeros((n_steps, obs_dim), device=device)
        self.acts = torch.zeros((n_steps, act_dim), device=device)
        self.rews = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.vals = torch.zeros(n_steps, device=device)
        self.logprobs = torch.zeros(n_steps, device=device)
        self.returns = torch.zeros(n_steps, device=device)
        self.advantages = torch.zeros(n_steps, device=device)
        self.vals_old = torch.zeros(n_steps, device=device)

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
    def __init__(
        self,
        env,
        n_steps: int = 2048,
        batch_size: int = 64,
        policy_hidden_sizes: List[int] = [64, 64],
        value_hidden_sizes: List[int] = [64, 64],
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        lam: float = 0.97,
        target_kl: float = 0.01,
        default_device: torch.device = torch.device('cpu'),
        processing_device: torch.device = torch.device('cuda:0')
    ):
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.target_kl = target_kl
        self.default_device = default_device
        self.processing_device = processing_device

        self.metrics = {'pi_loss': [], 'v_loss': [], 'approx_kl': [], 'entropy': [], 'clipfrac': [], 'rewards': []}
        self.total_timesteps = 0

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        # Initialize networks on CPU
        self.actor = Actor(obs_dim, act_dim, policy_hidden_sizes).to(default_device)
        self.critic = Critic(obs_dim, value_hidden_sizes).to(default_device)
        self.pi_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.vf_optimizer = torch.optim.Adam(self.critic.parameters(), lr=vf_lr)
        
        # Initialize buffer
        self.buffer = PPOBuffer(
            n_steps=n_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            gamma=gamma,
            lam=lam,
            device=default_device,
            processing_device=processing_device
        )
        
        # Initialize scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if processing_device.type == 'cuda' else None

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.default_device)
            dist = self.actor(obs_tensor)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            value = self.critic(obs_tensor)
            return action.numpy(), value.item(), logp.item()

    def learn(self, total_timesteps: int):
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:
            # Reset buffer at the start of each iteration
            self.buffer.reset()
            
            batch_rewards = []
            obs, _ = self.env.reset()
            episode_return = 0
            
            # Collect experience
            for t in range(self.n_steps):
                action, value, logp = self.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                episode_return += reward
                
                # Store experience in buffer (on CPU)
                self.buffer.store(
                    torch.as_tensor(obs, device=self.default_device),
                    torch.as_tensor(action, device=self.default_device),
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
            
            timesteps_so_far += self.n_steps
            self.total_timesteps += self.n_steps
            
            if batch_rewards:
                self.metrics['rewards'].extend(batch_rewards)
            
            self.update()

    def update(self):
        # Move networks to GPU for updates
        self.actor.to(self.processing_device)
        self.critic.to(self.processing_device)
        
        try:
            data = self.buffer.get()
            all_indices = torch.randperm(self.buffer.ptr, device=self.processing_device)
            
            # Policy updates
            for i in range(self.train_pi_iters):
                for start in range(0, len(all_indices), self.batch_size):
                    idx = all_indices[start:start + self.batch_size]
                    minibatch = {k: v[idx] for k, v in data.items()}
                    
                    with torch.cuda.amp.autocast():
                        loss_pi, pi_info = self.compute_loss_pi(minibatch)
                    
                    self.pi_optimizer.zero_grad()
                    self.scaler.scale(loss_pi).backward()
                    self.scaler.step(self.pi_optimizer)
                    self.scaler.update()
                    
                    if pi_info['kl'] > 1.5 * self.target_kl:
                        break
            
            # Value function updates
            for _ in range(self.train_v_iters):
                for start in range(0, len(all_indices), self.batch_size):
                    idx = all_indices[start:start + self.batch_size]
                    minibatch = {k: v[idx] for k, v in data.items()}
                    
                    with torch.cuda.amp.autocast():
                        loss_v = self.compute_loss_v(minibatch)
                    
                    self.vf_optimizer.zero_grad()
                    self.scaler.scale(loss_v).backward()
                    self.scaler.step(self.vf_optimizer)
                    self.scaler.update()
        
        finally:
            # Move networks back to CPU
            self.actor.to(self.default_device)
            self.critic.to(self.default_device)
            torch.cuda.empty_cache()

    def compute_loss_pi(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        obs, acts, advantages, logp_old = data['obs'], data['acts'], data['advantages'], data['logprobs']
        
        dist = self.actor(obs)
        logp = dist.log_prob(acts).sum(-1)
        ratio = torch.exp(logp - logp_old)
        
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()
        
        approx_kl = (logp_old - logp).mean().item()
        ent = dist.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        
        return loss_pi, dict(kl=approx_kl, ent=ent, cf=clipfrac)

    def compute_loss_v(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs, ret = data['obs'], data['returns']
        return ((self.critic(obs) - ret)**2).mean()

    def update(self):
        # Prepare data
        self.buffer.prepare_buffer()
        all_indices = torch.randperm(self.n_steps)
        update_metrics = {'pi_loss': [], 'v_loss': [], 'approx_kl': [], 'entropy': [], 'clipfrac': []}
        
        # Move networks to GPU for updates
        self.actor.to(self.processing_device)
        self.critic.to(self.processing_device)
        
        try:
            # Policy updates
            for i in range(self.train_pi_iters):
                for start in range(0, self.n_steps, self.batch_size):
                    mb_inds = all_indices[start:start + self.batch_size]
                    batch = self.buffer.get_minibatch(mb_inds)
                    
                    loss_pi, pi_info = self.compute_loss_pi(batch)
                    update_metrics['pi_loss'].append(loss_pi.item())
                    update_metrics['approx_kl'].append(pi_info['kl'])
                    update_metrics['entropy'].append(pi_info['ent'])
                    update_metrics['clipfrac'].append(pi_info['cf'])
                    
                    self.pi_optimizer.zero_grad(set_to_none=True)
                    
                    if self.scaler is not None:
                        # Use gradient scaling
                        self.scaler.scale(loss_pi).backward()
                        # Move gradients to CPU for clipping
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                param.grad = param.grad.cpu()
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        # Move back to GPU
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                param.grad = param.grad.to(self.processing_device)
                        # Step optimizer
                        self.scaler.step(self.pi_optimizer)
                        self.scaler.update()
                    else:
                        loss_pi.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        self.pi_optimizer.step()
                    
                    if pi_info['kl'] > 1.5 * self.target_kl:
                        break
                if pi_info['kl'] > 1.5 * self.target_kl:
                    break
            
            # Value function updates
            for _ in range(self.train_v_iters):
                for start in range(0, self.n_steps, self.batch_size):
                    mb_inds = all_indices[start:start + self.batch_size]
                    batch = self.buffer.get_minibatch(mb_inds)
                    loss_v = self.compute_loss_v(batch)
                    update_metrics['v_loss'].append(loss_v.item())
                    
                    self.vf_optimizer.zero_grad(set_to_none=True)
                    if self.scaler is not None:
                        self.scaler.scale(loss_v).backward()
                        # Move gradients to CPU for clipping
                        for param in self.critic.parameters():
                            if param.grad is not None:
                                param.grad = param.grad.cpu()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        # Move back to GPU
                        for param in self.critic.parameters():
                            if param.grad is not None:
                                param.grad = param.grad.to(self.processing_device)
                        self.scaler.step(self.vf_optimizer)
                        self.scaler.update()
                    else:
                        loss_v.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        self.vf_optimizer.step()
        
        finally:
            # Move networks back to CPU
            self.actor.to(self.default_device)
            self.critic.to(self.default_device)
            
        return {k: np.mean(v) for k, v in update_metrics.items() if v}
    def learn(self, total_timesteps: int):
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:
            batch_rewards = []
            obs, _ = self.env.reset()
            episode_return = 0
            episode_steps = 0
            
            # Data collection on CPU
            for t in range(self.n_steps):
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.default_device)
                    dist = self.actor(obs_tensor)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum(-1)
                    value = self.critic(obs_tensor)
                    
                next_obs, reward, done, info = self.env.step(action.numpy())
                episode_return += reward
                episode_steps += 1
                
                # Store everything on CPU
                self.buffer.store(
                    torch.as_tensor(obs, device=self.default_device),
                    action.to(self.default_device),  # Make sure action is on CPU
                    reward,
                    done,
                    value.item(),
                    logp.item()
                )
                
                obs = next_obs
                
                if done or (t == self.n_steps - 1):
                    with torch.no_grad():
                        last_val = 0 if done else self.critic(
                            torch.as_tensor(obs, dtype=torch.float32, device=self.default_device)
                        ).item()
                    self.buffer.finish_path(last_val)
                    if done:
                        batch_rewards.append(episode_return)
                    obs, _ = self.env.reset()
                    episode_return = 0
                    episode_steps = 0
            
            timesteps_so_far += self.n_steps
            self.total_timesteps += self.n_steps
            
            if batch_rewards:
                self.metrics['rewards'].extend(batch_rewards)
            
            _ = self.update()

    def save(self, save_path: str):
        # Move models to CPU to avoid GPU OOM during saving
        self.actor.to(self.default_device)
        self.critic.to(self.default_device)
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
        checkpoint = torch.load(load_path, map_location=self.default_device)
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