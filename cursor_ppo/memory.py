import torch
import torch.nn.functional as F

class StreamingFeatureMemory:
    """Maintains a streaming memory of feature prototypes and their rewards"""
    
    def __init__(self, feature_dim: int, momentum: float = 0.99, beta: float = 0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.momentum = momentum
        self.beta = beta
        self.feature_dim = feature_dim
        
        # Running statistics
        self.running_mean = torch.zeros(feature_dim, device=self.device)
        self.running_variance = torch.ones(feature_dim, device=self.device)
        self.running_reward = torch.tensor(0.0, device=self.device)
        
        # Prototypes
        self.cancer_prototype = None
        self.non_cancer_prototype = None
        self.prototype_momentum = 0.95
        
        # Feature history
        self.history_size = 32
        self.history_features = torch.zeros((self.history_size, feature_dim), device=self.device)
        self.history_rewards = torch.zeros(self.history_size, device=self.device)
        self.history_idx = 0
        self.history_filled = False
        
        self.count = 0

    def update(self, features: torch.Tensor, reward: float, cluster_centers: torch.Tensor):
        """Update memory with new features and their reward"""
        features_flat = features.reshape(-1)
        reward_tensor = torch.tensor(reward, device=self.device)
        
        # First update
        if self.count == 0:
            self.running_mean = features_flat
            self.running_reward = reward_tensor
            self.cancer_prototype = cluster_centers[0]
            self.non_cancer_prototype = cluster_centers[1]
        else:
            # Update running statistics
            delta = features_flat - self.running_mean
            self.running_mean += (1 - self.momentum) * delta
            self.running_variance = self.momentum * self.running_variance + \
                                  (1 - self.momentum) * delta * delta
            
            # Update prototypes with adaptive momentum
            sim_0_cancer = F.cosine_similarity(
                cluster_centers[0].unsqueeze(0),
                self.cancer_prototype.unsqueeze(0)
            )
            sim_0_non = F.cosine_similarity(
                cluster_centers[0].unsqueeze(0),
                self.non_cancer_prototype.unsqueeze(0)
            )
            
            confidence = torch.abs(sim_0_cancer - sim_0_non).item()
            effective_momentum = self.prototype_momentum * (1 + confidence)
            effective_momentum = min(effective_momentum, 0.99)
            
            if sim_0_cancer > sim_0_non:
                self.cancer_prototype = (effective_momentum * self.cancer_prototype + 
                                      (1 - effective_momentum) * cluster_centers[0])
                self.non_cancer_prototype = (effective_momentum * self.non_cancer_prototype + 
                                          (1 - effective_momentum) * cluster_centers[1])
            else:
                self.cancer_prototype = (effective_momentum * self.cancer_prototype + 
                                      (1 - effective_momentum) * cluster_centers[1])
                self.non_cancer_prototype = (effective_momentum * self.non_cancer_prototype + 
                                          (1 - effective_momentum) * cluster_centers[0])
            
            self.running_reward = self.momentum * self.running_reward + \
                                (1 - self.momentum) * reward_tensor
        
        # Update history buffer with high-reward features
        if reward > self.running_reward.item():
            self.history_features[self.history_idx] = features_flat
            self.history_rewards[self.history_idx] = reward_tensor
            self.history_idx = (self.history_idx + 1) % self.history_size
            if not self.history_filled and self.history_idx == 0:
                self.history_filled = True
        
        self.count += 1

    def query(self, features: torch.Tensor) -> torch.Tensor:
        """Query memory to get similarity score for features"""
        if self.count == 0:
            return torch.tensor(0.0, device=features.device)
        
        features_flat = features.reshape(-1)
        
        # Normalize features
        normalized_features = (features_flat - self.running_mean) / \
                            (torch.sqrt(self.running_variance) + 1e-8)
        
        # Compute main similarity
        main_similarity = F.cosine_similarity(
            normalized_features.unsqueeze(0),
            self.running_mean.unsqueeze(0)
        )
        
        # Compute history similarity
        if self.history_filled:
            history_end = self.history_size
        else:
            history_end = self.history_idx
        
        if history_end > 0:
            history_sims = F.cosine_similarity(
                features_flat.unsqueeze(0).expand(history_end, -1),
                self.history_features[:history_end],
                dim=1
            )
            history_weights = F.softmax(self.history_rewards[:history_end] / 0.1, dim=0)
            history_similarity = (history_sims * history_weights).sum()
        else:
            history_similarity = torch.tensor(0.0, device=self.device)
        
        # Combine similarities
        combined_similarity = 0.7 * main_similarity + 0.3 * history_similarity
        
        return combined_similarity * torch.sigmoid(self.running_reward)