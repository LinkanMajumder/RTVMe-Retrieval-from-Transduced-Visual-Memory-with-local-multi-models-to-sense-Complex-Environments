import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

class ReasoningModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Reasoning layers for CoT
        self.cot_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_dim'],
                nhead=8,
                dim_feedforward=config['hidden_dim'] * 4,
                dropout=0.1
            ) for _ in range(config['num_cot_steps'])
        ])
        
        # Self-consistency voting layer
        self.sc_voting = nn.Sequential(
            nn.Linear(config['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, config['num_trajectory_samples'])
        )
        
        # Tree of Thoughts components
        self.tot_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['hidden_dim'],
                nhead=8,
                dim_feedforward=config['hidden_dim'] * 4,
                dropout=0.1
            ),
            num_layers=3
        )
        
        self.tot_scoring = nn.Sequential(
            nn.Linear(config['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def chain_of_thought(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Implements Chain of Thought reasoning."""
        intermediate_states = []
        current_state = features
        
        for layer in self.cot_layers:
            # Add positional encoding
            pos_encoding = self._get_positional_encoding(current_state)
            current_state = current_state + pos_encoding
            
            # Apply transformer layer
            current_state = layer(current_state)
            intermediate_states.append(current_state)
        
        return intermediate_states
    
    def self_consistency(self, features: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
        """Implements Self-Consistency by generating multiple trajectories and voting."""
        batch_size = features.size(0)
        
        # Generate multiple trajectory samples
        trajectory_samples = []
        for _ in range(num_samples):
            # Add noise for diversity
            noisy_features = features + torch.randn_like(features) * 0.1
            trajectory = self.sc_voting(noisy_features)
            trajectory_samples.append(trajectory)
        
        # Stack samples
        trajectory_samples = torch.stack(trajectory_samples, dim=1)  # [batch, num_samples, traj_dim]
        
        # Weighted voting
        weights = F.softmax(trajectory_samples.mean(dim=-1), dim=1)  # [batch, num_samples]
        final_trajectory = (trajectory_samples * weights.unsqueeze(-1)).sum(dim=1)
        
        return final_trajectory
    
    def tree_of_thoughts(self, features: torch.Tensor, num_branches: int = 3) -> torch.Tensor:
        """Implements Tree of Thoughts reasoning."""
        batch_size = features.size(0)
        
        # Generate multiple thought branches
        branches = []
        branch_scores = []
        
        for _ in range(num_branches):
            # Add noise for diversity
            noisy_features = features + torch.randn_like(features) * 0.1
            
            # Encode branch
            branch = self.tot_encoder(noisy_features)
            branches.append(branch)
            
            # Score branch
            score = self.tot_scoring(branch.mean(dim=1))
            branch_scores.append(score)
        
        # Stack branches and scores
        branches = torch.stack(branches, dim=1)  # [batch, num_branches, seq_len, hidden_dim]
        branch_scores = torch.stack(branch_scores, dim=1)  # [batch, num_branches, 1]
        
        # Select best branch
        best_branch_idx = branch_scores.argmax(dim=1)  # [batch, 1]
        best_branches = torch.gather(
            branches, 
            1, 
            best_branch_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, branches.size(2), -1)
        ).squeeze(1)
        
        return best_branches
    
    def _get_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding for the sequence."""
        seq_len, hidden_dim = x.size(1), x.size(2)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, device=x.device) * -(np.log(10000.0) / hidden_dim))
        
        pos_encoding = torch.zeros(seq_len, hidden_dim, device=x.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # [1, seq_len, hidden_dim]