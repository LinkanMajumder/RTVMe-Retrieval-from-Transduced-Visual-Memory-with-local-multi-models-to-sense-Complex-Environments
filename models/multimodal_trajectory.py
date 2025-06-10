import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from timm import create_model
from .reasoning import ReasoningModule

class MultimodalTrajectoryPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Vision encoder
        self.vision_encoder = create_model(
            config['vision_model_name'],
            pretrained=True,
            num_classes=0
        )
        
        # Language encoder
        self.language_encoder = AutoModel.from_pretrained(config['language_model_name'])
        
        # Trajectory encoder
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(config['trajectory_input_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Feature projection layers
        self.vision_projection = nn.Linear(768, config['hidden_dim'])
        self.language_projection = nn.Linear(768, config['hidden_dim'])
        self.trajectory_projection = nn.Linear(512, config['hidden_dim'])
        
        # Reasoning module
        self.reasoning = ReasoningModule(config)
        
        # Trajectory decoder
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(config['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, config['future_timesteps'] * 2)  # x, y coordinates for each timestep
        )
        
    def forward(self, images, text, past_trajectory, reasoning_type='cot'):
        # Encode visual features
        visual_features = self.vision_encoder(images)
        visual_features = self.vision_projection(visual_features)
        
        # Encode language features
        language_features = self.language_encoder(**text).last_hidden_state[:, 0, :]
        language_features = self.language_projection(language_features)
        
        # Encode past trajectory
        trajectory_features = self.trajectory_encoder(past_trajectory)
        trajectory_features = self.trajectory_projection(trajectory_features)
        
        # Concatenate features
        combined_features = torch.stack([
            visual_features,
            language_features,
            trajectory_features
        ], dim=1)  # [batch, 3, hidden_dim]
        
        # Apply reasoning based on type
        if reasoning_type == 'cot':
            # Chain of Thought reasoning
            intermediate_states = self.reasoning.chain_of_thought(combined_features)
            final_features = intermediate_states[-1].mean(dim=1)  # Use last state
            
        elif reasoning_type == 'cot_sc':
            # Chain of Thought with Self-Consistency
            intermediate_states = self.reasoning.chain_of_thought(combined_features)
            final_features = self.reasoning.self_consistency(intermediate_states[-1])
            
        elif reasoning_type == 'tot':
            # Tree of Thoughts reasoning
            final_features = self.reasoning.tree_of_thoughts(combined_features)
            final_features = final_features.mean(dim=1)
            
        else:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")
        
        # Decode future trajectory
        future_trajectory = self.trajectory_decoder(final_features)
        
        return future_trajectory

    def predict(self, images, text, past_trajectory, reasoning_type='cot'):
        self.eval()
        with torch.no_grad():
            return self.forward(images, text, past_trajectory, reasoning_type)