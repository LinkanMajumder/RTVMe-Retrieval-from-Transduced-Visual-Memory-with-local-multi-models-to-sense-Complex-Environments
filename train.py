import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multimodal_trajectory import MultimodalTrajectoryPredictor
from data.nuscenes_dataset import NuScenesTrajectoryDataset
import wandb
from tqdm import tqdm
import argparse
import yaml
import os

def train(config):
    # Initialize wandb
    wandb.init(project="multimodal-trajectory", config=config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    train_dataset = NuScenesTrajectoryDataset(config, split='train')
    val_dataset = NuScenesTrajectoryDataset(config, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)
    
    # Initialize model
    model = MultimodalTrajectoryPredictor(config).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            # Move data to device
            images = batch['image'].to(device)
            text = {k: v.to(device) for k, v in batch['text'].items()}
            past_trajectory = batch['past_trajectory'].to(device)
            future_trajectory = batch['future_trajectory'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_trajectory = model(images, text, past_trajectory)
            
            # Reshape predicted trajectory to match target shape
            pred_trajectory = pred_trajectory.view(-1, config.future_timesteps, 2)
            
            # Calculate loss
            loss = criterion(pred_trajectory, future_trajectory)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Log batch metrics
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0]
            })
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = batch['image'].to(device)
                text = {k: v.to(device) for k, v in batch['text'].items()}
                past_trajectory = batch['past_trajectory'].to(device)
                future_trajectory = batch['future_trajectory'].to(device)
                
                # Forward pass
                pred_trajectory = model(images, text, past_trajectory)
                
                # Reshape predicted trajectory to match target shape
                pred_trajectory = pred_trajectory.view(-1, config.future_timesteps, 2)
                
                # Calculate loss
                loss = criterion(pred_trajectory, future_trajectory)
                
                val_loss += loss.item()
                val_steps += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch metrics
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "epoch": epoch + 1
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Convert config dict to object
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    config = Config(**config)
    
    # Start training
    train(config)

if __name__ == '__main__':
    main() 