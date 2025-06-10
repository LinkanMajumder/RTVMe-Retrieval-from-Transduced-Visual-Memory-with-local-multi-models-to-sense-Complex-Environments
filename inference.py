import torch
import yaml
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.multimodal_trajectory import MultimodalTrajectoryPredictor
from data.nuscenes_dataset import NuScenesTrajectoryDataset
from nuscenes.nuscenes import NuScenes
from transformers import AutoTokenizer
from torchvision import transforms

def load_model(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalTrajectoryPredictor(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def preprocess_image(image_path, config):
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def preprocess_text(text, config):
    tokenizer = AutoTokenizer.from_pretrained(config['language_model_name'])
    tokens = tokenizer(
        text,
        padding='max_length',
        max_length=config['max_text_length'],
        truncation=True,
        return_tensors='pt'
    )
    return tokens

def visualize_trajectory(past_trajectory, predicted_trajectory, save_path=None):
    plt.figure(figsize=(10, 10))
    
    # Plot past trajectory
    plt.plot(past_trajectory[:, 0], past_trajectory[:, 1], 
             'b-', label='Past Trajectory', marker='o')
    
    # Plot predicted trajectory
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 
             'r--', label='Predicted Trajectory', marker='x')
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Trajectory Prediction')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Inference for Multimodal Trajectory Prediction')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_4.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--sample_token', type=str, required=True,
                        help='NuScenes sample token for inference')
    parser.add_argument('--instance_token', type=str, required=True,
                        help='NuScenes instance token for inference')
    parser.add_argument('--output', type=str, default='prediction.png',
                        help='Path to save visualization')
    args = parser.parse_args()

    # Load config as a dictionary
    config = load_config(args.config)

    # Initialize NuScenes
    nusc = NuScenes(
        version=config['nuscenes_version'],
        dataroot=config['nuscenes_root'],
        verbose=False
    )
    
    # Load model
    model, device = load_model(config, args.checkpoint)
    
    # Get sample data
    sample = nusc.get('sample', args.sample_token)
    
    # Initialize dataset
    dataset = NuScenesTrajectoryDataset(config, split='val')
    
    # Get past trajectory
    past_trajectory = dataset._get_past_trajectory(sample, args.instance_token)
    past_trajectory_tensor = torch.tensor(past_trajectory, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get camera image
    cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    image_path = nusc.get_sample_data_path(cam_front_data['token'])
    image_tensor = preprocess_image(image_path, config).to(device)
    
    # Get scene description
    scene = nusc.get('scene', sample['scene_token'])
    scene_description = scene['description']
    text_tokens = preprocess_text(scene_description, config)
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    
    # Perform inference
    with torch.no_grad():
        predicted_trajectory = model(image_tensor, text_tokens, past_trajectory_tensor)
        predicted_trajectory = predicted_trajectory.cpu().numpy()[0]  # First batch item
    
    # Visualize results
    visualize_trajectory(past_trajectory, predicted_trajectory, args.output)
    print(f"Prediction visualization saved to {args.output}")
    
    # Print trajectory coordinates
    print("\nPredicted trajectory coordinates:")
    for i, (x, y) in enumerate(predicted_trajectory):
        print(f"Step {i+1}: ({x:.2f}, {y:.2f})")

if __name__ == '__main__':
    main()