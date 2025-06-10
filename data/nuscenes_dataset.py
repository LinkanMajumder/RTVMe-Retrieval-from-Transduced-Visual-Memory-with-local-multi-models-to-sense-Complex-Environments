import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
import numpy as np
from PIL import Image
import cv2
from transformers import AutoTokenizer

class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.nusc = NuScenes(version=config['nuscenes_version'], dataroot=config['nuscenes_root'])
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)
        
        # Get all scenes for the specified split
        self.scenes = self._get_scenes()
        self.samples = self._get_samples()
        
    def _get_scenes(self):
        # For nuScenes mini, use all scenes regardless of split
        return self.nusc.scene
    
    def _get_samples(self):
        samples = []
        for scene in self.scenes:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                samples.append(sample)
                sample_token = sample['next']
        return samples
    
    def _get_past_trajectory(self, sample, instance_token):
        past_trajectory = []
        current_sample = sample
        
        try:
            # Get past trajectory points
            for _ in range(self.config.past_timesteps):
                if current_sample is None:
                    break
                    
                ann = self.nusc.get('sample_annotation', 
                                  self.nusc.get('instance', instance_token)['first_annotation_token'])
                
                box = Box(ann['translation'], ann['size'], ann['rotation'])
                past_trajectory.append(box.center[:2])  # Only x, y coordinates
                
                current_sample = self.nusc.get('sample', current_sample['prev']) if current_sample['prev'] else None
                
            # Pad if necessary
            while len(past_trajectory) < self.config.past_timesteps:
                past_trajectory.insert(0, past_trajectory[0])
                
            return np.array(past_trajectory)
        except KeyError:
            # Return a default trajectory if the instance is invalid
            return np.zeros((self.config.past_timesteps, 2))
    
    def _get_future_trajectory(self, sample, instance_token):
        future_trajectory = []
        current_sample = sample
        
        try:
            # Get future trajectory points
            for _ in range(self.config.future_timesteps):
                if current_sample is None:
                    break
                    
                ann = self.nusc.get('sample_annotation', 
                                  self.nusc.get('instance', instance_token)['first_annotation_token'])
                
                box = Box(ann['translation'], ann['size'], ann['rotation'])
                future_trajectory.append(box.center[:2])  # Only x, y coordinates
                
                current_sample = self.nusc.get('sample', current_sample['next']) if current_sample['next'] else None
                
            # Pad if necessary
            while len(future_trajectory) < self.config.future_timesteps:
                future_trajectory.append(future_trajectory[-1])
                
            return np.array(future_trajectory)
        except KeyError:
            # Return a default trajectory if the instance is invalid
            return np.zeros((self.config.future_timesteps, 2))
    
    def _get_camera_image(self, sample):
        # Get front camera image
        cam_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        image_path = self.nusc.get_sample_data_path(cam_data['token'])
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config.image_size, self.config.image_size))
        image = image / 255.0  # Normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image
    
    def _get_scene_description(self, sample):
        # Create a natural language description of the scene
        scene = self.nusc.get('scene', sample['scene_token'])
        return f"Scene: {scene['description']}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get a random instance from the sample
        instance_token = np.random.choice(sample['anns'])
        
        # Get past and future trajectories
        past_trajectory = self._get_past_trajectory(sample, instance_token)
        future_trajectory = self._get_future_trajectory(sample, instance_token)
        
        # Get camera image
        image = self._get_camera_image(sample)
        
        # Get scene description
        text = self._get_scene_description(sample)
        text_tokens = self.tokenizer(text, padding='max_length', 
                                   truncation=True, max_length=self.config.max_text_length,
                                   return_tensors='pt')
        # Remove the batch dimension from each tensor
        text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}
        
        return {
            'image': image,
            'text': text_tokens,
            'past_trajectory': torch.from_numpy(past_trajectory).float().view(-1),
            'future_trajectory': torch.from_numpy(future_trajectory).float()
        }