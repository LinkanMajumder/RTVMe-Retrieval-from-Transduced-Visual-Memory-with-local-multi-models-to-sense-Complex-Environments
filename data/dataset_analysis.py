import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nuscenes.nuscenes import NuScenes
import pandas as pd
from collections import Counter
import yaml
from pathlib import Path

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def analyze_nuscenes_dataset():
    config = load_config()
    nusc = NuScenes(version=config['nuscenes_version'], dataroot=config['nuscenes_root'], verbose=True)
    
    # Collect statistics
    stats = {
        'total_scenes': len(nusc.scene),
        'total_samples': len(nusc.sample),
        'total_annotations': len(nusc.sample_annotation),
        'total_instances': len(nusc.instance)
    }
    
    # Analyze scene descriptions
    scene_descriptions = [scene['description'] for scene in nusc.scene]
    
    # Analyze timestamps and scene lengths
    scene_lengths = []
    time_of_day = []
    for scene in nusc.scene:
        first_sample = nusc.get('sample', scene['first_sample_token'])
        last_sample = nusc.get('sample', scene['last_sample_token'])
        scene_lengths.append(scene['nbr_samples'])
        
        # Determine time of day from scene description
        if any(time in scene['description'].lower() for time in ['night', 'evening', 'dark']):
            time_of_day.append('Night')
        else:
            time_of_day.append('Day')
    
    # Analyze annotations per category
    category_counts = Counter()
    for annotation in nusc.sample_annotation:
        # Get instance first, then get category from instance
        instance = nusc.get('instance', annotation['instance_token'])
        category = nusc.get('category', instance['category_token'])['name']
        category_counts[category] += 1
    
    # Create visualizations
    plt.style.use('seaborn')
    
    # 1. Scene Lengths Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(scene_lengths, bins=20)
    plt.title('Distribution of Scene Lengths')
    plt.xlabel('Number of Samples per Scene')
    plt.ylabel('Count')
    plt.savefig('scene_lengths_distribution.png')
    plt.close()
    
    # 2. Time of Day Distribution
    plt.figure(figsize=(8, 8))
    time_counts = Counter(time_of_day)
    plt.pie([time_counts['Day'], time_counts['Night']], 
            labels=['Day', 'Night'],
            autopct='%1.1f%%',
            colors=['lightblue', 'darkblue'])
    plt.title('Distribution of Day/Night Scenes')
    plt.savefig('time_of_day_distribution.png')
    plt.close()
    
    # 3. Top 10 Categories Bar Plot
    plt.figure(figsize=(15, 6))
    top_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    sns.barplot(x=list(top_categories.keys()), y=list(top_categories.values()))
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Object Categories in Dataset')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('top_categories.png')
    plt.close()
    
    # Print statistics
    print("\nNuScenes Dataset Statistics:")
    print("============================")
    print(f"Total Scenes: {stats['total_scenes']}")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Total Annotations: {stats['total_annotations']}")
    print(f"Total Instances: {stats['total_instances']}")
    print("\nAverage Samples per Scene:", np.mean(scene_lengths))
    print("Average Annotations per Sample:", stats['total_annotations'] / stats['total_samples'])
    
    # Analyze sensor coverage
    sensor_stats = {
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 0,
        'CAM_FRONT_RIGHT': 0,
        'CAM_BACK': 0,
        'CAM_BACK_LEFT': 0,
        'CAM_BACK_RIGHT': 0,
        'LIDAR_TOP': 0
    }
    
    for sample in nusc.sample:
        for sensor in sensor_stats.keys():
            if sample['data'].get(sensor):
                sensor_stats[sensor] += 1
    
    # 4. Sensor Coverage Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(sensor_stats.keys()), y=list(sensor_stats.values()))
    plt.xticks(rotation=45, ha='right')
    plt.title('Sensor Data Availability')
    plt.xlabel('Sensor')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig('sensor_coverage.png')
    plt.close()
    
    # Print sensor coverage statistics
    print("\nSensor Coverage Statistics:")
    print("==========================")
    for sensor, count in sensor_stats.items():
        print(f"{sensor}: {count} samples ({count/stats['total_samples']*100:.1f}%)")

if __name__ == '__main__':
    analyze_nuscenes_dataset()