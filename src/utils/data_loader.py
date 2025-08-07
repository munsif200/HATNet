"""
Synthetic data loader for ReST model training and testing
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for ReST model training and testing
    Generates spatial-temporal data for demonstration purposes
    """
    
    def __init__(self, num_samples=1000, mode='SG', device='cuda', seed=42):
        """
        Args:
            num_samples: Number of samples to generate
            mode: 'SG' for spatial graph, 'TG' for temporal graph
            device: Device to create tensors on
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.mode = mode
        self.device = device
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate synthetic data
        self.data = self._generate_data()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single data sample"""
        if self.mode == 'SG':
            return self._get_spatial_sample(idx)
        else:
            return self._get_temporal_sample(idx)
    
    def _generate_data(self):
        """Generate synthetic spatial-temporal data"""
        data = []
        
        for i in range(self.num_samples):
            if self.mode == 'SG':
                sample = self._create_spatial_sample()
            else:
                sample = self._create_temporal_sample()
            data.append(sample)
        
        return data
    
    def _create_spatial_sample(self):
        """Create a single spatial sample"""
        # Random number of objects (3-8)
        num_objects = np.random.randint(3, 9)
        
        # Spatial features (e.g., appearance features)
        spatial_features = torch.randn(num_objects, 32, device=self.device)
        
        # Object positions in 2D space
        positions = torch.randn(num_objects, 2, device=self.device) * 10
        
        # Create ground truth associations
        # For simplicity, create some random associations
        associations = torch.zeros(num_objects, num_objects, device=self.device)
        
        # Add some positive associations (same object across views)
        for i in range(min(3, num_objects)):
            if i < num_objects - 1:
                associations[i, i+1] = 1.0  # Link consecutive objects
        
        return {
            'spatial_features': spatial_features,
            'positions': positions,
            'targets': associations
        }
    
    def _create_temporal_sample(self):
        """Create a single temporal sample"""
        # Random sequence length (3-7 time steps)
        time_steps = np.random.randint(3, 8)
        num_objects = np.random.randint(2, 6)
        
        # Spatial features across time
        spatial_features = torch.randn(time_steps, num_objects, 32, device=self.device)
        
        # Object positions across time (with some temporal consistency)
        positions = torch.zeros(time_steps, num_objects, 2, device=self.device)
        
        # Create trajectories with some noise
        for obj in range(num_objects):
            start_pos = torch.randn(2, device=self.device) * 5
            velocity = torch.randn(2, device=self.device) * 0.5
            
            for t in range(time_steps):
                noise = torch.randn(2, device=self.device) * 0.1
                positions[t, obj] = start_pos + t * velocity + noise
        
        # Temporal information
        velocities = torch.zeros(time_steps, num_objects, 2, device=self.device)
        for t in range(1, time_steps):
            velocities[t] = positions[t] - positions[t-1]
        
        # Distances between objects
        distances = torch.zeros(time_steps-1, num_objects, num_objects, device=self.device)
        for t in range(time_steps-1):
            for i in range(num_objects):
                for j in range(num_objects):
                    if i != j:
                        dist = torch.norm(positions[t+1, j] - positions[t, i])
                        distances[t, i, j] = dist
        
        # Appearance similarity (simplified)
        appearance_sim = torch.ones(time_steps-1, num_objects, num_objects, device=self.device)
        for t in range(time_steps-1):
            for i in range(num_objects):
                for j in range(num_objects):
                    if i == j:
                        appearance_sim[t, i, j] = 1.0  # Same object
                    else:
                        appearance_sim[t, i, j] = torch.rand(1, device=self.device) * 0.5
        
        # Confidences
        confidences = torch.ones(time_steps, num_objects, device=self.device)
        
        temporal_info = {
            'velocities': velocities,
            'distances': distances,
            'appearance_sim': appearance_sim,
            'confidences': confidences
        }
        
        # Create ground truth targets (identity matrix for perfect tracking)
        targets = torch.eye(num_objects, device=self.device)
        
        return {
            'spatial_features': spatial_features,
            'positions': positions,
            'temporal_info': temporal_info,
            'targets': targets
        }
    
    def _get_spatial_sample(self, idx):
        """Get spatial sample by index"""
        return self.data[idx % len(self.data)]
    
    def _get_temporal_sample(self, idx):
        """Get temporal sample by index"""
        return self.data[idx % len(self.data)]


class MultiCameraDataset(Dataset):
    """
    Multi-camera synthetic dataset for spatial associations
    """
    
    def __init__(self, num_samples=500, num_cameras=4, device='cuda'):
        self.num_samples = num_samples
        self.num_cameras = num_cameras
        self.device = device
        
        self.data = self._generate_multicamera_data()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx % len(self.data)]
    
    def _generate_multicamera_data(self):
        """Generate multi-camera spatial data"""
        data = []
        
        for i in range(self.num_samples):
            # Number of objects in the scene
            num_objects = np.random.randint(2, 7)
            
            # Generate object features for each camera
            camera_features = []
            camera_positions = []
            
            for cam in range(self.num_cameras):
                # Each camera sees some subset of objects
                visible_objects = np.random.choice(
                    num_objects, 
                    size=np.random.randint(1, num_objects+1), 
                    replace=False
                )
                
                # Features for visible objects
                features = torch.randn(len(visible_objects), 32, device=self.device)
                
                # Positions with camera-specific transformation
                base_positions = torch.randn(len(visible_objects), 2, device=self.device) * 5
                cam_offset = torch.tensor([cam * 2.0, 0.0], device=self.device)
                positions = base_positions + cam_offset
                
                camera_features.append(features)
                camera_positions.append(positions)
            
            # Create association targets between cameras
            # For simplicity, assume perfect knowledge
            associations = self._create_camera_associations(camera_features, num_objects)
            
            sample = {
                'camera_features': camera_features,
                'camera_positions': camera_positions,
                'num_cameras': self.num_cameras,
                'targets': associations
            }
            
            data.append(sample)
        
        return data
    
    def _create_camera_associations(self, camera_features, num_objects):
        """Create ground truth associations between cameras"""
        # Simplified: return identity associations
        max_detections = max(len(feats) for feats in camera_features)
        associations = torch.eye(max_detections, device=self.device)
        return associations