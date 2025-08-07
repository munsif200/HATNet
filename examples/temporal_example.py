#!/usr/bin/env python3
"""
Basic ReST Temporal Graph Example
This example demonstrates how to use the ReST temporal graph for object tracking.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs import cfg
from src.models import ReSTModel


def run_temporal_example():
    """Run a basic temporal graph example"""
    print("ReST Temporal Graph Example")
    print("=" * 40)
    
    # Configure for temporal graph
    cfg.defrost()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.SOLVER.TYPE = 'TG'
    cfg.freeze()
    
    # Create model
    model = ReSTModel(cfg)
    model.eval()
    
    # Create synthetic trajectory data: 3 objects across 4 time steps
    time_steps = 4
    num_objects = 3
    
    # Create features for each time step
    spatial_features = torch.randn(time_steps, num_objects, 32)
    
    # Create realistic trajectories
    positions = torch.zeros(time_steps, num_objects, 2)
    
    # Object 1: Moving right
    positions[:, 0, :] = torch.tensor([
        [0.0, 0.0], [1.0, 0.1], [2.0, 0.2], [3.0, 0.3]
    ])
    
    # Object 2: Moving diagonally  
    positions[:, 1, :] = torch.tensor([
        [0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [3.0, 2.0]
    ])
    
    # Object 3: Moving up
    positions[:, 2, :] = torch.tensor([
        [5.0, 0.0], [5.1, 1.0], [5.0, 2.0], [4.9, 3.0]
    ])
    
    print(f"Input: {num_objects} objects tracked across {time_steps} time steps")
    print("\nObject trajectories:")
    for obj in range(num_objects):
        print(f"  Object {obj+1}:")
        for t in range(time_steps):
            x, y = positions[t, obj]
            print(f"    t={t}: ({x:.1f}, {y:.1f})")
    
    # Create temporal information
    velocities = torch.zeros(time_steps, num_objects, 2)
    for t in range(1, time_steps):
        velocities[t] = positions[t] - positions[t-1]
    
    # Distance matrix between consecutive frames
    distances = torch.zeros(time_steps-1, num_objects, num_objects)
    for t in range(time_steps-1):
        for i in range(num_objects):
            for j in range(num_objects):
                distances[t, i, j] = torch.norm(positions[t+1, j] - positions[t, i])
    
    temporal_info = {
        'velocities': velocities,
        'distances': distances,
        'appearance_sim': torch.ones(time_steps-1, num_objects, num_objects),
        'confidences': torch.ones(time_steps, num_objects)
    }
    
    # Create input data
    data = {
        'spatial_features': spatial_features,
        'positions': positions,
        'temporal_info': temporal_info
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(data)
    
    # Analyze results
    print(f"\nTemporal Association Results:")
    
    if 'trajectories' in outputs:
        trajectories = outputs['trajectories']
        print(f"Found {len(trajectories)} trajectories:")
        
        for i, traj in enumerate(trajectories):
            print(f"  Trajectory {i+1}: {traj}")
            if len(traj) > 1:
                print(f"    Length: {len(traj)} detections")
    else:
        print("No trajectories generated")
    
    # Show tracking scores for first time step transition
    if 'temporal_outputs' in outputs:
        tracking_scores = outputs['temporal_outputs']['tracking_scores']
        if tracking_scores.numel() > 0:
            print(f"\nTracking scores (t=0 -> t=1):")
            print("(Higher scores indicate stronger temporal associations)")
            
            for i in range(num_objects):
                for j in range(num_objects):
                    score = tracking_scores[0, i, j].item()
                    dist = distances[0, i, j].item()
                    print(f"  Object {i+1}(t=0) -> Object {j+1}(t=1): score={score:.3f}, distance={dist:.2f}")
    
    print("\nTemporal Graph Example Complete!")


if __name__ == "__main__":
    run_temporal_example()