#!/usr/bin/env python3
"""
Basic ReST Spatial Graph Example
This example demonstrates how to use the ReST spatial graph for object association.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs import cfg
from src.models import ReSTModel


def run_spatial_example():
    """Run a basic spatial graph example"""
    print("ReST Spatial Graph Example")
    print("=" * 40)
    
    # Configure for spatial graph
    cfg.defrost()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.SOLVER.TYPE = 'SG'
    cfg.freeze()
    
    # Create model
    model = ReSTModel(cfg)
    model.eval()
    
    # Create synthetic data: 4 objects in 2D space
    num_objects = 4
    spatial_features = torch.randn(num_objects, 32)  # 32-dim appearance features
    positions = torch.tensor([
        [0.0, 0.0],    # Object 1 at origin
        [1.0, 0.0],    # Object 2 nearby
        [5.0, 5.0],    # Object 3 far away
        [5.1, 5.2],    # Object 4 close to object 3
    ])
    
    print(f"Input: {num_objects} objects with positions:")
    for i, pos in enumerate(positions):
        print(f"  Object {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Create input data
    data = {
        'spatial_features': spatial_features,
        'positions': positions
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(data)
    
    # Analyze results
    association_scores = outputs['association_scores']
    print(f"\nAssociation Scores ({association_scores.shape}):")
    print("(Higher scores indicate stronger associations)")
    
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                score = association_scores[i, j].item()
                print(f"  Object {i+1} -> Object {j+1}: {score:.3f}")
    
    # Find strong associations (threshold = 0.6)
    threshold = 0.6
    print(f"\nStrong associations (score > {threshold}):")
    found_associations = False
    
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j and association_scores[i, j] > threshold:
                score = association_scores[i, j].item()
                dist = torch.norm(positions[i] - positions[j]).item()
                print(f"  Object {i+1} <-> Object {j+1}: score={score:.3f}, distance={dist:.2f}")
                found_associations = True
    
    if not found_associations:
        print("  No strong associations found with current threshold")
        
    print("\nSpatial Graph Example Complete!")


if __name__ == "__main__":
    run_spatial_example()