"""
Metrics computation for ReST model evaluation
"""

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(outputs, targets, threshold=0.5):
    """
    Compute evaluation metrics for ReST model
    Args:
        outputs: Model outputs dictionary
        targets: Ground truth targets
        threshold: Association threshold
    Returns:
        Dictionary with computed metrics
    """
    metrics = {}
    
    # Spatial association metrics
    if 'spatial_outputs' in outputs:
        spatial_metrics = compute_spatial_metrics(
            outputs['spatial_outputs'], targets, threshold
        )
        metrics.update({f'spatial_{k}': v for k, v in spatial_metrics.items()})
    
    # Temporal tracking metrics
    if 'temporal_outputs' in outputs:
        temporal_metrics = compute_temporal_metrics(
            outputs['temporal_outputs'], targets, threshold
        )
        metrics.update({f'temporal_{k}': v for k, v in temporal_metrics.items()})
    
    # Trajectory metrics
    if 'trajectories' in outputs:
        traj_metrics = compute_trajectory_metrics(outputs['trajectories'])
        metrics.update({f'trajectory_{k}': v for k, v in traj_metrics.items()})
    
    # Overall metrics (average of spatial and temporal if both exist)
    if 'spatial_accuracy' in metrics and 'temporal_accuracy' in metrics:
        metrics['accuracy'] = (metrics['spatial_accuracy'] + metrics['temporal_accuracy']) / 2
        metrics['precision'] = (metrics['spatial_precision'] + metrics['temporal_precision']) / 2
        metrics['recall'] = (metrics['spatial_recall'] + metrics['temporal_recall']) / 2
    elif 'spatial_accuracy' in metrics:
        metrics['accuracy'] = metrics['spatial_accuracy']
        metrics['precision'] = metrics['spatial_precision']
        metrics['recall'] = metrics['spatial_recall']
    elif 'temporal_accuracy' in metrics:
        metrics['accuracy'] = metrics['temporal_accuracy']
        metrics['precision'] = metrics['temporal_precision']
        metrics['recall'] = metrics['temporal_recall']
    else:
        metrics['accuracy'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
    
    return metrics


def compute_spatial_metrics(spatial_outputs, targets, threshold=0.5):
    """
    Compute metrics for spatial associations
    """
    association_scores = spatial_outputs['association_scores']
    
    if association_scores.numel() == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Convert to binary predictions
    predictions = (association_scores > threshold).float()
    
    # Create targets if not provided
    if isinstance(targets, torch.Tensor) and targets.dim() == 2:
        gt_associations = targets.float()
    else:
        # Default to identity matrix
        n = association_scores.shape[0]
        gt_associations = torch.eye(n, device=association_scores.device)
    
    # Ensure same size
    if gt_associations.shape != predictions.shape:
        min_size = min(gt_associations.shape[0], predictions.shape[0])
        gt_associations = gt_associations[:min_size, :min_size]
        predictions = predictions[:min_size, :min_size]
    
    # Convert to numpy for sklearn metrics
    pred_np = predictions.detach().cpu().numpy().flatten()
    gt_np = gt_associations.detach().cpu().numpy().flatten()
    
    # Compute metrics
    accuracy = np.mean(pred_np == gt_np)
    
    try:
        precision = precision_score(gt_np, pred_np, average='binary', zero_division=0)
        recall = recall_score(gt_np, pred_np, average='binary', zero_division=0)
        f1 = f1_score(gt_np, pred_np, average='binary', zero_division=0)
    except:
        precision = recall = f1 = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_temporal_metrics(temporal_outputs, targets, threshold=0.5):
    """
    Compute metrics for temporal tracking
    """
    tracking_scores = temporal_outputs['tracking_scores']
    
    if tracking_scores.numel() == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Convert to binary predictions
    predictions = (tracking_scores > threshold).float()
    
    # Create simple temporal targets (consecutive matching)
    T, N, _ = tracking_scores.shape
    temporal_targets = torch.zeros_like(tracking_scores)
    for t in range(T):
        for i in range(min(N, tracking_scores.shape[2])):
            temporal_targets[t, i, i] = 1.0
    
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy().flatten()
    gt_np = temporal_targets.detach().cpu().numpy().flatten()
    
    # Compute metrics
    accuracy = np.mean(pred_np == gt_np)
    
    try:
        precision = precision_score(gt_np, pred_np, average='binary', zero_division=0)
        recall = recall_score(gt_np, pred_np, average='binary', zero_division=0)
        f1 = f1_score(gt_np, pred_np, average='binary', zero_division=0)
    except:
        precision = recall = f1 = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_trajectory_metrics(trajectories):
    """
    Compute trajectory-specific metrics
    """
    if not trajectories:
        return {
            'num_trajectories': 0,
            'avg_length': 0.0,
            'max_length': 0,
            'min_length': 0
        }
    
    lengths = [len(traj) for traj in trajectories]
    
    return {
        'num_trajectories': len(trajectories),
        'avg_length': np.mean(lengths),
        'max_length': max(lengths),
        'min_length': min(lengths)
    }


def compute_association_accuracy(pred_associations, gt_associations, threshold=0.5):
    """
    Compute association accuracy for object tracking
    """
    pred_binary = (pred_associations > threshold).float()
    accuracy = torch.mean((pred_binary == gt_associations).float())
    return accuracy.item()


def compute_tracking_mota(trajectories, gt_trajectories):
    """
    Compute Multiple Object Tracking Accuracy (MOTA)
    Simplified version for demonstration
    """
    if not trajectories or not gt_trajectories:
        return 0.0
    
    # Simplified MOTA computation
    # In practice, this would involve more complex matching and error counting
    pred_length = sum(len(traj) for traj in trajectories)
    gt_length = sum(len(traj) for traj in gt_trajectories)
    
    if gt_length == 0:
        return 0.0
    
    # Simple overlap-based metric
    overlap = min(pred_length, gt_length)
    mota = overlap / gt_length
    
    return mota


def compute_id_switches(trajectories):
    """
    Count identity switches in trajectories
    Simplified version for demonstration
    """
    if not trajectories:
        return 0
    
    # Count trajectory fragments as potential ID switches
    total_fragments = len(trajectories)
    expected_objects = max(max(traj) for traj in trajectories if traj) + 1 if trajectories else 0
    
    # More fragments than objects suggests ID switches
    id_switches = max(0, total_fragments - expected_objects)
    
    return id_switches