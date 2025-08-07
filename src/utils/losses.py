"""
Loss functions for ReST model training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReSTLoss(nn.Module):
    """
    Combined loss function for ReST model
    Includes spatial association loss and temporal tracking loss
    """
    
    def __init__(self, cfg):
        super(ReSTLoss, self).__init__()
        self.cfg = cfg
        self.mode = cfg.SOLVER.TYPE
        
        # Loss weights
        self.spatial_weight = 1.0
        self.temporal_weight = 1.0
        self.trajectory_weight = 0.5
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
    def forward(self, outputs, targets):
        """
        Compute combined loss
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth targets
        Returns:
            Total loss
        """
        # Get device from outputs
        device = 'cpu'
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            device = subvalue.device
                            break
                elif torch.is_tensor(value):
                    device = value.device
                    break
                if device != 'cpu':
                    break
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict = {}
        
        # Spatial loss
        if 'spatial_outputs' in outputs:
            spatial_loss = self.compute_spatial_loss(
                outputs['spatial_outputs'], targets
            )
            total_loss += self.spatial_weight * spatial_loss
            loss_dict['spatial'] = spatial_loss
        
        # Temporal loss
        if 'temporal_outputs' in outputs:
            temporal_loss = self.compute_temporal_loss(
                outputs['temporal_outputs'], targets
            )
            total_loss += self.temporal_weight * temporal_loss
            loss_dict['temporal'] = temporal_loss
        
        # Trajectory loss
        if 'trajectories' in outputs:
            traj_loss = self.compute_trajectory_loss(
                outputs['trajectories'], targets
            )
            total_loss += self.trajectory_weight * traj_loss
            loss_dict['trajectory'] = traj_loss
        
        return total_loss
    
    def compute_spatial_loss(self, spatial_outputs, targets):
        """
        Compute spatial association loss
        """
        association_scores = spatial_outputs['association_scores']
        
        # Create target associations (identity matrix for perfect associations)
        if isinstance(targets, torch.Tensor) and targets.dim() == 2:
            target_associations = targets
        else:
            # Default to identity matrix
            n = association_scores.shape[0]
            target_associations = torch.eye(n, device=association_scores.device)
        
        # Ensure same size
        if target_associations.shape != association_scores.shape:
            min_size = min(target_associations.shape[0], association_scores.shape[0])
            target_associations = target_associations[:min_size, :min_size]
            association_scores = association_scores[:min_size, :min_size]
        
        # Use focal loss for imbalanced associations
        return self.focal_loss(association_scores, target_associations)
    
    def compute_temporal_loss(self, temporal_outputs, targets):
        """
        Compute temporal tracking loss
        """
        tracking_scores = temporal_outputs['tracking_scores']
        
        if tracking_scores.numel() == 0:
            return torch.tensor(0.0, device=tracking_scores.device)
        
        # Create temporal targets (consecutive matching)
        T, N, _ = tracking_scores.shape
        temporal_targets = torch.zeros_like(tracking_scores)
        
        # Simple consecutive matching for demonstration
        for t in range(T):
            for i in range(min(N, tracking_scores.shape[2])):
                temporal_targets[t, i, i] = 1.0
        
        return self.bce_loss(tracking_scores, temporal_targets)
    
    def compute_trajectory_loss(self, trajectories, targets):
        """
        Compute trajectory consistency loss
        """
        if not trajectories:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simple trajectory length penalty (encourage longer trajectories)
        total_length = sum(len(traj) for traj in trajectories)
        expected_length = len(trajectories) * 3  # Expected average length
        
        length_loss = abs(total_length - expected_length) / expected_length
        return torch.tensor(length_loss, device='cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in association learning
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Compute focal loss
        Args:
            inputs: Predicted probabilities [N, ...]
            targets: Ground truth labels [N, ...]
        Returns:
            Focal loss value
        """
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal weights
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weights = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weights
        focal_loss = focal_weights * bce_loss
        
        return focal_loss.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Loss for temporal consistency in tracking
    """
    
    def __init__(self, lambda_temporal=1.0):
        super(TemporalConsistencyLoss, self).__init__()
        self.lambda_temporal = lambda_temporal
    
    def forward(self, embeddings):
        """
        Compute temporal consistency loss
        Args:
            embeddings: [T, N, D] temporal embeddings
        Returns:
            Consistency loss
        """
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute differences between consecutive embeddings
        diff = embeddings[1:] - embeddings[:-1]
        
        # Encourage smoothness
        smoothness_loss = torch.mean(torch.norm(diff, dim=-1))
        
        return self.lambda_temporal * smoothness_loss