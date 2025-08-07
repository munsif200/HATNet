"""
ReST Model: Reconfigurable Spatial-Temporal Graph Model
Main model that combines spatial and temporal graphs
"""

import torch
import torch.nn as nn
from .spatial_graph import SpatialGraph
from .temporal_graph import TemporalGraph


class ReSTModel(nn.Module):
    """
    ReST: A Reconfigurable Spatial-Temporal Graph Model
    
    This model implements the two-stage approach:
    1. Spatial Graph: Associates objects across spatial views
    2. Temporal Graph: Associates spatial detections across time
    """
    
    def __init__(self, cfg):
        super(ReSTModel, self).__init__()
        
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.mode = cfg.SOLVER.TYPE  # 'SG' or 'TG'
        
        # Initialize spatial and temporal graphs
        self.spatial_graph = SpatialGraph(cfg)
        self.temporal_graph = TemporalGraph(cfg)
        
        # Feature fusion layers
        self.spatial_temporal_fusion = nn.Sequential(
            nn.Linear(cfg.GRAPH.NODE_DIM * 2, cfg.GRAPH.NODE_DIM),
            nn.ReLU(),
            nn.Linear(cfg.GRAPH.NODE_DIM, cfg.GRAPH.NODE_DIM),
            nn.LayerNorm(cfg.GRAPH.NODE_DIM)
        )
        
        self.to(self.device)
    
    def forward(self, data):
        """
        Forward pass of the ReST model
        Args:
            data: Dictionary containing:
                - spatial_features: [B, T, N, spatial_dim] or [N, spatial_dim]
                - positions: [B, T, N, 2] or [N, 2] object positions
                - temporal_info: Dictionary with temporal information (for TG mode)
        Returns:
            Dictionary with model outputs
        """
        if self.mode == 'SG':
            return self.forward_spatial(data)
        elif self.mode == 'TG':
            return self.forward_temporal(data)
        else:
            return self.forward_joint(data)
    
    def forward_spatial(self, data):
        """
        Forward pass for spatial graph only
        Args:
            data: Dictionary with spatial_features and positions
        Returns:
            Spatial graph outputs
        """
        spatial_features = data['spatial_features']
        positions = data['positions']
        
        # Handle batch dimension
        if spatial_features.dim() == 4:  # [B, T, N, D]
            batch_size, time_steps, num_objects = spatial_features.shape[:3]
            outputs = []
            
            for b in range(batch_size):
                for t in range(time_steps):
                    sp_feat = spatial_features[b, t]
                    pos = positions[b, t]
                    
                    # Skip empty frames
                    if sp_feat.shape[0] > 0:
                        output = self.spatial_graph(sp_feat, pos)
                        outputs.append(output)
            
            return self.aggregate_spatial_outputs(outputs)
        
        elif spatial_features.dim() == 3:  # [T, N, D] - temporal sequence
            time_steps, num_objects = spatial_features.shape[:2]
            outputs = []
            
            for t in range(time_steps):
                sp_feat = spatial_features[t]
                pos = positions[t]
                
                # Skip empty frames
                if sp_feat.shape[0] > 0:
                    output = self.spatial_graph(sp_feat, pos)
                    outputs.append(output)
            
            return self.aggregate_spatial_outputs(outputs)
        
        else:  # [N, D]
            return self.spatial_graph(spatial_features, positions)
    
    def forward_temporal(self, data):
        """
        Forward pass for temporal graph only
        Args:
            data: Dictionary with spatial embeddings and temporal info
        Returns:
            Temporal graph outputs
        """
        # First get spatial embeddings
        spatial_outputs = self.forward_spatial(data)
        spatial_embeddings = spatial_outputs['node_embeddings']
        
        # Reshape for temporal processing
        if spatial_embeddings.dim() == 2:  # [N, D]
            # Add time dimension
            spatial_embeddings = spatial_embeddings.unsqueeze(0)  # [1, N, D]
        
        temporal_info = data.get('temporal_info', {})
        
        # Apply temporal graph
        temporal_outputs = self.temporal_graph(spatial_embeddings, temporal_info)
        
        return {
            'spatial_outputs': spatial_outputs,
            'temporal_outputs': temporal_outputs,
            'final_embeddings': temporal_outputs['temporal_embeddings'],
            'trajectories': temporal_outputs['trajectories']
        }
    
    def forward_joint(self, data):
        """
        Forward pass for joint spatial-temporal processing
        Args:
            data: Dictionary with all necessary data
        Returns:
            Joint model outputs
        """
        # Get spatial outputs
        spatial_outputs = self.forward_spatial(data)
        
        # Get temporal outputs
        temporal_outputs = self.forward_temporal(data)
        
        # Fuse spatial and temporal features
        spatial_emb = spatial_outputs['node_embeddings']
        temporal_emb = temporal_outputs['temporal_outputs']['temporal_embeddings']
        
        # Handle dimension mismatch
        if temporal_emb.dim() == 3 and spatial_emb.dim() == 2:
            # Take last time step from temporal embeddings
            temporal_emb = temporal_emb[-1]
        
        if spatial_emb.shape[0] == temporal_emb.shape[0]:
            # Fuse features
            fused_features = torch.cat([spatial_emb, temporal_emb], dim=1)
            final_embeddings = self.spatial_temporal_fusion(fused_features)
        else:
            # Fall back to temporal embeddings
            final_embeddings = temporal_emb
        
        return {
            'spatial_outputs': spatial_outputs,
            'temporal_outputs': temporal_outputs,
            'final_embeddings': final_embeddings,
            'trajectories': temporal_outputs['trajectories']
        }
    
    def aggregate_spatial_outputs(self, outputs_list):
        """
        Aggregate outputs from multiple spatial processing steps
        """
        if not outputs_list:
            return {
                'node_embeddings': torch.empty((0, self.cfg.GRAPH.NODE_DIM), device=self.device),
                'association_scores': torch.empty((0, 0), device=self.device),
                'graph': None
            }
        
        # Concatenate all embeddings
        all_embeddings = torch.cat([out['node_embeddings'] for out in outputs_list], dim=0)
        
        # Average association scores (if same size)
        try:
            all_scores = torch.stack([out['association_scores'] for out in outputs_list])
            avg_scores = all_scores.mean(dim=0)
        except:
            avg_scores = outputs_list[0]['association_scores']
        
        return {
            'node_embeddings': all_embeddings,
            'association_scores': avg_scores,
            'graph': outputs_list[0]['graph']  # Return first graph as representative
        }
    
    def predict_associations(self, data, threshold=0.5):
        """
        Predict object associations using trained model
        Args:
            data: Input data dictionary
            threshold: Association threshold
        Returns:
            Predicted associations and trajectories
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
            
            associations = {}
            
            if 'spatial_outputs' in outputs:
                spatial_scores = outputs['spatial_outputs']['association_scores']
                spatial_associations = (spatial_scores > threshold).cpu().numpy()
                associations['spatial'] = spatial_associations
            
            if 'trajectories' in outputs:
                associations['trajectories'] = outputs['trajectories']
            
            return associations