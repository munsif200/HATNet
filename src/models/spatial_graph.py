"""
Spatial Graph Model for ReST
Handles spatial associations across multiple views/cameras
"""

import torch
import torch.nn as nn
from .mpn import MultiLayerMPN, SimpleGraph


class SpatialGraph(nn.Module):
    """
    Spatial Graph Model for associating objects across different spatial views.
    This module creates spatial associations between detected objects.
    """
    
    def __init__(self, cfg):
        super(SpatialGraph, self).__init__()
        
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.spatial_dim = cfg.ST.SPATIAL_DIM
        self.node_dim = cfg.GRAPH.NODE_DIM
        self.edge_dim = cfg.GRAPH.EDGE_DIM
        
        # Feature projection layers
        self.spatial_projector = nn.Sequential(
            nn.Linear(self.spatial_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.node_dim),
            nn.LayerNorm(self.node_dim)
        )
        
        # Edge feature encoder for spatial relationships
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, 16),  # x, y, distance, angle
            nn.ReLU(),
            nn.Linear(16, self.edge_dim),
            nn.Tanh()
        )
        
        # Multi-layer Message Passing Network
        self.mpn = MultiLayerMPN(cfg)
        
        # Classification head for spatial association
        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
    
    def create_spatial_graph(self, spatial_features, positions):
        """
        Create a spatial graph from object detections
        Args:
            spatial_features: [N, spatial_dim] object features
            positions: [N, 2] object positions (x, y)
        Returns:
            SimpleGraph with spatial connections
        """
        num_nodes = spatial_features.shape[0]
        
        # Create fully connected graph for spatial reasoning
        edges = []
        edge_features = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edges.append((i, j))
                    
                    # Compute spatial edge features
                    pos_i, pos_j = positions[i], positions[j]
                    distance = torch.norm(pos_i - pos_j)
                    angle = torch.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
                    
                    # Ensure all tensors are scalars
                    edge_feat = torch.tensor([
                        pos_i[0].item(), pos_i[1].item(), 
                        distance.item(), angle.item()
                    ], device=self.device, dtype=torch.float32)
                    edge_features.append(edge_feat)
        
        # Create SimpleGraph
        graph = SimpleGraph(edges, num_nodes, self.device)
        
        if edge_features:
            edge_features = torch.stack(edge_features)
        else:
            edge_features = torch.empty((0, 4), device=self.device)
            
        return graph, edge_features
    
    def forward(self, spatial_features, positions):
        """
        Forward pass for spatial graph processing
        Args:
            spatial_features: [N, spatial_dim] spatial features of objects
            positions: [N, 2] positions of objects
        Returns:
            Dictionary containing:
                - node_embeddings: Updated node embeddings
                - association_scores: Pairwise association scores
                - graph: The spatial graph
        """
        batch_size = spatial_features.shape[0]
        
        if batch_size == 0:
            return {
                'node_embeddings': torch.empty((0, self.node_dim), device=self.device),
                'association_scores': torch.empty((0, 0), device=self.device),
                'graph': None
            }
        
        # Project spatial features to node space
        node_features = self.spatial_projector(spatial_features)
        
        # Create spatial graph
        graph, edge_positions = self.create_spatial_graph(spatial_features, positions)
        
        if graph.num_edges() > 0:
            # Encode edge features
            edge_features = self.edge_encoder(edge_positions)
            
            # Apply message passing
            updated_node_features = self.mpn(graph, node_features, edge_features)
        else:
            # No edges case
            updated_node_features = node_features
        
        # Compute pairwise association scores
        association_scores = self.compute_association_scores(updated_node_features)
        
        return {
            'node_embeddings': updated_node_features,
            'association_scores': association_scores,
            'graph': graph
        }
    
    def compute_association_scores(self, node_embeddings):
        """
        Compute pairwise association scores between nodes
        Args:
            node_embeddings: [N, node_dim] node embeddings
        Returns:
            [N, N] association score matrix
        """
        num_nodes = node_embeddings.shape[0]
        scores = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Concatenate embeddings and classify
                    pair_embedding = torch.cat([node_embeddings[i], node_embeddings[j]])
                    score = self.classifier(pair_embedding)
                    scores[i, j] = score.squeeze()
        
        return scores