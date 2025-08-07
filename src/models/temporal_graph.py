"""
Temporal Graph Model for ReST
Handles temporal associations across time steps
"""

import torch
import torch.nn as nn
from .mpn import MultiLayerMPN, SimpleGraph


class TemporalGraph(nn.Module):
    """
    Temporal Graph Model for tracking objects across time.
    This module creates temporal associations between spatial detections.
    """
    
    def __init__(self, cfg):
        super(TemporalGraph, self).__init__()
        
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.temporal_dim = cfg.ST.TEMPORAL_DIM
        self.node_dim = cfg.GRAPH.NODE_DIM
        self.edge_dim = cfg.GRAPH.EDGE_DIM
        
        # Temporal feature encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.temporal_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.node_dim),
            nn.LayerNorm(self.node_dim)
        )
        
        # Edge feature encoder for temporal relationships
        self.temporal_edge_encoder = nn.Sequential(
            nn.Linear(6, 16),  # time_diff, velocity_x, velocity_y, distance, appearance_sim, confidence
            nn.ReLU(),
            nn.Linear(16, self.edge_dim),
            nn.Tanh()
        )
        
        # Multi-layer Message Passing Network
        self.mpn = MultiLayerMPN(cfg)
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.node_dim,
            hidden_size=self.node_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Temporal association classifier
        self.temporal_classifier = nn.Sequential(
            nn.Linear(self.node_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
    
    def create_temporal_graph(self, node_features, temporal_info):
        """
        Create a temporal graph from spatial detections across time
        Args:
            node_features: [T, N, node_dim] features across time steps
            temporal_info: Dictionary with temporal information
        Returns:
            SimpleGraph with temporal connections
        """
        T, N = node_features.shape[:2]
        total_nodes = T * N
        
        edges = []
        edge_features = []
        
        # Create temporal connections between consecutive time steps
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    # Connect nodes from time t to time t+1
                    src_idx = t * N + i
                    dst_idx = (t + 1) * N + j
                    
                    edges.append((src_idx, dst_idx))
                    
                    # Compute temporal edge features
                    time_diff = 1.0  # One time step
                    velocity_x = temporal_info.get('velocities', torch.zeros(T, N, 2))[t, i, 0]
                    velocity_y = temporal_info.get('velocities', torch.zeros(T, N, 2))[t, i, 1]
                    distance = temporal_info.get('distances', torch.zeros(T-1, N, N))[t, i, j]
                    appearance_sim = temporal_info.get('appearance_sim', torch.ones(T-1, N, N))[t, i, j]
                    confidence = temporal_info.get('confidences', torch.ones(T, N))[t, i]
                    
                    # Ensure all tensors are scalars
                    edge_feat = torch.tensor([
                        time_diff, velocity_x.item(), velocity_y.item(), 
                        distance.item(), appearance_sim.item(), confidence.item()
                    ], device=self.device, dtype=torch.float32)
                    edge_features.append(edge_feat)
        
        # Create SimpleGraph
        graph = SimpleGraph(edges, total_nodes, self.device)
        
        if edge_features:
            edge_features = torch.stack(edge_features)
        else:
            edge_features = torch.empty((0, 6), device=self.device)
            
        return graph, edge_features
    
    def forward(self, spatial_embeddings, temporal_info):
        """
        Forward pass for temporal graph processing
        Args:
            spatial_embeddings: [T, N, node_dim] spatial embeddings across time
            temporal_info: Dictionary with temporal information
        Returns:
            Dictionary containing:
                - temporal_embeddings: Updated temporal embeddings
                - tracking_scores: Temporal association scores
                - trajectories: Predicted object trajectories
        """
        T, N = spatial_embeddings.shape[:2]
        
        if T == 0 or N == 0:
            return {
                'temporal_embeddings': torch.empty((0, 0, self.node_dim), device=self.device),
                'tracking_scores': torch.empty((0, 0, 0), device=self.device),
                'trajectories': []
            }
        
        # Encode temporal features
        temporal_embeddings = self.temporal_encoder(spatial_embeddings.view(-1, self.node_dim))
        temporal_embeddings = temporal_embeddings.view(T, N, self.node_dim)
        
        # Create temporal graph
        graph, edge_features = self.create_temporal_graph(temporal_embeddings, temporal_info)
        
        if graph.num_edges() > 0:
            # Flatten for graph processing
            flat_embeddings = temporal_embeddings.view(-1, self.node_dim)
            
            # Encode edge features
            temporal_edge_features = self.temporal_edge_encoder(edge_features)
            
            # Apply message passing
            updated_embeddings = self.mpn(graph, flat_embeddings, temporal_edge_features)
            updated_embeddings = updated_embeddings.view(T, N, self.node_dim)
        else:
            updated_embeddings = temporal_embeddings
        
        # Apply LSTM for sequence modeling
        lstm_input = updated_embeddings.permute(1, 0, 2)  # [N, T, node_dim]
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output.permute(1, 0, 2)  # [T, N, node_dim]
        
        # Compute temporal association scores
        tracking_scores = self.compute_tracking_scores(lstm_output)
        
        # Generate trajectories
        trajectories = self.generate_trajectories(tracking_scores)
        
        return {
            'temporal_embeddings': lstm_output,
            'tracking_scores': tracking_scores,
            'trajectories': trajectories
        }
    
    def compute_tracking_scores(self, temporal_embeddings):
        """
        Compute temporal association scores between consecutive time steps
        Args:
            temporal_embeddings: [T, N, node_dim] temporal embeddings
        Returns:
            [T-1, N, N] tracking score tensor
        """
        T, N = temporal_embeddings.shape[:2]
        tracking_scores = torch.zeros((T-1, N, N), device=self.device)
        
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    # Concatenate embeddings from consecutive time steps
                    pair_embedding = torch.cat([
                        temporal_embeddings[t, i],
                        temporal_embeddings[t+1, j]
                    ])
                    score = self.temporal_classifier(pair_embedding)
                    tracking_scores[t, i, j] = score.squeeze()
        
        return tracking_scores
    
    def generate_trajectories(self, tracking_scores, threshold=0.5):
        """
        Generate object trajectories from tracking scores
        Args:
            tracking_scores: [T-1, N, N] tracking scores
            threshold: Association threshold
        Returns:
            List of trajectories (each trajectory is a list of node indices)
        """
        T_minus_1, N, _ = tracking_scores.shape
        T = T_minus_1 + 1
        
        trajectories = []
        used_nodes = set()
        
        # Start from first time step
        for start_node in range(N):
            if start_node in used_nodes:
                continue
                
            trajectory = [start_node]
            current_node = start_node
            used_nodes.add(start_node)
            
            # Follow best associations
            for t in range(T_minus_1):
                best_score = 0
                best_next = -1
                
                for next_node in range(N):
                    score = tracking_scores[t, current_node, next_node].item()
                    if score > threshold and score > best_score:
                        next_key = f"{t+1}_{next_node}"
                        if next_key not in used_nodes:
                            best_score = score
                            best_next = next_node
                
                if best_next != -1:
                    trajectory.append(best_next)
                    used_nodes.add(f"{t+1}_{best_next}")
                    current_node = best_next
                else:
                    # End trajectory if no good association
                    break
            
            if len(trajectory) > 1:  # Only keep trajectories with multiple detections
                trajectories.append(trajectory)
        
        return trajectories