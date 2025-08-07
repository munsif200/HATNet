"""
Message Passing Network (MPN) implementation for ReST
Based on the original ReST paper: A Reconfigurable Spatial-Temporal Graph Model
"""

import torch
import torch.nn as nn

# Simplified graph representation without DGL dependency
class SimpleGraph:
    """Simple graph representation"""
    def __init__(self, edges, num_nodes, device='cuda'):
        self.edges = edges  # List of (src, dst) tuples
        self.num_nodes = num_nodes
        self.device = device
        self._edge_index = None
        
    @property
    def edge_index(self):
        if self._edge_index is None and self.edges:
            src, dst = zip(*self.edges)
            self._edge_index = torch.tensor([src, dst], device=self.device)
        elif not self.edges:
            self._edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        return self._edge_index
    
    def num_edges(self):
        return len(self.edges)


class MPN(nn.Module):
    """Message Passing Neural Network for graph-based spatial-temporal modeling."""

    def __init__(self, cfg):
        super(MPN, self).__init__()
        
        # Configuration
        self.node_dim = cfg.GRAPH.NODE_DIM
        self.edge_dim = cfg.GRAPH.EDGE_DIM
        self.message_dim = cfg.GRAPH.MESSAGE_DIM
        self.device = cfg.MODEL.DEVICE
        
        # Node message encoder: processes node features + edge messages
        self.node_msg_encoder = nn.Sequential(
            nn.Linear(self.node_dim + self.message_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.message_dim),
            nn.ReLU()
        )
        
        # Edge message encoder: processes source + destination + edge features
        self.edge_msg_encoder = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.message_dim),
            nn.ReLU()
        )
        
        self.to(self.device)

    def forward(self, graph, node_features, edge_features):
        """
        Forward pass of the Message Passing Network
        Args:
            graph: SimpleGraph object
            node_features: Node feature tensor [N, node_dim]
            edge_features: Edge feature tensor [E, edge_dim]
        Returns:
            Updated node features
        """
        if graph.num_edges() == 0:
            return node_features
        
        # Get edge indices
        edge_index = graph.edge_index  # [2, num_edges]
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        
        # Get source and destination node features
        src_features = node_features[src_nodes]  # [num_edges, node_dim]
        dst_features = node_features[dst_nodes]  # [num_edges, node_dim]
        
        # Compute edge messages
        edge_input = torch.cat([src_features, dst_features, edge_features], dim=1)
        edge_messages = self.edge_msg_encoder(edge_input)
        
        # Compute node messages
        node_input = torch.cat([dst_features, edge_messages], dim=1)
        node_messages = self.node_msg_encoder(node_input)
        
        # Aggregate messages for each node
        updated_features = node_features.clone()
        for i in range(node_features.shape[0]):
            # Find all messages for node i
            mask = (dst_nodes == i)
            if mask.any():
                aggregated_msg = node_messages[mask].sum(dim=0)
                updated_features[i] = updated_features[i] + aggregated_msg
        
        return updated_features


class MultiLayerMPN(nn.Module):
    """Multi-layer Message Passing Network with residual connections."""
    
    def __init__(self, cfg):
        super(MultiLayerMPN, self).__init__()
        
        self.num_layers = cfg.ST.NUM_LAYERS
        self.node_dim = cfg.GRAPH.NODE_DIM
        
        # Stack of MPN layers
        self.mpn_layers = nn.ModuleList([
            MPN(cfg) for _ in range(self.num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.node_dim) for _ in range(self.num_layers)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(cfg.ST.DROPOUT)
        
    def forward(self, graph, node_features, edge_features):
        """
        Forward pass through multiple MPN layers with residual connections
        """
        h = node_features
        
        for i, (mpn_layer, layer_norm) in enumerate(zip(self.mpn_layers, self.layer_norms)):
            # Apply MPN layer
            h_new = mpn_layer(graph, h, edge_features)
            
            # Residual connection + layer norm + dropout
            h = layer_norm(h + self.dropout(h_new))
            
        return h