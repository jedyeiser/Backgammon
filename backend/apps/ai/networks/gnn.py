"""
Graph Neural Network layers and architectures.

Implements GNN layers for processing graph-structured game states
like Catan boards. Supports heterogeneous graphs with multiple
node types (hexes, vertices, edges).

Key components:
- GNNLayer: Basic message passing layer
- HeteroGNNLayer: For heterogeneous graphs
- CatanGNN: Complete architecture for Catan
- GlobalPooling: Aggregate graph to fixed-size vector

Reference:
    Kipf, T. N., & Welling, M. (2016).
    Semi-supervised classification with graph convolutional networks.
    arXiv preprint arXiv:1609.02907.
"""
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MessagePassingLayer(nn.Module):
    """
    Basic message passing GNN layer.

    Implements: h_v' = UPDATE(h_v, AGGREGATE({h_u : u in N(v)}))

    Uses a simple GCN-style aggregation with learnable transformations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregation: str = 'mean',
        activation: str = 'relu',
    ):
        """
        Initialize the message passing layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            aggregation: 'mean', 'sum', or 'max'.
            activation: Activation function name.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregation = aggregation

        # Transform for node features
        self.node_transform = nn.Linear(in_features, out_features)

        # Transform for messages
        self.message_transform = nn.Linear(in_features, out_features)

        # Combine node with aggregated messages
        self.update_transform = nn.Linear(out_features * 2, out_features)

        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_features].
            edge_index: Edge indices [2, num_edges].

        Returns:
            Updated node features [num_nodes, out_features].
        """
        num_nodes = x.size(0)

        if edge_index.size(1) == 0:
            # No edges - just transform nodes
            return self.activation(self.node_transform(x))

        # Transform node features
        h_nodes = self.node_transform(x)

        # Create messages
        source = edge_index[0]
        target = edge_index[1]
        messages = self.message_transform(x[source])

        # Aggregate messages at each node
        aggregated = self._aggregate(messages, target, num_nodes)

        # Combine and update
        combined = torch.cat([h_nodes, aggregated], dim=-1)
        output = self.update_transform(combined)

        return self.activation(output)

    def _aggregate(
        self,
        messages: 'torch.Tensor',
        target: 'torch.Tensor',
        num_nodes: int,
    ) -> 'torch.Tensor':
        """Aggregate messages at target nodes."""
        out = torch.zeros(num_nodes, messages.size(-1), device=messages.device)

        if self.aggregation == 'sum':
            out.scatter_add_(0, target.unsqueeze(-1).expand_as(messages), messages)
        elif self.aggregation == 'mean':
            # Sum then divide by count
            out.scatter_add_(0, target.unsqueeze(-1).expand_as(messages), messages)
            count = torch.zeros(num_nodes, device=messages.device)
            count.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float))
            count = count.clamp(min=1)
            out = out / count.unsqueeze(-1)
        elif self.aggregation == 'max':
            out.fill_(-float('inf'))
            out.scatter_reduce_(0, target.unsqueeze(-1).expand_as(messages), messages, reduce='amax')
            out[out == -float('inf')] = 0

        return out


class HeteroMessagePassingLayer(nn.Module):
    """
    Heterogeneous message passing for graphs with multiple node types.

    Maintains separate transformations for different edge types
    (e.g., hex_to_vertex, vertex_to_vertex, vertex_to_edge).
    """

    def __init__(
        self,
        node_dims: Dict[str, int],
        hidden_dim: int,
        edge_types: List[Tuple[str, str]],
    ):
        """
        Initialize heterogeneous message passing.

        Args:
            node_dims: Dict mapping node type to feature dimension.
            hidden_dim: Hidden dimension for all transformations.
            edge_types: List of (source_type, target_type) tuples.
        """
        super().__init__()
        self.node_dims = node_dims
        self.hidden_dim = hidden_dim
        self.edge_types = edge_types

        # Node transformations (one per node type)
        self.node_transforms = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim)
            for node_type, dim in node_dims.items()
        })

        # Message transformations (one per edge type)
        self.message_transforms = nn.ModuleDict()
        for src_type, tgt_type in edge_types:
            key = f"{src_type}_to_{tgt_type}"
            self.message_transforms[key] = nn.Linear(hidden_dim, hidden_dim)

        # Update transformations
        self.update_transforms = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim * 2, hidden_dim)
            for node_type in node_dims
        })

        self.activation = nn.ReLU()

    def forward(
        self,
        x_dict: Dict[str, 'torch.Tensor'],
        edge_index_dict: Dict[str, 'torch.Tensor'],
    ) -> Dict[str, 'torch.Tensor']:
        """
        Forward pass for heterogeneous graph.

        Args:
            x_dict: Dict mapping node type to features.
            edge_index_dict: Dict mapping edge type to edge indices.

        Returns:
            Dict mapping node type to updated features.
        """
        # Transform all node features to hidden dim
        h_dict = {
            node_type: self.node_transforms[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Collect messages for each node type
        messages_dict = {node_type: [] for node_type in self.node_dims}

        for (src_type, tgt_type), edge_key in zip(
            self.edge_types,
            [f"{s}_to_{t}" for s, t in self.edge_types]
        ):
            if edge_key not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_key]
            if edge_index.size(1) == 0:
                continue

            source = edge_index[0]
            target = edge_index[1]

            # Get source features
            src_features = h_dict[src_type][source]

            # Transform messages
            messages = self.message_transforms[edge_key](src_features)

            # Aggregate at target nodes
            num_tgt = h_dict[tgt_type].size(0)
            aggregated = torch.zeros(
                num_tgt, self.hidden_dim,
                device=messages.device
            )
            aggregated.scatter_add_(
                0,
                target.unsqueeze(-1).expand_as(messages),
                messages
            )

            messages_dict[tgt_type].append(aggregated)

        # Combine messages and update
        output_dict = {}
        for node_type, h in h_dict.items():
            if messages_dict[node_type]:
                # Sum all message types
                total_messages = sum(messages_dict[node_type])
                combined = torch.cat([h, total_messages], dim=-1)
            else:
                combined = torch.cat([h, torch.zeros_like(h)], dim=-1)

            output_dict[node_type] = self.activation(
                self.update_transforms[node_type](combined)
            )

        return output_dict


class GlobalPooling(nn.Module):
    """
    Pool graph to fixed-size representation.

    Combines node features across the graph using attention-weighted
    averaging or simple statistics.
    """

    def __init__(
        self,
        hidden_dim: int,
        pooling_type: str = 'attention',
    ):
        """
        Initialize global pooling.

        Args:
            hidden_dim: Feature dimension.
            pooling_type: 'attention', 'mean', or 'mean_max'.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Pool node features to single vector.

        Args:
            x: Node features [num_nodes, hidden_dim].

        Returns:
            Pooled features [hidden_dim] or [hidden_dim * 2].
        """
        if self.pooling_type == 'mean':
            return x.mean(dim=0)

        elif self.pooling_type == 'mean_max':
            return torch.cat([x.mean(dim=0), x.max(dim=0)[0]], dim=-1)

        elif self.pooling_type == 'attention':
            # Attention-weighted sum
            weights = self.attention(x)
            weights = F.softmax(weights, dim=0)
            return (weights * x).sum(dim=0)

        return x.mean(dim=0)


class CatanGNN(nn.Module):
    """
    Graph Neural Network for Catan position evaluation.

    Processes the heterogeneous Catan graph (hexes, vertices, edges)
    through multiple message passing layers, then pools to a
    fixed-size representation for value prediction.

    Architecture:
        1. Embed each node type to hidden dimension
        2. Apply N heterogeneous message passing layers
        3. Pool each node type separately
        4. Concatenate with global features
        5. MLP to output value

    Example:
        gnn = CatanGNN(
            hex_dim=13,
            vertex_dim=12,
            edge_dim=6,
            global_dim=20,
            hidden_dim=64,
            num_layers=3,
        )

        graph_tensors = encoder.encode_graph_tensors(game_state, player)
        value = gnn(graph_tensors)
    """

    def __init__(
        self,
        hex_dim: int = 13,
        vertex_dim: int = 12,
        edge_dim: int = 6,
        global_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 3,
        pooling_type: str = 'attention',
        dropout: float = 0.1,
    ):
        """
        Initialize the Catan GNN.

        Args:
            hex_dim: Hex node feature dimension.
            vertex_dim: Vertex node feature dimension.
            edge_dim: Edge node feature dimension.
            global_dim: Global feature dimension.
            hidden_dim: Hidden dimension for all layers.
            num_layers: Number of message passing layers.
            pooling_type: Type of global pooling.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial embeddings
        self.hex_embed = nn.Linear(hex_dim, hidden_dim)
        self.vertex_embed = nn.Linear(vertex_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        # Message passing layers
        node_dims = {
            'hex': hidden_dim,
            'vertex': hidden_dim,
            'edge': hidden_dim,
        }
        edge_types = [
            ('hex', 'vertex'),
            ('vertex', 'hex'),
            ('vertex', 'vertex'),
            ('vertex', 'edge'),
            ('edge', 'vertex'),
        ]

        self.gnn_layers = nn.ModuleList([
            HeteroMessagePassingLayer(node_dims, hidden_dim, edge_types)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                'hex': nn.LayerNorm(hidden_dim),
                'vertex': nn.LayerNorm(hidden_dim),
                'edge': nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

        # Pooling
        self.pooling_type = pooling_type
        pool_out = hidden_dim * 2 if pooling_type == 'mean_max' else hidden_dim
        self.pool_hex = GlobalPooling(hidden_dim, pooling_type)
        self.pool_vertex = GlobalPooling(hidden_dim, pooling_type)
        self.pool_edge = GlobalPooling(hidden_dim, pooling_type)

        # Output MLP
        mlp_input_dim = pool_out * 3 + global_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Store architecture info for serialization
        self.architecture = {
            'name': 'CatanGNN',
            'hex_dim': hex_dim,
            'vertex_dim': vertex_dim,
            'edge_dim': edge_dim,
            'global_dim': global_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'pooling_type': pooling_type,
            'dropout': dropout,
        }

    def forward(
        self,
        graph_data: Dict[str, 'torch.Tensor'],
    ) -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            graph_data: Dictionary with:
                - hex_features: [num_hexes, hex_dim]
                - vertex_features: [num_vertices, vertex_dim]
                - edge_features: [num_edges, edge_dim]
                - hex_to_vertex: [2, num_h2v_edges]
                - vertex_to_vertex: [2, num_v2v_edges]
                - vertex_to_edge: [2, num_v2e_edges]
                - global_features: [global_dim]

        Returns:
            Value prediction [1] (batch_size=1) or [batch, 1].
        """
        # Initial embeddings
        h_hex = self.hex_embed(graph_data['hex_features'])
        h_vertex = self.vertex_embed(graph_data['vertex_features'])
        h_edge = self.edge_embed(graph_data['edge_features'])

        x_dict = {'hex': h_hex, 'vertex': h_vertex, 'edge': h_edge}

        # Build edge index dict
        edge_index_dict = {
            'hex_to_vertex': graph_data.get('hex_to_vertex', torch.zeros(2, 0, dtype=torch.long)),
            'vertex_to_hex': graph_data.get('hex_to_vertex', torch.zeros(2, 0, dtype=torch.long)).flip(0),
            'vertex_to_vertex': graph_data.get('vertex_to_vertex', torch.zeros(2, 0, dtype=torch.long)),
            'vertex_to_edge': graph_data.get('vertex_to_edge', torch.zeros(2, 0, dtype=torch.long)),
            'edge_to_vertex': graph_data.get('vertex_to_edge', torch.zeros(2, 0, dtype=torch.long)).flip(0),
        }

        # Message passing layers with residual connections
        for layer, norms in zip(self.gnn_layers, self.layer_norms):
            x_new = layer(x_dict, edge_index_dict)

            # Residual + LayerNorm
            for node_type in x_dict:
                x_dict[node_type] = norms[node_type](
                    x_dict[node_type] + x_new[node_type]
                )

        # Global pooling
        pooled_hex = self.pool_hex(x_dict['hex'])
        pooled_vertex = self.pool_vertex(x_dict['vertex'])
        pooled_edge = self.pool_edge(x_dict['edge'])

        # Concatenate with global features
        global_feats = graph_data.get('global_features', torch.zeros(20))
        if global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0) if pooled_hex.dim() > 1 else global_feats

        combined = torch.cat([
            pooled_hex,
            pooled_vertex,
            pooled_edge,
            global_feats.squeeze() if global_feats.dim() > 1 else global_feats,
        ], dim=-1)

        # Output value
        value = self.output_mlp(combined)

        return value


def create_catan_gnn(
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs,
) -> CatanGNN:
    """
    Factory function for creating Catan GNN.

    Args:
        hidden_dim: Hidden dimension.
        num_layers: Number of GNN layers.
        **kwargs: Additional arguments.

    Returns:
        Configured CatanGNN instance.
    """
    return CatanGNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs,
    )
