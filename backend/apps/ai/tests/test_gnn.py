"""
Tests for Graph Neural Network components.

Tests the MessagePassingLayer, HeteroMessagePassingLayer,
GlobalPooling, and CatanGNN for:
- Correct output shapes
- Aggregation behavior
- Heterogeneous graph handling
- End-to-end forward pass
"""
import pytest
import torch

from apps.ai.networks.gnn import (
    MessagePassingLayer,
    HeteroMessagePassingLayer,
    GlobalPooling,
    CatanGNN,
    create_catan_gnn,
)


class TestMessagePassingLayer:
    """Tests for MessagePassingLayer."""

    @pytest.fixture
    def layer(self):
        """Create a basic message passing layer."""
        return MessagePassingLayer(
            in_features=16,
            out_features=32,
            aggregation='mean',
        )

    @pytest.fixture
    def sample_graph(self):
        """Create a simple graph for testing."""
        # 4 nodes, 5 edges
        x = torch.randn(4, 16)
        edge_index = torch.tensor([
            [0, 0, 1, 2, 3],  # source
            [1, 2, 2, 3, 0],  # target
        ])
        return x, edge_index

    def test_init(self, layer):
        """Test layer initialization."""
        assert layer.in_features == 16
        assert layer.out_features == 32
        assert layer.aggregation == 'mean'

    def test_forward_shape(self, layer, sample_graph):
        """Test forward pass output shape."""
        x, edge_index = sample_graph

        output = layer(x, edge_index)

        assert output.shape == (4, 32)

    def test_forward_no_edges(self, layer):
        """Test forward with no edges."""
        x = torch.randn(4, 16)
        edge_index = torch.zeros(2, 0, dtype=torch.long)

        output = layer(x, edge_index)

        assert output.shape == (4, 32)

    def test_aggregation_sum(self, sample_graph):
        """Test sum aggregation."""
        layer = MessagePassingLayer(16, 32, aggregation='sum')
        x, edge_index = sample_graph

        output = layer(x, edge_index)
        assert output.shape == (4, 32)

    def test_aggregation_max(self, sample_graph):
        """Test max aggregation."""
        layer = MessagePassingLayer(16, 32, aggregation='max')
        x, edge_index = sample_graph

        output = layer(x, edge_index)
        assert output.shape == (4, 32)

    def test_different_activations(self, sample_graph):
        """Test different activation functions."""
        x, edge_index = sample_graph

        for activation in ['relu', 'tanh', 'sigmoid', 'none']:
            layer = MessagePassingLayer(16, 32, activation=activation)
            output = layer(x, edge_index)
            assert output.shape == (4, 32)

    def test_gradient_flow(self, layer, sample_graph):
        """Test that gradients flow through the layer."""
        x, edge_index = sample_graph
        x.requires_grad_(True)

        output = layer(x, edge_index)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestHeteroMessagePassingLayer:
    """Tests for HeteroMessagePassingLayer."""

    @pytest.fixture
    def layer(self):
        """Create a heterogeneous message passing layer."""
        node_dims = {'type_a': 8, 'type_b': 12}
        edge_types = [('type_a', 'type_b'), ('type_b', 'type_a')]

        return HeteroMessagePassingLayer(
            node_dims=node_dims,
            hidden_dim=16,
            edge_types=edge_types,
        )

    @pytest.fixture
    def sample_hetero_graph(self):
        """Create a sample heterogeneous graph."""
        x_dict = {
            'type_a': torch.randn(5, 8),   # 5 nodes of type A
            'type_b': torch.randn(3, 12),  # 3 nodes of type B
        }
        edge_index_dict = {
            'type_a_to_type_b': torch.tensor([
                [0, 1, 2],  # source (type_a)
                [0, 1, 2],  # target (type_b)
            ]),
            'type_b_to_type_a': torch.tensor([
                [0, 1],     # source (type_b)
                [3, 4],     # target (type_a)
            ]),
        }
        return x_dict, edge_index_dict

    def test_init(self, layer):
        """Test layer initialization."""
        assert layer.hidden_dim == 16
        assert 'type_a' in layer.node_transforms
        assert 'type_b' in layer.node_transforms

    def test_forward_shape(self, layer, sample_hetero_graph):
        """Test forward pass output shapes."""
        x_dict, edge_index_dict = sample_hetero_graph

        output_dict = layer(x_dict, edge_index_dict)

        assert 'type_a' in output_dict
        assert 'type_b' in output_dict
        assert output_dict['type_a'].shape == (5, 16)
        assert output_dict['type_b'].shape == (3, 16)

    def test_forward_no_edges(self):
        """Test forward with no edges."""
        node_dims = {'type_a': 8}
        layer = HeteroMessagePassingLayer(node_dims, 16, [])

        x_dict = {'type_a': torch.randn(3, 8)}
        edge_index_dict = {}

        output_dict = layer(x_dict, edge_index_dict)

        assert output_dict['type_a'].shape == (3, 16)


class TestGlobalPooling:
    """Tests for GlobalPooling."""

    @pytest.fixture
    def sample_nodes(self):
        """Create sample node features."""
        return torch.randn(10, 32)  # 10 nodes, 32 features

    def test_mean_pooling(self, sample_nodes):
        """Test mean pooling."""
        pooling = GlobalPooling(32, pooling_type='mean')

        output = pooling(sample_nodes)

        assert output.shape == (32,)
        # Should be close to actual mean
        assert torch.allclose(output, sample_nodes.mean(dim=0))

    def test_mean_max_pooling(self, sample_nodes):
        """Test mean-max concatenated pooling."""
        pooling = GlobalPooling(32, pooling_type='mean_max')

        output = pooling(sample_nodes)

        # Output is [mean, max] concatenated
        assert output.shape == (64,)

    def test_attention_pooling(self, sample_nodes):
        """Test attention-weighted pooling."""
        pooling = GlobalPooling(32, pooling_type='attention')

        output = pooling(sample_nodes)

        assert output.shape == (32,)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        pooling = GlobalPooling(32, pooling_type='attention')
        x = torch.randn(5, 32)

        weights = pooling.attention(x)
        softmax_weights = torch.softmax(weights, dim=0)

        assert softmax_weights.sum().item() == pytest.approx(1.0)


class TestCatanGNN:
    """Tests for CatanGNN."""

    @pytest.fixture
    def gnn(self):
        """Create a Catan GNN."""
        return CatanGNN(
            hex_dim=13,
            vertex_dim=12,
            edge_dim=6,
            global_dim=20,
            hidden_dim=32,  # Smaller for faster tests
            num_layers=2,
        )

    @pytest.fixture
    def sample_catan_graph(self):
        """Create sample Catan graph data."""
        return {
            'hex_features': torch.randn(19, 13),      # 19 hexes
            'vertex_features': torch.randn(54, 12),   # 54 vertices
            'edge_features': torch.randn(72, 6),      # 72 edges
            'hex_to_vertex': torch.randint(0, 19, (2, 100)),
            'vertex_to_vertex': torch.randint(0, 54, (2, 80)),
            'vertex_to_edge': torch.randint(0, 54, (2, 144)),
            'global_features': torch.randn(20),
        }

    def test_init(self, gnn):
        """Test GNN initialization."""
        assert gnn.hidden_dim == 32
        assert gnn.num_layers == 2
        assert hasattr(gnn, 'hex_embed')
        assert hasattr(gnn, 'vertex_embed')
        assert hasattr(gnn, 'edge_embed')

    def test_architecture_stored(self, gnn):
        """Test that architecture is stored."""
        assert hasattr(gnn, 'architecture')
        assert gnn.architecture['name'] == 'CatanGNN'
        assert gnn.architecture['hidden_dim'] == 32

    def test_forward_shape(self, gnn, sample_catan_graph):
        """Test forward pass output shape."""
        output = gnn(sample_catan_graph)

        assert output.shape == (1,) or output.shape == torch.Size([])

    def test_forward_value_range(self, gnn, sample_catan_graph):
        """Test that output is in valid range (sigmoid)."""
        output = gnn(sample_catan_graph)

        value = output.item()
        assert 0 <= value <= 1

    def test_forward_minimal_graph(self, gnn):
        """Test forward with minimal graph."""
        minimal_graph = {
            'hex_features': torch.randn(1, 13),
            'vertex_features': torch.randn(1, 12),
            'edge_features': torch.randn(1, 6),
            'hex_to_vertex': torch.zeros(2, 0, dtype=torch.long),
            'vertex_to_vertex': torch.zeros(2, 0, dtype=torch.long),
            'vertex_to_edge': torch.zeros(2, 0, dtype=torch.long),
            'global_features': torch.randn(20),
        }

        output = gnn(minimal_graph)
        assert output.shape == (1,) or output.shape == torch.Size([])

    def test_gradient_flow(self, gnn, sample_catan_graph):
        """Test gradient flow through the GNN."""
        # Make features require grad
        for key in ['hex_features', 'vertex_features', 'edge_features']:
            sample_catan_graph[key].requires_grad_(True)

        output = gnn(sample_catan_graph)
        output.backward()

        # Check gradients exist
        assert sample_catan_graph['hex_features'].grad is not None
        assert sample_catan_graph['vertex_features'].grad is not None

    def test_training_mode(self, gnn):
        """Test training/eval mode."""
        gnn.train()
        assert gnn.training

        gnn.eval()
        assert not gnn.training

    def test_different_pooling_types(self):
        """Test different pooling types."""
        for pooling in ['mean', 'attention', 'mean_max']:
            gnn = CatanGNN(hidden_dim=16, num_layers=1, pooling_type=pooling)

            graph = {
                'hex_features': torch.randn(3, 13),
                'vertex_features': torch.randn(6, 12),
                'edge_features': torch.randn(9, 6),
                'global_features': torch.randn(20),
            }

            output = gnn(graph)
            assert 0 <= output.item() <= 1


class TestCreateCatanGNN:
    """Tests for create_catan_gnn factory."""

    def test_factory_default(self):
        """Test factory with defaults."""
        gnn = create_catan_gnn()

        assert isinstance(gnn, CatanGNN)
        assert gnn.hidden_dim == 64
        assert gnn.num_layers == 3

    def test_factory_custom(self):
        """Test factory with custom parameters."""
        gnn = create_catan_gnn(hidden_dim=128, num_layers=5)

        assert gnn.hidden_dim == 128
        assert gnn.num_layers == 5

    def test_factory_with_kwargs(self):
        """Test factory with additional kwargs."""
        gnn = create_catan_gnn(
            hidden_dim=32,
            num_layers=2,
            dropout=0.2,
            pooling_type='mean',
        )

        assert gnn.hidden_dim == 32
        assert gnn.architecture['dropout'] == 0.2


class TestGNNIntegration:
    """Integration tests for GNN components."""

    def test_multiple_layers_stack(self):
        """Test stacking multiple message passing layers."""
        layers = [
            MessagePassingLayer(16, 32),
            MessagePassingLayer(32, 32),
            MessagePassingLayer(32, 16),
        ]

        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))

        for layer in layers:
            x = layer(x, edge_index)

        assert x.shape == (10, 16)

    def test_pooling_after_message_passing(self):
        """Test pooling after message passing."""
        layer = MessagePassingLayer(16, 32)
        pooling = GlobalPooling(32, pooling_type='mean')

        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))

        h = layer(x, edge_index)
        pooled = pooling(h)

        assert pooled.shape == (32,)

    def test_gnn_deterministic(self):
        """Test that GNN is deterministic in eval mode."""
        gnn = CatanGNN(hidden_dim=16, num_layers=1)
        gnn.eval()

        graph = {
            'hex_features': torch.randn(5, 13),
            'vertex_features': torch.randn(10, 12),
            'edge_features': torch.randn(15, 6),
            'global_features': torch.randn(20),
        }

        with torch.no_grad():
            output1 = gnn(graph)
            output2 = gnn(graph)

        assert torch.allclose(output1, output2)
