"""
Tests for neural network infrastructure.

Tests the NetworkBuilder, DynamicNetwork, and architecture presets for:
- JSON to PyTorch conversion
- Weight serialization/deserialization
- Architecture validation
- Network cloning
"""
import pytest
import torch
import numpy as np

from apps.ai.networks.builder import NetworkBuilder, DynamicNetwork
from apps.ai.networks.architectures import (
    td_gammon_architecture,
    modern_backgammon_architecture,
    create_mlp_architecture,
    minimal_architecture,
    two_headed_architecture,
)


class TestNetworkBuilder:
    """Tests for NetworkBuilder."""

    @pytest.fixture
    def builder(self):
        """Create builder instance."""
        return NetworkBuilder()

    def test_from_json_creates_network(self, builder, minimal_architecture):
        """Test that from_json creates a valid network."""
        network = builder.from_json(minimal_architecture)
        assert isinstance(network, DynamicNetwork)

    def test_from_json_preserves_architecture(self, builder, minimal_architecture):
        """Test that network stores its architecture."""
        network = builder.from_json(minimal_architecture)
        assert network.architecture == minimal_architecture

    def test_from_json_forward_pass(self, builder, minimal_architecture):
        """Test forward pass through built network."""
        network = builder.from_json(minimal_architecture)
        x = torch.randn(1, 198)
        output = network(x)
        assert output.shape == (1, 1)

    def test_from_json_batch_forward(self, builder, minimal_architecture):
        """Test batch forward pass."""
        network = builder.from_json(minimal_architecture)
        x = torch.randn(32, 198)
        output = network(x)
        assert output.shape == (32, 1)

    def test_from_json_td_gammon(self, builder, td_gammon_architecture):
        """Test building TD-Gammon architecture."""
        network = builder.from_json(td_gammon_architecture)
        x = torch.randn(1, 198)
        output = network(x)
        assert output.shape == (1, 1)
        # Output should be in [0, 1] due to sigmoid
        assert 0 <= output.item() <= 1

    def test_from_json_validates_architecture(self, builder):
        """Test that invalid architectures raise errors."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            builder.from_json("not a dict")

        with pytest.raises(ValueError, match="must have 'layers'"):
            builder.from_json({})

        with pytest.raises(ValueError, match="must be a list"):
            builder.from_json({'layers': 'not a list'})

    def test_from_json_layer_validation(self, builder):
        """Test that invalid layers raise errors."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            builder.from_json({'layers': ['not a dict']})

        with pytest.raises(ValueError, match="must have 'type'"):
            builder.from_json({'layers': [{'id': 'no_type'}]})

    def test_from_json_unknown_layer_type(self, builder):
        """Test that unknown layer types raise errors."""
        arch = {
            'layers': [
                {'id': 'bad', 'type': 'unknown_layer'}
            ]
        }
        with pytest.raises(ValueError, match="Unknown layer type"):
            builder.from_json(arch)

    def test_from_json_unknown_activation(self, builder):
        """Test that unknown activations raise errors."""
        arch = {
            'layers': [
                {'id': 'act', 'type': 'activation', 'fn': 'unknown_fn'}
            ]
        }
        with pytest.raises(ValueError, match="Unknown activation"):
            builder.from_json(arch)

    def test_from_json_all_activation_types(self, builder):
        """Test all supported activation functions."""
        for activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softplus']:
            arch = {
                'input_size': 10,
                'output_size': 1,
                'layers': [
                    {'id': 'fc', 'type': 'linear', 'in': 10, 'out': 5},
                    {'id': 'act', 'type': 'activation', 'fn': activation},
                ]
            }
            network = builder.from_json(arch)
            x = torch.randn(1, 10)
            output = network(x)
            assert output.shape == (1, 5)

    def test_from_json_with_batchnorm(self, builder):
        """Test network with batch normalization."""
        arch = {
            'layers': [
                {'id': 'fc', 'type': 'linear', 'in': 10, 'out': 5},
                {'id': 'bn', 'type': 'batchnorm', 'features': 5},
                {'id': 'act', 'type': 'activation', 'fn': 'relu'},
            ]
        }
        network = builder.from_json(arch)
        network.train()  # BatchNorm behaves differently in train/eval
        x = torch.randn(4, 10)  # Need batch size > 1 for batchnorm
        output = network(x)
        assert output.shape == (4, 5)

    def test_from_json_with_dropout(self, builder):
        """Test network with dropout."""
        arch = {
            'layers': [
                {'id': 'fc', 'type': 'linear', 'in': 10, 'out': 5},
                {'id': 'drop', 'type': 'dropout', 'p': 0.5},
            ]
        }
        network = builder.from_json(arch)
        x = torch.randn(1, 10)
        output = network(x)
        assert output.shape == (1, 5)

    def test_to_json_dynamic_network(self, builder, minimal_architecture):
        """Test converting DynamicNetwork back to JSON."""
        network = builder.from_json(minimal_architecture)
        json_arch = builder.to_json(network)
        assert json_arch == minimal_architecture

    def test_serialize_weights(self, builder, minimal_architecture):
        """Test weight serialization produces bytes."""
        network = builder.from_json(minimal_architecture)
        weights = builder.serialize_weights(network)
        assert isinstance(weights, bytes)
        assert len(weights) > 0

    def test_deserialize_weights(self, builder, minimal_architecture):
        """Test weight deserialization."""
        network = builder.from_json(minimal_architecture)

        # Get initial output
        x = torch.randn(1, 198)
        initial_output = network(x).detach().clone()

        # Serialize weights
        weights = builder.serialize_weights(network)

        # Create new network and load weights
        new_network = builder.from_json(minimal_architecture)
        builder.deserialize_weights(weights, new_network)

        # Outputs should match
        new_output = new_network(x)
        assert torch.allclose(initial_output, new_output)

    def test_serialize_deserialize_roundtrip(self, builder, td_gammon_architecture):
        """Test complete round-trip serialization."""
        network = builder.from_json(td_gammon_architecture)

        # Set specific weights for verification
        with torch.no_grad():
            for param in network.parameters():
                param.fill_(0.5)

        weights = builder.serialize_weights(network)
        new_network = builder.from_json(td_gammon_architecture)
        builder.deserialize_weights(weights, new_network)

        # Verify weights match
        for p1, p2 in zip(network.parameters(), new_network.parameters()):
            assert torch.allclose(p1, p2)

    def test_clone_network(self, builder, minimal_architecture):
        """Test network cloning."""
        network = builder.from_json(minimal_architecture)
        clone = builder.clone_network(network)

        # Verify clone is different object
        assert clone is not network

        # Verify weights match
        x = torch.randn(1, 198)
        original_out = network(x)
        clone_out = clone(x)
        assert torch.allclose(original_out, clone_out)

    def test_clone_network_independence(self, builder, minimal_architecture):
        """Test that cloned network is independent."""
        network = builder.from_json(minimal_architecture)
        clone = builder.clone_network(network)

        # Modify clone weights
        with torch.no_grad():
            for param in clone.parameters():
                param.add_(1.0)

        # Original should be unchanged
        x = torch.randn(1, 198)
        original_out = network(x)
        clone_out = clone(x)
        assert not torch.allclose(original_out, clone_out)

    def test_get_parameter_count(self, builder):
        """Test parameter counting."""
        arch = minimal_architecture()
        network = builder.from_json(arch)
        count = builder.get_parameter_count(network)

        # 198 inputs * 1 output + 1 bias = 199 parameters
        assert count == 199

    def test_get_parameter_count_larger(self, builder):
        """Test parameter counting for larger network."""
        arch = td_gammon_architecture()
        network = builder.from_json(arch)
        count = builder.get_parameter_count(network)

        # Hidden layer: 198*80 + 80 bias = 15920
        # Output layer: 80*1 + 1 bias = 81
        # Total: 16001
        assert count == 16001

    def test_get_layer_info(self, builder, minimal_architecture):
        """Test getting layer information."""
        network = builder.from_json(minimal_architecture)
        info = builder.get_layer_info(network)

        assert len(info) == 2  # 2 layers in minimal architecture
        assert info[0]['id'] == 'output'
        assert info[0]['type'] == 'Linear'


class TestDynamicNetwork:
    """Tests for DynamicNetwork."""

    @pytest.fixture
    def network(self):
        """Create a test network."""
        builder = NetworkBuilder()
        return builder.from_json(td_gammon_architecture())

    def test_layer_ids(self, network):
        """Test getting layer IDs."""
        ids = network.layer_ids()
        assert 'hidden' in ids
        assert 'output' in ids

    def test_get_layer(self, network):
        """Test getting a layer by ID."""
        layer = network.get_layer('hidden')
        assert layer is not None
        assert isinstance(layer, torch.nn.Linear)

    def test_get_layer_nonexistent(self, network):
        """Test getting a nonexistent layer."""
        layer = network.get_layer('nonexistent')
        assert layer is None

    def test_forward(self, network):
        """Test forward pass."""
        x = torch.randn(1, 198)
        output = network(x)
        assert output.shape == (1, 1)

    def test_training_mode(self, network):
        """Test training/eval mode."""
        network.train()
        assert network.training

        network.eval()
        assert not network.training


class TestArchitectures:
    """Tests for architecture presets."""

    def test_td_gammon_architecture_default(self):
        """Test default TD-Gammon architecture."""
        arch = td_gammon_architecture()

        assert arch['input_size'] == 198
        assert arch['output_size'] == 1
        assert len(arch['layers']) == 4

    def test_td_gammon_architecture_custom(self):
        """Test custom TD-Gammon architecture."""
        arch = td_gammon_architecture(input_size=100, hidden_size=40)

        assert arch['input_size'] == 100
        assert arch['layers'][0]['in'] == 100
        assert arch['layers'][0]['out'] == 40

    def test_modern_backgammon_architecture_default(self):
        """Test default modern architecture."""
        arch = modern_backgammon_architecture()

        assert arch['input_size'] == 198
        assert arch['output_size'] == 1
        assert 'layers' in arch
        assert len(arch['layers']) > 4  # More complex than TD-Gammon

    def test_modern_backgammon_architecture_with_dropout(self):
        """Test modern architecture with dropout."""
        arch = modern_backgammon_architecture(dropout=0.5)

        # Should have dropout layers
        dropout_layers = [l for l in arch['layers'] if l['type'] == 'dropout']
        assert len(dropout_layers) > 0

    def test_modern_backgammon_architecture_no_batchnorm(self):
        """Test modern architecture without batchnorm."""
        arch = modern_backgammon_architecture(use_batchnorm=False)

        # Should not have batchnorm layers
        bn_layers = [l for l in arch['layers'] if l['type'] == 'batchnorm']
        assert len(bn_layers) == 0

    def test_create_mlp_architecture(self):
        """Test generic MLP creation."""
        arch = create_mlp_architecture(
            input_size=100,
            output_size=10,
            hidden_sizes=[64, 32],
        )

        assert arch['input_size'] == 100
        assert arch['output_size'] == 10

        # Check structure
        builder = NetworkBuilder()
        network = builder.from_json(arch)
        x = torch.randn(1, 100)
        output = network(x)
        assert output.shape == (1, 10)

    def test_create_mlp_with_output_activation(self):
        """Test MLP with output activation."""
        arch = create_mlp_architecture(
            input_size=10,
            output_size=3,
            hidden_sizes=[5],
            output_activation='softmax',
        )

        builder = NetworkBuilder()
        network = builder.from_json(arch)
        x = torch.randn(1, 10)
        output = network(x)

        # Softmax output should sum to 1
        assert output.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_minimal_architecture(self):
        """Test minimal architecture."""
        arch = minimal_architecture()

        assert arch['input_size'] == 198
        assert len(arch['layers']) == 2  # Just linear + sigmoid

    def test_two_headed_architecture(self):
        """Test two-headed architecture."""
        arch = two_headed_architecture()

        assert arch['input_size'] == 198
        assert arch['output_size'] == 1

        # Should have multiple layers for backbone + value head
        assert len(arch['layers']) >= 4

    def test_all_architectures_buildable(self):
        """Test that all architecture presets can be built."""
        builder = NetworkBuilder()

        architectures = [
            td_gammon_architecture(),
            modern_backgammon_architecture(),
            create_mlp_architecture(100, 10, [50]),
            minimal_architecture(),
            two_headed_architecture(),
        ]

        for arch in architectures:
            network = builder.from_json(arch)
            x = torch.randn(1, arch['input_size'])
            output = network(x)
            assert output.shape[1] == arch['output_size']
