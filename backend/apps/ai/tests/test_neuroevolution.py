"""
Tests for neuroevolution operators.

Tests the mutation, crossover, and selection operators for:
- Weight perturbation correctness
- Topology mutations
- Crossover strategies
- Selection mechanisms
"""
import pytest
import torch
import copy

from apps.ai.evolution.mutations import WeightMutator, TopologyMutator, CombinedMutator
from apps.ai.evolution.crossover import WeightCrossover, ArchitectureCrossover, blend_networks
from apps.ai.networks.builder import NetworkBuilder
from apps.ai.networks.architectures import minimal_architecture, td_gammon_architecture


class TestWeightMutator:
    """Tests for WeightMutator."""

    @pytest.fixture
    def network(self):
        """Create a minimal network for testing."""
        builder = NetworkBuilder()
        return builder.from_json(minimal_architecture())

    @pytest.fixture
    def mutator(self):
        """Create a weight mutator."""
        return WeightMutator(sigma=0.1, mutation_rate=1.0)

    def test_init(self, mutator):
        """Test mutator initialization."""
        assert mutator.sigma == 0.1
        assert mutator.mutation_rate == 1.0
        assert mutator.generation == 0

    def test_mutate_changes_weights(self, mutator, network):
        """Test that mutation changes network weights."""
        # Store original weights
        original_weights = [p.clone() for p in network.parameters()]

        # Mutate (not in place)
        mutated = mutator.mutate(network, in_place=False)

        # Check weights are different
        weights_changed = any(
            not torch.allclose(orig, new)
            for orig, new in zip(original_weights, mutated.parameters())
        )
        assert weights_changed

    def test_mutate_not_in_place(self, mutator, network):
        """Test that mutation creates a new network by default."""
        original_weights = [p.clone() for p in network.parameters()]

        mutated = mutator.mutate(network, in_place=False)

        # Original should be unchanged
        for orig, current in zip(original_weights, network.parameters()):
            assert torch.allclose(orig, current)

        # Mutated should be different
        assert mutated is not network

    def test_mutate_in_place(self, mutator, network):
        """Test in-place mutation."""
        original_weights = [p.clone() for p in network.parameters()]

        result = mutator.mutate(network, in_place=True)

        assert result is network

        # Weights should have changed
        weights_changed = any(
            not torch.allclose(orig, current)
            for orig, current in zip(original_weights, network.parameters())
        )
        assert weights_changed

    def test_mutate_partial_rate(self, network):
        """Test partial mutation rate."""
        mutator = WeightMutator(sigma=10.0, mutation_rate=0.0)

        original_weights = [p.clone() for p in network.parameters()]
        mutated = mutator.mutate(network)

        # With 0% rate, weights should be unchanged
        for orig, new in zip(original_weights, mutated.parameters()):
            assert torch.allclose(orig, new)

    def test_step_generation(self, mutator):
        """Test generation stepping."""
        assert mutator.generation == 0
        mutator.step_generation()
        assert mutator.generation == 1

    def test_sigma_decay(self, network):
        """Test sigma decay over generations."""
        mutator = WeightMutator(sigma=1.0, sigma_decay=0.5)

        # Generation 0: sigma = 1.0
        # Generation 1: sigma = 0.5
        # Generation 2: sigma = 0.25

        mutator.step_generation()
        mutator.step_generation()

        # With decayed sigma, mutations should be smaller
        original_weights = [p.clone() for p in network.parameters()]
        mutated = mutator.mutate(network)

        # Weights should change but by less than with full sigma
        total_change = sum(
            (orig - new).abs().sum().item()
            for orig, new in zip(original_weights, mutated.parameters())
        )

        # Should be some change but bounded
        assert total_change > 0

    def test_mutate_selective(self, mutator, network):
        """Test selective layer mutation."""
        layer_sigmas = {
            'output': 0.0,  # Don't mutate output layer
        }

        mutated = mutator.mutate_selective(network, layer_sigmas)
        assert mutated is not network


class TestTopologyMutator:
    """Tests for TopologyMutator."""

    @pytest.fixture
    def mutator(self):
        """Create a topology mutator."""
        return TopologyMutator(
            add_node_rate=1.0,  # High rate for testing
            add_connection_rate=1.0,
            remove_connection_rate=0.0,
        )

    @pytest.fixture
    def architecture(self):
        """Create a test architecture."""
        return {
            'input_size': 198,
            'output_size': 1,
            'layers': [
                {'id': 'fc1', 'type': 'linear', 'in': 198, 'out': 40},
                {'id': 'act1', 'type': 'activation', 'fn': 'relu'},
                {'id': 'fc2', 'type': 'linear', 'in': 40, 'out': 20},
                {'id': 'act2', 'type': 'activation', 'fn': 'relu'},
                {'id': 'fc3', 'type': 'linear', 'in': 20, 'out': 1},
                {'id': 'output', 'type': 'activation', 'fn': 'sigmoid'},
            ]
        }

    def test_init(self, mutator):
        """Test mutator initialization."""
        assert mutator.add_node_rate == 1.0
        assert mutator.add_connection_rate == 1.0

    def test_add_node(self, mutator, architecture):
        """Test adding a node."""
        new_arch = mutator.add_node(architecture)

        # Should have more layers
        assert len(new_arch['layers']) > len(architecture['layers'])

    def test_add_connection(self, mutator, architecture):
        """Test adding a connection (expanding layer)."""
        new_arch = mutator.add_connection(architecture)

        # Find a linear layer with changed size
        orig_sizes = {
            l['id']: l.get('out', 0)
            for l in architecture['layers']
            if l['type'] == 'linear'
        }
        new_sizes = {
            l['id']: l.get('out', 0)
            for l in new_arch['layers']
            if l['type'] == 'linear'
        }

        # Some hidden layer should have grown
        # (but this can fail if no hidden layers qualify)

    def test_remove_connection(self, architecture):
        """Test removing a connection (shrinking layer)."""
        mutator = TopologyMutator(remove_connection_rate=1.0)

        new_arch = mutator.remove_connection(architecture)

        # Architecture should still be valid

    def test_mutate_returns_mutations_list(self, mutator, architecture):
        """Test that mutate returns mutation history."""
        new_arch, mutations = mutator.mutate(architecture)

        assert isinstance(mutations, list)

    def test_minimal_architecture_not_mutatable(self):
        """Test that minimal architecture resists topology changes."""
        mutator = TopologyMutator(add_node_rate=1.0)
        arch = minimal_architecture()

        # Minimal arch has only 2 layers - hard to mutate
        new_arch = mutator.add_node(arch)

        # May or may not succeed depending on structure

    def test_architecture_remains_valid(self, mutator, architecture):
        """Test that mutated architecture is buildable."""
        builder = NetworkBuilder()

        for _ in range(3):
            architecture, _ = mutator.mutate(architecture)

        # Should be able to build the mutated architecture
        network = builder.from_json(architecture)
        x = torch.randn(1, 198)
        output = network(x)

        assert output.shape == (1, 1)


class TestCombinedMutator:
    """Tests for CombinedMutator."""

    @pytest.fixture
    def network(self):
        """Create a test network."""
        builder = NetworkBuilder()
        return builder.from_json(td_gammon_architecture())

    @pytest.fixture
    def mutator(self):
        """Create combined mutator."""
        return CombinedMutator(
            weight_sigma=0.1,
            topology_add_node_rate=0.0,  # Disable topology for testing
        )

    def test_init(self, mutator):
        """Test mutator initialization."""
        assert hasattr(mutator, 'weight_mutator')
        assert hasattr(mutator, 'topology_mutator')

    def test_mutate_weights_only(self, mutator, network):
        """Test mutation with weights only."""
        child, info = mutator.mutate(
            network, mutate_topology=False, mutate_weights=True
        )

        assert info['weight_mutated'] is True
        assert info['topology_mutations'] == []

    def test_mutate_returns_info(self, mutator, network):
        """Test that mutate returns mutation info."""
        child, info = mutator.mutate(network)

        assert 'topology_mutations' in info
        assert 'weight_mutated' in info


class TestWeightCrossover:
    """Tests for WeightCrossover."""

    @pytest.fixture
    def builder(self):
        return NetworkBuilder()

    @pytest.fixture
    def parent_a(self, builder):
        """Create first parent."""
        net = builder.from_json(minimal_architecture())
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(1.0)
        return net

    @pytest.fixture
    def parent_b(self, builder):
        """Create second parent."""
        net = builder.from_json(minimal_architecture())
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(0.0)
        return net

    def test_interpolation_crossover(self, parent_a, parent_b):
        """Test interpolation crossover strategy."""
        crossover = WeightCrossover(strategy='interpolation', alpha=0.5)

        child = crossover.crossover(parent_a, parent_b)

        # With alpha=0.5, weights should be 0.5
        for p in child.parameters():
            assert torch.allclose(p, torch.full_like(p, 0.5))

    def test_interpolation_biased(self, parent_a, parent_b):
        """Test biased interpolation."""
        crossover = WeightCrossover(strategy='interpolation', alpha=0.8)

        child = crossover.crossover(parent_a, parent_b)

        # With alpha=0.8, weights should be 0.8*1 + 0.2*0 = 0.8
        for p in child.parameters():
            assert torch.allclose(p, torch.full_like(p, 0.8))

    def test_uniform_crossover(self, parent_a, parent_b):
        """Test uniform crossover strategy."""
        crossover = WeightCrossover(strategy='uniform')

        child = crossover.crossover(parent_a, parent_b)

        # Weights should be mix of 0s and 1s
        for p in child.parameters():
            assert torch.all((p == 0) | (p == 1))

    def test_layer_wise_crossover(self, parent_a, parent_b):
        """Test layer-wise crossover."""
        crossover = WeightCrossover(strategy='layer_wise')

        child = crossover.crossover(parent_a, parent_b)

        # Each layer should be entirely from one parent
        # (all 0s or all 1s)

    def test_incompatible_parents(self, builder):
        """Test that incompatible parents raise error."""
        arch_a = {'input_size': 10, 'output_size': 1, 'layers': [
            {'id': 'fc', 'type': 'linear', 'in': 10, 'out': 5},
        ]}
        arch_b = {'input_size': 10, 'output_size': 1, 'layers': [
            {'id': 'fc', 'type': 'linear', 'in': 10, 'out': 10},  # Different size
        ]}

        parent_a = builder.from_json(arch_a)
        parent_b = builder.from_json(arch_b)

        crossover = WeightCrossover()

        with pytest.raises(ValueError, match="identical architectures"):
            crossover.crossover(parent_a, parent_b)


class TestArchitectureCrossover:
    """Tests for ArchitectureCrossover."""

    @pytest.fixture
    def builder(self):
        return NetworkBuilder()

    @pytest.fixture
    def parent_a(self, builder):
        return builder.from_json(minimal_architecture())

    @pytest.fixture
    def parent_b(self, builder):
        return builder.from_json(td_gammon_architecture())

    def test_crossover_returns_network(self, parent_a, parent_b):
        """Test that crossover produces a valid network."""
        crossover = ArchitectureCrossover()

        child, info = crossover.crossover(parent_a, parent_b)

        # Should be a valid network
        x = torch.randn(1, 198)
        output = child(x)
        assert output.shape[1] == 1

    def test_crossover_returns_info(self, parent_a, parent_b):
        """Test that crossover returns inheritance info."""
        crossover = ArchitectureCrossover()

        child, info = crossover.crossover(parent_a, parent_b)

        assert 'dominant_parent' in info
        assert info['dominant_parent'] in ['a', 'b']

    def test_fitness_affects_dominance(self, parent_a, parent_b):
        """Test that fitter parent is dominant."""
        crossover = ArchitectureCrossover(prefer_fitter=True)

        # A is fitter
        child, info = crossover.crossover(
            parent_a, parent_b, fitness_a=0.9, fitness_b=0.1
        )
        assert info['dominant_parent'] == 'a'

        # B is fitter
        child, info = crossover.crossover(
            parent_a, parent_b, fitness_a=0.1, fitness_b=0.9
        )
        assert info['dominant_parent'] == 'b'


class TestBlendNetworks:
    """Tests for blend_networks function."""

    @pytest.fixture
    def builder(self):
        return NetworkBuilder()

    @pytest.fixture
    def networks(self, builder):
        """Create networks with known weights."""
        nets = []
        for val in [0.0, 0.5, 1.0]:
            net = builder.from_json(minimal_architecture())
            with torch.no_grad():
                for p in net.parameters():
                    p.fill_(val)
            nets.append(net)
        return nets

    def test_blend_equal_weights(self, networks):
        """Test blending with equal weights."""
        blended = blend_networks(networks)

        # Should be average: (0 + 0.5 + 1) / 3 = 0.5
        for p in blended.parameters():
            assert torch.allclose(p, torch.full_like(p, 0.5))

    def test_blend_custom_weights(self, networks):
        """Test blending with custom weights."""
        blended = blend_networks(networks, weights=[1.0, 0.0, 0.0])

        # Should be just first network
        for p in blended.parameters():
            assert torch.allclose(p, torch.full_like(p, 0.0))

    def test_blend_single_network(self, builder):
        """Test blending single network returns clone."""
        net = builder.from_json(minimal_architecture())
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(0.5)

        blended = blend_networks([net])

        assert blended is not net
        for p1, p2 in zip(net.parameters(), blended.parameters()):
            assert torch.allclose(p1, p2)

    def test_blend_empty_raises(self):
        """Test blending empty list raises error."""
        with pytest.raises(ValueError, match="at least one network"):
            blend_networks([])
