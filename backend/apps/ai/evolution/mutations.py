"""
Network mutation operators for neuroevolution.

Implements two types of mutations:
1. Weight perturbation: Add Gaussian noise to weights
2. Topology mutations (NEAT-style): Add/remove nodes and connections

These mutations can evolve both the weights and structure of
neural networks, enabling the discovery of novel architectures.

Reference:
    Stanley, K. O., & Miikkulainen, R. (2002).
    Evolving neural networks through augmenting topologies.
    Evolutionary computation, 10(2), 99-127.
"""
import copy
import random
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class WeightMutator:
    """
    Weight perturbation mutation operator.

    Adds Gaussian noise to network weights. This is the simplest
    form of neuroevolution - keeping topology fixed while
    evolving weights.

    Attributes:
        sigma: Standard deviation of Gaussian noise.
        mutation_rate: Probability of mutating each weight.

    Example:
        mutator = WeightMutator(sigma=0.1, mutation_rate=1.0)
        child_network = mutator.mutate(parent_network)
    """

    def __init__(
        self,
        sigma: float = 0.1,
        mutation_rate: float = 1.0,
        sigma_decay: float = 1.0,
    ):
        """
        Initialize the weight mutator.

        Args:
            sigma: Standard deviation of Gaussian noise.
            mutation_rate: Probability of mutating each weight (0-1).
                          1.0 = mutate all weights
                          0.1 = mutate 10% of weights
            sigma_decay: Multiplicative decay for sigma over generations.
        """
        self.sigma = sigma
        self.mutation_rate = mutation_rate
        self.sigma_decay = sigma_decay
        self.generation = 0

    def mutate(
        self,
        network: 'nn.Module',
        in_place: bool = False,
    ) -> 'nn.Module':
        """
        Apply weight perturbation to a network.

        Args:
            network: The network to mutate.
            in_place: If True, modify network in place.
                     If False, return a new mutated copy.

        Returns:
            Mutated network (same object if in_place=True).
        """
        import torch

        if not in_place:
            network = self._clone_network(network)

        current_sigma = self.sigma * (self.sigma_decay ** self.generation)

        with torch.no_grad():
            for param in network.parameters():
                if self.mutation_rate >= 1.0:
                    # Mutate all weights
                    noise = torch.randn_like(param) * current_sigma
                    param.add_(noise)
                else:
                    # Mutate subset of weights
                    mask = torch.rand_like(param) < self.mutation_rate
                    noise = torch.randn_like(param) * current_sigma
                    param.add_(noise * mask.float())

        return network

    def mutate_selective(
        self,
        network: 'nn.Module',
        layer_sigmas: Dict[str, float],
    ) -> 'nn.Module':
        """
        Apply different mutation rates to different layers.

        Args:
            network: Network to mutate.
            layer_sigmas: Dict mapping layer names to sigma values.

        Returns:
            New mutated network.
        """
        import torch

        network = self._clone_network(network)

        with torch.no_grad():
            for name, param in network.named_parameters():
                # Find matching sigma
                sigma = self.sigma
                for layer_name, layer_sigma in layer_sigmas.items():
                    if layer_name in name:
                        sigma = layer_sigma
                        break

                noise = torch.randn_like(param) * sigma
                param.add_(noise)

        return network

    def _clone_network(self, network: 'nn.Module') -> 'nn.Module':
        """Create a deep copy of a network."""
        from ..networks import NetworkBuilder

        builder = NetworkBuilder()

        # If it's a DynamicNetwork, use architecture to rebuild
        if hasattr(network, 'architecture'):
            new_network = builder.from_json(network.architecture)
            weights = builder.serialize_weights(network)
            builder.deserialize_weights(weights, new_network)
            return new_network

        # Otherwise, use state dict
        import copy
        new_network = copy.deepcopy(network)
        return new_network

    def step_generation(self) -> None:
        """Increment generation counter for sigma decay."""
        self.generation += 1


class TopologyMutator:
    """
    NEAT-style topology mutation operator.

    Can add/remove nodes and connections to evolve network structure.
    This allows networks to start simple and grow in complexity.

    Key operations:
    - add_node: Split a connection by inserting a new node
    - add_connection: Add a new connection between nodes
    - remove_connection: Remove an existing connection

    Example:
        mutator = TopologyMutator(add_node_rate=0.03, add_conn_rate=0.05)
        new_arch = mutator.mutate(architecture)
    """

    def __init__(
        self,
        add_node_rate: float = 0.03,
        add_connection_rate: float = 0.05,
        remove_connection_rate: float = 0.01,
        activation_options: List[str] = None,
    ):
        """
        Initialize the topology mutator.

        Args:
            add_node_rate: Probability of adding a node.
            add_connection_rate: Probability of adding a connection.
            remove_connection_rate: Probability of removing a connection.
            activation_options: Activation functions to use for new nodes.
        """
        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate
        self.remove_connection_rate = remove_connection_rate
        self.activation_options = activation_options or ['relu', 'sigmoid', 'tanh']

        # Counter for unique node IDs
        self._node_counter = 0

    def mutate(
        self,
        architecture: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Apply topology mutations to an architecture.

        Args:
            architecture: JSON architecture specification.

        Returns:
            Tuple of (new architecture, list of mutations applied).
        """
        arch = copy.deepcopy(architecture)
        mutations_applied = []

        # Try each mutation type
        if random.random() < self.add_node_rate:
            arch, success = self._add_node(arch)
            if success:
                mutations_applied.append('add_node')

        if random.random() < self.add_connection_rate:
            arch, success = self._add_connection(arch)
            if success:
                mutations_applied.append('add_connection')

        if random.random() < self.remove_connection_rate:
            arch, success = self._remove_connection(arch)
            if success:
                mutations_applied.append('remove_connection')

        return arch, mutations_applied

    def add_node(
        self,
        architecture: Dict[str, Any],
        target_layer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a new node by splitting a linear layer.

        Inserts a new hidden layer between two existing layers,
        initializing weights to preserve the original behavior.

        Args:
            architecture: Architecture to modify.
            target_layer_id: Specific layer to split (random if None).

        Returns:
            New architecture with added node.
        """
        arch, _ = self._add_node(copy.deepcopy(architecture), target_layer_id)
        return arch

    def _add_node(
        self,
        arch: Dict[str, Any],
        target_layer_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """Internal add node implementation."""
        layers = arch.get('layers', [])

        # Find linear layers that can be split
        linear_layers = [
            (i, layer) for i, layer in enumerate(layers)
            if layer.get('type') == 'linear'
        ]

        if len(linear_layers) <= 1:
            # Need at least 2 linear layers (input and output)
            return arch, False

        # Don't split the output layer
        splittable = linear_layers[:-1]

        if not splittable:
            return arch, False

        # Choose layer to split
        if target_layer_id:
            target = next(
                ((i, l) for i, l in splittable if l.get('id') == target_layer_id),
                None
            )
            if target is None:
                return arch, False
            idx, layer = target
        else:
            idx, layer = random.choice(splittable)

        # Create new node (splitting the layer)
        in_features = layer['in']
        out_features = layer['out']
        new_size = max(1, out_features // 2)  # New intermediate size

        self._node_counter += 1
        new_layer_id = f"evolved_{self._node_counter}"

        # Replace original layer with two layers
        new_layer_1 = {
            'id': layer['id'],
            'type': 'linear',
            'in': in_features,
            'out': new_size,
        }

        new_activation = {
            'id': f"{new_layer_id}_act",
            'type': 'activation',
            'fn': random.choice(self.activation_options),
        }

        new_layer_2 = {
            'id': new_layer_id,
            'type': 'linear',
            'in': new_size,
            'out': out_features,
        }

        # Insert new layers
        layers[idx] = new_layer_1
        layers.insert(idx + 1, new_activation)
        layers.insert(idx + 2, new_layer_2)

        return arch, True

    def add_connection(
        self,
        architecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Add a new connection (expand a layer's width).

        For feedforward networks, this increases the width
        of a hidden layer.

        Args:
            architecture: Architecture to modify.

        Returns:
            New architecture with expanded layer.
        """
        arch, _ = self._add_connection(copy.deepcopy(architecture))
        return arch

    def _add_connection(
        self,
        arch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """Internal add connection implementation."""
        layers = arch.get('layers', [])

        # Find hidden linear layers (not input or output)
        linear_layers = [
            (i, layer) for i, layer in enumerate(layers)
            if layer.get('type') == 'linear'
        ]

        if len(linear_layers) < 3:
            return arch, False

        # Choose a hidden layer to expand
        hidden_layers = linear_layers[1:-1]

        if not hidden_layers:
            return arch, False

        idx, layer = random.choice(hidden_layers)

        # Increase layer width
        old_out = layer['out']
        new_out = old_out + max(1, old_out // 4)  # Increase by 25%

        layer['out'] = new_out

        # Update next layer's input size
        for next_layer in layers[idx + 1:]:
            if next_layer.get('type') == 'linear':
                next_layer['in'] = new_out
                break
            elif next_layer.get('type') == 'batchnorm':
                next_layer['features'] = new_out

        return arch, True

    def remove_connection(
        self,
        architecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Remove a connection (reduce layer width).

        Args:
            architecture: Architecture to modify.

        Returns:
            New architecture with reduced layer.
        """
        arch, _ = self._remove_connection(copy.deepcopy(architecture))
        return arch

    def _remove_connection(
        self,
        arch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """Internal remove connection implementation."""
        layers = arch.get('layers', [])

        # Find hidden linear layers
        linear_layers = [
            (i, layer) for i, layer in enumerate(layers)
            if layer.get('type') == 'linear'
        ]

        if len(linear_layers) < 3:
            return arch, False

        hidden_layers = linear_layers[1:-1]

        if not hidden_layers:
            return arch, False

        idx, layer = random.choice(hidden_layers)

        # Decrease layer width (minimum 1)
        old_out = layer['out']
        new_out = max(1, old_out - max(1, old_out // 4))

        if new_out == old_out:
            return arch, False

        layer['out'] = new_out

        # Update next layer's input size
        for next_layer in layers[idx + 1:]:
            if next_layer.get('type') == 'linear':
                next_layer['in'] = new_out
                break
            elif next_layer.get('type') == 'batchnorm':
                next_layer['features'] = new_out

        return arch, True


class CombinedMutator:
    """
    Combined weight and topology mutation.

    Applies both types of mutations with configurable rates,
    enabling full neuroevolution of both weights and structure.
    """

    def __init__(
        self,
        weight_sigma: float = 0.1,
        weight_mutation_rate: float = 1.0,
        topology_add_node_rate: float = 0.03,
        topology_add_conn_rate: float = 0.05,
        topology_remove_conn_rate: float = 0.01,
    ):
        """
        Initialize the combined mutator.

        Args:
            weight_sigma: Sigma for weight mutations.
            weight_mutation_rate: Rate for weight mutations.
            topology_add_node_rate: Rate for add node mutations.
            topology_add_conn_rate: Rate for add connection mutations.
            topology_remove_conn_rate: Rate for remove connection mutations.
        """
        self.weight_mutator = WeightMutator(
            sigma=weight_sigma,
            mutation_rate=weight_mutation_rate,
        )
        self.topology_mutator = TopologyMutator(
            add_node_rate=topology_add_node_rate,
            add_connection_rate=topology_add_conn_rate,
            remove_connection_rate=topology_remove_conn_rate,
        )

    def mutate(
        self,
        network: 'nn.Module',
        mutate_topology: bool = True,
        mutate_weights: bool = True,
    ) -> Tuple['nn.Module', Dict[str, Any]]:
        """
        Apply combined mutations.

        Args:
            network: Network to mutate.
            mutate_topology: Whether to apply topology mutations.
            mutate_weights: Whether to apply weight mutations.

        Returns:
            Tuple of (mutated network, mutation info dict).
        """
        from ..networks import NetworkBuilder

        mutation_info = {
            'topology_mutations': [],
            'weight_mutated': False,
        }

        builder = NetworkBuilder()

        # Get architecture
        if hasattr(network, 'architecture'):
            arch = copy.deepcopy(network.architecture)
        else:
            arch = builder.to_json(network)

        # Apply topology mutation
        if mutate_topology:
            arch, mutations = self.topology_mutator.mutate(arch)
            mutation_info['topology_mutations'] = mutations

        # Build new network from architecture
        new_network = builder.from_json(arch)

        # Try to preserve compatible weights
        if hasattr(network, 'architecture'):
            self._transfer_compatible_weights(network, new_network)

        # Apply weight mutation
        if mutate_weights:
            self.weight_mutator.mutate(new_network, in_place=True)
            mutation_info['weight_mutated'] = True

        return new_network, mutation_info

    def _transfer_compatible_weights(
        self,
        source: 'nn.Module',
        target: 'nn.Module',
    ) -> None:
        """Transfer weights where shapes match."""
        import torch

        source_state = source.state_dict()
        target_state = target.state_dict()

        for name, target_param in target_state.items():
            if name in source_state:
                source_param = source_state[name]
                if source_param.shape == target_param.shape:
                    target_state[name] = source_param.clone()
                else:
                    # Partial transfer for resized layers
                    min_shape = [
                        min(s, t) for s, t in
                        zip(source_param.shape, target_param.shape)
                    ]
                    slices = tuple(slice(0, s) for s in min_shape)
                    target_state[name][slices] = source_param[slices].clone()

        target.load_state_dict(target_state)
