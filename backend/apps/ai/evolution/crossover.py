"""
Crossover operators for network evolution.

Implements methods for combining two parent networks to
create offspring, enabling sexual reproduction in the
evolutionary process.

For neural networks, crossover is tricky because:
- Networks may have different topologies
- Weight alignment matters for functionality
- Simple crossover can be destructive

This module provides several crossover strategies:
- Weight interpolation (same topology)
- Layer-wise crossover (same topology)
- Architecture inheritance (different topologies)
"""
import copy
import random
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class WeightCrossover:
    """
    Weight-level crossover for networks with identical topology.

    Creates offspring by combining weights from two parents.
    Requires both parents to have the same architecture.

    Strategies:
    - uniform: Each weight randomly from parent A or B
    - interpolation: Weighted average of parent weights
    - layer_wise: Each layer randomly from parent A or B
    """

    def __init__(
        self,
        strategy: str = 'interpolation',
        alpha: float = 0.5,
    ):
        """
        Initialize the crossover operator.

        Args:
            strategy: 'uniform', 'interpolation', or 'layer_wise'.
            alpha: Interpolation weight (0.5 = equal contribution).
        """
        self.strategy = strategy
        self.alpha = alpha

    def crossover(
        self,
        parent_a: 'nn.Module',
        parent_b: 'nn.Module',
    ) -> 'nn.Module':
        """
        Create offspring from two parent networks.

        Args:
            parent_a: First parent network.
            parent_b: Second parent network.

        Returns:
            Child network combining traits from both parents.

        Raises:
            ValueError: If parents have different architectures.
        """
        import torch

        # Verify same architecture
        if not self._check_compatible(parent_a, parent_b):
            raise ValueError("Parents must have identical architectures")

        # Clone parent A as base
        child = self._clone_network(parent_a)

        state_a = parent_a.state_dict()
        state_b = parent_b.state_dict()
        child_state = child.state_dict()

        with torch.no_grad():
            if self.strategy == 'interpolation':
                # Weighted average of weights
                for name in child_state:
                    child_state[name] = (
                        self.alpha * state_a[name] +
                        (1 - self.alpha) * state_b[name]
                    )

            elif self.strategy == 'uniform':
                # Each weight randomly from A or B
                for name in child_state:
                    mask = torch.rand_like(state_a[name]) < 0.5
                    child_state[name] = torch.where(
                        mask, state_a[name], state_b[name]
                    )

            elif self.strategy == 'layer_wise':
                # Each layer randomly from A or B
                for name in child_state:
                    if random.random() < 0.5:
                        child_state[name] = state_a[name].clone()
                    else:
                        child_state[name] = state_b[name].clone()

        child.load_state_dict(child_state)
        return child

    def _check_compatible(
        self,
        net_a: 'nn.Module',
        net_b: 'nn.Module',
    ) -> bool:
        """Check if two networks have compatible architectures."""
        state_a = net_a.state_dict()
        state_b = net_b.state_dict()

        if state_a.keys() != state_b.keys():
            return False

        for name in state_a:
            if state_a[name].shape != state_b[name].shape:
                return False

        return True

    def _clone_network(self, network: 'nn.Module') -> 'nn.Module':
        """Create a deep copy of a network."""
        from ..networks import NetworkBuilder

        if hasattr(network, 'architecture'):
            builder = NetworkBuilder()
            new_network = builder.from_json(network.architecture)
            weights = builder.serialize_weights(network)
            builder.deserialize_weights(weights, new_network)
            return new_network

        import copy
        return copy.deepcopy(network)


class ArchitectureCrossover:
    """
    Architecture-level crossover for networks with different topologies.

    Creates offspring by inheriting architectural traits from parents.
    Can combine layer structures from both networks.
    """

    def __init__(
        self,
        prefer_fitter: bool = True,
        inherit_weights: bool = True,
    ):
        """
        Initialize architecture crossover.

        Args:
            prefer_fitter: If True, prefer structure from fitter parent.
            inherit_weights: If True, try to preserve parent weights.
        """
        self.prefer_fitter = prefer_fitter
        self.inherit_weights = inherit_weights

    def crossover(
        self,
        parent_a: 'nn.Module',
        parent_b: 'nn.Module',
        fitness_a: float = 0.5,
        fitness_b: float = 0.5,
    ) -> Tuple['nn.Module', Dict[str, Any]]:
        """
        Create offspring from two parent networks.

        Args:
            parent_a: First parent network.
            parent_b: Second parent network.
            fitness_a: Fitness score of parent A.
            fitness_b: Fitness score of parent B.

        Returns:
            Tuple of (child network, inheritance info).
        """
        from ..networks import NetworkBuilder

        builder = NetworkBuilder()

        # Get architectures
        arch_a = self._get_architecture(parent_a)
        arch_b = self._get_architecture(parent_b)

        # Determine dominant parent (fitter one)
        if self.prefer_fitter:
            if fitness_a >= fitness_b:
                dominant_arch = arch_a
                dominant_parent = parent_a
                recessive_arch = arch_b
            else:
                dominant_arch = arch_b
                dominant_parent = parent_b
                recessive_arch = arch_a
        else:
            # Random dominance
            if random.random() < 0.5:
                dominant_arch = arch_a
                dominant_parent = parent_a
                recessive_arch = arch_b
            else:
                dominant_arch = arch_b
                dominant_parent = parent_b
                recessive_arch = arch_a

        # Create child architecture by combining
        child_arch = self._combine_architectures(dominant_arch, recessive_arch)

        # Build child network
        child = builder.from_json(child_arch)

        inheritance_info = {
            'dominant_parent': 'a' if dominant_parent == parent_a else 'b',
            'architecture_source': 'combined',
        }

        # Inherit weights from dominant parent where possible
        if self.inherit_weights:
            self._transfer_weights(dominant_parent, child)
            inheritance_info['weights_from'] = 'dominant'

        return child, inheritance_info

    def _get_architecture(self, network: 'nn.Module') -> Dict[str, Any]:
        """Get architecture from network."""
        from ..networks import NetworkBuilder

        if hasattr(network, 'architecture'):
            return copy.deepcopy(network.architecture)

        builder = NetworkBuilder()
        return builder.to_json(network)

    def _combine_architectures(
        self,
        dominant: Dict[str, Any],
        recessive: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine two architectures.

        Strategy: Use dominant structure, but randomly inherit
        layer sizes from recessive parent (except for output layer).
        """
        result = copy.deepcopy(dominant)

        dom_layers = result.get('layers', [])
        rec_layers = recessive.get('layers', [])

        # Find matching layer types and randomly inherit sizes
        rec_linears = [l for l in rec_layers if l.get('type') == 'linear']

        # Find the last linear layer (output layer) - we should not modify its output size
        linear_indices = [i for i, l in enumerate(dom_layers) if l.get('type') == 'linear']
        last_linear_idx = linear_indices[-1] if linear_indices else -1

        for i, layer in enumerate(dom_layers):
            if layer.get('type') == 'linear' and rec_linears:
                # Skip the output layer - don't change its output size
                if i == last_linear_idx:
                    continue
                # 30% chance to inherit size from recessive
                if random.random() < 0.3:
                    rec_layer = random.choice(rec_linears)
                    # Only inherit output size (input must match previous)
                    if 'out' in rec_layer:
                        layer['out'] = rec_layer['out']

        # Repair architecture (ensure layer sizes match)
        self._repair_architecture(result)

        return result

    def _repair_architecture(self, arch: Dict[str, Any]) -> None:
        """Ensure layer sizes are consistent."""
        layers = arch.get('layers', [])

        # Track expected input size
        expected_input = arch.get('input_size', 198)

        for i, layer in enumerate(layers):
            if layer.get('type') == 'linear':
                # Ensure input matches expected
                layer['in'] = expected_input
                # Update expected for next layer
                expected_input = layer['out']

            elif layer.get('type') == 'batchnorm':
                layer['features'] = expected_input

    def _transfer_weights(
        self,
        source: 'nn.Module',
        target: 'nn.Module',
    ) -> None:
        """Transfer compatible weights from source to target."""
        import torch

        source_state = source.state_dict()
        target_state = target.state_dict()

        for name, target_param in target_state.items():
            if name in source_state:
                source_param = source_state[name]

                if source_param.shape == target_param.shape:
                    target_state[name] = source_param.clone()
                else:
                    # Partial transfer
                    min_shape = [
                        min(s, t) for s, t in
                        zip(source_param.shape, target_param.shape)
                    ]
                    if all(s > 0 for s in min_shape):
                        slices = tuple(slice(0, s) for s in min_shape)
                        target_state[name][slices] = source_param[slices].clone()

        target.load_state_dict(target_state)


def blend_networks(
    networks: List['nn.Module'],
    weights: Optional[List[float]] = None,
) -> 'nn.Module':
    """
    Blend multiple networks into one.

    Creates a new network whose weights are a weighted average
    of the input networks. All networks must have identical topology.

    Args:
        networks: List of networks to blend.
        weights: Optional weights for each network (sum to 1).
                If None, equal weights are used.

    Returns:
        Blended network.

    Raises:
        ValueError: If networks have different architectures.
    """
    import torch
    from ..networks import NetworkBuilder

    if not networks:
        raise ValueError("Need at least one network to blend")

    if len(networks) == 1:
        # Clone single network
        builder = NetworkBuilder()
        if hasattr(networks[0], 'architecture'):
            result = builder.from_json(networks[0].architecture)
            weights_bytes = builder.serialize_weights(networks[0])
            builder.deserialize_weights(weights_bytes, result)
            return result
        return copy.deepcopy(networks[0])

    # Normalize weights
    if weights is None:
        weights = [1.0 / len(networks)] * len(networks)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    # Clone first network as base
    builder = NetworkBuilder()
    if hasattr(networks[0], 'architecture'):
        result = builder.from_json(networks[0].architecture)
    else:
        result = copy.deepcopy(networks[0])

    result_state = result.state_dict()

    # Blend weights
    with torch.no_grad():
        for name in result_state:
            blended = torch.zeros_like(result_state[name])
            for net, weight in zip(networks, weights):
                net_state = net.state_dict()
                if name in net_state and net_state[name].shape == result_state[name].shape:
                    blended += weight * net_state[name]
                else:
                    raise ValueError(f"Incompatible layer: {name}")
            result_state[name] = blended

    result.load_state_dict(result_state)
    return result
