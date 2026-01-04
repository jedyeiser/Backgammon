"""
Network builder for converting between JSON and PyTorch models.

This module provides the core infrastructure for:
- Building PyTorch networks from JSON architecture specs
- Converting PyTorch networks back to JSON
- Serializing/deserializing network weights
- Supporting topology mutations for evolution
"""
import gzip
import io
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DynamicNetwork(nn.Module):
    """
    A PyTorch network built from a JSON architecture specification.

    This class wraps an nn.Sequential or custom module structure
    that was built from a JSON architecture definition.

    Attributes:
        architecture: The JSON architecture this network was built from.
        layers: OrderedDict of layer_id -> nn.Module.
    """

    def __init__(self, layers: 'OrderedDict[str, nn.Module]', architecture: Dict[str, Any]):
        super().__init__()
        self.architecture = architecture
        self._layers = nn.ModuleDict(layers)
        self._layer_order = list(layers.keys())

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass through all layers in order."""
        for layer_id in self._layer_order:
            x = self._layers[layer_id](x)
        return x

    def get_layer(self, layer_id: str) -> Optional[nn.Module]:
        """Get a layer by its ID."""
        return self._layers.get(layer_id)

    def layer_ids(self) -> List[str]:
        """Return list of layer IDs in order."""
        return self._layer_order.copy()


class NetworkBuilder:
    """
    Build PyTorch networks from JSON architecture specifications.

    The JSON format is designed to be:
    - Human-readable and editable
    - Suitable for storing in a database (JSONField)
    - Evolvable (can add/remove layers programmatically)

    Architecture Format:
        {
            "input_size": 198,
            "output_size": 1,
            "layers": [
                {"id": "layer_0", "type": "linear", "in": 198, "out": 80},
                {"id": "act_0", "type": "activation", "fn": "sigmoid"},
                {"id": "layer_1", "type": "linear", "in": 80, "out": 1},
                {"id": "output", "type": "activation", "fn": "sigmoid"}
            ]
        }

    Example:
        builder = NetworkBuilder()
        network = builder.from_json(architecture)
        weights = builder.serialize_weights(network)
        builder.deserialize_weights(weights, network)
    """

    # Supported activation functions
    ACTIVATIONS = {
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'softmax': lambda: nn.Softmax(dim=-1),
        'softplus': nn.Softplus,
        'identity': nn.Identity,
    }

    def __init__(self):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for NetworkBuilder")

    def from_json(self, architecture: Dict[str, Any]) -> DynamicNetwork:
        """
        Build a PyTorch network from a JSON architecture specification.

        Args:
            architecture: Dictionary with 'input_size', 'output_size', and 'layers'.

        Returns:
            A DynamicNetwork instance.

        Raises:
            ValueError: If architecture is invalid.
        """
        self._validate_architecture(architecture)

        layers = OrderedDict()

        for layer_spec in architecture.get('layers', []):
            layer_id = layer_spec.get('id', f"layer_{len(layers)}")
            layer = self._build_layer(layer_spec)
            layers[layer_id] = layer

        return DynamicNetwork(layers, architecture)

    def _build_layer(self, layer_spec: Dict[str, Any]) -> nn.Module:
        """
        Build a single layer from its specification.

        Args:
            layer_spec: Dictionary with layer type and parameters.

        Returns:
            A PyTorch nn.Module.

        Raises:
            ValueError: If layer type is unknown.
        """
        layer_type = layer_spec.get('type', '')

        if layer_type == 'linear':
            in_features = layer_spec['in']
            out_features = layer_spec['out']
            bias = layer_spec.get('bias', True)
            return nn.Linear(in_features, out_features, bias=bias)

        elif layer_type == 'activation':
            fn_name = layer_spec.get('fn', 'relu')
            if fn_name not in self.ACTIVATIONS:
                raise ValueError(f"Unknown activation function: {fn_name}")
            return self.ACTIVATIONS[fn_name]()

        elif layer_type == 'dropout':
            p = layer_spec.get('p', 0.5)
            return nn.Dropout(p=p)

        elif layer_type == 'batchnorm':
            features = layer_spec['features']
            return nn.BatchNorm1d(features)

        elif layer_type == 'layernorm':
            features = layer_spec['features']
            return nn.LayerNorm(features)

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def _validate_architecture(self, architecture: Dict[str, Any]) -> None:
        """Validate that an architecture specification is well-formed."""
        if not isinstance(architecture, dict):
            raise ValueError("Architecture must be a dictionary")

        if 'layers' not in architecture:
            raise ValueError("Architecture must have 'layers' key")

        if not isinstance(architecture['layers'], list):
            raise ValueError("'layers' must be a list")

        for i, layer in enumerate(architecture['layers']):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer {i} must be a dictionary")
            if 'type' not in layer:
                raise ValueError(f"Layer {i} must have 'type' key")

    def to_json(self, network: nn.Module) -> Dict[str, Any]:
        """
        Convert a PyTorch network to JSON architecture.

        Note: This works best with networks created via from_json().
        For arbitrary networks, the conversion may be imperfect.

        Args:
            network: A PyTorch network.

        Returns:
            JSON architecture dictionary.
        """
        if isinstance(network, DynamicNetwork):
            return network.architecture.copy()

        # Try to reverse-engineer from Sequential
        layers = []
        input_size = None
        output_size = None

        for i, (name, module) in enumerate(network.named_modules()):
            if name == '':  # Skip the root module
                continue

            layer_spec = self._module_to_spec(module, f"layer_{i}")
            if layer_spec:
                layers.append(layer_spec)

                # Track sizes
                if isinstance(module, nn.Linear):
                    if input_size is None:
                        input_size = module.in_features
                    output_size = module.out_features

        return {
            'input_size': input_size,
            'output_size': output_size,
            'layers': layers,
        }

    def _module_to_spec(self, module: nn.Module, layer_id: str) -> Optional[Dict[str, Any]]:
        """Convert a PyTorch module to a layer specification."""
        if isinstance(module, nn.Linear):
            return {
                'id': layer_id,
                'type': 'linear',
                'in': module.in_features,
                'out': module.out_features,
                'bias': module.bias is not None,
            }
        elif isinstance(module, nn.Sigmoid):
            return {'id': layer_id, 'type': 'activation', 'fn': 'sigmoid'}
        elif isinstance(module, nn.Tanh):
            return {'id': layer_id, 'type': 'activation', 'fn': 'tanh'}
        elif isinstance(module, nn.ReLU):
            return {'id': layer_id, 'type': 'activation', 'fn': 'relu'}
        elif isinstance(module, nn.LeakyReLU):
            return {'id': layer_id, 'type': 'activation', 'fn': 'leaky_relu'}
        elif isinstance(module, nn.Dropout):
            return {'id': layer_id, 'type': 'dropout', 'p': module.p}
        elif isinstance(module, nn.BatchNorm1d):
            return {'id': layer_id, 'type': 'batchnorm', 'features': module.num_features}
        elif isinstance(module, nn.LayerNorm):
            return {'id': layer_id, 'type': 'layernorm', 'features': module.normalized_shape[0]}
        return None

    def serialize_weights(self, network: nn.Module) -> bytes:
        """
        Serialize network weights to compressed bytes.

        The weights are stored as a gzip-compressed pickle of a
        dictionary mapping parameter names to numpy arrays.

        Args:
            network: A PyTorch network.

        Returns:
            Compressed bytes containing the weights.
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for weight serialization")

        # Extract weights as numpy arrays
        weights_dict = {}
        for name, param in network.named_parameters():
            weights_dict[name] = param.detach().cpu().numpy()

        # Pickle and compress
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
            pickle.dump(weights_dict, f)

        return buffer.getvalue()

    def deserialize_weights(self, data: bytes, network: nn.Module) -> None:
        """
        Load weights from compressed bytes into a network.

        Args:
            data: Compressed bytes from serialize_weights().
            network: The network to load weights into.

        Raises:
            ValueError: If weights don't match network structure.
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for weight deserialization")

        # Decompress and unpickle
        buffer = io.BytesIO(data)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
            weights_dict = pickle.load(f)

        # Load weights into network
        state_dict = {}
        for name, np_array in weights_dict.items():
            state_dict[name] = torch.from_numpy(np_array)

        network.load_state_dict(state_dict, strict=True)

    def clone_network(self, network: nn.Module) -> nn.Module:
        """
        Create a deep copy of a network with the same weights.

        Args:
            network: Network to clone.

        Returns:
            A new network with identical weights.
        """
        if isinstance(network, DynamicNetwork):
            new_network = self.from_json(network.architecture)
        else:
            # For non-dynamic networks, serialize and rebuild
            arch = self.to_json(network)
            new_network = self.from_json(arch)

        # Copy weights
        weights = self.serialize_weights(network)
        self.deserialize_weights(weights, new_network)

        return new_network

    def get_parameter_count(self, network: nn.Module) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in network.parameters() if p.requires_grad)

    def get_layer_info(self, network: nn.Module) -> List[Dict[str, Any]]:
        """Get information about each layer in the network."""
        info = []

        if isinstance(network, DynamicNetwork):
            for layer_id in network.layer_ids():
                layer = network.get_layer(layer_id)
                layer_info = {
                    'id': layer_id,
                    'type': type(layer).__name__,
                    'parameters': sum(p.numel() for p in layer.parameters()),
                }
                if isinstance(layer, nn.Linear):
                    layer_info['in_features'] = layer.in_features
                    layer_info['out_features'] = layer.out_features
                info.append(layer_info)
        else:
            for name, module in network.named_modules():
                if name == '':
                    continue
                info.append({
                    'id': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                })

        return info
