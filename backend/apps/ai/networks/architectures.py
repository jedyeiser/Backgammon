"""
Preset neural network architectures for games.

This module provides ready-to-use architecture specifications
for different games and network types.
"""
from typing import Any, Dict, List, Optional


def td_gammon_architecture(
    input_size: int = 198,
    hidden_size: int = 80,
) -> Dict[str, Any]:
    """
    Classic TD-Gammon architecture.

    The original TD-Gammon used 40-80 hidden units with sigmoid
    activations. This simple architecture achieved expert-level play.

    Architecture:
        Input (198) -> Hidden (80, sigmoid) -> Output (1, sigmoid)

    Args:
        input_size: Input feature dimension (default 198 for backgammon).
        hidden_size: Number of hidden units (40-80 recommended).

    Returns:
        JSON architecture specification.

    Reference:
        Tesauro, G. (1995). Temporal difference learning and TD-Gammon.
    """
    return {
        'name': 'TD-Gammon',
        'input_size': input_size,
        'output_size': 1,
        'layers': [
            {'id': 'hidden', 'type': 'linear', 'in': input_size, 'out': hidden_size},
            {'id': 'hidden_act', 'type': 'activation', 'fn': 'sigmoid'},
            {'id': 'output', 'type': 'linear', 'in': hidden_size, 'out': 1},
            {'id': 'output_act', 'type': 'activation', 'fn': 'sigmoid'},
        ],
    }


def modern_backgammon_architecture(
    input_size: int = 198,
    hidden_sizes: List[int] = None,
    use_batchnorm: bool = True,
    dropout: float = 0.0,
) -> Dict[str, Any]:
    """
    Modern backgammon architecture with improvements.

    Incorporates modern deep learning techniques:
    - ReLU activations (faster training)
    - Batch normalization (stable training)
    - Optional dropout (regularization)
    - Deeper network (more capacity)

    Architecture:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Output

    Args:
        input_size: Input feature dimension.
        hidden_sizes: List of hidden layer sizes. Default [256, 128, 64].
        use_batchnorm: Whether to use batch normalization.
        dropout: Dropout probability (0 = no dropout).

    Returns:
        JSON architecture specification.
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]

    layers = []
    prev_size = input_size

    for i, hidden_size in enumerate(hidden_sizes):
        # Linear layer
        layers.append({
            'id': f'linear_{i}',
            'type': 'linear',
            'in': prev_size,
            'out': hidden_size,
        })

        # Batch normalization
        if use_batchnorm:
            layers.append({
                'id': f'bn_{i}',
                'type': 'batchnorm',
                'features': hidden_size,
            })

        # Activation
        layers.append({
            'id': f'act_{i}',
            'type': 'activation',
            'fn': 'relu',
        })

        # Dropout
        if dropout > 0:
            layers.append({
                'id': f'dropout_{i}',
                'type': 'dropout',
                'p': dropout,
            })

        prev_size = hidden_size

    # Output layer
    layers.append({
        'id': 'output',
        'type': 'linear',
        'in': prev_size,
        'out': 1,
    })
    layers.append({
        'id': 'output_act',
        'type': 'activation',
        'fn': 'sigmoid',
    })

    return {
        'name': 'Modern Backgammon',
        'input_size': input_size,
        'output_size': 1,
        'layers': layers,
    }


def create_mlp_architecture(
    input_size: int,
    output_size: int,
    hidden_sizes: List[int],
    activation: str = 'relu',
    output_activation: Optional[str] = None,
    use_batchnorm: bool = False,
    dropout: float = 0.0,
) -> Dict[str, Any]:
    """
    Create a generic MLP (Multi-Layer Perceptron) architecture.

    Flexible function for creating feedforward networks with
    customizable depth, width, and regularization.

    Args:
        input_size: Input feature dimension.
        output_size: Output dimension.
        hidden_sizes: List of hidden layer sizes.
        activation: Activation function for hidden layers.
        output_activation: Activation for output (None = no activation).
        use_batchnorm: Whether to use batch normalization.
        dropout: Dropout probability.

    Returns:
        JSON architecture specification.

    Example:
        # 3-layer MLP for classification
        arch = create_mlp_architecture(
            input_size=100,
            output_size=10,
            hidden_sizes=[64, 32],
            activation='relu',
            output_activation='softmax',
        )
    """
    layers = []
    prev_size = input_size

    for i, hidden_size in enumerate(hidden_sizes):
        layers.append({
            'id': f'linear_{i}',
            'type': 'linear',
            'in': prev_size,
            'out': hidden_size,
        })

        if use_batchnorm:
            layers.append({
                'id': f'bn_{i}',
                'type': 'batchnorm',
                'features': hidden_size,
            })

        layers.append({
            'id': f'act_{i}',
            'type': 'activation',
            'fn': activation,
        })

        if dropout > 0:
            layers.append({
                'id': f'dropout_{i}',
                'type': 'dropout',
                'p': dropout,
            })

        prev_size = hidden_size

    # Output layer
    layers.append({
        'id': 'output',
        'type': 'linear',
        'in': prev_size,
        'out': output_size,
    })

    if output_activation:
        layers.append({
            'id': 'output_act',
            'type': 'activation',
            'fn': output_activation,
        })

    return {
        'name': f'MLP-{"-".join(map(str, hidden_sizes))}',
        'input_size': input_size,
        'output_size': output_size,
        'layers': layers,
    }


def minimal_architecture(input_size: int = 198) -> Dict[str, Any]:
    """
    Minimal architecture for testing.

    Just a single linear layer with sigmoid output.
    Useful for testing infrastructure without training overhead.

    Args:
        input_size: Input feature dimension.

    Returns:
        JSON architecture specification.
    """
    return {
        'name': 'Minimal',
        'input_size': input_size,
        'output_size': 1,
        'layers': [
            {'id': 'output', 'type': 'linear', 'in': input_size, 'out': 1},
            {'id': 'output_act', 'type': 'activation', 'fn': 'sigmoid'},
        ],
    }


def two_headed_architecture(
    input_size: int = 198,
    hidden_size: int = 128,
    value_head_size: int = 32,
    policy_head_size: int = 32,
    num_actions: int = 100,
) -> Dict[str, Any]:
    """
    Two-headed architecture with value and policy outputs.

    Similar to AlphaGo/AlphaZero architecture with separate
    value and policy heads sharing a common backbone.

    Note: This is for future use - current players only use value head.

    Architecture:
        Input -> Shared backbone -> Value head (scalar)
                                 -> Policy head (action logits)

    Args:
        input_size: Input feature dimension.
        hidden_size: Shared backbone hidden size.
        value_head_size: Value head hidden size.
        policy_head_size: Policy head hidden size.
        num_actions: Number of possible actions.

    Returns:
        JSON architecture specification (value head only for now).
    """
    # For now, just return value head architecture
    # Full two-headed support would require custom forward logic
    return {
        'name': 'Two-Headed (Value Only)',
        'input_size': input_size,
        'output_size': 1,
        'layers': [
            {'id': 'backbone', 'type': 'linear', 'in': input_size, 'out': hidden_size},
            {'id': 'backbone_act', 'type': 'activation', 'fn': 'relu'},
            {'id': 'value_hidden', 'type': 'linear', 'in': hidden_size, 'out': value_head_size},
            {'id': 'value_act', 'type': 'activation', 'fn': 'relu'},
            {'id': 'value_out', 'type': 'linear', 'in': value_head_size, 'out': 1},
            {'id': 'value_sigmoid', 'type': 'activation', 'fn': 'sigmoid'},
        ],
    }
