"""
Neural network visualization utilities.

Generate visualizations of network structure and internals:
- Layer architecture diagrams
- Weight distribution histograms
- Activation patterns on sample inputs
- Network complexity metrics
"""
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if TYPE_CHECKING:
    import torch.nn as nn


def visualize_architecture(
    architecture: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    show_sizes: bool = True,
) -> Optional[str]:
    """
    Visualize neural network architecture as a diagram.

    Args:
        architecture: JSON architecture specification.
        save_path: Path to save figure.
        figsize: Figure size.
        show_sizes: Whether to show layer sizes.

    Returns:
        Path to saved figure if save_path provided.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")

    layers = architecture.get('layers', [])
    if not layers:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Calculate positions
    n_layers = len([l for l in layers if l.get('type') == 'linear'])
    x_positions = np.linspace(1, 9, max(n_layers + 1, 2))

    layer_boxes = []
    current_x_idx = 0

    # Input layer
    input_size = architecture.get('input_size', 198)
    input_box = FancyBboxPatch(
        (x_positions[0] - 0.3, 4), 0.6, 2,
        boxstyle="round,pad=0.02",
        facecolor='lightblue',
        edgecolor='blue',
        linewidth=2,
    )
    ax.add_patch(input_box)
    ax.text(
        x_positions[0], 5,
        f'Input\n{input_size}',
        ha='center', va='center', fontsize=10,
    )
    layer_boxes.append((x_positions[0], 5))
    current_x_idx += 1

    # Hidden and output layers
    for layer in layers:
        layer_type = layer.get('type', '')

        if layer_type == 'linear':
            out_size = layer.get('out', 0)
            in_size = layer.get('in', 0)

            x = x_positions[current_x_idx]
            height = min(2, out_size / 50)  # Scale by size

            color = 'lightgreen' if current_x_idx < len(x_positions) - 1 else 'lightyellow'
            box = FancyBboxPatch(
                (x - 0.3, 5 - height/2), 0.6, height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='green' if color == 'lightgreen' else 'orange',
                linewidth=2,
            )
            ax.add_patch(box)

            label = f'Linear\n{out_size}'
            if layer.get('id'):
                label = f"{layer['id']}\n{out_size}"

            ax.text(x, 5, label, ha='center', va='center', fontsize=9)

            # Draw connection from previous layer
            if layer_boxes:
                prev_x, prev_y = layer_boxes[-1]
                ax.annotate(
                    '',
                    xy=(x - 0.3, 5),
                    xytext=(prev_x + 0.3, prev_y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                )

            layer_boxes.append((x, 5))
            current_x_idx += 1

        elif layer_type == 'activation':
            fn = layer.get('fn', 'relu')
            if layer_boxes:
                x = layer_boxes[-1][0]
                ax.text(
                    x, 3.5,
                    fn.upper(),
                    ha='center', va='center',
                    fontsize=8, color='purple',
                    style='italic',
                )

    # Title
    name = architecture.get('name', 'Neural Network')
    ax.set_title(f"Architecture: {name}", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def plot_weight_distributions(
    network: 'nn.Module',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[str]:
    """
    Plot histograms of weight distributions for each layer.

    Args:
        network: PyTorch network.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        raise ImportError("matplotlib and numpy required")

    import torch

    # Collect weights by layer
    weight_data = {}
    for name, param in network.named_parameters():
        if 'weight' in name:
            weights = param.detach().cpu().numpy().flatten()
            weight_data[name] = weights

    if not weight_data:
        return None

    n_layers = len(weight_data)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_layers > 1 else [axes]

    for i, (name, weights) in enumerate(weight_data.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        ax.hist(weights, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')

        # Add statistics
        mean = np.mean(weights)
        std = np.std(weights)
        ax.axvline(mean, color='red', linestyle='--', label=f'mean={mean:.3f}')
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Weight Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def plot_gradient_flow(
    network: 'nn.Module',
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Plot gradient magnitudes through the network.

    Useful for detecting vanishing/exploding gradients.

    Args:
        network: PyTorch network (after backward pass).
        save_path: Path to save figure.

    Returns:
        Path to saved figure.
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        raise ImportError("matplotlib and numpy required")

    layers = []
    grad_magnitudes = []

    for name, param in network.named_parameters():
        if param.grad is not None and 'weight' in name:
            layers.append(name)
            grad_magnitudes.append(param.grad.abs().mean().item())

    if not layers:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(layers))
    ax.bar(x, grad_magnitudes, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Gradient Magnitude')
    ax.set_title('Gradient Flow Through Network')
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def get_network_stats(network: 'nn.Module') -> Dict[str, Any]:
    """
    Compute statistics about a network.

    Args:
        network: PyTorch network.

    Returns:
        Dictionary of network statistics.
    """
    import torch

    total_params = 0
    trainable_params = 0
    weight_stats = {}

    for name, param in network.named_parameters():
        n_params = param.numel()
        total_params += n_params
        if param.requires_grad:
            trainable_params += n_params

        if 'weight' in name:
            weights = param.detach().cpu()
            weight_stats[name] = {
                'shape': list(param.shape),
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'min': weights.min().item(),
                'max': weights.max().item(),
            }

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'num_layers': len([n for n in network.named_modules() if n[0]]),
        'weight_stats': weight_stats,
    }


def format_network_summary(
    network: 'nn.Module',
    architecture: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a text summary of the network.

    Args:
        network: PyTorch network.
        architecture: Optional architecture dict.

    Returns:
        Formatted text summary.
    """
    stats = get_network_stats(network)

    lines = [
        "=" * 50,
        "NETWORK SUMMARY",
        "=" * 50,
        "",
    ]

    if architecture:
        lines.append(f"Name: {architecture.get('name', 'Unknown')}")
        lines.append(f"Input size: {architecture.get('input_size', 'N/A')}")
        lines.append(f"Output size: {architecture.get('output_size', 'N/A')}")
        lines.append("")

    lines.extend([
        f"Total parameters: {stats['total_parameters']:,}",
        f"Trainable parameters: {stats['trainable_parameters']:,}",
        "",
        "Layer weights:",
    ])

    for name, w_stats in stats['weight_stats'].items():
        lines.append(f"  {name}: {w_stats['shape']}")
        lines.append(f"    mean={w_stats['mean']:.4f}, std={w_stats['std']:.4f}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)
