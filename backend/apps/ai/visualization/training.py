"""
Training visualization utilities.

Generate plots and reports for monitoring training progress:
- Loss curves over time
- Win rate vs training games
- Learning rate schedules
- Comparison between model versions

All visualizations can be saved to files or returned as data
for frontend rendering.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class TrainingCurve:
    """Data for a single training curve."""
    name: str
    steps: List[int]
    values: List[float]
    color: str = 'blue'
    style: str = '-'


def plot_training_curves(
    curves: List[TrainingCurve],
    title: str = 'Training Progress',
    xlabel: str = 'Games',
    ylabel: str = 'Value',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[str]:
    """
    Plot multiple training curves on a single figure.

    Args:
        curves: List of TrainingCurve objects.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Path to save figure (None = don't save).
        figsize: Figure size in inches.

    Returns:
        Path to saved figure if save_path provided.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    for curve in curves:
        ax.plot(
            curve.steps,
            curve.values,
            label=curve.name,
            color=curve.color,
            linestyle=curve.style,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def plot_win_rate_history(
    evaluations: List[Dict[str, Any]],
    title: str = 'Win Rate vs Random',
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Plot win rate progression during training.

    Args:
        evaluations: List of dicts with 'game' and 'win_rate' keys.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Path to saved figure.
    """
    if not evaluations:
        return None

    games = [e.get('game', 0) for e in evaluations]
    win_rates = [e.get('win_rate', 0) for e in evaluations]

    curve = TrainingCurve(
        name='Win Rate',
        steps=games,
        values=win_rates,
        color='green',
    )

    return plot_training_curves(
        [curve],
        title=title,
        xlabel='Training Games',
        ylabel='Win Rate',
        save_path=save_path,
    )


def plot_training_comparison(
    experiments: Dict[str, List[Dict[str, Any]]],
    metric: str = 'win_rate',
    title: str = 'Training Comparison',
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Compare training progress across multiple experiments.

    Args:
        experiments: Dict mapping experiment name to evaluation list.
        metric: Which metric to compare.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Path to saved figure.
    """
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    curves = []

    for i, (name, evaluations) in enumerate(experiments.items()):
        games = [e.get('game', 0) for e in evaluations]
        values = [e.get(metric, 0) for e in evaluations]

        curves.append(TrainingCurve(
            name=name,
            steps=games,
            values=values,
            color=colors[i % len(colors)],
        ))

    return plot_training_curves(
        curves,
        title=title,
        xlabel='Training Games',
        ylabel=metric.replace('_', ' ').title(),
        save_path=save_path,
    )


def plot_learning_rate_schedule(
    games: List[int],
    learning_rates: List[float],
    title: str = 'Learning Rate Schedule',
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Plot learning rate schedule.

    Args:
        games: Training game numbers.
        learning_rates: Learning rate at each game.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Path to saved figure.
    """
    curve = TrainingCurve(
        name='Learning Rate',
        steps=games,
        values=learning_rates,
        color='red',
    )

    return plot_training_curves(
        [curve],
        title=title,
        xlabel='Training Games',
        ylabel='Learning Rate',
        save_path=save_path,
    )


def generate_training_report(
    stats: Dict[str, Any],
    evaluations: List[Dict[str, Any]],
    output_dir: str,
    experiment_name: str = 'training',
) -> Dict[str, str]:
    """
    Generate a complete training report with multiple plots.

    Args:
        stats: Training statistics dict.
        evaluations: Evaluation history.
        output_dir: Directory to save plots.
        experiment_name: Name for the experiment.

    Returns:
        Dict mapping plot name to file path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Win rate plot
    if evaluations:
        win_rate_path = output_path / f'{experiment_name}_win_rate.png'
        plot_win_rate_history(evaluations, save_path=str(win_rate_path))
        plots['win_rate'] = str(win_rate_path)

    # TD error plot (if available)
    if 'td_errors' in stats:
        td_path = output_path / f'{experiment_name}_td_error.png'
        td_curve = TrainingCurve(
            name='TD Error',
            steps=list(range(len(stats['td_errors']))),
            values=stats['td_errors'],
            color='orange',
        )
        plot_training_curves(
            [td_curve],
            title='TD Error Over Training',
            ylabel='Average TD Error',
            save_path=str(td_path),
        )
        plots['td_error'] = str(td_path)

    return plots


def format_training_summary(
    stats: Dict[str, Any],
    evaluations: List[Dict[str, Any]],
) -> str:
    """
    Generate a text summary of training.

    Args:
        stats: Training statistics.
        evaluations: Evaluation history.

    Returns:
        Formatted text summary.
    """
    lines = [
        "=" * 50,
        "TRAINING SUMMARY",
        "=" * 50,
        "",
        f"Total games played: {stats.get('games_played', 0):,}",
        f"Training time: {stats.get('training_time_seconds', 0)/60:.1f} minutes",
        "",
    ]

    if evaluations:
        final_eval = evaluations[-1] if evaluations else {}
        best_eval = max(evaluations, key=lambda e: e.get('win_rate', 0))

        lines.extend([
            "Performance:",
            f"  Final win rate: {final_eval.get('win_rate', 0):.2%}",
            f"  Best win rate: {best_eval.get('win_rate', 0):.2%} (game {best_eval.get('game', 0):,})",
            "",
        ])

    lines.extend([
        f"Final TD error: {stats.get('final_td_error', 0):.6f}",
        f"Checkpoint: {stats.get('checkpoint_path', 'N/A')}",
        "",
        "=" * 50,
    ])

    return "\n".join(lines)
