"""
Visualization and monitoring for ML experiments.

Provides plotting and reporting utilities for:
- Training progress (loss curves, win rates)
- Network structure and weight distributions
- Evolution lineage and fitness progression

All visualizations use matplotlib for static plots that can
be saved to files for reports or embedded in documentation.

Example usage:
    from apps.ai.visualization import (
        plot_win_rate_history,
        plot_fitness_over_generations,
        visualize_architecture,
    )

    # Training visualization
    plot_win_rate_history(evaluations, save_path='win_rate.png')

    # Network visualization
    visualize_architecture(architecture, save_path='network.png')

    # Evolution visualization
    plot_fitness_over_generations(stats, save_path='evolution.png')
"""
from .training import (
    TrainingCurve,
    plot_training_curves,
    plot_win_rate_history,
    plot_training_comparison,
    plot_learning_rate_schedule,
    generate_training_report,
    format_training_summary,
)
from .network import (
    visualize_architecture,
    plot_weight_distributions,
    plot_gradient_flow,
    get_network_stats,
    format_network_summary,
)
from .evolution import (
    plot_fitness_over_generations,
    plot_population_diversity,
    plot_mutation_distribution,
    plot_lineage_tree,
    generate_evolution_report,
    format_evolution_summary,
)

__all__ = [
    # Training
    'TrainingCurve',
    'plot_training_curves',
    'plot_win_rate_history',
    'plot_training_comparison',
    'plot_learning_rate_schedule',
    'generate_training_report',
    'format_training_summary',

    # Network
    'visualize_architecture',
    'plot_weight_distributions',
    'plot_gradient_flow',
    'get_network_stats',
    'format_network_summary',

    # Evolution
    'plot_fitness_over_generations',
    'plot_population_diversity',
    'plot_mutation_distribution',
    'plot_lineage_tree',
    'generate_evolution_report',
    'format_evolution_summary',
]
