"""
Evolution visualization utilities.

Generate visualizations for evolutionary processes:
- Fitness over generations
- Population diversity
- Family trees / lineage diagrams
- Mutation type analysis
"""
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def plot_fitness_over_generations(
    stats_history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show_range: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[str]:
    """
    Plot fitness progression over generations.

    Shows best, average, and optionally min fitness per generation.

    Args:
        stats_history: List of GenerationStats dicts.
        save_path: Path to save figure.
        show_range: If True, show min-max range as shaded area.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        raise ImportError("matplotlib and numpy required")

    if not stats_history:
        return None

    generations = [s.get('generation', i) for i, s in enumerate(stats_history)]
    best = [s.get('best_fitness', 0) for s in stats_history]
    avg = [s.get('avg_fitness', 0) for s in stats_history]
    min_fit = [s.get('min_fitness', 0) for s in stats_history]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(generations, best, 'g-', linewidth=2, label='Best')
    ax.plot(generations, avg, 'b-', linewidth=2, label='Average')

    if show_range:
        ax.fill_between(
            generations, min_fit, best,
            alpha=0.2, color='gray',
            label='Range'
        )

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Over Generations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def plot_population_diversity(
    diversity_history: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[str]:
    """
    Plot population diversity over generations.

    Diversity can be measured as average pairwise distance
    between individuals or fitness standard deviation.

    Args:
        diversity_history: Diversity values per generation.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    generations = range(len(diversity_history))
    ax.plot(generations, diversity_history, 'purple', linewidth=2)
    ax.fill_between(generations, 0, diversity_history, alpha=0.3, color='purple')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity')
    ax.set_title('Population Diversity Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def plot_mutation_distribution(
    mutation_counts: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[str]:
    """
    Plot distribution of mutation types.

    Args:
        mutation_counts: Dict mapping mutation type to count.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    if not mutation_counts:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    types = list(mutation_counts.keys())
    counts = list(mutation_counts.values())

    colors = plt.cm.Set3(range(len(types)))
    bars = ax.bar(types, counts, color=colors, edgecolor='black')

    ax.set_xlabel('Mutation Type')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Mutation Types')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{count}',
            ha='center', va='bottom',
        )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def plot_lineage_tree(
    individuals: List[Dict[str, Any]],
    max_depth: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[str]:
    """
    Plot a family tree of evolved individuals.

    Shows parent-child relationships and fitness values.

    Args:
        individuals: List of individual dicts with 'id', 'parent_ids', 'fitness'.
        max_depth: Maximum generations to show.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        raise ImportError("matplotlib and numpy required")

    if not individuals:
        return None

    # Build graph structure
    id_to_ind = {ind.get('id', str(i)): ind for i, ind in enumerate(individuals)}
    children = {ind_id: [] for ind_id in id_to_ind}

    for ind_id, ind in id_to_ind.items():
        for parent_id in ind.get('parent_ids', []):
            if parent_id in children:
                children[parent_id].append(ind_id)

    # Find root nodes (no parents)
    roots = [
        ind_id for ind_id, ind in id_to_ind.items()
        if not ind.get('parent_ids')
    ]

    if not roots:
        # Use first generation as roots
        min_gen = min(ind.get('generation', 0) for ind in individuals)
        roots = [
            ind.get('id', str(i))
            for i, ind in enumerate(individuals)
            if ind.get('generation', 0) == min_gen
        ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.axis('off')

    # Layout nodes by generation
    positions = {}
    nodes_by_gen = {}

    for ind_id, ind in id_to_ind.items():
        gen = ind.get('generation', 0)
        if gen not in nodes_by_gen:
            nodes_by_gen[gen] = []
        nodes_by_gen[gen].append(ind_id)

    # Assign positions
    for gen, node_ids in sorted(nodes_by_gen.items())[:max_depth]:
        x = 1 + gen * (8 / max_depth)
        n_nodes = len(node_ids)
        for i, node_id in enumerate(node_ids):
            y = 1 + (i + 0.5) * (8 / max(n_nodes, 1))
            positions[node_id] = (x, y)

    # Draw edges
    for parent_id, child_ids in children.items():
        if parent_id not in positions:
            continue
        px, py = positions[parent_id]
        for child_id in child_ids:
            if child_id not in positions:
                continue
            cx, cy = positions[child_id]
            ax.annotate(
                '',
                xy=(cx, cy),
                xytext=(px, py),
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    alpha=0.5,
                    connectionstyle='arc3,rad=0.1',
                ),
            )

    # Draw nodes
    for ind_id, (x, y) in positions.items():
        ind = id_to_ind.get(ind_id, {})
        fitness = ind.get('fitness', 0)

        # Color by fitness
        color = plt.cm.RdYlGn(fitness)

        circle = Circle(
            (x, y), 0.3,
            facecolor=color,
            edgecolor='black',
            linewidth=1,
        )
        ax.add_patch(circle)

        # Label with fitness
        ax.text(
            x, y,
            f'{fitness:.2f}',
            ha='center', va='center',
            fontsize=8,
        )

    ax.set_title('Evolution Lineage Tree', fontsize=14, fontweight='bold')
    ax.text(0.5, 9.5, 'Green = High Fitness, Red = Low Fitness',
            ha='left', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.close()
    return None


def generate_evolution_report(
    stats_history: List[Dict[str, Any]],
    individuals: List[Dict[str, Any]],
    output_dir: str,
    experiment_name: str = 'evolution',
) -> Dict[str, str]:
    """
    Generate a complete evolution report with multiple plots.

    Args:
        stats_history: Generation statistics.
        individuals: All individuals from evolution.
        output_dir: Directory to save plots.
        experiment_name: Name for the experiment.

    Returns:
        Dict mapping plot name to file path.
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Fitness progression
    if stats_history:
        fitness_path = output_path / f'{experiment_name}_fitness.png'
        plot_fitness_over_generations(stats_history, save_path=str(fitness_path))
        plots['fitness'] = str(fitness_path)

        # Diversity (using fitness std as proxy)
        diversity = [s.get('fitness_std', 0) for s in stats_history]
        if any(d > 0 for d in diversity):
            diversity_path = output_path / f'{experiment_name}_diversity.png'
            plot_population_diversity(diversity, save_path=str(diversity_path))
            plots['diversity'] = str(diversity_path)

    # Mutation distribution
    if individuals:
        mutation_counts = {}
        for ind in individuals:
            for mut in ind.get('mutation_history', []):
                mutation_counts[mut] = mutation_counts.get(mut, 0) + 1

        if mutation_counts:
            mutation_path = output_path / f'{experiment_name}_mutations.png'
            plot_mutation_distribution(mutation_counts, save_path=str(mutation_path))
            plots['mutations'] = str(mutation_path)

    # Lineage tree (last generation)
    if individuals:
        lineage_path = output_path / f'{experiment_name}_lineage.png'
        plot_lineage_tree(individuals, save_path=str(lineage_path))
        plots['lineage'] = str(lineage_path)

    return plots


def format_evolution_summary(
    stats_history: List[Dict[str, Any]],
    best_individual: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a text summary of evolution.

    Args:
        stats_history: Generation statistics.
        best_individual: Best individual found.

    Returns:
        Formatted text summary.
    """
    lines = [
        "=" * 50,
        "EVOLUTION SUMMARY",
        "=" * 50,
        "",
        f"Total generations: {len(stats_history)}",
    ]

    if stats_history:
        first = stats_history[0]
        last = stats_history[-1]

        lines.extend([
            "",
            "Fitness progression:",
            f"  Initial best: {first.get('best_fitness', 0):.3f}",
            f"  Final best: {last.get('best_fitness', 0):.3f}",
            f"  Final average: {last.get('avg_fitness', 0):.3f}",
            f"  Improvement: {last.get('best_fitness', 0) - first.get('best_fitness', 0):.3f}",
        ])

    if best_individual:
        lines.extend([
            "",
            "Best individual:",
            f"  ID: {best_individual.get('id', 'N/A')}",
            f"  Fitness: {best_individual.get('fitness', 0):.3f}",
            f"  Generation: {best_individual.get('generation', 0)}",
            f"  Mutations: {', '.join(best_individual.get('mutation_history', []))}",
        ])

    lines.extend(["", "=" * 50])

    return "\n".join(lines)
