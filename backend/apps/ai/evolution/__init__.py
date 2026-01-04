"""
Neuroevolution module for evolving neural network game players.

Implements two complementary evolution approaches:
1. Weight perturbation: Add Gaussian noise to weights
2. Topology evolution (NEAT-style): Add/remove nodes and connections

This module provides:
- Mutation operators (weight and topology)
- Crossover operators for combining networks
- Selection strategies (tournament, rank, truncation)
- Population management for complete evolution experiments

Example usage:
    from apps.ai.evolution import Population, EvolutionConfig
    from apps.ai.networks import td_gammon_architecture

    # Configure evolution
    config = EvolutionConfig(
        population_size=50,
        max_generations=100,
        mutation_rate=0.8,
        crossover_rate=0.5,
    )

    # Create population
    pop = Population(config, game_type='backgammon')
    pop.initialize_from_architecture(td_gammon_architecture())

    # Run evolution
    for gen in range(100):
        pop.evaluate_all()
        print(f"Gen {gen}: best={pop.best_fitness:.3f}")
        pop.evolve_generation()

    # Get best network
    best = pop.get_best()
    print(f"Best individual: {best.id} with fitness {best.fitness:.3f}")
"""
from .mutations import (
    WeightMutator,
    TopologyMutator,
    CombinedMutator,
)
from .crossover import (
    WeightCrossover,
    ArchitectureCrossover,
    blend_networks,
)
from .selection import (
    Individual,
    TournamentSelection,
    RankSelection,
    TruncationSelection,
    EliteSelection,
    DiversityAwareSelection,
    get_selection_strategy,
)
from .population import (
    Population,
    EvolutionConfig,
    GenerationStats,
)

__all__ = [
    # Mutations
    'WeightMutator',
    'TopologyMutator',
    'CombinedMutator',

    # Crossover
    'WeightCrossover',
    'ArchitectureCrossover',
    'blend_networks',

    # Selection
    'Individual',
    'TournamentSelection',
    'RankSelection',
    'TruncationSelection',
    'EliteSelection',
    'DiversityAwareSelection',
    'get_selection_strategy',

    # Population management
    'Population',
    'EvolutionConfig',
    'GenerationStats',
]
