"""
Population management for evolutionary algorithms.

Handles the lifecycle of a population of networks:
- Initialization (random or from templates)
- Evaluation (fitness calculation)
- Evolution (selection, crossover, mutation)
- Generation advancement

The population manager is the main interface for running
evolutionary experiments.
"""
import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .selection import Individual, TournamentSelection, EliteSelection
from .mutations import WeightMutator, TopologyMutator, CombinedMutator
from .crossover import WeightCrossover, ArchitectureCrossover

if TYPE_CHECKING:
    import torch.nn as nn
    from ..encoders.base import BaseEncoder
    from ..models import AIModel


@dataclass
class EvolutionConfig:
    """Configuration for an evolution experiment."""

    # Population
    population_size: int = 50
    elite_count: int = 2

    # Selection
    selection_strategy: str = 'tournament'
    tournament_size: int = 3

    # Mutation
    mutation_rate: float = 0.8
    weight_sigma: float = 0.1
    topology_mutation_rate: float = 0.1
    add_node_rate: float = 0.03
    add_connection_rate: float = 0.05

    # Crossover
    crossover_rate: float = 0.5
    crossover_strategy: str = 'interpolation'

    # Fitness evaluation
    games_per_evaluation: int = 100
    evaluation_opponent: str = 'random'  # 'random', 'self', 'best'

    # Evolution limits
    max_generations: int = 100
    target_fitness: float = 0.7  # Win rate to stop at

    # Persistence
    checkpoint_interval: int = 10
    checkpoint_dir: str = './evolution_checkpoints'


@dataclass
class GenerationStats:
    """Statistics for a generation."""
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    min_fitness: float = 0.0
    fitness_std: float = 0.0
    num_mutations: int = 0
    num_crossovers: int = 0
    num_new_individuals: int = 0


class Population:
    """
    Manages a population of evolving neural networks.

    Handles the complete evolutionary cycle:
    1. Initialize population (random networks or from template)
    2. Evaluate fitness (games against opponent)
    3. Select parents
    4. Create offspring (crossover + mutation)
    5. Replace old generation
    6. Repeat

    Example:
        config = EvolutionConfig(population_size=50)
        pop = Population(config, game_type='backgammon')

        # Initialize from architecture
        pop.initialize_from_architecture(td_gammon_architecture())

        # Run evolution
        for gen in range(100):
            pop.evaluate_all()
            pop.evolve_generation()
            print(f"Gen {gen}: best={pop.best_fitness:.3f}")
    """

    def __init__(
        self,
        config: EvolutionConfig,
        game_type: str = 'backgammon',
        fitness_function: Optional[Callable[['nn.Module'], float]] = None,
    ):
        """
        Initialize the population manager.

        Args:
            config: Evolution configuration.
            game_type: Type of game for evaluation.
            fitness_function: Optional custom fitness function.
                            If None, uses win rate vs random.
        """
        self.config = config
        self.game_type = game_type
        self.fitness_function = fitness_function

        # Population state
        self.individuals: List[Individual] = []
        self.generation = 0

        # Evolution operators
        self.mutator = CombinedMutator(
            weight_sigma=config.weight_sigma,
            topology_add_node_rate=config.add_node_rate,
            topology_add_conn_rate=config.add_connection_rate,
        )
        self.crossover = WeightCrossover(strategy=config.crossover_strategy)
        self.selection = TournamentSelection(tournament_size=config.tournament_size)
        self.elite_selection = EliteSelection(elite_count=config.elite_count)

        # Statistics
        self.stats_history: List[GenerationStats] = []

        # Set up encoder
        self.encoder = self._get_encoder()

    def _get_encoder(self):
        """Get encoder for game type."""
        if self.game_type == 'backgammon':
            from ..encoders import BackgammonEncoder
            return BackgammonEncoder()
        raise ValueError(f"Unknown game type: {self.game_type}")

    def initialize_random(
        self,
        architecture: Dict[str, Any],
    ) -> None:
        """
        Initialize population with random networks.

        Args:
            architecture: Architecture template for networks.
        """
        from ..networks import NetworkBuilder

        builder = NetworkBuilder()
        self.individuals = []

        for i in range(self.config.population_size):
            network = builder.from_json(architecture)
            # Weights are already randomly initialized by PyTorch
            ind = Individual(
                network=network,
                fitness=0.0,
                generation=0,
                id=f"ind_{i:04d}",
            )
            self.individuals.append(ind)

        self.generation = 0

    def initialize_from_architecture(
        self,
        architecture: Dict[str, Any],
        vary_weights: bool = True,
    ) -> None:
        """
        Initialize from a single architecture with varied weights.

        Args:
            architecture: Base architecture.
            vary_weights: If True, add variation to initial weights.
        """
        self.initialize_random(architecture)

        if vary_weights:
            # Add some initial variation
            for ind in self.individuals:
                self.mutator.weight_mutator.mutate(ind.network, in_place=True)

    def initialize_from_ai_models(
        self,
        ai_models: List['AIModel'],
    ) -> None:
        """
        Initialize population from existing AIModels.

        Args:
            ai_models: List of AIModels to seed population.
        """
        from ..players.neural import NeuralPlayer

        self.individuals = []

        for i, model in enumerate(ai_models):
            player = NeuralPlayer.from_ai_model(model)
            ind = Individual(
                network=player.network,
                fitness=0.0,
                generation=0,
                parent_ids=[str(model.id)],
                id=f"ind_{i:04d}",
            )
            self.individuals.append(ind)

        # Fill remaining slots with mutations of existing
        while len(self.individuals) < self.config.population_size:
            parent = random.choice(self.individuals[:len(ai_models)])
            child_network, _ = self.mutator.mutate(parent.network)
            ind = Individual(
                network=child_network,
                fitness=0.0,
                generation=0,
                parent_ids=[parent.id],
                mutation_history=['initial_variation'],
                id=f"ind_{len(self.individuals):04d}",
            )
            self.individuals.append(ind)

        self.generation = 0

    def evaluate_all(
        self,
        parallel: bool = False,
    ) -> GenerationStats:
        """
        Evaluate fitness for all individuals.

        Args:
            parallel: If True, evaluate in parallel (future).

        Returns:
            Generation statistics.
        """
        fitnesses = []

        for ind in self.individuals:
            if self.fitness_function:
                fitness = self.fitness_function(ind.network)
            else:
                fitness = self._default_fitness(ind.network)

            ind.fitness = fitness
            fitnesses.append(fitness)

        # Compute statistics
        stats = GenerationStats(
            generation=self.generation,
            best_fitness=max(fitnesses),
            avg_fitness=sum(fitnesses) / len(fitnesses),
            min_fitness=min(fitnesses),
            fitness_std=self._std(fitnesses),
        )

        self.stats_history.append(stats)
        return stats

    def _default_fitness(self, network: 'nn.Module') -> float:
        """
        Default fitness: win rate against random player.

        Args:
            network: Network to evaluate.

        Returns:
            Win rate (0-1).
        """
        from ..training.self_play import SelfPlayTrainer

        trainer = SelfPlayTrainer(
            network=network,
            encoder=self.encoder,
            game_type=self.game_type,
            temperature=0.0,  # Greedy for evaluation
        )

        win_rate = trainer.evaluate_vs_random(
            num_games=self.config.games_per_evaluation
        )

        return win_rate

    def evolve_generation(self) -> GenerationStats:
        """
        Create the next generation through evolution.

        Returns:
            Statistics for the new generation.
        """
        new_individuals = []
        stats = GenerationStats(generation=self.generation + 1)

        # Preserve elite
        elite = self.elite_selection.get_elite(self.individuals)
        for ind in elite:
            new_ind = Individual(
                network=self._clone_network(ind.network),
                fitness=ind.fitness,
                generation=self.generation + 1,
                parent_ids=[ind.id],
                mutation_history=['elite'],
                id=f"gen{self.generation + 1}_elite_{len(new_individuals):02d}",
            )
            new_individuals.append(new_ind)

        # Fill rest with offspring
        while len(new_individuals) < self.config.population_size:
            # Decide: crossover or mutation only
            if random.random() < self.config.crossover_rate:
                # Crossover
                parents = self.selection.select(self.individuals, 2)
                if len(parents) >= 2:
                    try:
                        child_network = self.crossover.crossover(
                            parents[0].network,
                            parents[1].network,
                        )
                        parent_ids = [parents[0].id, parents[1].id]
                        mutation_history = ['crossover']
                        stats.num_crossovers += 1
                    except ValueError:
                        # Incompatible architectures, fall back to mutation
                        parent = self.selection.select(self.individuals, 1)[0]
                        child_network = self._clone_network(parent.network)
                        parent_ids = [parent.id]
                        mutation_history = []
                else:
                    continue
            else:
                # Mutation only
                parent = self.selection.select(self.individuals, 1)[0]
                child_network = self._clone_network(parent.network)
                parent_ids = [parent.id]
                mutation_history = []

            # Apply mutation
            if random.random() < self.config.mutation_rate:
                use_topology = random.random() < self.config.topology_mutation_rate
                child_network, mutation_info = self.mutator.mutate(
                    child_network,
                    mutate_topology=use_topology,
                    mutate_weights=True,
                )
                mutation_history.extend(mutation_info.get('topology_mutations', []))
                if mutation_info.get('weight_mutated'):
                    mutation_history.append('weight_perturbation')
                stats.num_mutations += 1

            new_ind = Individual(
                network=child_network,
                fitness=0.0,
                generation=self.generation + 1,
                parent_ids=parent_ids,
                mutation_history=mutation_history,
                id=f"gen{self.generation + 1}_ind_{len(new_individuals):03d}",
            )
            new_individuals.append(new_ind)
            stats.num_new_individuals += 1

        # Replace population
        self.individuals = new_individuals
        self.generation += 1

        return stats

    def evolve(
        self,
        generations: Optional[int] = None,
        progress_callback: Optional[Callable[[int, GenerationStats], None]] = None,
    ) -> List[GenerationStats]:
        """
        Run full evolution loop.

        Args:
            generations: Number of generations (uses config if None).
            progress_callback: Called with (generation, stats) each gen.

        Returns:
            List of generation statistics.
        """
        generations = generations or self.config.max_generations
        all_stats = []

        for gen in range(generations):
            # Evaluate current generation
            eval_stats = self.evaluate_all()

            if progress_callback:
                progress_callback(gen, eval_stats)

            all_stats.append(eval_stats)

            # Check for target fitness
            if eval_stats.best_fitness >= self.config.target_fitness:
                break

            # Checkpoint
            if gen % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            # Evolve to next generation
            self.evolve_generation()

        return all_stats

    def get_best(self) -> Individual:
        """Get the best individual in the current population."""
        return max(self.individuals, key=lambda x: x.fitness)

    def get_top_n(self, n: int) -> List[Individual]:
        """Get the top n individuals by fitness."""
        return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[:n]

    @property
    def best_fitness(self) -> float:
        """Get the best fitness in the population."""
        return max(ind.fitness for ind in self.individuals) if self.individuals else 0.0

    @property
    def avg_fitness(self) -> float:
        """Get the average fitness in the population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    def _clone_network(self, network: 'nn.Module') -> 'nn.Module':
        """Create a deep copy of a network."""
        from ..networks import NetworkBuilder

        builder = NetworkBuilder()
        if hasattr(network, 'architecture'):
            new_network = builder.from_json(network.architecture)
            weights = builder.serialize_weights(network)
            builder.deserialize_weights(weights, new_network)
            return new_network
        return copy.deepcopy(network)

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def save_checkpoint(self, path: Optional[str] = None) -> Path:
        """
        Save population checkpoint.

        Args:
            path: Optional custom path.

        Returns:
            Path to saved checkpoint.
        """
        import torch
        from ..networks import NetworkBuilder

        checkpoint_dir = Path(path or self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        filepath = checkpoint_dir / f'population_gen{self.generation:05d}.pt'

        builder = NetworkBuilder()

        # Serialize each individual
        serialized = []
        for ind in self.individuals:
            ind_data = {
                'id': ind.id,
                'fitness': ind.fitness,
                'generation': ind.generation,
                'parent_ids': ind.parent_ids,
                'mutation_history': ind.mutation_history,
                'architecture': (
                    ind.network.architecture
                    if hasattr(ind.network, 'architecture')
                    else builder.to_json(ind.network)
                ),
                'weights': builder.serialize_weights(ind.network),
            }
            serialized.append(ind_data)

        checkpoint = {
            'generation': self.generation,
            'config': self.config,
            'individuals': serialized,
            'stats_history': self.stats_history,
        }

        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(self, path: str) -> None:
        """
        Load population from checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        import torch
        from ..networks import NetworkBuilder

        checkpoint = torch.load(path, weights_only=False)

        self.generation = checkpoint['generation']
        self.config = checkpoint.get('config', self.config)
        self.stats_history = checkpoint.get('stats_history', [])

        builder = NetworkBuilder()
        self.individuals = []

        for ind_data in checkpoint['individuals']:
            network = builder.from_json(ind_data['architecture'])
            builder.deserialize_weights(ind_data['weights'], network)

            ind = Individual(
                network=network,
                fitness=ind_data['fitness'],
                generation=ind_data['generation'],
                parent_ids=ind_data.get('parent_ids', []),
                mutation_history=ind_data.get('mutation_history', []),
                id=ind_data['id'],
            )
            self.individuals.append(ind)

    def save_best_to_ai_model(self, ai_model: 'AIModel') -> None:
        """
        Save the best individual to an AIModel.

        Args:
            ai_model: AIModel to save to.
        """
        from ..networks import NetworkBuilder

        best = self.get_best()
        builder = NetworkBuilder()

        ai_model.network_architecture = (
            best.network.architecture
            if hasattr(best.network, 'architecture')
            else builder.to_json(best.network)
        )
        ai_model.network_weights = builder.serialize_weights(best.network)
        ai_model.generation = self.generation

        ai_model.save()
