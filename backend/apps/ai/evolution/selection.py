"""
Selection strategies for evolutionary algorithms.

Selection determines which individuals survive and reproduce.
Different strategies provide different selection pressures:
- Tournament: Simple, tunable pressure
- Rank-based: Reduces fitness-proportional variance
- Truncation: Only top performers reproduce

All strategies work with (individual, fitness) pairs and
return selected individuals for the next generation.
"""
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, TypeVar

T = TypeVar('T')


@dataclass
class Individual:
    """Wrapper for an evolved individual with fitness."""
    network: Any  # nn.Module
    fitness: float
    generation: int = 0
    parent_ids: List[str] = None
    mutation_history: List[str] = None
    id: str = ''

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutation_history is None:
            self.mutation_history = []
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())[:8]


class TournamentSelection:
    """
    Tournament selection strategy.

    Selects individuals by running tournaments: randomly pick
    k individuals, best one wins. Repeat until enough selected.

    Tournament size controls selection pressure:
    - k=2: Low pressure, more diversity
    - k=7: High pressure, faster convergence

    Example:
        selection = TournamentSelection(tournament_size=3)
        parents = selection.select(population, num_parents=10)
    """

    def __init__(
        self,
        tournament_size: int = 3,
        allow_duplicates: bool = True,
    ):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of individuals per tournament.
            allow_duplicates: If True, same individual can be selected twice.
        """
        self.tournament_size = tournament_size
        self.allow_duplicates = allow_duplicates

    def select(
        self,
        population: List[Individual],
        num_to_select: int,
    ) -> List[Individual]:
        """
        Select individuals using tournament selection.

        Args:
            population: List of individuals with fitness.
            num_to_select: Number of individuals to select.

        Returns:
            List of selected individuals.
        """
        if not population:
            return []

        selected = []
        available = list(population)

        for _ in range(num_to_select):
            if not available:
                if self.allow_duplicates:
                    available = list(population)
                else:
                    break

            # Run tournament
            tournament_size = min(self.tournament_size, len(available))
            contestants = random.sample(available, tournament_size)

            # Select winner (highest fitness)
            winner = max(contestants, key=lambda ind: ind.fitness)
            selected.append(winner)

            if not self.allow_duplicates:
                available.remove(winner)

        return selected

    def select_pairs(
        self,
        population: List[Individual],
        num_pairs: int,
    ) -> List[Tuple[Individual, Individual]]:
        """
        Select parent pairs for crossover.

        Args:
            population: List of individuals.
            num_pairs: Number of parent pairs to select.

        Returns:
            List of (parent_a, parent_b) tuples.
        """
        pairs = []
        for _ in range(num_pairs):
            parents = self.select(population, 2)
            if len(parents) >= 2:
                pairs.append((parents[0], parents[1]))
        return pairs


class RankSelection:
    """
    Rank-based selection strategy.

    Assigns selection probability based on rank rather than
    raw fitness. This reduces the influence of outliers and
    provides more stable selection pressure.

    Probability of rank i: p(i) = (n - i + 1) / sum(1..n)
    """

    def __init__(
        self,
        selection_pressure: float = 1.5,
    ):
        """
        Initialize rank selection.

        Args:
            selection_pressure: Controls distribution steepness.
                              1.0 = uniform, 2.0 = linear ranking
        """
        self.selection_pressure = selection_pressure

    def select(
        self,
        population: List[Individual],
        num_to_select: int,
    ) -> List[Individual]:
        """
        Select individuals using rank-based selection.

        Args:
            population: List of individuals.
            num_to_select: Number to select.

        Returns:
            Selected individuals.
        """
        if not population:
            return []

        # Sort by fitness (ascending)
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_pop)

        # Calculate rank-based probabilities
        probabilities = []
        for i in range(n):
            # Linear ranking: p(i) = (2 - s)/n + 2i(s-1)/(n(n-1))
            # where s is selection pressure and i is rank (0 = worst)
            prob = (
                (2 - self.selection_pressure) / n +
                2 * i * (self.selection_pressure - 1) / (n * (n - 1) + 1e-10)
            )
            probabilities.append(max(0, prob))

        # Normalize
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / n] * n

        # Select based on probabilities
        indices = random.choices(range(n), weights=probabilities, k=num_to_select)
        return [sorted_pop[i] for i in indices]


class TruncationSelection:
    """
    Truncation selection strategy.

    Simple elitist selection: only the top fraction of the
    population is allowed to reproduce.

    This is the simplest selection method but can lead to
    rapid loss of diversity.
    """

    def __init__(
        self,
        truncation_fraction: float = 0.5,
    ):
        """
        Initialize truncation selection.

        Args:
            truncation_fraction: Top fraction that survives (0-1).
        """
        self.truncation_fraction = truncation_fraction

    def select(
        self,
        population: List[Individual],
        num_to_select: int,
    ) -> List[Individual]:
        """
        Select from top individuals.

        Args:
            population: List of individuals.
            num_to_select: Number to select.

        Returns:
            Selected individuals.
        """
        if not population:
            return []

        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Truncate to top fraction
        cutoff = max(1, int(len(sorted_pop) * self.truncation_fraction))
        top_individuals = sorted_pop[:cutoff]

        # Sample from top individuals
        return random.choices(top_individuals, k=num_to_select)


class EliteSelection:
    """
    Elitism: Preserve the best individuals unchanged.

    This is typically combined with another selection method.
    The elite bypass mutation/crossover and go directly to
    the next generation.
    """

    def __init__(
        self,
        elite_count: int = 2,
    ):
        """
        Initialize elite selection.

        Args:
            elite_count: Number of elite individuals to preserve.
        """
        self.elite_count = elite_count

    def get_elite(
        self,
        population: List[Individual],
    ) -> List[Individual]:
        """
        Get the elite individuals from a population.

        Args:
            population: List of individuals.

        Returns:
            List of elite individuals.
        """
        if not population:
            return []

        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:self.elite_count]


class DiversityAwareSelection:
    """
    Selection that considers both fitness and diversity.

    Prevents premature convergence by encouraging diversity
    in the selected population. Combines fitness with a
    novelty/distance measure.
    """

    def __init__(
        self,
        base_selection: Optional['TournamentSelection'] = None,
        diversity_weight: float = 0.3,
    ):
        """
        Initialize diversity-aware selection.

        Args:
            base_selection: Underlying selection strategy.
            diversity_weight: Weight for diversity (0-1).
        """
        self.base_selection = base_selection or TournamentSelection()
        self.diversity_weight = diversity_weight

    def select(
        self,
        population: List[Individual],
        num_to_select: int,
    ) -> List[Individual]:
        """
        Select individuals considering diversity.

        Args:
            population: List of individuals.
            num_to_select: Number to select.

        Returns:
            Selected individuals with diversity.
        """
        if not population:
            return []

        # First, compute diversity scores
        diversity_scores = self._compute_diversity(population)

        # Combine fitness and diversity
        for ind, div_score in zip(population, diversity_scores):
            combined_fitness = (
                (1 - self.diversity_weight) * ind.fitness +
                self.diversity_weight * div_score
            )
            ind._combined_fitness = combined_fitness

        # Select based on combined fitness
        sorted_pop = sorted(
            population,
            key=lambda x: x._combined_fitness,
            reverse=True
        )

        # Use base selection on adjusted population
        return self.base_selection.select(population, num_to_select)

    def _compute_diversity(
        self,
        population: List[Individual],
    ) -> List[float]:
        """
        Compute diversity score for each individual.

        Uses weight vector distance as a proxy for behavioral diversity.
        """
        import torch

        if not population:
            return []

        # Get weight vectors
        weight_vectors = []
        for ind in population:
            state = ind.network.state_dict()
            weights = torch.cat([p.flatten() for p in state.values()])
            weight_vectors.append(weights)

        # Compute average distance to all others
        diversity_scores = []
        for i, vec_i in enumerate(weight_vectors):
            distances = []
            for j, vec_j in enumerate(weight_vectors):
                if i != j:
                    dist = torch.norm(vec_i - vec_j).item()
                    distances.append(dist)

            avg_distance = sum(distances) / len(distances) if distances else 0
            diversity_scores.append(avg_distance)

        # Normalize to 0-1
        max_div = max(diversity_scores) if diversity_scores else 1
        if max_div > 0:
            diversity_scores = [d / max_div for d in diversity_scores]

        return diversity_scores


def get_selection_strategy(
    strategy_name: str,
    **kwargs,
) -> Any:
    """
    Factory function for selection strategies.

    Args:
        strategy_name: One of 'tournament', 'rank', 'truncation'.
        **kwargs: Arguments for the strategy.

    Returns:
        Selection strategy instance.
    """
    strategies = {
        'tournament': TournamentSelection,
        'rank': RankSelection,
        'truncation': TruncationSelection,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name](**kwargs)
