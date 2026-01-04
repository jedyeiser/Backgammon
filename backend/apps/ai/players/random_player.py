"""
Random player implementation.

A simple player that selects uniformly at random from legal actions.
Useful as a baseline for evaluating other players and for initial testing.
"""
import random
from typing import Any, Dict, List, Optional

from .base import BasePlayer


class RandomPlayer(BasePlayer):
    """
    A player that chooses actions uniformly at random.

    This is the simplest possible player implementation and serves as:
    - A baseline for evaluating AI strength (win rate vs random)
    - A test opponent for debugging game logic
    - A starting point for understanding the player interface

    Attributes:
        seed: Optional random seed for reproducibility.

    Example:
        player = RandomPlayer(player_id='random_1', game_type='backgammon')
        action = player.select_action(game_state, legal_actions)
    """

    def __init__(
        self,
        player_id: str,
        game_type: str,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a random player.

        Args:
            player_id: Unique identifier for this player.
            game_type: Game type this player plays.
            name: Optional display name. Defaults to 'Random Player'.
            seed: Optional random seed for reproducibility in testing.
        """
        super().__init__(
            player_id=player_id,
            game_type=game_type,
            name=name or 'Random Player',
        )
        self.seed = seed
        self._rng = random.Random(seed)

    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select a random action from legal actions.

        Args:
            game_state: Current game state (ignored by random player).
            legal_actions: List of valid actions to choose from.

        Returns:
            A randomly selected action from legal_actions.

        Raises:
            ValueError: If legal_actions is empty.
        """
        if not legal_actions:
            raise ValueError("Cannot select from empty action list")

        return self._rng.choice(legal_actions)

    def get_player_type(self) -> str:
        """Return 'random' as the player type."""
        return 'random'

    def get_config(self) -> Dict[str, Any]:
        """Return configuration including seed."""
        config = super().get_config()
        config['seed'] = self.seed
        return config

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """
        Reset the random number generator with a new seed.

        Args:
            seed: New random seed. If None, uses system entropy.
        """
        self.seed = seed
        self._rng = random.Random(seed)
