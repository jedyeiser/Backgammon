"""
Base player abstraction for all player types.

This module defines the interface that all players (human and AI) must implement
to participate in games. The key method is `select_action`, which takes a game
state and list of legal actions, returning the chosen action.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BasePlayer(ABC):
    """
    Abstract base class for all player types.

    All players (human, random, heuristic, neural, etc.) inherit from this class
    and implement the `select_action` method to choose moves in games.

    Attributes:
        player_id: Unique identifier for this player instance.
        game_type: The game type this player is configured for (e.g., 'backgammon').
        name: Human-readable name for display purposes.

    Example:
        class MyAIPlayer(BasePlayer):
            def select_action(self, game_state, legal_actions):
                # Evaluate each action and return the best one
                return max(legal_actions, key=self.evaluate)

            def get_player_type(self):
                return 'my_ai'
    """

    def __init__(
        self,
        player_id: str,
        game_type: str,
        name: Optional[str] = None,
    ):
        """
        Initialize a player.

        Args:
            player_id: Unique identifier for this player.
            game_type: Game type this player plays (e.g., 'backgammon', 'catan').
            name: Optional display name. Defaults to player_id if not provided.
        """
        self.player_id = player_id
        self.game_type = game_type
        self.name = name or player_id

    @abstractmethod
    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Choose an action from the list of legal actions.

        This is the core method that all players must implement. It receives
        the current game state and a list of all legal actions, and must
        return exactly one action from that list.

        Args:
            game_state: Current state of the game as a dictionary.
                       Format depends on game type.
            legal_actions: List of valid actions the player can take.
                          Each action is a dictionary with at minimum a 'type' key.

        Returns:
            A single action dictionary from legal_actions.

        Raises:
            ValueError: If legal_actions is empty (should not normally happen).

        Example (backgammon):
            game_state = {
                'points': {...},
                'bar': {'white': 0, 'black': 0},
                'home': {'white': 0, 'black': 0},
                'current_turn': 'white',
                'dice': [3, 5],
                'moves_remaining': [3, 5],
            }
            legal_actions = [
                {'type': 'move', 'from': 1, 'to': 4, 'die_used': 3},
                {'type': 'move', 'from': 1, 'to': 6, 'die_used': 5},
                ...
            ]
        """
        pass

    @abstractmethod
    def get_player_type(self) -> str:
        """
        Return the type identifier for this player.

        Used for serialization, logging, and player registry lookup.

        Returns:
            A string identifying this player type (e.g., 'random', 'neural', 'human').
        """
        pass

    def on_game_start(self, game_state: Dict[str, Any]) -> None:
        """
        Called when a new game starts.

        Override this method to perform any initialization needed
        at the start of a game (e.g., reset internal state).

        Args:
            game_state: Initial game state.
        """
        pass

    def on_game_end(
        self,
        game_state: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """
        Called when a game ends.

        Override this method to perform any cleanup or learning
        at the end of a game (e.g., update neural network weights).

        Args:
            game_state: Final game state.
            result: Dictionary with game result information:
                   - 'winner': player_id of winner
                   - 'win_type': type of win (e.g., 'normal', 'gammon')
                   - 'points': points won
        """
        pass

    def on_opponent_action(
        self,
        action: Dict[str, Any],
        game_state: Dict[str, Any],
    ) -> None:
        """
        Called when the opponent takes an action.

        Override this method to observe opponent moves for learning
        or strategy adaptation.

        Args:
            action: The action taken by the opponent.
            game_state: Game state after the opponent's action.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Return player configuration for serialization.

        Override to include additional configuration specific to
        your player implementation.

        Returns:
            Dictionary with player configuration.
        """
        return {
            'player_id': self.player_id,
            'game_type': self.game_type,
            'name': self.name,
            'type': self.get_player_type(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.player_id}, game={self.game_type})"

    def __str__(self) -> str:
        return f"{self.name} ({self.get_player_type()})"
