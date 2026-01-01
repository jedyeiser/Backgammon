"""
Abstract base class for game rule sets.

This module defines the interface that all game implementations must follow,
enabling a pluggable architecture for different board games.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseRuleSet(ABC):
    """
    Abstract base class defining the interface for game rule implementations.

    All game types (Backgammon, Catan, etc.) must implement this interface
    to work with the game engine and platform.

    Attributes:
        game_type: Unique identifier for this game type (e.g., 'backgammon').
        display_name: Human-readable name for UI display.
        min_players: Minimum number of players required.
        max_players: Maximum number of players allowed.
        requires_dice: Whether the game uses dice.

    Example:
        class MyGameRuleSet(BaseRuleSet):
            game_type = 'mygame'
            display_name = 'My Game'
            min_players = 2
            max_players = 4

            def get_initial_state(self) -> Dict[str, Any]:
                return {'board': [], 'scores': {}}
    """

    game_type: str = ''
    display_name: str = ''
    min_players: int = 2
    max_players: int = 2
    requires_dice: bool = False

    def __init__(self, game_state: Dict[str, Any]):
        """
        Initialize rule set with current game state.

        Args:
            game_state: Dictionary containing the current game state.
        """
        self.game_state = game_state

    @abstractmethod
    def get_initial_state(self) -> Dict[str, Any]:
        """
        Return the initial game state for a new game.

        Returns:
            Dict containing the initial game state in game-specific format.
        """
        pass

    @abstractmethod
    def get_legal_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """
        Get all legal actions for the specified player.

        Args:
            player_id: Identifier for the player (e.g., 'white', 'black', or user ID).

        Returns:
            List of dictionaries describing legal actions. Each action dict
            should have at minimum a 'type' key.
        """
        pass

    @abstractmethod
    def apply_action(self, player_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply an action and return the result.

        Args:
            player_id: Identifier for the player taking the action.
            action: Dictionary describing the action to take.

        Returns:
            Dictionary containing the result of the action and any relevant data.

        Raises:
            ValueError: If the action is invalid.
        """
        pass

    @abstractmethod
    def check_winner(self) -> Optional[str]:
        """
        Check if there's a winner.

        Returns:
            Player ID of the winner, or None if game is ongoing.
        """
        pass

    @abstractmethod
    def get_current_player(self) -> str:
        """
        Get the ID of the player whose turn it is.

        Returns:
            Player ID of the current player.
        """
        pass

    @abstractmethod
    def validate_state(self) -> bool:
        """
        Validate that the current game state is legal.

        Returns:
            True if state is valid, False otherwise.
        """
        pass

    def roll_dice(self) -> Dict[str, Any]:
        """
        Roll dice for the game (if applicable).

        Override this method for games that use dice.

        Returns:
            Dictionary containing dice values and moves remaining.
        """
        raise NotImplementedError("This game type doesn't use dice")

    def calculate_score(self, winner_id: str) -> Dict[str, int]:
        """
        Calculate scores for all players after game completion.

        Override this method for games with complex scoring systems.

        Args:
            winner_id: ID of the winning player.

        Returns:
            Dictionary mapping player IDs to their scores.
        """
        return {winner_id: 1}

    def serialize_state(self) -> Dict[str, Any]:
        """
        Serialize game state for storage/transmission.

        Returns:
            JSON-serializable dictionary of game state.
        """
        return self.game_state

    @classmethod
    def deserialize_state(cls, data: Dict[str, Any]) -> 'BaseRuleSet':
        """
        Create a rule set instance from serialized state.

        Args:
            data: Serialized game state.

        Returns:
            New instance with the given state.
        """
        return cls(data)
