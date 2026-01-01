"""
RuleSet registry for managing available game types.

Provides a central registry for registering and retrieving game rule
implementations, enabling plugin-style extensibility.
"""
from typing import Dict, Type, Optional, List

from .base import BaseRuleSet


class RuleSetRegistry:
    """
    Registry for game rule set implementations.

    Allows dynamic registration and retrieval of game types,
    enabling plugin-style extensibility.

    Usage:
        # Register a rule set
        RuleSetRegistry.register(BackgammonRuleSet)

        # Get a rule set by type
        ruleset_class = RuleSetRegistry.get('backgammon')
        ruleset = ruleset_class(game_state)

        # List available games
        games = RuleSetRegistry.get_available_games()
    """

    _registry: Dict[str, Type[BaseRuleSet]] = {}

    @classmethod
    def register(cls, ruleset_class: Type[BaseRuleSet]) -> None:
        """
        Register a rule set class.

        Args:
            ruleset_class: The RuleSet class to register.

        Raises:
            ValueError: If the game type is already registered.
        """
        game_type = ruleset_class.game_type
        if game_type in cls._registry:
            # Allow re-registration for testing/reloading
            pass
        cls._registry[game_type] = ruleset_class

    @classmethod
    def get(cls, game_type: str) -> Optional[Type[BaseRuleSet]]:
        """
        Get a rule set class by game type.

        Args:
            game_type: The game type identifier (e.g., 'backgammon').

        Returns:
            The RuleSet class, or None if not found.
        """
        return cls._registry.get(game_type)

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseRuleSet]]:
        """
        Get all registered rule sets.

        Returns:
            Dictionary mapping game types to their RuleSet classes.
        """
        return cls._registry.copy()

    @classmethod
    def get_available_games(cls) -> List[Dict[str, any]]:
        """
        Get list of available games for UI display.

        Returns:
            List of dictionaries with game info including:
            - type: Game type identifier
            - name: Display name
            - min_players: Minimum players required
            - max_players: Maximum players allowed
            - requires_dice: Whether game uses dice
        """
        return [
            {
                'type': ruleset.game_type,
                'name': ruleset.display_name,
                'min_players': ruleset.min_players,
                'max_players': ruleset.max_players,
                'requires_dice': ruleset.requires_dice,
            }
            for ruleset in cls._registry.values()
        ]

    @classmethod
    def is_registered(cls, game_type: str) -> bool:
        """
        Check if a game type is registered.

        Args:
            game_type: The game type identifier.

        Returns:
            True if registered, False otherwise.
        """
        return game_type in cls._registry

    @classmethod
    def unregister(cls, game_type: str) -> bool:
        """
        Unregister a game type.

        Args:
            game_type: The game type identifier.

        Returns:
            True if unregistered, False if not found.
        """
        if game_type in cls._registry:
            del cls._registry[game_type]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered game types. Useful for testing."""
        cls._registry.clear()


def _register_defaults() -> None:
    """Register built-in game types."""
    from .backgammon import BackgammonRuleSet
    from .catan import CatanRuleSet

    RuleSetRegistry.register(BackgammonRuleSet)
    RuleSetRegistry.register(CatanRuleSet)


# Auto-register built-in game types on module import
_register_defaults()
