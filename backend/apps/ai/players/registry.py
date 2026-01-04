"""
Player registry for creating and managing player instances.

This module provides a registry pattern for player types, allowing
new player implementations to be registered and instantiated by name.
"""
from typing import Any, Callable, Dict, Optional, Type

from .base import BasePlayer


class PlayerRegistry:
    """
    Registry for player types.

    Allows registration of player classes by type name and instantiation
    of players by type. Supports both built-in and custom player types.

    Example:
        # Register a custom player
        PlayerRegistry.register('my_ai', MyAIPlayer)

        # Create a player instance
        player = PlayerRegistry.create(
            player_type='my_ai',
            player_id='player_1',
            game_type='backgammon',
        )
    """

    _registry: Dict[str, Type[BasePlayer]] = {}

    @classmethod
    def register(
        cls,
        player_type: str,
        player_class: Type[BasePlayer],
    ) -> None:
        """
        Register a player class with a type name.

        Args:
            player_type: String identifier for this player type.
            player_class: The player class to register.

        Raises:
            ValueError: If player_type is already registered.
            TypeError: If player_class is not a BasePlayer subclass.
        """
        if player_type in cls._registry:
            raise ValueError(f"Player type '{player_type}' is already registered")

        if not issubclass(player_class, BasePlayer):
            raise TypeError(
                f"Player class must be a subclass of BasePlayer, "
                f"got {player_class.__name__}"
            )

        cls._registry[player_type] = player_class

    @classmethod
    def unregister(cls, player_type: str) -> None:
        """
        Unregister a player type.

        Args:
            player_type: The type name to unregister.
        """
        cls._registry.pop(player_type, None)

    @classmethod
    def get(cls, player_type: str) -> Optional[Type[BasePlayer]]:
        """
        Get a registered player class by type name.

        Args:
            player_type: The type name to look up.

        Returns:
            The player class, or None if not registered.
        """
        return cls._registry.get(player_type)

    @classmethod
    def create(
        cls,
        player_type: str,
        player_id: str,
        game_type: str,
        **kwargs: Any,
    ) -> BasePlayer:
        """
        Create a player instance by type name.

        Args:
            player_type: Registered type name of the player.
            player_id: Unique ID for the player instance.
            game_type: Game type the player will play.
            **kwargs: Additional arguments passed to player constructor.

        Returns:
            A new player instance.

        Raises:
            ValueError: If player_type is not registered.
        """
        player_class = cls.get(player_type)
        if player_class is None:
            available = ', '.join(cls._registry.keys()) or '(none)'
            raise ValueError(
                f"Unknown player type '{player_type}'. "
                f"Available types: {available}"
            )

        return player_class(
            player_id=player_id,
            game_type=game_type,
            **kwargs,
        )

    @classmethod
    def list_types(cls) -> list[str]:
        """
        List all registered player types.

        Returns:
            List of registered type names.
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered player types."""
        cls._registry.clear()


def register_player(player_type: str) -> Callable[[Type[BasePlayer]], Type[BasePlayer]]:
    """
    Decorator to register a player class.

    Example:
        @register_player('my_ai')
        class MyAIPlayer(BasePlayer):
            ...

    Args:
        player_type: Type name to register the class under.

    Returns:
        Decorator function.
    """
    def decorator(cls: Type[BasePlayer]) -> Type[BasePlayer]:
        PlayerRegistry.register(player_type, cls)
        return cls
    return decorator


def get_player(
    player_type: str,
    player_id: str,
    game_type: str,
    **kwargs: Any,
) -> BasePlayer:
    """
    Convenience function to create a player by type.

    Args:
        player_type: Registered type name of the player.
        player_id: Unique ID for the player instance.
        game_type: Game type the player will play.
        **kwargs: Additional arguments passed to player constructor.

    Returns:
        A new player instance.
    """
    return PlayerRegistry.create(
        player_type=player_type,
        player_id=player_id,
        game_type=game_type,
        **kwargs,
    )


# Register built-in player types
def _register_builtin_players() -> None:
    """Register the built-in player types."""
    from .random_player import RandomPlayer
    from .heuristic import HeuristicPlayer
    from .neural import NeuralPlayer
    from .behavioral import BehavioralPlayer

    PlayerRegistry.register('random', RandomPlayer)
    PlayerRegistry.register('heuristic', HeuristicPlayer)
    PlayerRegistry.register('neural', NeuralPlayer)
    PlayerRegistry.register('behavioral', BehavioralPlayer)


# Auto-register on module import
_register_builtin_players()
