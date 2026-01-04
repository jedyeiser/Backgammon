"""
Player abstractions for AI and human players.

This module provides a unified interface for all player types,
enabling consistent game interaction regardless of whether the
player is human, random AI, heuristic AI, or neural network based.
"""
from .base import BasePlayer
from .random_player import RandomPlayer
from .registry import PlayerRegistry, get_player, register_player

__all__ = [
    'BasePlayer',
    'RandomPlayer',
    'PlayerRegistry',
    'get_player',
    'register_player',
]
