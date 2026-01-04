"""
Board state encoders for neural network input.

Encoders transform game states into fixed-size feature vectors
suitable for neural network input.
"""
from .base import BaseEncoder
from .backgammon import BackgammonEncoder

__all__ = [
    'BaseEncoder',
    'BackgammonEncoder',
]
