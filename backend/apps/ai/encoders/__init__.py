"""
Board state encoders for neural network input.

Encoders transform game states into feature representations
suitable for neural network input.

Includes:
- BackgammonEncoder: Fixed 198-feature TD-Gammon encoding
- CatanGraphEncoder: Graph-structured encoding for GNNs
"""
from .base import BaseEncoder
from .backgammon import BackgammonEncoder
from .catan_gnn import CatanGraphEncoder

__all__ = [
    'BaseEncoder',
    'BackgammonEncoder',
    'CatanGraphEncoder',
]
