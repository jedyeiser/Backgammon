"""
Neural network infrastructure for AI players.

This module provides:
- NetworkBuilder: Convert between JSON architecture and PyTorch models
- Preset architectures for different games
- Weight serialization/deserialization
"""
from .builder import NetworkBuilder
from .architectures import (
    td_gammon_architecture,
    modern_backgammon_architecture,
    create_mlp_architecture,
)

__all__ = [
    'NetworkBuilder',
    'td_gammon_architecture',
    'modern_backgammon_architecture',
    'create_mlp_architecture',
]
