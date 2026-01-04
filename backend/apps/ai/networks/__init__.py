"""
Neural network infrastructure for AI players.

This module provides:
- NetworkBuilder: Convert between JSON architecture and PyTorch models
- Preset architectures for different games
- Weight serialization/deserialization
- Graph Neural Networks for games with variable board structures
"""
from .builder import NetworkBuilder
from .architectures import (
    td_gammon_architecture,
    modern_backgammon_architecture,
    create_mlp_architecture,
)
from .gnn import (
    MessagePassingLayer,
    HeteroMessagePassingLayer,
    GlobalPooling,
    CatanGNN,
    create_catan_gnn,
)

__all__ = [
    # Builder
    'NetworkBuilder',

    # MLP Architectures
    'td_gammon_architecture',
    'modern_backgammon_architecture',
    'create_mlp_architecture',

    # GNN Components
    'MessagePassingLayer',
    'HeteroMessagePassingLayer',
    'GlobalPooling',
    'CatanGNN',
    'create_catan_gnn',
]
