"""
Position evaluation functions for games.

These functions provide hand-coded heuristics for evaluating
game positions, useful for:
- Heuristic players
- Training signal validation
- Understanding what makes positions good/bad
"""
from .backgammon import (
    evaluate_position,
    pip_count,
    blot_count,
    made_points_count,
    home_board_strength,
    race_position,
)

__all__ = [
    'evaluate_position',
    'pip_count',
    'blot_count',
    'made_points_count',
    'home_board_strength',
    'race_position',
]
