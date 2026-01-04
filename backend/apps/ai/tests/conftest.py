"""
Pytest fixtures for AI app tests.

Provides fixtures for:
- Network architectures
- Sample game states
- AI models
- Training configurations
"""
import pytest
from typing import Dict, Any, List


@pytest.fixture
def sample_backgammon_state() -> Dict[str, Any]:
    """Return a sample backgammon game state for testing."""
    return {
        'points': {
            '1': -2,   # 2 black checkers
            '6': 5,    # 5 white checkers
            '8': 3,    # 3 white checkers
            '12': -5,  # 5 black checkers
            '13': 5,   # 5 white checkers
            '17': -3,  # 3 black checkers
            '19': -5,  # 5 black checkers
            '24': 2,   # 2 white checkers
        },
        'bar': {'white': 0, 'black': 0},
        'home': {'white': 0, 'black': 0},
        'current_turn': 'white',
        'dice': [3, 5],
    }


@pytest.fixture
def initial_backgammon_state() -> Dict[str, Any]:
    """Return the standard starting position."""
    return {
        'points': {
            '1': -2,
            '6': 5,
            '8': 3,
            '12': -5,
            '13': 5,
            '17': -3,
            '19': -5,
            '24': 2,
        },
        'bar': {'white': 0, 'black': 0},
        'home': {'white': 0, 'black': 0},
        'current_turn': 'white',
    }


@pytest.fixture
def empty_board_state() -> Dict[str, Any]:
    """Return an empty board (for testing edge cases)."""
    return {
        'points': {},
        'bar': {'white': 0, 'black': 0},
        'home': {'white': 15, 'black': 15},
        'current_turn': 'white',
    }


@pytest.fixture
def bearing_off_state() -> Dict[str, Any]:
    """Return a state where white is bearing off."""
    return {
        'points': {
            '1': 3,
            '2': 4,
            '3': 4,
            '4': 2,
            '5': 1,
            '6': 1,
            '19': -5,
            '20': -5,
            '21': -5,
        },
        'bar': {'white': 0, 'black': 0},
        'home': {'white': 0, 'black': 0},
        'current_turn': 'white',
    }


@pytest.fixture
def minimal_architecture() -> Dict[str, Any]:
    """Return a minimal neural network architecture for testing."""
    return {
        'input_size': 198,
        'output_size': 1,
        'layers': [
            {'id': 'fc1', 'type': 'linear', 'in': 198, 'out': 10},
            {'id': 'act1', 'type': 'activation', 'fn': 'relu'},
            {'id': 'fc2', 'type': 'linear', 'in': 10, 'out': 1},
            {'id': 'output', 'type': 'activation', 'fn': 'sigmoid'},
        ]
    }


@pytest.fixture
def td_gammon_architecture() -> Dict[str, Any]:
    """Return TD-Gammon style architecture."""
    return {
        'input_size': 198,
        'output_size': 1,
        'layers': [
            {'id': 'fc1', 'type': 'linear', 'in': 198, 'out': 80},
            {'id': 'act1', 'type': 'activation', 'fn': 'sigmoid'},
            {'id': 'fc2', 'type': 'linear', 'in': 80, 'out': 1},
            {'id': 'output', 'type': 'activation', 'fn': 'sigmoid'},
        ]
    }


@pytest.fixture
def sample_legal_actions() -> List[Dict[str, Any]]:
    """Return sample legal actions for a backgammon turn."""
    return [
        {'type': 'move', 'from': 6, 'to': 3, 'dice': 3},
        {'type': 'move', 'from': 6, 'to': 1, 'dice': 5},
        {'type': 'move', 'from': 8, 'to': 5, 'dice': 3},
        {'type': 'move', 'from': 8, 'to': 3, 'dice': 5},
        {'type': 'move', 'from': 13, 'to': 10, 'dice': 3},
        {'type': 'move', 'from': 13, 'to': 8, 'dice': 5},
    ]
