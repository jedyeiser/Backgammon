"""
Heuristic player implementation.

A player that uses hand-coded evaluation functions to select moves.
Useful as a stronger baseline than random and for understanding
what makes positions good or bad.
"""
import copy
from typing import Any, Dict, List, Optional

from .base import BasePlayer


class HeuristicPlayer(BasePlayer):
    """
    A player that selects moves using heuristic evaluation.

    Evaluates all legal moves by simulating them and scoring the
    resulting positions using game-specific heuristics. Selects
    the move with the best evaluation.

    This is stronger than random play but weaker than trained
    neural networks. Good for:
    - Testing game logic
    - Baseline comparisons
    - Understanding position evaluation

    Attributes:
        weights: Custom weights for evaluation factors.
        depth: Search depth (currently only 1-ply supported).

    Example:
        player = HeuristicPlayer(
            player_id='heuristic_1',
            game_type='backgammon',
        )
        action = player.select_action(game_state, legal_actions)
    """

    def __init__(
        self,
        player_id: str,
        game_type: str,
        name: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize a heuristic player.

        Args:
            player_id: Unique identifier for this player.
            game_type: Game type ('backgammon', 'catan', etc.).
            name: Optional display name.
            weights: Optional custom weights for evaluation.
        """
        super().__init__(
            player_id=player_id,
            game_type=game_type,
            name=name or 'Heuristic Player',
        )
        self.weights = weights
        self._evaluator = self._get_evaluator()

    def _get_evaluator(self):
        """Get the appropriate evaluator for this game type."""
        if self.game_type == 'backgammon':
            from ..evaluation.backgammon import evaluate_position
            return evaluate_position
        else:
            # Default: return 0 for all positions
            return lambda state, player, weights=None: 0.0

    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select the best action using heuristic evaluation.

        Evaluates each legal action by simulating it and scoring
        the resulting position. Returns the action with the highest
        evaluation.

        Args:
            game_state: Current game state.
            legal_actions: List of legal actions.

        Returns:
            The action with the best evaluation.

        Raises:
            ValueError: If legal_actions is empty.
        """
        if not legal_actions:
            raise ValueError("Cannot select from empty action list")

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Get current player
        player = self._get_current_player(game_state)

        # Evaluate each action
        best_action = legal_actions[0]
        best_value = float('-inf')

        for action in legal_actions:
            # Simulate the action
            new_state = self._simulate_action(game_state, action, player)

            # Evaluate the resulting position
            value = self._evaluator(new_state, player, self.weights)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _get_current_player(self, game_state: Dict[str, Any]) -> str:
        """Get the current player from game state."""
        # Different games store this differently
        if 'current_turn' in game_state:
            return game_state['current_turn']
        elif 'current_player_index' in game_state:
            # Catan style
            player_order = game_state.get('player_order', [])
            idx = game_state['current_player_index']
            if player_order and 0 <= idx < len(player_order):
                return player_order[idx]
        return 'white'  # Default

    def _simulate_action(
        self,
        game_state: Dict[str, Any],
        action: Dict[str, Any],
        player: str,
    ) -> Dict[str, Any]:
        """
        Simulate an action and return the resulting state.

        Creates a deep copy of the state and applies the action.

        Args:
            game_state: Current game state.
            action: Action to simulate.
            player: Current player.

        Returns:
            New game state after the action.
        """
        # Deep copy to avoid modifying original
        new_state = copy.deepcopy(game_state)

        # Apply action based on game type
        if self.game_type == 'backgammon':
            self._apply_backgammon_action(new_state, action, player)
        # Add other game types as needed

        return new_state

    def _apply_backgammon_action(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        player: str,
    ) -> None:
        """
        Apply a backgammon action to a state (in place).

        Args:
            state: Game state to modify.
            action: Action to apply.
            player: Current player.
        """
        action_type = action.get('type')

        if action_type != 'move':
            return  # Only simulate moves

        from_point = action.get('from')
        to_point = action.get('to')
        die_used = action.get('die_used')

        points = state.get('points', {})
        bar = state.get('bar', {'white': 0, 'black': 0})
        home = state.get('home', {'white': 0, 'black': 0})

        # Determine direction based on player
        is_white = player == 'white'
        sign = 1 if is_white else -1

        # Handle moving from bar
        if from_point == 0 and is_white:  # White bar
            bar['white'] = max(0, bar['white'] - 1)
        elif from_point == 25 and not is_white:  # Black bar
            bar['black'] = max(0, bar['black'] - 1)
        else:
            # Moving from a point
            from_key = str(from_point)
            current = points.get(from_key, 0)
            points[from_key] = current - sign

        # Handle bearing off
        if to_point == 26 and is_white:  # White home
            home['white'] = home.get('white', 0) + 1
        elif to_point == 27 and not is_white:  # Black home
            home['black'] = home.get('black', 0) + 1
        else:
            # Moving to a point
            to_key = str(to_point)
            current = points.get(to_key, 0)

            # Check for hitting opponent blot
            if is_white and current == -1:
                # Hit black blot
                points[to_key] = 1
                bar['black'] = bar.get('black', 0) + 1
            elif not is_white and current == 1:
                # Hit white blot
                points[to_key] = -1
                bar['white'] = bar.get('white', 0) + 1
            else:
                # Normal move
                points[to_key] = current + sign

        # Update moves remaining
        moves_remaining = state.get('moves_remaining', [])
        if die_used in moves_remaining:
            moves_remaining.remove(die_used)

    def get_player_type(self) -> str:
        """Return 'heuristic' as the player type."""
        return 'heuristic'

    def get_config(self) -> Dict[str, Any]:
        """Return configuration including weights."""
        config = super().get_config()
        config['weights'] = self.weights
        return config
