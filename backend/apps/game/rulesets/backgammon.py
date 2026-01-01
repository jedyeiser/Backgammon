"""
Backgammon-specific rule implementation.

This module contains the complete rules for backgammon, including:
- Board setup and representation
- Movement rules and validation
- Bearing off
- Doubling cube
- Win condition detection

Board Representation:
    Points are numbered 1-24.
    Point 0: White's bar
    Point 25: Black's bar
    Point 26: White's home (bearoff)
    Point 27: Black's home (bearoff)

    Positive values: White checkers
    Negative values: Black checkers
"""
import random
from typing import List, Dict, Any, Optional, Tuple

from .base import BaseRuleSet


class BackgammonRuleSet(BaseRuleSet):
    """
    Backgammon game rules implementation.

    Implements the full rules of backgammon including:
    - 24-point board with bar and home areas
    - Dice rolling and movement
    - Hitting blots and entering from bar
    - Bearing off when all checkers are in home board
    - Doubling cube management
    """

    game_type = 'backgammon'
    display_name = 'Backgammon'
    min_players = 2
    max_players = 2
    requires_dice = True

    # Board constants
    TOTAL_POINTS = 24
    WHITE_BAR = 0
    BLACK_BAR = 25
    WHITE_HOME = 26
    BLACK_HOME = 27
    CHECKERS_PER_PLAYER = 15

    def get_initial_state(self) -> Dict[str, Any]:
        """Return standard backgammon starting position."""
        return {
            'points': {
                '1': 2,     # White: 2 checkers
                '6': -5,    # Black: 5 checkers
                '8': -3,    # Black: 3 checkers
                '12': 5,    # White: 5 checkers
                '13': -5,   # Black: 5 checkers
                '17': 3,    # White: 3 checkers
                '19': 5,    # White: 5 checkers
                '24': -2,   # Black: 2 checkers
            },
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
            'current_turn': 'white',
            'dice': [],
            'moves_remaining': [],
            'cube_value': 1,
            'cube_owner': 'center',
            'double_offered': False,
        }

    def roll_dice(self) -> Dict[str, Any]:
        """Roll two dice and calculate available moves."""
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)

        # Doubles give four moves
        if die1 == die2:
            self.game_state['moves_remaining'] = [die1] * 4
        else:
            self.game_state['moves_remaining'] = [die1, die2]

        self.game_state['dice'] = [die1, die2]

        return {
            'dice': [die1, die2],
            'moves_remaining': self.game_state['moves_remaining'],
            'legal_moves': self.get_legal_actions(self.game_state['current_turn'])
        }

    def get_legal_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get all legal moves for the current player."""
        actions = []
        color = self.game_state.get('current_turn', 'white')

        if player_id != color:
            return []

        moves_remaining = self.game_state.get('moves_remaining', [])
        if not moves_remaining:
            return []

        # Check if must enter from bar
        bar_count = self.game_state['bar'].get(color, 0)
        if bar_count > 0:
            return self._get_bar_entry_actions(color)

        # Get normal move actions
        can_bear_off = self._can_bear_off(color)

        for die in set(moves_remaining):
            for from_point in self._get_player_points(color):
                to_point = self._calculate_destination(from_point, die, color)
                if to_point and self._is_valid_move(from_point, to_point, color, can_bear_off):
                    actions.append({
                        'type': 'move',
                        'from': from_point,
                        'to': to_point,
                        'die_used': die
                    })

        return actions

    def apply_action(self, player_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return the result."""
        action_type = action.get('type')

        if action_type == 'roll':
            return self.roll_dice()

        elif action_type == 'move':
            return self._execute_move(action['from'], action['to'], player_id)

        elif action_type == 'double':
            return self._offer_double(player_id)

        elif action_type == 'accept_double':
            return self._accept_double(player_id)

        elif action_type == 'reject_double':
            return self._reject_double(player_id)

        raise ValueError(f"Unknown action type: {action_type}")

    def check_winner(self) -> Optional[str]:
        """Check if a player has borne off all checkers."""
        if self.game_state['home'].get('white', 0) == self.CHECKERS_PER_PLAYER:
            return 'white'
        if self.game_state['home'].get('black', 0) == self.CHECKERS_PER_PLAYER:
            return 'black'
        return None

    def get_current_player(self) -> str:
        """Return whose turn it is."""
        return self.game_state.get('current_turn', 'white')

    def validate_state(self) -> bool:
        """Validate the board state is legal."""
        white_count = self._count_checkers('white')
        black_count = self._count_checkers('black')
        return white_count == self.CHECKERS_PER_PLAYER and black_count == self.CHECKERS_PER_PLAYER

    def calculate_score(self, winner_id: str) -> Dict[str, int]:
        """Calculate score based on win type and cube."""
        loser = 'black' if winner_id == 'white' else 'white'
        loser_home = self.game_state['home'].get(loser, 0)

        # Determine multiplier
        if loser_home == 0:
            loser_bar = self.game_state['bar'].get(loser, 0)
            if loser_bar > 0 or self._has_checker_in_home_board(loser, winner_id):
                multiplier = 3  # Backgammon
            else:
                multiplier = 2  # Gammon
        else:
            multiplier = 1  # Normal

        score = self.game_state.get('cube_value', 1) * multiplier
        return {winner_id: score, loser: 0}

    # Private helper methods

    def _get_bar_entry_actions(self, color: str) -> List[Dict[str, Any]]:
        """Get legal moves for entering from bar."""
        actions = []
        bar = self.WHITE_BAR if color == 'white' else self.BLACK_BAR

        for die in set(self.game_state.get('moves_remaining', [])):
            if color == 'white':
                entry_point = die
            else:
                entry_point = 25 - die

            if self._can_land_on(entry_point, color):
                actions.append({
                    'type': 'move',
                    'from': bar,
                    'to': entry_point,
                    'die_used': die
                })

        return actions

    def _get_player_points(self, color: str) -> List[int]:
        """Get all points where player has checkers."""
        points = []
        for point_str, count in self.game_state.get('points', {}).items():
            point = int(point_str)
            if color == 'white' and count > 0:
                points.append(point)
            elif color == 'black' and count < 0:
                points.append(point)
        return points

    def _calculate_destination(self, from_point: int, die: int, color: str) -> Optional[int]:
        """Calculate destination point for a move."""
        if color == 'white':
            to_point = from_point + die
            if to_point > 24:
                return self.WHITE_HOME
        else:
            to_point = from_point - die
            if to_point < 1:
                return self.BLACK_HOME
        return to_point

    def _can_land_on(self, point: int, color: str) -> bool:
        """Check if player can land on a point."""
        if point in (self.WHITE_HOME, self.BLACK_HOME):
            return True

        count = self.game_state.get('points', {}).get(str(point), 0)
        if color == 'white':
            return count >= -1
        return count <= 1

    def _can_bear_off(self, color: str) -> bool:
        """Check if player can bear off."""
        if self.game_state['bar'].get(color, 0) > 0:
            return False

        for point_str, count in self.game_state.get('points', {}).items():
            point = int(point_str)
            if color == 'white':
                if count > 0 and point < 19:
                    return False
            else:
                if count < 0 and point > 6:
                    return False

        return True

    def _is_valid_move(self, from_point: int, to_point: int, color: str, can_bear_off: bool) -> bool:
        """Validate a move."""
        if to_point in (self.WHITE_HOME, self.BLACK_HOME):
            return can_bear_off
        return self._can_land_on(to_point, color)

    def _execute_move(self, from_point: int, to_point: int, color: str) -> Dict[str, Any]:
        """Execute a move and update state."""
        points = self.game_state.get('points', {})
        direction = 1 if color == 'white' else -1

        # Handle bar entry
        if from_point in (self.WHITE_BAR, self.BLACK_BAR):
            self.game_state['bar'][color] -= 1
        else:
            # Remove checker from source
            from_key = str(from_point)
            current = points.get(from_key, 0)
            points[from_key] = current - direction
            if points[from_key] == 0:
                del points[from_key]

        # Handle bearing off
        if to_point in (self.WHITE_HOME, self.BLACK_HOME):
            self.game_state['home'][color] += 1
        else:
            to_key = str(to_point)
            current = points.get(to_key, 0)

            # Check for hitting opponent's blot
            if color == 'white' and current == -1:
                points[to_key] = 1
                self.game_state['bar']['black'] += 1
            elif color == 'black' and current == 1:
                points[to_key] = -1
                self.game_state['bar']['white'] += 1
            else:
                points[to_key] = current + direction

        self.game_state['points'] = points

        # Calculate and remove used die
        die_value = self._calculate_die_used(from_point, to_point, color)
        if die_value in self.game_state.get('moves_remaining', []):
            self.game_state['moves_remaining'].remove(die_value)

        return {'success': True, 'die_used': die_value}

    def _calculate_die_used(self, from_point: int, to_point: int, color: str) -> int:
        """Calculate which die value was used for a move."""
        if from_point == self.WHITE_BAR:
            return to_point
        elif from_point == self.BLACK_BAR:
            return 25 - to_point
        elif to_point == self.WHITE_HOME:
            return 25 - from_point
        elif to_point == self.BLACK_HOME:
            return from_point
        else:
            return abs(to_point - from_point)

    def _count_checkers(self, color: str) -> int:
        """Count total checkers for a player."""
        total = self.game_state['bar'].get(color, 0)
        total += self.game_state['home'].get(color, 0)

        for count in self.game_state.get('points', {}).values():
            if color == 'white' and count > 0:
                total += count
            elif color == 'black' and count < 0:
                total += abs(count)

        return total

    def _has_checker_in_home_board(self, loser: str, winner: str) -> bool:
        """Check if loser has checker in winner's home board."""
        for point_str, count in self.game_state.get('points', {}).items():
            point = int(point_str)
            if winner == 'white' and point >= 19 and count < 0:
                return True
            if winner == 'black' and point <= 6 and count > 0:
                return True
        return False

    def _offer_double(self, player_id: str) -> Dict[str, Any]:
        """Offer to double the stakes."""
        if self.game_state.get('double_offered', False):
            raise ValueError("A double has already been offered.")

        cube_owner = self.game_state.get('cube_owner', 'center')
        if cube_owner != 'center' and cube_owner != player_id:
            raise ValueError("You don't have the cube.")

        self.game_state['double_offered'] = True
        new_value = self.game_state.get('cube_value', 1) * 2

        return {'double_offered': True, 'cube_value': new_value}

    def _accept_double(self, player_id: str) -> Dict[str, Any]:
        """Accept the offered double."""
        if not self.game_state.get('double_offered', False):
            raise ValueError("No double has been offered.")

        current_turn = self.game_state.get('current_turn', 'white')
        if current_turn == player_id:
            raise ValueError("You cannot accept your own double.")

        self.game_state['cube_value'] = self.game_state.get('cube_value', 1) * 2
        self.game_state['cube_owner'] = player_id
        self.game_state['double_offered'] = False

        return {'accepted': True, 'cube_value': self.game_state['cube_value']}

    def _reject_double(self, player_id: str) -> Dict[str, Any]:
        """Reject the offered double (concede the game)."""
        if not self.game_state.get('double_offered', False):
            raise ValueError("No double has been offered.")

        current_turn = self.game_state.get('current_turn', 'white')
        if current_turn == player_id:
            raise ValueError("You cannot reject your own double.")

        opponent = 'white' if player_id == 'black' else 'black'
        return {'rejected': True, 'winner': opponent}

    def switch_turn(self) -> None:
        """Switch to the other player's turn."""
        current = self.game_state.get('current_turn', 'white')
        self.game_state['current_turn'] = 'black' if current == 'white' else 'white'
        self.game_state['dice'] = []
        self.game_state['moves_remaining'] = []
