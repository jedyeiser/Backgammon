"""
Backgammon Game Engine.

Handles all game logic including:
- Dice rolling
- Move validation
- Board state management
- Win condition checking
"""
import random
from typing import List, Tuple, Optional

from ..models import Game, Move


class GameEngine:
    """
    Core game engine for backgammon.

    Manages board state and validates moves according to backgammon rules.
    """

    # Board constants
    TOTAL_POINTS = 24
    WHITE_BAR = 0
    BLACK_BAR = 25
    WHITE_HOME = 26
    BLACK_HOME = 27
    CHECKERS_PER_PLAYER = 15

    def __init__(self, game: Game):
        """Initialize engine with a game instance."""
        self.game = game
        self.board_state = game.board_state

    def roll_dice(self) -> dict:
        """Roll the dice and calculate available moves."""
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)

        # Doubles give four moves
        if die1 == die2:
            moves_remaining = [die1, die1, die1, die1]
        else:
            moves_remaining = [die1, die2]

        self.game.dice = [die1, die2]
        self.game.moves_remaining = moves_remaining
        self.game.save(update_fields=['dice', 'moves_remaining', 'version'])

        # Record the roll
        self._record_move(
            Move.MoveType.ROLL,
            dice_values=[die1, die2]
        )

        return {
            'dice': [die1, die2],
            'moves_remaining': moves_remaining,
            'legal_moves': self.get_legal_moves()
        }

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Calculate all legal moves for the current player.

        Returns list of (from_point, to_point) tuples.
        """
        legal_moves = []
        color = self.game.current_turn
        board = self.board_state

        # Get available dice values
        moves_remaining = self.game.moves_remaining
        if not moves_remaining:
            return []

        # Check if player has checkers on the bar
        bar_count = board['bar'].get(color, 0)
        if bar_count > 0:
            # Must enter from bar first
            return self._get_bar_entry_moves(color, moves_remaining)

        # Get all points with player's checkers
        player_points = self._get_player_points(color)

        # Check if player can bear off
        can_bear_off = self._can_bear_off(color)

        # Calculate moves for each unique die value
        unique_dice = set(moves_remaining)
        for die_value in unique_dice:
            for from_point in player_points:
                to_point = self._calculate_destination(from_point, die_value, color)

                if to_point is not None:
                    if self._is_valid_move(from_point, to_point, color, can_bear_off):
                        legal_moves.append((from_point, to_point))

        return legal_moves

    def _get_bar_entry_moves(self, color: str, dice: List[int]) -> List[Tuple[int, int]]:
        """Get legal moves for entering from the bar."""
        moves = []
        bar = self.WHITE_BAR if color == 'white' else self.BLACK_BAR

        for die in set(dice):
            if color == 'white':
                entry_point = die  # White enters on points 1-6
            else:
                entry_point = 25 - die  # Black enters on points 24-19

            if self._can_land_on(entry_point, color):
                moves.append((bar, entry_point))

        return moves

    def _get_player_points(self, color: str) -> List[int]:
        """Get all points where the player has checkers."""
        points = []
        board_points = self.board_state.get('points', {})

        for point_str, count in board_points.items():
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
                return self.WHITE_HOME  # Bearing off
        else:
            to_point = from_point - die
            if to_point < 1:
                return self.BLACK_HOME  # Bearing off

        return to_point

    def _can_land_on(self, point: int, color: str) -> bool:
        """Check if a player can land on a point."""
        if point in (self.WHITE_HOME, self.BLACK_HOME):
            return True

        board_points = self.board_state.get('points', {})
        count = board_points.get(str(point), 0)

        if color == 'white':
            # White can land if no more than 1 black checker
            return count >= -1
        else:
            # Black can land if no more than 1 white checker
            return count <= 1

    def _can_bear_off(self, color: str) -> bool:
        """Check if a player can bear off (all checkers in home board)."""
        board_points = self.board_state.get('points', {})
        bar_count = self.board_state['bar'].get(color, 0)

        if bar_count > 0:
            return False

        if color == 'white':
            # All white checkers must be on points 19-24
            for point_str, count in board_points.items():
                point = int(point_str)
                if count > 0 and point < 19:
                    return False
        else:
            # All black checkers must be on points 1-6
            for point_str, count in board_points.items():
                point = int(point_str)
                if count < 0 and point > 6:
                    return False

        return True

    def _is_valid_move(
        self,
        from_point: int,
        to_point: int,
        color: str,
        can_bear_off: bool
    ) -> bool:
        """Validate a single checker move."""
        # Bearing off validation
        if to_point in (self.WHITE_HOME, self.BLACK_HOME):
            if not can_bear_off:
                return False
            # Additional bearing off logic can be added here

        return self._can_land_on(to_point, color)

    def make_moves(
        self,
        player,
        checker_moves: List[List[int]]
    ) -> dict:
        """
        Execute a sequence of checker moves.

        Args:
            player: The user making the move
            checker_moves: List of [from_point, to_point] pairs
        """
        color = self.game.get_player_color(player)

        if color != self.game.current_turn:
            raise ValueError("It's not your turn.")

        # Validate and execute each move
        for from_point, to_point in checker_moves:
            self._execute_move(from_point, to_point, color)

        # Record the move
        self._record_move(
            Move.MoveType.MOVE,
            checker_moves=checker_moves
        )

        # Check for win
        winner = self._check_winner()
        if winner:
            win_type = self._determine_win_type(winner)
            self.game.complete_game(
                self.game.white_player if winner == 'white' else self.game.black_player,
                win_type
            )
            return {'winner': winner, 'win_type': win_type}

        # Switch turns if all moves used
        if not self.game.moves_remaining:
            self._switch_turn()

        return {'success': True}

    def _execute_move(self, from_point: int, to_point: int, color: str) -> None:
        """Execute a single move and update board state."""
        board_points = self.board_state.get('points', {})
        direction = 1 if color == 'white' else -1

        # Handle bar entry
        if from_point in (self.WHITE_BAR, self.BLACK_BAR):
            self.board_state['bar'][color] -= 1
        else:
            # Remove checker from source
            from_key = str(from_point)
            current = board_points.get(from_key, 0)
            board_points[from_key] = current - direction

            # Clean up empty points
            if board_points[from_key] == 0:
                del board_points[from_key]

        # Handle bearing off
        if to_point in (self.WHITE_HOME, self.BLACK_HOME):
            self.board_state['home'][color] += 1
        else:
            to_key = str(to_point)
            current = board_points.get(to_key, 0)

            # Check for hitting opponent's blot
            if color == 'white' and current == -1:
                board_points[to_key] = 1
                self.board_state['bar']['black'] += 1
            elif color == 'black' and current == 1:
                board_points[to_key] = -1
                self.board_state['bar']['white'] += 1
            else:
                board_points[to_key] = current + direction

        self.board_state['points'] = board_points

        # Calculate used die value and remove from moves_remaining
        die_value = self._calculate_die_used(from_point, to_point, color)
        if die_value in self.game.moves_remaining:
            self.game.moves_remaining.remove(die_value)

        # Save updated state
        self.game.board_state = self.board_state
        self.game.save(update_fields=['board_state', 'moves_remaining', 'version'])

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

    def _check_winner(self) -> Optional[str]:
        """Check if either player has won."""
        if self.board_state['home'].get('white', 0) == self.CHECKERS_PER_PLAYER:
            return 'white'
        if self.board_state['home'].get('black', 0) == self.CHECKERS_PER_PLAYER:
            return 'black'
        return None

    def _determine_win_type(self, winner: str) -> str:
        """Determine if the win is normal, gammon, or backgammon."""
        loser = 'black' if winner == 'white' else 'white'
        loser_home = self.board_state['home'].get(loser, 0)
        loser_bar = self.board_state['bar'].get(loser, 0)

        if loser_home == 0:
            # Check for backgammon (checker on bar or in winner's home board)
            if loser_bar > 0 or self._has_checker_in_home_board(loser, winner):
                return Game.WinType.BACKGAMMON
            return Game.WinType.GAMMON

        return Game.WinType.NORMAL

    def _has_checker_in_home_board(self, loser: str, winner: str) -> bool:
        """Check if loser has a checker in winner's home board."""
        board_points = self.board_state.get('points', {})

        for point_str, count in board_points.items():
            point = int(point_str)
            if winner == 'white':
                # White's home board is 19-24
                if point >= 19 and count < 0:
                    return True
            else:
                # Black's home board is 1-6
                if point <= 6 and count > 0:
                    return True

        return False

    def _switch_turn(self) -> None:
        """Switch to the other player's turn."""
        self.game.current_turn = 'black' if self.game.current_turn == 'white' else 'white'
        self.game.dice = []
        self.game.moves_remaining = []
        self.game.save(update_fields=['current_turn', 'dice', 'moves_remaining', 'version'])

    def offer_double(self, player) -> dict:
        """Offer to double the stakes."""
        color = self.game.get_player_color(player)

        if self.game.double_offered:
            raise ValueError("A double has already been offered.")

        if self.game.cube_owner != 'center' and self.game.cube_owner != color:
            raise ValueError("You don't have the cube.")

        self.game.double_offered = True
        self.game.save(update_fields=['double_offered', 'version'])

        self._record_move(Move.MoveType.DOUBLE)

        return {'double_offered': True, 'cube_value': self.game.cube_value * 2}

    def accept_double(self, player) -> dict:
        """Accept the offered double."""
        color = self.game.get_player_color(player)
        opponent_color = 'white' if color == 'black' else 'black'

        if not self.game.double_offered:
            raise ValueError("No double has been offered.")

        if self.game.current_turn == color:
            raise ValueError("You cannot accept your own double.")

        self.game.cube_value *= 2
        self.game.cube_owner = color
        self.game.double_offered = False
        self.game.save(update_fields=['cube_value', 'cube_owner', 'double_offered', 'version'])

        self._record_move(Move.MoveType.ACCEPT_DOUBLE)

        return {'accepted': True, 'cube_value': self.game.cube_value}

    def reject_double(self, player) -> dict:
        """Reject the offered double (concede the game)."""
        color = self.game.get_player_color(player)

        if not self.game.double_offered:
            raise ValueError("No double has been offered.")

        if self.game.current_turn == color:
            raise ValueError("You cannot reject your own double.")

        opponent_color = 'white' if color == 'black' else 'black'
        winner = self.game.white_player if opponent_color == 'white' else self.game.black_player

        self._record_move(Move.MoveType.REJECT_DOUBLE)

        self.game.complete_game(winner, Game.WinType.RESIGN)

        return {'rejected': True, 'winner': opponent_color}

    def resign(self, player) -> dict:
        """Resign from the game."""
        color = self.game.get_player_color(player)
        opponent_color = 'white' if color == 'black' else 'black'
        winner = self.game.white_player if opponent_color == 'white' else self.game.black_player

        self._record_move(Move.MoveType.RESIGN)

        self.game.complete_game(winner, Game.WinType.RESIGN)

        return {'resigned': True, 'winner': opponent_color}

    def _record_move(
        self,
        move_type: str,
        dice_values: List[int] = None,
        checker_moves: List[List[int]] = None
    ) -> Move:
        """Record a move in the database."""
        # Get the current player
        if self.game.current_turn == 'white':
            player = self.game.white_player
        else:
            player = self.game.black_player

        # Get the next move number
        last_move = self.game.moves.order_by('-move_number').first()
        move_number = (last_move.move_number + 1) if last_move else 1

        return Move.objects.create(
            game=self.game,
            player=player,
            move_number=move_number,
            move_type=move_type,
            dice_values=dice_values,
            checker_moves=checker_moves,
            board_state_after=self.board_state
        )
