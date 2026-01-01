"""
Game Engine - Facade for game rule implementations.

This module provides a unified interface for game operations, delegating
game-specific logic to the appropriate RuleSet implementation while
handling persistence, move recording, and player management.

The engine acts as a facade that:
- Loads the appropriate RuleSet based on game type
- Delegates game logic (moves, validation, win detection) to the ruleset
- Handles database persistence and optimistic locking
- Records move history for replay and analysis
"""
from typing import List, Optional, Dict, Any

from ..models import Game, Move
from ..rulesets import RuleSetRegistry, BaseRuleSet


class GameEngine:
    """
    Facade for game operations.

    Provides a unified API for all game types while delegating
    game-specific logic to the appropriate RuleSet implementation.

    Handles:
    - Loading and initializing the correct ruleset
    - Persisting state changes to the database
    - Recording moves for history and replay
    - Turn management and player validation
    """

    def __init__(self, game: Game):
        """
        Initialize engine with a game instance.

        Args:
            game: The Game model instance to operate on.
        """
        self.game = game
        self.board_state = game.board_state

        # Load the appropriate ruleset
        ruleset_class = RuleSetRegistry.get(game.game_type_id)
        if ruleset_class is None:
            raise ValueError(f"Unknown game type: {game.game_type_id}")

        self.ruleset: BaseRuleSet = ruleset_class(self.board_state)

    def roll_dice(self) -> Dict[str, Any]:
        """
        Roll dice and calculate available moves.

        Returns:
            Dictionary containing dice values, moves remaining, and legal moves.
        """
        result = self.ruleset.roll_dice()

        # Update game state from ruleset
        self.game.dice = result.get('dice', [])
        self.game.moves_remaining = result.get('moves_remaining', [])
        self._sync_board_state()
        self.game.save(update_fields=['dice', 'moves_remaining', 'board_state', 'version'])

        # Record the roll
        self._record_move(
            Move.MoveType.ROLL,
            dice_values=result.get('dice')
        )

        # Add legal moves to result
        result['legal_moves'] = self.get_legal_moves()
        return result

    def get_legal_moves(self) -> List[Dict[str, Any]]:
        """
        Get all legal moves for the current player.

        Returns:
            List of legal action dictionaries.
        """
        current_player = self.game.current_turn
        return self.ruleset.get_legal_actions(current_player)

    def make_moves(
        self,
        player,
        checker_moves: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Execute a sequence of checker moves.

        Args:
            player: The user making the move.
            checker_moves: List of [from_point, to_point] pairs.

        Returns:
            Dictionary with result info, or winner info if game ended.

        Raises:
            ValueError: If it's not the player's turn.
        """
        color = self.game.get_player_color(player)

        if color != self.game.current_turn:
            raise ValueError("It's not your turn.")

        # Execute each move through the ruleset
        for from_point, to_point in checker_moves:
            die_used = self._calculate_die_used(from_point, to_point, color)
            action = {
                'type': 'move',
                'from': from_point,
                'to': to_point,
                'die_used': die_used
            }
            self.ruleset.apply_action(color, action)

            # Remove used die from moves remaining
            if die_used in self.game.moves_remaining:
                self.game.moves_remaining.remove(die_used)

        # Sync state and save
        self._sync_board_state()
        self.game.save(update_fields=['board_state', 'moves_remaining', 'version'])

        # Record the move
        self._record_move(
            Move.MoveType.MOVE,
            checker_moves=checker_moves
        )

        # Check for winner
        winner = self.ruleset.check_winner()
        if winner:
            win_type = self._determine_win_type(winner)
            winner_user = self.game.white_player if winner == 'white' else self.game.black_player
            self.game.complete_game(winner_user, win_type)
            return {'winner': winner, 'win_type': win_type}

        # Switch turns if all moves used
        if not self.game.moves_remaining:
            self._switch_turn()

        return {'success': True}

    def _calculate_die_used(self, from_point: int, to_point: int, color: str) -> int:
        """Calculate which die value was used for a move."""
        # Bar constants for backgammon
        WHITE_BAR = 0
        BLACK_BAR = 25
        WHITE_HOME = 26
        BLACK_HOME = 27

        if from_point == WHITE_BAR:
            return to_point
        elif from_point == BLACK_BAR:
            return 25 - to_point
        elif to_point == WHITE_HOME:
            return 25 - from_point
        elif to_point == BLACK_HOME:
            return from_point
        else:
            return abs(to_point - from_point)

    def _determine_win_type(self, winner: str) -> str:
        """Determine if the win is normal, gammon, or backgammon."""
        # Delegate to ruleset if it has scoring info
        if hasattr(self.ruleset, 'calculate_score'):
            scores = self.ruleset.calculate_score(winner)
            score = scores.get(winner, 1)
            if score >= 3:
                return Game.WinType.BACKGAMMON
            elif score >= 2:
                return Game.WinType.GAMMON
        return Game.WinType.NORMAL

    def _switch_turn(self) -> None:
        """Switch to the other player's turn."""
        self.game.current_turn = 'black' if self.game.current_turn == 'white' else 'white'
        self.game.dice = []
        self.game.moves_remaining = []

        # Also update ruleset state
        if hasattr(self.ruleset, 'switch_turn'):
            self.ruleset.switch_turn()

        self.game.save(update_fields=['current_turn', 'dice', 'moves_remaining', 'version'])

    def _sync_board_state(self) -> None:
        """Sync board state from ruleset to game model."""
        self.board_state = self.ruleset.game_state
        self.game.board_state = self.board_state

    def offer_double(self, player) -> Dict[str, Any]:
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

    def accept_double(self, player) -> Dict[str, Any]:
        """Accept the offered double."""
        color = self.game.get_player_color(player)

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

    def reject_double(self, player) -> Dict[str, Any]:
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

    def resign(self, player) -> Dict[str, Any]:
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
        """
        Record a move in the database.

        Args:
            move_type: The type of move being recorded.
            dice_values: Dice values for roll moves.
            checker_moves: List of from/to pairs for checker moves.

        Returns:
            The created Move instance.
        """
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

    # Game-type specific action dispatcher
    def apply_action(self, player, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a game action through the ruleset.

        This is a generic action dispatcher that routes actions to
        the appropriate handler based on action type.

        Args:
            player: The user taking the action.
            action: Action dictionary with at least a 'type' key.

        Returns:
            Result dictionary from the action.
        """
        action_type = action.get('type')

        if action_type == 'roll':
            return self.roll_dice()
        elif action_type == 'move':
            moves = action.get('moves', [[action.get('from'), action.get('to')]])
            return self.make_moves(player, moves)
        elif action_type == 'double':
            return self.offer_double(player)
        elif action_type == 'accept_double':
            return self.accept_double(player)
        elif action_type == 'reject_double':
            return self.reject_double(player)
        elif action_type == 'resign':
            return self.resign(player)
        else:
            # Delegate unknown actions to ruleset
            color = self.game.get_player_color(player)
            result = self.ruleset.apply_action(color, action)
            self._sync_board_state()
            self.game.save(update_fields=['board_state', 'version'])
            return result
