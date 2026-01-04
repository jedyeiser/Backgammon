"""
Match runner for playing games between players.

Orchestrates games between any combination of human and AI players,
supporting single games, matches, and tournaments.

Features:
- Play any player type against any other
- Record game history and statistics
- Update ELO ratings after matches
- Support for multiple game types
"""
import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .elo import EloCalculator, EloResult

if TYPE_CHECKING:
    from ..players.base import BasePlayer


@dataclass
class GameResult:
    """Result of a single game."""
    winner: Optional[str] = None  # Player ID or None for draw
    loser: Optional[str] = None
    is_draw: bool = False
    num_moves: int = 0
    final_state: Optional[Dict[str, Any]] = None
    player_colors: Dict[str, str] = field(default_factory=dict)
    game_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MatchResult:
    """Result of a match (multiple games)."""
    player_a_id: str
    player_b_id: str
    player_a_wins: int = 0
    player_b_wins: int = 0
    draws: int = 0
    games: List[GameResult] = field(default_factory=list)
    player_a_rating_change: int = 0
    player_b_rating_change: int = 0

    @property
    def total_games(self) -> int:
        return self.player_a_wins + self.player_b_wins + self.draws

    @property
    def player_a_score(self) -> float:
        """Score for player A (1 per win, 0.5 per draw)."""
        return self.player_a_wins + 0.5 * self.draws

    @property
    def player_b_score(self) -> float:
        """Score for player B."""
        return self.player_b_wins + 0.5 * self.draws


class MatchRunner:
    """
    Run matches between players.

    Handles the complete game flow:
    1. Initialize game state
    2. Alternate turns between players
    3. Apply actions and check for winner
    4. Record results and update ratings

    Supports any game type with a corresponding RuleSet.

    Example:
        runner = MatchRunner(game_type='backgammon')

        # Create players
        player_a = NeuralPlayer.from_ai_model(model_a)
        player_b = RandomPlayer('random', 'backgammon')

        # Run a match
        result = runner.run_match(player_a, player_b, num_games=10)

        print(f"Player A wins: {result.player_a_wins}")
        print(f"Player B wins: {result.player_b_wins}")
    """

    def __init__(
        self,
        game_type: str = 'backgammon',
        update_ratings: bool = True,
        record_history: bool = False,
        max_moves_per_game: int = 1000,
    ):
        """
        Initialize the match runner.

        Args:
            game_type: Type of game to run.
            update_ratings: Whether to update ELO after games.
            record_history: Whether to record full game history.
            max_moves_per_game: Maximum moves before declaring draw.
        """
        self.game_type = game_type
        self.update_ratings = update_ratings
        self.record_history = record_history
        self.max_moves_per_game = max_moves_per_game

        self.elo_calculator = EloCalculator()
        self._ruleset_class = self._get_ruleset_class()

    def _get_ruleset_class(self):
        """Get the ruleset class for the game type."""
        if self.game_type == 'backgammon':
            from apps.game.rulesets.backgammon import BackgammonRuleSet
            return BackgammonRuleSet
        elif self.game_type == 'catan':
            from apps.game.rulesets.catan import CatanRuleSet
            return CatanRuleSet
        else:
            raise ValueError(f"Unknown game type: {self.game_type}")

    def run_game(
        self,
        player_a: 'BasePlayer',
        player_b: 'BasePlayer',
        swap_colors: bool = False,
    ) -> GameResult:
        """
        Run a single game between two players.

        Args:
            player_a: First player.
            player_b: Second player.
            swap_colors: If True, swap who plays white.

        Returns:
            GameResult with winner, moves, etc.
        """
        result = GameResult()

        # Assign colors
        if swap_colors:
            white_player, black_player = player_b, player_a
        else:
            white_player, black_player = player_a, player_b

        result.player_colors = {
            white_player.player_id: 'white',
            black_player.player_id: 'black',
        }

        # Initialize game
        ruleset = self._ruleset_class({})
        state = ruleset.get_initial_state()

        # Set up players in state
        state['player_order'] = [white_player.player_id, black_player.player_id]
        state['current_player_index'] = 0

        players = {
            'white': white_player,
            'black': black_player,
        }

        # Game loop
        move_count = 0
        game_history = []

        while move_count < self.max_moves_per_game:
            # Get current player
            current_color = state.get('current_turn', 'white')
            current_player = players[current_color]

            # Get legal actions
            ruleset = self._ruleset_class(state)

            if self.game_type == 'backgammon':
                # Roll dice if needed
                if not state.get('dice') or not state.get('moves_remaining'):
                    dice = [random.randint(1, 6), random.randint(1, 6)]
                    state['dice'] = dice
                    if dice[0] == dice[1]:
                        state['moves_remaining'] = [dice[0]] * 4
                    else:
                        state['moves_remaining'] = list(dice)

            legal_actions = ruleset.get_legal_actions(current_player.player_id)

            if not legal_actions:
                # No legal moves - switch turn or end game
                if self.game_type == 'backgammon':
                    state['moves_remaining'] = []
                    state['current_turn'] = 'black' if current_color == 'white' else 'white'
                    continue
                break

            # Player selects action
            action = current_player.select_action(state, legal_actions)

            if self.record_history:
                game_history.append({
                    'player': current_player.player_id,
                    'color': current_color,
                    'action': action,
                    'state': copy.deepcopy(state),
                })

            # Apply action
            ruleset.apply_action(current_player.player_id, action)
            state = ruleset.game_state
            move_count += 1

            # Check for winner
            winner_id = ruleset.check_winner()
            if winner_id:
                winner_color = current_color
                loser_color = 'black' if winner_color == 'white' else 'white'

                result.winner = players[winner_color].player_id
                result.loser = players[loser_color].player_id
                result.num_moves = move_count
                result.final_state = state
                result.game_history = game_history
                return result

            # Switch turn if moves exhausted
            if self.game_type == 'backgammon' and not state.get('moves_remaining'):
                state['current_turn'] = 'black' if current_color == 'white' else 'white'

        # Game reached move limit - declare draw
        result.is_draw = True
        result.num_moves = move_count
        result.final_state = state
        result.game_history = game_history
        return result

    def run_match(
        self,
        player_a: 'BasePlayer',
        player_b: 'BasePlayer',
        num_games: int = 1,
        alternate_colors: bool = True,
        rating_a: Optional[int] = None,
        rating_b: Optional[int] = None,
        games_played_a: int = 100,
        games_played_b: int = 100,
    ) -> MatchResult:
        """
        Run a match of multiple games.

        Args:
            player_a: First player.
            player_b: Second player.
            num_games: Number of games in the match.
            alternate_colors: If True, alternate who plays white.
            rating_a: Player A's ELO rating.
            rating_b: Player B's ELO rating.
            games_played_a: Prior games for A (for K-factor).
            games_played_b: Prior games for B (for K-factor).

        Returns:
            MatchResult with scores and rating changes.
        """
        result = MatchResult(
            player_a_id=player_a.player_id,
            player_b_id=player_b.player_id,
        )

        for i in range(num_games):
            swap = alternate_colors and (i % 2 == 1)
            game_result = self.run_game(player_a, player_b, swap_colors=swap)
            result.games.append(game_result)

            if game_result.is_draw:
                result.draws += 1
            elif game_result.winner == player_a.player_id:
                result.player_a_wins += 1
            else:
                result.player_b_wins += 1

        # Update ratings if requested
        if self.update_ratings and rating_a is not None and rating_b is not None:
            new_a, new_b = self.elo_calculator.update_from_match(
                rating_a=rating_a,
                rating_b=rating_b,
                wins_a=result.player_a_wins,
                wins_b=result.player_b_wins,
                draws=result.draws,
                games_a=games_played_a,
                games_b=games_played_b,
            )
            result.player_a_rating_change = new_a - rating_a
            result.player_b_rating_change = new_b - rating_b

        return result

    def evaluate_player(
        self,
        player: 'BasePlayer',
        opponent: 'BasePlayer',
        num_games: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate a player's performance against an opponent.

        Plays games as both colors and reports statistics.

        Args:
            player: Player to evaluate.
            opponent: Opponent to play against.
            num_games: Number of games (half as each color).

        Returns:
            Dictionary with evaluation statistics.
        """
        result = self.run_match(
            player_a=player,
            player_b=opponent,
            num_games=num_games,
            alternate_colors=True,
        )

        total = result.total_games
        wins = result.player_a_wins
        losses = result.player_b_wins
        draws = result.draws

        return {
            'total_games': total,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / total if total > 0 else 0.0,
            'loss_rate': losses / total if total > 0 else 0.0,
            'draw_rate': draws / total if total > 0 else 0.0,
            'score': (wins + 0.5 * draws) / total if total > 0 else 0.0,
        }


def run_benchmark(
    player: 'BasePlayer',
    game_type: str = 'backgammon',
    num_games: int = 100,
) -> Dict[str, float]:
    """
    Quick benchmark against random player.

    Args:
        player: Player to benchmark.
        game_type: Game type.
        num_games: Number of games.

    Returns:
        Benchmark statistics.
    """
    from ..players.random_player import RandomPlayer

    random_opponent = RandomPlayer(
        player_id='random_benchmark',
        game_type=game_type,
    )

    runner = MatchRunner(game_type=game_type, update_ratings=False)
    return runner.evaluate_player(player, random_opponent, num_games)
