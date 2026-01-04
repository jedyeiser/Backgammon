"""
Self-play training for neural network game players.

Self-play is a powerful technique where an agent plays against
itself (or copies of itself) to generate training data. This
approach was used successfully by TD-Gammon and later by
AlphaGo/AlphaZero.

The key insight is that by playing against itself, the agent
can generate an unlimited amount of training data without
needing human games or hand-coded opponents.
"""
import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from ..encoders.base import BaseEncoder
    from ..players.neural import NeuralPlayer


@dataclass
class GameRecord:
    """Record of a single self-play game."""
    states: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[str] = None
    num_moves: int = 0
    final_state: Optional[Dict[str, Any]] = None


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    games_played: int = 0
    total_moves: int = 0
    wins_by_player: Dict[str, int] = field(default_factory=dict)
    avg_game_length: float = 0.0
    avg_td_error: float = 0.0
    win_rate_vs_random: float = 0.0
    evaluations: List[Dict[str, Any]] = field(default_factory=list)


class SelfPlayTrainer:
    """
    Train neural networks through self-play.

    Runs games where the network plays against itself (or a copy),
    using TD(λ) learning to update weights after each game.

    The training loop:
    1. Play a game (network vs network)
    2. Record states visited and final outcome
    3. Apply TD updates to network weights
    4. Periodically evaluate against baseline (random player)
    5. Save checkpoints

    Attributes:
        network: The neural network being trained.
        encoder: Board state encoder.
        td_learner: TD(λ) learner for weight updates.
        game_simulator: Function to simulate game actions.

    Example:
        trainer = SelfPlayTrainer(
            network=network,
            encoder=BackgammonEncoder(),
            game_type='backgammon',
            alpha=0.01,
            lambda_=0.7,
        )

        # Train for 10000 games, evaluate every 1000
        stats = trainer.train(
            num_games=10000,
            eval_interval=1000,
        )
    """

    def __init__(
        self,
        network: 'nn.Module',
        encoder: 'BaseEncoder',
        game_type: str = 'backgammon',
        alpha: float = 0.01,
        lambda_: float = 0.7,
        gamma: float = 1.0,
        temperature: float = 0.1,
        annealing_schedule: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the self-play trainer.

        Args:
            network: Neural network to train.
            encoder: Encoder for converting game states to tensors.
            game_type: Type of game ('backgammon', etc.).
            alpha: Learning rate for TD updates.
            lambda_: Eligibility trace decay.
            gamma: Discount factor.
            temperature: Softmax temperature for move selection.
                        Higher = more exploration, lower = more greedy.
            annealing_schedule: Optional learning rate schedule.
                               {'initial_alpha': 0.1, 'final_alpha': 0.001}
        """
        from .td_learner import TDLearner

        self.network = network
        self.encoder = encoder
        self.game_type = game_type
        self.temperature = temperature
        self.annealing_schedule = annealing_schedule

        self.td_learner = TDLearner(
            network=network,
            alpha=alpha,
            lambda_=lambda_,
            gamma=gamma,
        )

        # Game-specific functions
        self._setup_game_functions()

        # Statistics
        self.stats = TrainingStats()

    def _setup_game_functions(self) -> None:
        """Set up game-specific simulation functions."""
        if self.game_type == 'backgammon':
            self._get_initial_state = self._backgammon_initial_state
            self._get_legal_actions = self._backgammon_legal_actions
            self._apply_action = self._backgammon_apply_action
            self._is_terminal = self._backgammon_is_terminal
            self._get_winner = self._backgammon_get_winner
            self._roll_dice = self._backgammon_roll_dice
        else:
            raise ValueError(f"Unsupported game type: {self.game_type}")

    def train(
        self,
        num_games: int,
        eval_interval: int = 1000,
        eval_games: int = 100,
        checkpoint_interval: int = 5000,
        checkpoint_callback: Optional[Callable[['nn.Module', int], None]] = None,
        progress_callback: Optional[Callable[[int, TrainingStats], None]] = None,
    ) -> TrainingStats:
        """
        Run the self-play training loop.

        Args:
            num_games: Total number of games to play.
            eval_interval: Evaluate against random every N games.
            eval_games: Number of games per evaluation.
            checkpoint_interval: Save checkpoint every N games.
            checkpoint_callback: Called with (network, game_num) for saving.
            progress_callback: Called with (game_num, stats) for logging.

        Returns:
            Training statistics.
        """
        self.network.train()

        for game_num in range(1, num_games + 1):
            # Anneal learning rate if schedule provided
            if self.annealing_schedule:
                progress = game_num / num_games
                self.td_learner.anneal_learning_rate(
                    initial_alpha=self.annealing_schedule['initial_alpha'],
                    final_alpha=self.annealing_schedule['final_alpha'],
                    progress=progress,
                )

            # Play one game
            record = self._play_game()
            self.stats.games_played += 1
            self.stats.total_moves += record.num_moves

            # Track wins
            if record.winner:
                self.stats.wins_by_player[record.winner] = (
                    self.stats.wins_by_player.get(record.winner, 0) + 1
                )

            # Update average game length
            self.stats.avg_game_length = (
                self.stats.total_moves / self.stats.games_played
            )

            # Periodic evaluation
            if game_num % eval_interval == 0:
                win_rate = self.evaluate_vs_random(num_games=eval_games)
                self.stats.win_rate_vs_random = win_rate
                self.stats.evaluations.append({
                    'game': game_num,
                    'win_rate': win_rate,
                    'td_error': self.td_learner.get_stats()['avg_td_error'],
                })

                if progress_callback:
                    progress_callback(game_num, self.stats)

            # Checkpoint
            if checkpoint_callback and game_num % checkpoint_interval == 0:
                checkpoint_callback(self.network, game_num)

        return self.stats

    def _play_game(self) -> GameRecord:
        """
        Play a single self-play game.

        The network plays both sides, with TD updates applied
        at the end based on the outcome.

        Returns:
            Record of the game.
        """
        import torch

        record = GameRecord()

        # Initialize game
        state = self._get_initial_state()
        current_player = 'white'

        # Track states for each player (for TD updates)
        player_states = {'white': [], 'black': []}

        self.td_learner.new_episode()

        while not self._is_terminal(state):
            # Roll dice (for backgammon)
            dice = self._roll_dice()
            state['dice'] = dice
            state['moves_remaining'] = self._get_moves_remaining(dice)

            # Get legal actions
            legal_actions = self._get_legal_actions(state, current_player)

            if not legal_actions:
                # No legal moves, switch player
                current_player = 'black' if current_player == 'white' else 'white'
                continue

            # Select action using the network
            action = self._select_action(state, legal_actions, current_player)

            # Record state before action
            features = self.encoder.encode_tensor(state, current_player)
            player_states[current_player].append(features)

            # Apply action
            state = self._apply_action(state, action, current_player)
            record.states.append(copy.deepcopy(state))
            record.actions.append(action)
            record.num_moves += 1

            # Check if player gets another turn (doubles in backgammon)
            # For now, always switch players after move sequence
            if not state.get('moves_remaining'):
                current_player = 'black' if current_player == 'white' else 'white'

        # Game over - get winner and apply TD updates
        winner = self._get_winner(state)
        record.winner = winner
        record.final_state = state

        # TD update for each player
        for player, states in player_states.items():
            if states:
                final_value = 1.0 if player == winner else 0.0
                self.td_learner.train_from_trajectory(states, final_value)

        return record

    def _select_action(
        self,
        state: Dict[str, Any],
        legal_actions: List[Dict[str, Any]],
        player: str,
    ) -> Dict[str, Any]:
        """
        Select an action using the neural network.

        Uses softmax over position evaluations with temperature
        for exploration.
        """
        import torch
        import numpy as np

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Evaluate each action
        values = []

        with torch.no_grad():
            for action in legal_actions:
                # Simulate action
                new_state = self._simulate_action(state, action, player)
                features = self.encoder.encode_tensor(new_state, player)

                if features.dim() == 1:
                    features = features.unsqueeze(0)

                value = self.network(features).item()
                values.append(value)

        # Softmax selection with temperature
        if self.temperature > 0:
            values_array = np.array(values)
            exp_values = np.exp(values_array / self.temperature)
            probs = exp_values / exp_values.sum()
            idx = np.random.choice(len(legal_actions), p=probs)
        else:
            # Greedy
            idx = max(range(len(values)), key=lambda i: values[i])

        return legal_actions[idx]

    def _simulate_action(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        player: str,
    ) -> Dict[str, Any]:
        """Simulate an action without modifying original state."""
        new_state = copy.deepcopy(state)
        return self._apply_action(new_state, action, player)

    def evaluate_vs_random(
        self,
        num_games: int = 100,
    ) -> float:
        """
        Evaluate the network against a random player.

        Plays games with the network as both white and black
        to get an unbiased win rate estimate.

        Args:
            num_games: Number of games to play (half as each color).

        Returns:
            Win rate (0.0 to 1.0).
        """
        import torch

        self.network.eval()
        wins = 0
        games_per_side = num_games // 2

        # Play as white
        for _ in range(games_per_side):
            winner = self._play_vs_random(network_player='white')
            if winner == 'white':
                wins += 1

        # Play as black
        for _ in range(games_per_side):
            winner = self._play_vs_random(network_player='black')
            if winner == 'black':
                wins += 1

        self.network.train()
        return wins / num_games

    def _play_vs_random(self, network_player: str) -> str:
        """
        Play a game with network vs random player.

        Args:
            network_player: Which color the network plays ('white' or 'black').

        Returns:
            The winning player.
        """
        import torch
        import random as py_random

        state = self._get_initial_state()
        current_player = 'white'

        while not self._is_terminal(state):
            dice = self._roll_dice()
            state['dice'] = dice
            state['moves_remaining'] = self._get_moves_remaining(dice)

            legal_actions = self._get_legal_actions(state, current_player)

            if not legal_actions:
                current_player = 'black' if current_player == 'white' else 'white'
                continue

            # Select action based on player type
            if current_player == network_player:
                action = self._select_action(state, legal_actions, current_player)
            else:
                action = py_random.choice(legal_actions)

            state = self._apply_action(state, action, current_player)

            if not state.get('moves_remaining'):
                current_player = 'black' if current_player == 'white' else 'white'

        return self._get_winner(state)

    # =========================================
    # Backgammon-specific game functions
    # =========================================

    def _backgammon_initial_state(self) -> Dict[str, Any]:
        """Create initial backgammon state."""
        # Standard starting position
        points = {}

        # White (positive) starting positions
        points['24'] = 2   # 2 white on point 24
        points['13'] = 5   # 5 white on point 13
        points['8'] = 3    # 3 white on point 8
        points['6'] = 5    # 5 white on point 6

        # Black (negative) starting positions
        points['1'] = -2   # 2 black on point 1
        points['12'] = -5  # 5 black on point 12
        points['17'] = -3  # 3 black on point 17
        points['19'] = -5  # 5 black on point 19

        return {
            'points': points,
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
            'current_turn': 'white',
            'dice': [],
            'moves_remaining': [],
        }

    def _backgammon_roll_dice(self) -> List[int]:
        """Roll two dice."""
        import random
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        return [d1, d2]

    def _get_moves_remaining(self, dice: List[int]) -> List[int]:
        """Get moves remaining from dice roll."""
        if len(dice) == 2 and dice[0] == dice[1]:
            # Doubles - four moves of the same value
            return [dice[0]] * 4
        return list(dice)

    def _backgammon_legal_actions(
        self,
        state: Dict[str, Any],
        player: str,
    ) -> List[Dict[str, Any]]:
        """Get legal actions for backgammon."""
        from ..evaluation.backgammon import generate_all_moves

        return generate_all_moves(state, player)

    def _backgammon_apply_action(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        player: str,
    ) -> Dict[str, Any]:
        """Apply a backgammon action to the state."""
        # Similar to NeuralPlayer._apply_backgammon_action
        action_type = action.get('type')

        if action_type != 'move':
            return state

        from_point = action.get('from')
        to_point = action.get('to')
        die_used = action.get('die_used')

        points = state.get('points', {})
        bar = state.get('bar', {'white': 0, 'black': 0})
        home = state.get('home', {'white': 0, 'black': 0})

        is_white = player == 'white'
        sign = 1 if is_white else -1

        # Handle moving from bar
        if from_point == 0 and is_white:
            bar['white'] = max(0, bar['white'] - 1)
        elif from_point == 25 and not is_white:
            bar['black'] = max(0, bar['black'] - 1)
        else:
            from_key = str(from_point)
            current = points.get(from_key, 0)
            points[from_key] = current - sign

        # Handle bearing off
        if to_point == 26 and is_white:
            home['white'] = home.get('white', 0) + 1
        elif to_point == 27 and not is_white:
            home['black'] = home.get('black', 0) + 1
        else:
            to_key = str(to_point)
            current = points.get(to_key, 0)

            # Check for hitting opponent blot
            if is_white and current == -1:
                points[to_key] = 1
                bar['black'] = bar.get('black', 0) + 1
            elif not is_white and current == 1:
                points[to_key] = -1
                bar['white'] = bar.get('white', 0) + 1
            else:
                points[to_key] = current + sign

        # Update moves remaining
        moves_remaining = state.get('moves_remaining', [])
        if die_used in moves_remaining:
            moves_remaining.remove(die_used)

        state['moves_remaining'] = moves_remaining
        return state

    def _backgammon_is_terminal(self, state: Dict[str, Any]) -> bool:
        """Check if the game is over."""
        home = state.get('home', {'white': 0, 'black': 0})
        return home.get('white', 0) == 15 or home.get('black', 0) == 15

    def _backgammon_get_winner(self, state: Dict[str, Any]) -> str:
        """Get the winner of the game."""
        home = state.get('home', {'white': 0, 'black': 0})
        if home.get('white', 0) == 15:
            return 'white'
        elif home.get('black', 0) == 15:
            return 'black'
        return ''
