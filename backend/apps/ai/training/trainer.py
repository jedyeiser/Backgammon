"""
Main training orchestration module.

Provides high-level interfaces for training neural network
game players with various methods:
- Self-play with TD learning
- Evolution-based training
- Supervised learning from game records

This module ties together the TD learner, self-play engine,
and checkpoint management into a cohesive training workflow.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from ..encoders.base import BaseEncoder
    from ..models import AIModel


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    # Network settings
    architecture_name: str = 'td_gammon'
    hidden_size: int = 80

    # TD learning parameters
    alpha: float = 0.01
    lambda_: float = 0.7
    gamma: float = 1.0

    # Learning rate annealing
    initial_alpha: float = 0.1
    final_alpha: float = 0.001
    anneal: bool = True

    # Training duration
    num_games: int = 100000
    eval_interval: int = 1000
    eval_games: int = 100
    checkpoint_interval: int = 5000

    # Exploration
    initial_temperature: float = 0.1
    final_temperature: float = 0.0
    temperature_anneal: bool = True

    # Paths
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    experiment_name: str = 'training'

    # Game settings
    game_type: str = 'backgammon'


@dataclass
class TrainingResult:
    """Results from a training run."""
    games_played: int = 0
    final_win_rate: float = 0.0
    best_win_rate: float = 0.0
    best_game: int = 0
    training_time_seconds: float = 0.0
    final_td_error: float = 0.0
    checkpoint_path: Optional[str] = None
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)


class Trainer:
    """
    High-level training interface for game AI.

    Combines network creation, TD learning, self-play, checkpointing,
    and logging into a simple training workflow.

    Example:
        config = TrainingConfig(
            num_games=50000,
            alpha=0.01,
            eval_interval=1000,
        )

        trainer = Trainer(config)
        result = trainer.train()

        print(f"Final win rate: {result.final_win_rate:.2%}")
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.network = None
        self.encoder = None
        self.self_play_trainer = None
        self.checkpoint_manager = None
        self.logger = None

    def train(
        self,
        resume_from: Optional[str] = None,
        progress_callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> TrainingResult:
        """
        Run the full training loop.

        Args:
            resume_from: Path to checkpoint to resume from.
            progress_callback: Called with (game_num, stats) during training.

        Returns:
            Training results.
        """
        import time

        # Initialize components
        self._setup()

        start_game = 0
        if resume_from:
            start_game = self._resume(resume_from)

        start_time = time.time()
        result = TrainingResult()

        # Run self-play training
        def checkpoint_callback(network, game_num):
            stats = self.self_play_trainer.stats
            path = self.checkpoint_manager.save(
                network=network,
                td_learner=self.self_play_trainer.td_learner,
                game_num=game_num,
                stats={
                    'win_rate': stats.win_rate_vs_random,
                    'avg_td_error': self.self_play_trainer.td_learner.get_stats()['avg_td_error'],
                    'avg_game_length': stats.avg_game_length,
                },
            )
            result.checkpoint_path = str(path)

        def inner_progress_callback(game_num, stats):
            # Log metrics
            self.logger.log(
                step=game_num,
                metrics={
                    'win_rate': stats.win_rate_vs_random,
                    'avg_game_length': stats.avg_game_length,
                    'games_played': stats.games_played,
                },
            )

            # Update result tracking
            if stats.win_rate_vs_random > result.best_win_rate:
                result.best_win_rate = stats.win_rate_vs_random
                result.best_game = game_num

            result.evaluation_history.append({
                'game': game_num,
                'win_rate': stats.win_rate_vs_random,
            })

            # Call user callback
            if progress_callback:
                progress_callback(game_num, {
                    'win_rate': stats.win_rate_vs_random,
                    'games_played': stats.games_played,
                    'avg_game_length': stats.avg_game_length,
                })

        # Run training
        stats = self.self_play_trainer.train(
            num_games=self.config.num_games - start_game,
            eval_interval=self.config.eval_interval,
            eval_games=self.config.eval_games,
            checkpoint_interval=self.config.checkpoint_interval,
            checkpoint_callback=checkpoint_callback,
            progress_callback=inner_progress_callback,
        )

        end_time = time.time()

        # Populate result
        result.games_played = stats.games_played + start_game
        result.final_win_rate = stats.win_rate_vs_random
        result.training_time_seconds = end_time - start_time
        result.final_td_error = self.self_play_trainer.td_learner.get_stats()['avg_td_error']

        # Final checkpoint
        final_path = self.checkpoint_manager.save(
            network=self.network,
            td_learner=self.self_play_trainer.td_learner,
            game_num=result.games_played,
            stats={
                'win_rate': result.final_win_rate,
                'best_win_rate': result.best_win_rate,
            },
            metadata={'final': True},
        )
        result.checkpoint_path = str(final_path)

        return result

    def _setup(self) -> None:
        """Set up training components."""
        from ..networks import NetworkBuilder, td_gammon_architecture, modern_backgammon_architecture
        from ..encoders import BackgammonEncoder
        from .self_play import SelfPlayTrainer
        from .checkpoints import CheckpointManager, TrainingLogger

        # Create network
        builder = NetworkBuilder()

        if self.config.architecture_name == 'td_gammon':
            arch = td_gammon_architecture(hidden_size=self.config.hidden_size)
        elif self.config.architecture_name == 'modern':
            arch = modern_backgammon_architecture()
        else:
            arch = td_gammon_architecture()

        self.network = builder.from_json(arch)

        # Create encoder
        if self.config.game_type == 'backgammon':
            self.encoder = BackgammonEncoder()
        else:
            raise ValueError(f"Unsupported game type: {self.config.game_type}")

        # Create annealing schedule
        annealing_schedule = None
        if self.config.anneal:
            annealing_schedule = {
                'initial_alpha': self.config.initial_alpha,
                'final_alpha': self.config.final_alpha,
            }

        # Create self-play trainer
        self.self_play_trainer = SelfPlayTrainer(
            network=self.network,
            encoder=self.encoder,
            game_type=self.config.game_type,
            alpha=self.config.alpha,
            lambda_=self.config.lambda_,
            gamma=self.config.gamma,
            temperature=self.config.initial_temperature,
            annealing_schedule=annealing_schedule,
        )

        # Create checkpoint manager
        checkpoint_dir = Path(self.config.checkpoint_dir) / self.config.experiment_name
        self.checkpoint_manager = CheckpointManager(str(checkpoint_dir))

        # Create logger
        self.logger = TrainingLogger(
            log_dir=self.config.log_dir,
            experiment_name=self.config.experiment_name,
        )

    def _resume(self, checkpoint_path: str) -> int:
        """Resume training from a checkpoint."""
        checkpoint = self.checkpoint_manager.load(checkpoint_path)
        game_num = self.checkpoint_manager.restore(
            checkpoint=checkpoint,
            network=self.network,
            td_learner=self.self_play_trainer.td_learner,
        )

        # Also restore stats
        if 'stats' in checkpoint:
            self.self_play_trainer.stats.win_rate_vs_random = checkpoint['stats'].get('win_rate', 0.0)

        self.logger.load()  # Load existing log entries

        return game_num

    def save_to_ai_model(
        self,
        ai_model: 'AIModel',
        save_weights: bool = True,
    ) -> None:
        """
        Save the trained network to an AIModel.

        Args:
            ai_model: AIModel instance to save to.
            save_weights: Whether to save network weights.
        """
        from ..networks import NetworkBuilder

        if self.network is None:
            raise ValueError("No network to save. Run training first.")

        builder = NetworkBuilder()

        # Save architecture
        ai_model.network_architecture = builder.to_json(self.network)

        # Save weights
        if save_weights:
            ai_model.network_weights = builder.serialize_weights(self.network)

        ai_model.save()

    @classmethod
    def from_ai_model(
        cls,
        ai_model: 'AIModel',
        config: Optional[TrainingConfig] = None,
    ) -> 'Trainer':
        """
        Create a trainer from an existing AIModel.

        Args:
            ai_model: AIModel with network architecture.
            config: Optional training configuration.

        Returns:
            Trainer initialized with the model.
        """
        from ..networks import NetworkBuilder

        config = config or TrainingConfig()
        trainer = cls(config)

        # Load network from model
        builder = NetworkBuilder()
        trainer.network = builder.from_json(ai_model.network_architecture)

        if ai_model.network_weights:
            builder.deserialize_weights(ai_model.network_weights, trainer.network)

        return trainer


def quick_train(
    num_games: int = 10000,
    eval_interval: int = 500,
    hidden_size: int = 80,
    alpha: float = 0.01,
    experiment_name: str = 'quick_train',
    verbose: bool = True,
) -> TrainingResult:
    """
    Quick training utility for experimentation.

    Creates a default configuration and runs training with
    progress output to console.

    Args:
        num_games: Number of games to train.
        eval_interval: Evaluation interval.
        hidden_size: Hidden layer size.
        alpha: Learning rate.
        experiment_name: Name for checkpoints/logs.
        verbose: Print progress to console.

    Returns:
        Training results.
    """
    config = TrainingConfig(
        num_games=num_games,
        eval_interval=eval_interval,
        hidden_size=hidden_size,
        alpha=alpha,
        experiment_name=experiment_name,
    )

    trainer = Trainer(config)

    def progress_callback(game_num, stats):
        if verbose:
            print(
                f"Game {game_num:6d} | "
                f"Win rate: {stats['win_rate']:.2%} | "
                f"Avg length: {stats['avg_game_length']:.1f}"
            )

    result = trainer.train(progress_callback=progress_callback)

    if verbose:
        print("\n=== Training Complete ===")
        print(f"Games played: {result.games_played}")
        print(f"Final win rate: {result.final_win_rate:.2%}")
        print(f"Best win rate: {result.best_win_rate:.2%} (game {result.best_game})")
        print(f"Training time: {result.training_time_seconds/60:.1f} minutes")
        print(f"Checkpoint: {result.checkpoint_path}")

    return result
