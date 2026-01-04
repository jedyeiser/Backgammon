"""
Training infrastructure for game AI.

This module provides the complete training pipeline:
- TD(λ) learning for temporal difference updates
- Self-play game generation
- Checkpoint management for resumable training
- Training orchestration with logging

Example usage:
    from apps.ai.training import Trainer, TrainingConfig, quick_train

    # Quick training for experimentation
    result = quick_train(num_games=10000, verbose=True)

    # Full training with configuration
    config = TrainingConfig(
        num_games=100000,
        alpha=0.01,
        lambda_=0.7,
        eval_interval=1000,
        experiment_name='my_td_gammon',
    )
    trainer = Trainer(config)
    result = trainer.train()
"""
from .td_learner import TDLearner, TDLambdaWithTraces
from .self_play import SelfPlayTrainer, GameRecord, TrainingStats
from .checkpoints import CheckpointManager, TrainingLogger
from .trainer import Trainer, TrainingConfig, TrainingResult, quick_train

__all__ = [
    # TD Learning
    'TDLearner',
    'TDLambdaWithTraces',

    # Self-play
    'SelfPlayTrainer',
    'GameRecord',
    'TrainingStats',

    # Checkpoints
    'CheckpointManager',
    'TrainingLogger',

    # Orchestration
    'Trainer',
    'TrainingConfig',
    'TrainingResult',
    'quick_train',
]
