"""
Checkpoint management for training sessions.

Save and load training state including:
- Network weights
- Optimizer state
- Training statistics
- Learning rate schedule progress

Checkpoints enable:
- Resume training after interruption
- Compare networks at different training stages
- Track training progress over time
"""
import gzip
import io
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from .td_learner import TDLearner


class CheckpointManager:
    """
    Manage training checkpoints for neural networks.

    Saves and loads complete training state including network
    weights, optimizer state, and training statistics.

    Attributes:
        checkpoint_dir: Directory for storing checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep.

    Example:
        manager = CheckpointManager('./checkpoints/td_gammon')

        # Save during training
        manager.save(
            network=network,
            td_learner=learner,
            game_num=5000,
            stats={'win_rate': 0.55},
        )

        # Resume training later
        state = manager.load_latest()
        manager.restore(
            checkpoint=state,
            network=network,
            td_learner=learner,
        )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 10,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            max_checkpoints: Maximum checkpoints to keep (0 = unlimited).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        network: 'nn.Module',
        td_learner: Optional['TDLearner'] = None,
        game_num: int = 0,
        stats: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            network: The neural network.
            td_learner: Optional TD learner with training state.
            game_num: Current game number in training.
            stats: Optional training statistics.
            metadata: Optional additional metadata.

        Returns:
            Path to the saved checkpoint file.
        """
        import torch

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'checkpoint_{game_num:08d}_{timestamp}.pt'
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'game_num': game_num,
            'timestamp': timestamp,
            'network_state_dict': network.state_dict(),
        }

        # Save optimizer state if available
        if td_learner is not None:
            checkpoint['optimizer_state_dict'] = td_learner.optimizer.state_dict()
            checkpoint['td_learner_stats'] = td_learner.get_stats()
            checkpoint['alpha'] = td_learner.alpha
            checkpoint['lambda'] = td_learner.lambda_

        # Save stats and metadata
        if stats:
            checkpoint['stats'] = stats
        if metadata:
            checkpoint['metadata'] = metadata

        # Save network architecture if available
        if hasattr(network, 'architecture'):
            checkpoint['architecture'] = network.architecture

        torch.save(checkpoint, filepath)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return filepath

    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Load a specific checkpoint.

        Args:
            filepath: Path to checkpoint file.

        Returns:
            Checkpoint dictionary.
        """
        import torch
        return torch.load(filepath, weights_only=False)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
            Checkpoint dictionary or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return self.load(checkpoints[-1])

    def load_best(self, metric: str = 'win_rate') -> Optional[Dict[str, Any]]:
        """
        Load the checkpoint with the best metric value.

        Args:
            metric: The metric key in stats to use for comparison.

        Returns:
            Best checkpoint or None.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        best_checkpoint = None
        best_value = float('-inf')

        for cp_path in checkpoints:
            cp = self.load(cp_path)
            stats = cp.get('stats', {})
            value = stats.get(metric, float('-inf'))

            if value > best_value:
                best_value = value
                best_checkpoint = cp

        return best_checkpoint

    def restore(
        self,
        checkpoint: Dict[str, Any],
        network: 'nn.Module',
        td_learner: Optional['TDLearner'] = None,
    ) -> int:
        """
        Restore training state from a checkpoint.

        Args:
            checkpoint: Checkpoint dictionary from load().
            network: Network to restore weights into.
            td_learner: Optional TD learner to restore optimizer state.

        Returns:
            The game number at the checkpoint.
        """
        # Restore network weights
        network.load_state_dict(checkpoint['network_state_dict'])

        # Restore optimizer state
        if td_learner is not None and 'optimizer_state_dict' in checkpoint:
            td_learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore TD learner settings
            if 'alpha' in checkpoint:
                td_learner.set_learning_rate(checkpoint['alpha'])
            if 'lambda' in checkpoint:
                td_learner.lambda_ = checkpoint['lambda']

            # Restore statistics
            if 'td_learner_stats' in checkpoint:
                stats = checkpoint['td_learner_stats']
                td_learner.episodes_trained = stats.get('episodes_trained', 0)

        return checkpoint.get('game_num', 0)

    def list_checkpoints(self) -> list[Path]:
        """
        List all checkpoint files sorted by game number.

        Returns:
            List of checkpoint file paths in order.
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pt'))
        # Sort by game number (extracted from filename)
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))
        return checkpoints

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if over the limit."""
        if self.max_checkpoints <= 0:
            return

        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()

    def get_checkpoint_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get metadata about a checkpoint without loading weights.

        Args:
            filepath: Path to checkpoint file.

        Returns:
            Dictionary with checkpoint metadata.
        """
        import torch

        # Load only the non-tensor parts
        cp = torch.load(filepath, weights_only=False)

        return {
            'game_num': cp.get('game_num', 0),
            'timestamp': cp.get('timestamp', ''),
            'stats': cp.get('stats', {}),
            'metadata': cp.get('metadata', {}),
            'has_optimizer': 'optimizer_state_dict' in cp,
            'has_architecture': 'architecture' in cp,
        }


class TrainingLogger:
    """
    Log training progress for visualization and analysis.

    Records metrics at regular intervals for later plotting
    and analysis of training dynamics.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = 'training',
    ):
        """
        Initialize the training logger.

        Args:
            log_dir: Directory for log files.
            experiment_name: Name of the experiment.
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f'{experiment_name}.jsonl'
        self.entries: list[Dict[str, Any]] = []

    def log(
        self,
        step: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metrics for a training step.

        Args:
            step: Training step (game number, epoch, etc.).
            metrics: Dictionary of metric names to values.
            metadata: Optional additional metadata.
        """
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }
        if metadata:
            entry['metadata'] = metadata

        self.entries.append(entry)

        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_metric_history(self, metric: str) -> tuple[list[int], list[float]]:
        """
        Get the history of a specific metric.

        Args:
            metric: Name of the metric.

        Returns:
            Tuple of (steps, values).
        """
        steps = []
        values = []

        for entry in self.entries:
            if metric in entry.get('metrics', {}):
                steps.append(entry['step'])
                values.append(entry['metrics'][metric])

        return steps, values

    def load(self) -> None:
        """Load entries from the log file."""
        if not self.log_file.exists():
            return

        self.entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the training run."""
        if not self.entries:
            return {}

        first_entry = self.entries[0]
        last_entry = self.entries[-1]

        # Collect all metrics
        all_metrics = set()
        for entry in self.entries:
            all_metrics.update(entry.get('metrics', {}).keys())

        # Compute final values and trends
        summary = {
            'experiment_name': self.experiment_name,
            'total_steps': last_entry['step'],
            'num_log_entries': len(self.entries),
            'start_time': first_entry['timestamp'],
            'end_time': last_entry['timestamp'],
            'final_metrics': last_entry.get('metrics', {}),
        }

        return summary
