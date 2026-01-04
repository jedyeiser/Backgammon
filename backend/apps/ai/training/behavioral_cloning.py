"""
Behavioral cloning trainer for player daemons.

This module provides the infrastructure to train neural networks
that mimic a specific player's behavior using supervised learning
on their historical game data.

The key difference from traditional game AI training:
- We're not trying to learn "optimal" play
- We're learning to predict what a specific human would do
- Accuracy is measured by matching the human's actual choices
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from django.contrib.auth import get_user_model
    User = get_user_model()

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for behavioral cloning training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    min_games: int = 10
    device: str = 'cpu'


@dataclass
class TrainingResult:
    """Results from training a behavioral cloning model."""
    epochs_trained: int
    final_loss: float
    final_accuracy: float
    best_accuracy: float
    validation_loss: float
    validation_accuracy: float
    total_samples: int
    training_samples: int
    validation_samples: int


class PlayerMoveDataset:
    """
    PyTorch Dataset for loading a player's historical moves.

    Extracts (state, action) pairs from the Move model for a specific user.
    The state is encoded using BackgammonEncoder, and the action is converted
    to target indices for the three prediction heads.

    Attributes:
        user: The user whose moves to load.
        encoder: Board state encoder.
        samples: List of (features, targets) tuples.
    """

    # Action type mapping (must match BehavioralPlayer)
    ACTION_TYPES = ['move', 'roll', 'double', 'accept_double', 'reject_double', 'resign']
    NUM_ACTION_TYPES = len(ACTION_TYPES)
    NUM_POINTS = 26

    def __init__(
        self,
        user: 'User',
        game_type: str = 'backgammon',
        min_games: int = 10,
    ):
        """
        Initialize the dataset.

        Args:
            user: The user whose moves to load.
            game_type: Game type to filter.
            min_games: Minimum games required for training.

        Raises:
            ValueError: If user has fewer than min_games completed games.
        """
        from ..encoders import BackgammonEncoder

        self.user = user
        self.game_type = game_type
        self.encoder = BackgammonEncoder()
        self.samples: List[Tuple['torch.Tensor', 'torch.Tensor']] = []

        self._load_samples(min_games)

    def _load_samples(self, min_games: int) -> None:
        """Load all samples from the user's game history."""
        import torch
        from django.db import models
        from apps.game.models import Move, Game

        # Get completed games where user participated
        games = Game.objects.filter(
            status=Game.Status.COMPLETED,
            game_type_id=self.game_type,
        ).filter(
            models.Q(white_player=self.user) | models.Q(black_player=self.user)
        ).prefetch_related('moves')

        if games.count() < min_games:
            raise ValueError(
                f"User {self.user.username} has only {games.count()} games, "
                f"need at least {min_games} for training"
            )

        logger.info(f"Loading moves from {games.count()} games for {self.user.username}")

        for game in games:
            player_color = game.get_player_color(self.user)
            if not player_color:
                continue

            # Get all moves by this user in this game
            moves = game.moves.filter(
                player=self.user,
                move_type=Move.MoveType.MOVE,  # Only train on actual moves, not rolls
            ).order_by('move_number')

            # For each move, we need the state BEFORE the move
            # and the action taken
            prev_state = game.get_initial_board()
            prev_move_num = 0

            all_moves = list(game.moves.order_by('move_number'))

            for move in all_moves:
                if move.player == self.user and move.move_type == Move.MoveType.MOVE:
                    # Encode the state before this move
                    features = self._encode_state(prev_state, player_color)

                    # Encode the action
                    targets = self._encode_action(move)

                    if features is not None and targets is not None:
                        self.samples.append((
                            torch.tensor(features, dtype=torch.float32),
                            torch.tensor(targets, dtype=torch.long),
                        ))

                # Update state for next iteration
                if move.board_state_after:
                    prev_state = move.board_state_after

        logger.info(f"Loaded {len(self.samples)} training samples")

    def _encode_state(
        self,
        board_state: Dict[str, Any],
        player_color: str,
    ) -> Optional[List[float]]:
        """Encode a board state to features."""
        try:
            # Add required fields if missing
            state = {
                'points': board_state.get('points', {}),
                'bar': board_state.get('bar', {'white': 0, 'black': 0}),
                'home': board_state.get('home', {'white': 0, 'black': 0}),
                'current_turn': player_color,
            }
            return self.encoder.encode(state, player_color)
        except Exception as e:
            logger.warning(f"Failed to encode state: {e}")
            return None

    def _encode_action(self, move: 'Move') -> Optional[List[int]]:
        """
        Encode an action to target indices.

        Returns:
            List of [action_type_idx, from_point_idx, to_point_idx]
        """
        try:
            # Action type
            move_type = move.move_type
            if move_type not in self.ACTION_TYPES:
                move_type = 'move'
            action_type_idx = self.ACTION_TYPES.index(move_type)

            # For move actions, extract from/to points
            from_idx = 0
            to_idx = 0

            if move.checker_moves and len(move.checker_moves) > 0:
                # Take the first checker move (most important)
                first_move = move.checker_moves[0]
                if isinstance(first_move, (list, tuple)) and len(first_move) >= 2:
                    from_point, to_point = first_move[0], first_move[1]
                    from_idx = self._point_to_index(from_point)
                    to_idx = self._point_to_index(to_point)
                elif isinstance(first_move, dict):
                    from_point = first_move.get('from', 0)
                    to_point = first_move.get('to', 0)
                    from_idx = self._point_to_index(from_point)
                    to_idx = self._point_to_index(to_point)

            return [action_type_idx, from_idx, to_idx]

        except Exception as e:
            logger.warning(f"Failed to encode action: {e}")
            return None

    def _point_to_index(self, point: int) -> int:
        """Convert a board point to network index."""
        if point == 0 or point == 25:  # Bar
            return 24
        elif point >= 26:  # Bear-off
            return 25
        elif 1 <= point <= 24:
            return point - 1
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple['torch.Tensor', 'torch.Tensor']:
        return self.samples[idx]


class BehavioralCloningTrainer:
    """
    Trainer for behavioral cloning models.

    Uses supervised learning to train a policy network that predicts
    the actions a specific human would take.

    The loss is a combination of cross-entropy losses for each head:
    - Action type prediction
    - From-point prediction (for moves)
    - To-point prediction (for moves)

    Example:
        trainer = BehavioralCloningTrainer(user, config)
        result = trainer.train()
        trainer.save_model()
    """

    def __init__(
        self,
        user: 'User',
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize the trainer.

        Args:
            user: The user whose play style to learn.
            config: Training configuration.
        """
        import torch

        self.user = user
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        self.network: Optional['nn.Module'] = None
        self.optimizer = None
        self.dataset: Optional[PlayerMoveDataset] = None
        self.train_loader = None
        self.val_loader = None

    def prepare(self) -> None:
        """Prepare the dataset, network, and data loaders."""
        import torch
        from torch.utils.data import DataLoader, random_split
        from ..networks import NetworkBuilder
        from ..players.behavioral import BehavioralPlayer

        logger.info(f"Preparing training for {self.user.username}")

        # Load dataset
        self.dataset = PlayerMoveDataset(
            self.user,
            min_games=self.config.min_games,
        )

        # Split into train/validation
        total = len(self.dataset)
        val_size = int(total * self.config.validation_split)
        train_size = total - val_size

        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Create network
        builder = NetworkBuilder()
        architecture = BehavioralPlayer.create_network_architecture()
        self.network = builder.from_json(architecture)
        self.network.to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        logger.info(
            f"Prepared: {train_size} training samples, {val_size} validation samples"
        )

    def train(self) -> TrainingResult:
        """
        Train the behavioral cloning model.

        Returns:
            TrainingResult with training statistics.
        """
        import torch
        import torch.nn as nn

        if self.network is None:
            self.prepare()

        logger.info(f"Starting training for {self.config.epochs} epochs")

        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        patience_counter = 0

        # Loss functions for each head
        action_type_loss = nn.CrossEntropyLoss()
        from_point_loss = nn.CrossEntropyLoss()
        to_point_loss = nn.CrossEntropyLoss()

        num_action_types = PlayerMoveDataset.NUM_ACTION_TYPES
        num_points = PlayerMoveDataset.NUM_POINTS

        for epoch in range(self.config.epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for features, targets in self.train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output = self.network(features)

                # Split outputs
                action_logits = output[:, :num_action_types]
                from_logits = output[:, num_action_types:num_action_types + num_points]
                to_logits = output[:, num_action_types + num_points:]

                # Split targets
                action_targets = targets[:, 0]
                from_targets = targets[:, 1]
                to_targets = targets[:, 2]

                # Compute losses
                loss1 = action_type_loss(action_logits, action_targets)
                loss2 = from_point_loss(from_logits, from_targets)
                loss3 = to_point_loss(to_logits, to_targets)

                # Combined loss (weighted)
                loss = loss1 + 0.5 * (loss2 + loss3)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * features.size(0)

                # Accuracy (based on action type)
                _, predicted = action_logits.max(1)
                train_correct += predicted.eq(action_targets).sum().item()
                train_total += features.size(0)

            train_loss /= train_total
            train_accuracy = train_correct / train_total

            # Validation phase
            val_loss, val_accuracy = self._validate(
                action_type_loss, from_point_loss, to_point_loss,
                num_action_types, num_points
            )

            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
                )

            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)

        return TrainingResult(
            epochs_trained=epoch + 1,
            final_loss=train_loss,
            final_accuracy=train_accuracy,
            best_accuracy=best_val_accuracy,
            validation_loss=best_val_loss,
            validation_accuracy=best_val_accuracy,
            total_samples=train_size + val_size,
            training_samples=train_size,
            validation_samples=val_size,
        )

    def _validate(
        self,
        action_type_loss,
        from_point_loss,
        to_point_loss,
        num_action_types: int,
        num_points: int,
    ) -> Tuple[float, float]:
        """Run validation and return loss and accuracy."""
        import torch

        self.network.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                output = self.network(features)

                action_logits = output[:, :num_action_types]
                from_logits = output[:, num_action_types:num_action_types + num_points]
                to_logits = output[:, num_action_types + num_points:]

                action_targets = targets[:, 0]
                from_targets = targets[:, 1]
                to_targets = targets[:, 2]

                loss1 = action_type_loss(action_logits, action_targets)
                loss2 = from_point_loss(from_logits, from_targets)
                loss3 = to_point_loss(to_logits, to_targets)
                loss = loss1 + 0.5 * (loss2 + loss3)

                val_loss += loss.item() * features.size(0)

                _, predicted = action_logits.max(1)
                val_correct += predicted.eq(action_targets).sum().item()
                val_total += features.size(0)

        return val_loss / val_total, val_correct / val_total

    def save_model(self) -> 'AIModel':
        """
        Save the trained model to the database.

        Returns:
            The created/updated AIModel instance.
        """
        from ..models import AIModel
        from ..networks import NetworkBuilder
        from ..players.behavioral import BehavioralPlayer

        if self.network is None:
            raise ValueError("No trained network to save")

        builder = NetworkBuilder()
        architecture = BehavioralPlayer.create_network_architecture()
        weights = builder.serialize_weights(self.network)

        # Get or create the model
        ai_model, created = AIModel.objects.update_or_create(
            owner=self.user,
            model_type='behavioral',
            game_type_id='backgammon',
            defaults={
                'name': f"{self.user.username}'s Daemon",
                'network_architecture': architecture,
                'network_weights': weights,
            }
        )

        action = "Created" if created else "Updated"
        logger.info(f"{action} behavioral model for {self.user.username}")

        return ai_model

    def evaluate_accuracy(
        self,
        test_games: Optional[List] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model accuracy on held-out games.

        Args:
            test_games: Optional list of Game objects to evaluate on.
                       If None, uses validation set.

        Returns:
            Dictionary with accuracy metrics.
        """
        import torch

        if self.network is None:
            raise ValueError("No trained network to evaluate")

        self.network.eval()

        # Use validation loader if no test games provided
        if test_games is None and self.val_loader is not None:
            correct = 0
            total = 0
            from_correct = 0
            to_correct = 0

            num_action_types = PlayerMoveDataset.NUM_ACTION_TYPES
            num_points = PlayerMoveDataset.NUM_POINTS

            with torch.no_grad():
                for features, targets in self.val_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)

                    output = self.network(features)

                    action_logits = output[:, :num_action_types]
                    from_logits = output[:, num_action_types:num_action_types + num_points]
                    to_logits = output[:, num_action_types + num_points:]

                    action_targets = targets[:, 0]
                    from_targets = targets[:, 1]
                    to_targets = targets[:, 2]

                    _, pred_action = action_logits.max(1)
                    _, pred_from = from_logits.max(1)
                    _, pred_to = to_logits.max(1)

                    correct += pred_action.eq(action_targets).sum().item()
                    from_correct += pred_from.eq(from_targets).sum().item()
                    to_correct += pred_to.eq(to_targets).sum().item()
                    total += features.size(0)

            return {
                'action_type_accuracy': correct / total,
                'from_point_accuracy': from_correct / total,
                'to_point_accuracy': to_correct / total,
                'total_samples': total,
            }

        return {'error': 'No validation data available'}


def train_player_daemon(
    user: 'User',
    epochs: int = 100,
    min_games: int = 10,
    **kwargs,
) -> TrainingResult:
    """
    High-level function to train a player daemon.

    Args:
        user: The user whose play style to learn.
        epochs: Number of training epochs.
        min_games: Minimum games required.
        **kwargs: Additional config overrides.

    Returns:
        TrainingResult with training statistics.
    """
    config = TrainingConfig(
        epochs=epochs,
        min_games=min_games,
        **kwargs,
    )

    trainer = BehavioralCloningTrainer(user, config)
    trainer.prepare()
    result = trainer.train()
    trainer.save_model()

    return result
