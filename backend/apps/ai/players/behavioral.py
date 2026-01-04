"""
Behavioral cloning player that mimics a human player's style.

This player learns to play like a specific human by training on their
historical game data. It uses supervised learning (behavioral cloning)
to predict the actions a human would take in any given position.

The key insight is that instead of learning "optimal play", we learn
"how player X would play" - capturing their personal style, preferences,
and even their mistakes.
"""
import copy
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BasePlayer

if TYPE_CHECKING:
    import torch.nn as nn
    from ..encoders.base import BaseEncoder
    from django.contrib.auth import get_user_model
    User = get_user_model()


class BehavioralPlayer(BasePlayer):
    """
    A player that mimics a specific human player's behavior.

    Uses a neural network trained via behavioral cloning on the human's
    game history. The network learns to predict the actions the human
    would take given any board position.

    Unlike value-based players (like NeuralPlayer which evaluates positions),
    this player directly predicts actions - making it a policy network.

    Architecture:
        The network has multiple output heads:
        - Move type head: Predicts action type (move, double, etc.)
        - From-point head: Predicts source point for moves
        - To-point head: Predicts destination point for moves

    Attributes:
        network: Policy network for action prediction.
        encoder: Board state encoder.
        user_id: ID of the human player being mimicked.
        temperature: Sampling temperature for action selection.

    Example:
        # Create from a trained model
        player = BehavioralPlayer.from_user(user, player_id='daemon_1')

        # Use in a game
        action = player.select_action(game_state, legal_actions)
    """

    # Action type mapping
    ACTION_TYPES = ['move', 'roll', 'double', 'accept_double', 'reject_double', 'resign']
    NUM_ACTION_TYPES = len(ACTION_TYPES)

    # Point indices: 0-23 (board), 24 (bar), 25 (bear-off)
    NUM_POINTS = 26

    def __init__(
        self,
        player_id: str,
        game_type: str,
        network: 'nn.Module',
        encoder: 'BaseEncoder',
        user_id: Optional[int] = None,
        name: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize a behavioral cloning player.

        Args:
            player_id: Unique identifier for this player.
            game_type: Game type (e.g., 'backgammon').
            network: Trained policy network.
            encoder: Board state encoder.
            user_id: ID of the human player being mimicked.
            name: Display name (defaults to "Daemon of {username}").
            temperature: Sampling temperature (0 = greedy, >0 = stochastic).
        """
        super().__init__(
            player_id=player_id,
            game_type=game_type,
            name=name or 'Behavioral Player',
        )
        self.network = network
        self.encoder = encoder
        self.user_id = user_id
        self.temperature = temperature

        # Put network in eval mode
        self.network.eval()

    @classmethod
    def from_user(
        cls,
        user: 'User',
        player_id: Optional[str] = None,
        temperature: float = 0.1,
    ) -> 'BehavioralPlayer':
        """
        Create a BehavioralPlayer from a user's trained daemon model.

        Loads the daemon model associated with this user, or raises
        an error if no daemon has been trained.

        Args:
            user: The user whose play style to mimic.
            player_id: Optional player ID.
            temperature: Sampling temperature.

        Returns:
            A BehavioralPlayer configured to play like the user.

        Raises:
            ValueError: If no daemon model exists for this user.
        """
        from ..models import AIModel
        from ..networks import NetworkBuilder
        from ..encoders import BackgammonEncoder

        # Look for the user's daemon model
        try:
            ai_model = AIModel.objects.get(
                owner=user,
                model_type='behavioral',
                game_type_id='backgammon',
            )
        except AIModel.DoesNotExist:
            raise ValueError(f"No behavioral daemon trained for user {user.username}")

        if not ai_model.network_architecture:
            raise ValueError(f"Daemon model for {user.username} has no architecture")

        # Build network
        builder = NetworkBuilder()
        network = builder.from_json(ai_model.network_architecture)

        # Load weights
        if ai_model.network_weights:
            builder.deserialize_weights(ai_model.network_weights, network)

        return cls(
            player_id=player_id or f'daemon_{user.id}',
            game_type='backgammon',
            network=network,
            encoder=BackgammonEncoder(),
            user_id=user.id,
            name=f"Daemon of {user.username}",
            temperature=temperature,
        )

    @classmethod
    def create_network_architecture(cls, input_size: int = 198) -> Dict[str, Any]:
        """
        Create the architecture for a behavioral cloning network.

        The architecture uses a shared backbone with three output heads:
        - Action type (6 classes)
        - From point (26 options)
        - To point (26 options)

        Args:
            input_size: Input feature dimension.

        Returns:
            Network architecture specification.
        """
        return {
            'name': 'BehavioralCloning',
            'input_size': input_size,
            'output_size': cls.NUM_ACTION_TYPES + 2 * cls.NUM_POINTS,  # 58 total
            'layers': [
                # Shared backbone
                {'id': 'fc1', 'type': 'linear', 'in': input_size, 'out': 256},
                {'id': 'act1', 'type': 'activation', 'fn': 'relu'},
                {'id': 'drop1', 'type': 'dropout', 'p': 0.2},
                {'id': 'fc2', 'type': 'linear', 'in': 256, 'out': 128},
                {'id': 'act2', 'type': 'activation', 'fn': 'relu'},
                {'id': 'drop2', 'type': 'dropout', 'p': 0.2},
                {'id': 'fc3', 'type': 'linear', 'in': 128, 'out': 64},
                {'id': 'act3', 'type': 'activation', 'fn': 'relu'},
                # Combined output (we'll split in forward pass interpretation)
                {'id': 'output', 'type': 'linear', 'in': 64, 'out': cls.NUM_ACTION_TYPES + 2 * cls.NUM_POINTS},
            ],
        }

    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select an action by predicting what the human would do.

        Uses the policy network to score each legal action based on
        how likely the mimicked human would choose it, then selects
        based on temperature-controlled sampling.

        Args:
            game_state: Current game state.
            legal_actions: List of legal actions.

        Returns:
            The predicted action the human would take.

        Raises:
            ValueError: If legal_actions is empty.
        """
        import torch
        import numpy as np

        if not legal_actions:
            raise ValueError("Cannot select from empty action list")

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Get current player perspective
        player = self._get_current_player(game_state)

        # Encode the current state
        with torch.no_grad():
            features = self.encoder.encode_tensor(game_state, player)
            features = features.unsqueeze(0)  # Add batch dimension

            # Get network prediction
            output = self.network(features)

            # Split output into heads
            action_type_logits = output[0, :self.NUM_ACTION_TYPES]
            from_point_logits = output[0, self.NUM_ACTION_TYPES:self.NUM_ACTION_TYPES + self.NUM_POINTS]
            to_point_logits = output[0, self.NUM_ACTION_TYPES + self.NUM_POINTS:]

        # Score each legal action
        scores = []
        for action in legal_actions:
            score = self._score_action(
                action,
                action_type_logits,
                from_point_logits,
                to_point_logits,
            )
            scores.append(score)

        scores = np.array(scores)

        # Temperature-controlled selection
        if self.temperature <= 0:
            # Greedy
            best_idx = np.argmax(scores)
        else:
            # Softmax sampling
            exp_scores = np.exp((scores - scores.max()) / self.temperature)
            probs = exp_scores / exp_scores.sum()
            best_idx = np.random.choice(len(legal_actions), p=probs)

        return legal_actions[best_idx]

    def _score_action(
        self,
        action: Dict[str, Any],
        action_type_logits: 'torch.Tensor',
        from_point_logits: 'torch.Tensor',
        to_point_logits: 'torch.Tensor',
    ) -> float:
        """
        Score an action based on network predictions.

        Combines the log-probabilities from each head to get
        an overall score for the action.

        Args:
            action: The action to score.
            action_type_logits: Logits for action type prediction.
            from_point_logits: Logits for from-point prediction.
            to_point_logits: Logits for to-point prediction.

        Returns:
            Combined score (log-probability) for the action.
        """
        import torch.nn.functional as F

        score = 0.0

        # Score action type
        action_type = action.get('type', 'move')
        if action_type in self.ACTION_TYPES:
            type_idx = self.ACTION_TYPES.index(action_type)
            type_probs = F.log_softmax(action_type_logits, dim=0)
            score += type_probs[type_idx].item()

        # Score from point (for move actions)
        if action_type == 'move':
            from_point = action.get('from', 0)
            from_idx = self._point_to_index(from_point)
            from_probs = F.log_softmax(from_point_logits, dim=0)
            score += from_probs[from_idx].item()

            # Score to point
            to_point = action.get('to', 0)
            to_idx = self._point_to_index(to_point)
            to_probs = F.log_softmax(to_point_logits, dim=0)
            score += to_probs[to_idx].item()

        return score

    def _point_to_index(self, point: int) -> int:
        """Convert a board point to network index."""
        # Points 1-24 -> indices 0-23
        # Bar (0 for white, 25 for black) -> index 24
        # Bear-off (26 for white, 27 for black) -> index 25
        if point == 0 or point == 25:  # Bar
            return 24
        elif point >= 26:  # Bear-off
            return 25
        else:
            return point - 1

    def _get_current_player(self, game_state: Dict[str, Any]) -> str:
        """Get the current player from game state."""
        return game_state.get('current_turn', 'white')

    def get_action_distribution(
        self,
        game_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get the full action distribution for analysis.

        Useful for understanding how the daemon "thinks" and
        for evaluating prediction accuracy.

        Args:
            game_state: Current game state.

        Returns:
            Dictionary with probability distributions for each head.
        """
        import torch
        import torch.nn.functional as F

        player = self._get_current_player(game_state)

        with torch.no_grad():
            features = self.encoder.encode_tensor(game_state, player)
            features = features.unsqueeze(0)
            output = self.network(features)

            action_type_logits = output[0, :self.NUM_ACTION_TYPES]
            from_point_logits = output[0, self.NUM_ACTION_TYPES:self.NUM_ACTION_TYPES + self.NUM_POINTS]
            to_point_logits = output[0, self.NUM_ACTION_TYPES + self.NUM_POINTS:]

            action_type_probs = F.softmax(action_type_logits, dim=0).cpu().numpy()
            from_point_probs = F.softmax(from_point_logits, dim=0).cpu().numpy()
            to_point_probs = F.softmax(to_point_logits, dim=0).cpu().numpy()

        return {
            'action_types': dict(zip(self.ACTION_TYPES, action_type_probs.tolist())),
            'from_points': from_point_probs.tolist(),
            'to_points': to_point_probs.tolist(),
        }

    def get_player_type(self) -> str:
        """Return 'behavioral' as the player type."""
        return 'behavioral'

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        config = super().get_config()
        config['user_id'] = self.user_id
        config['temperature'] = self.temperature
        return config

    def set_temperature(self, temperature: float) -> None:
        """
        Set the sampling temperature.

        Args:
            temperature: New temperature value.
                        0 = greedy (always pick most likely)
                        Higher = more random/exploratory
        """
        self.temperature = temperature

    def save_to_ai_model(self, ai_model: 'AIModel') -> None:
        """
        Save the network weights to an AIModel.

        Args:
            ai_model: The AIModel to save to.
        """
        from ..networks import NetworkBuilder

        builder = NetworkBuilder()
        ai_model.network_weights = builder.serialize_weights(self.network)
        ai_model.save(update_fields=['network_weights', 'updated_at'])
