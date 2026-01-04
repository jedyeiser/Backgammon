"""
Neural network player implementation.

A player that uses a neural network to evaluate positions and select moves.
This is the foundation for TD-Gammon style learning and evolved networks.
"""
import copy
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BasePlayer

if TYPE_CHECKING:
    import torch.nn as nn
    from ..encoders.base import BaseEncoder
    from ..models import AIModel


class NeuralPlayer(BasePlayer):
    """
    A player that uses a neural network for position evaluation.

    Evaluates all legal moves by simulating them, encoding the
    resulting positions, and scoring them with the neural network.
    Selects the move leading to the best evaluated position.

    The network outputs a value in [0, 1] representing the probability
    of winning from that position (from the current player's perspective).

    Attributes:
        network: The PyTorch neural network.
        encoder: The board state encoder.
        temperature: Softmax temperature for move selection (0 = greedy).

    Example:
        from apps.ai.networks import NetworkBuilder, td_gammon_architecture
        from apps.ai.encoders import BackgammonEncoder

        # Create network
        builder = NetworkBuilder()
        network = builder.from_json(td_gammon_architecture())

        # Create player
        player = NeuralPlayer(
            player_id='neural_1',
            game_type='backgammon',
            network=network,
            encoder=BackgammonEncoder(),
        )

        # Or load from AIModel
        player = NeuralPlayer.from_ai_model(ai_model)
    """

    def __init__(
        self,
        player_id: str,
        game_type: str,
        network: 'nn.Module',
        encoder: 'BaseEncoder',
        name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize a neural network player.

        Args:
            player_id: Unique identifier for this player.
            game_type: Game type ('backgammon', 'catan', etc.).
            network: PyTorch neural network for position evaluation.
            encoder: Board state encoder matching the network's input.
            name: Optional display name.
            temperature: Softmax temperature for move selection.
                        0 = always pick best (greedy)
                        >0 = sample proportionally to exp(value/temp)
        """
        super().__init__(
            player_id=player_id,
            game_type=game_type,
            name=name or 'Neural Player',
        )
        self.network = network
        self.encoder = encoder
        self.temperature = temperature

        # Put network in eval mode by default
        self.network.eval()

    @classmethod
    def from_ai_model(
        cls,
        ai_model: 'AIModel',
        player_id: Optional[str] = None,
    ) -> 'NeuralPlayer':
        """
        Create a NeuralPlayer from a database AIModel.

        Args:
            ai_model: The AIModel instance with architecture and weights.
            player_id: Optional player ID (defaults to model UUID).

        Returns:
            A NeuralPlayer configured with the model's network.

        Raises:
            ValueError: If model has no architecture or weights.
        """
        from ..networks import NetworkBuilder
        from ..encoders import BackgammonEncoder

        if not ai_model.network_architecture:
            raise ValueError(f"AIModel {ai_model.id} has no network architecture")

        # Build network from architecture
        builder = NetworkBuilder()
        network = builder.from_json(ai_model.network_architecture)

        # Load weights if available
        if ai_model.network_weights:
            builder.deserialize_weights(ai_model.network_weights, network)

        # Get appropriate encoder for game type
        game_type = ai_model.game_type_id or 'backgammon'
        encoder = cls._get_encoder_for_game(game_type)

        return cls(
            player_id=player_id or str(ai_model.id),
            game_type=game_type,
            network=network,
            encoder=encoder,
            name=ai_model.name,
        )

    @staticmethod
    def _get_encoder_for_game(game_type: str) -> 'BaseEncoder':
        """Get the appropriate encoder for a game type."""
        if game_type == 'backgammon':
            from ..encoders import BackgammonEncoder
            return BackgammonEncoder()
        else:
            raise ValueError(f"No encoder available for game type: {game_type}")

    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select the best action using neural network evaluation.

        For each legal action:
        1. Simulate the action to get the resulting state
        2. Encode the state as features
        3. Evaluate with the neural network
        4. Select the action with the highest value

        Args:
            game_state: Current game state.
            legal_actions: List of legal actions.

        Returns:
            The action with the best neural network evaluation.

        Raises:
            ValueError: If legal_actions is empty.
        """
        import torch

        if not legal_actions:
            raise ValueError("Cannot select from empty action list")

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Get current player for encoding perspective
        player = self._get_current_player(game_state)

        # Evaluate each action
        values = []

        with torch.no_grad():
            for action in legal_actions:
                # Simulate the action
                new_state = self._simulate_action(game_state, action, player)

                # Encode from current player's perspective
                features = self.encoder.encode_tensor(new_state, player)

                # Add batch dimension
                features = features.unsqueeze(0)

                # Evaluate with network
                value = self.network(features).item()
                values.append(value)

        # Select action
        if self.temperature <= 0:
            # Greedy selection
            best_idx = max(range(len(values)), key=lambda i: values[i])
            return legal_actions[best_idx]
        else:
            # Softmax sampling with temperature
            import numpy as np
            values_array = np.array(values)
            exp_values = np.exp(values_array / self.temperature)
            probs = exp_values / exp_values.sum()
            idx = np.random.choice(len(legal_actions), p=probs)
            return legal_actions[idx]

    def evaluate_position(
        self,
        game_state: Dict[str, Any],
        player: Optional[str] = None,
    ) -> float:
        """
        Evaluate a position using the neural network.

        Args:
            game_state: Game state to evaluate.
            player: Player perspective (defaults to current turn).

        Returns:
            Value in [0, 1] representing win probability.
        """
        import torch

        if player is None:
            player = self._get_current_player(game_state)

        with torch.no_grad():
            features = self.encoder.encode_tensor(game_state, player)
            features = features.unsqueeze(0)
            value = self.network(features).item()

        return value

    def _get_current_player(self, game_state: Dict[str, Any]) -> str:
        """Get the current player from game state."""
        if 'current_turn' in game_state:
            return game_state['current_turn']
        elif 'current_player_index' in game_state:
            player_order = game_state.get('player_order', [])
            idx = game_state['current_player_index']
            if player_order and 0 <= idx < len(player_order):
                return player_order[idx]
        return 'white'

    def _simulate_action(
        self,
        game_state: Dict[str, Any],
        action: Dict[str, Any],
        player: str,
    ) -> Dict[str, Any]:
        """
        Simulate an action and return the resulting state.

        Args:
            game_state: Current game state.
            action: Action to simulate.
            player: Current player.

        Returns:
            New game state after the action.
        """
        new_state = copy.deepcopy(game_state)

        if self.game_type == 'backgammon':
            self._apply_backgammon_action(new_state, action, player)

        return new_state

    def _apply_backgammon_action(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        player: str,
    ) -> None:
        """Apply a backgammon action to a state (in place)."""
        action_type = action.get('type')

        if action_type != 'move':
            return

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

    def get_player_type(self) -> str:
        """Return 'neural' as the player type."""
        return 'neural'

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        config = super().get_config()
        config['temperature'] = self.temperature
        config['encoder'] = self.encoder.__class__.__name__
        return config

    def set_training_mode(self, training: bool = True) -> None:
        """
        Set the network to training or evaluation mode.

        Args:
            training: True for training mode, False for evaluation.
        """
        if training:
            self.network.train()
        else:
            self.network.eval()

    def get_network(self) -> 'nn.Module':
        """Return the underlying neural network."""
        return self.network

    def save_to_ai_model(self, ai_model: 'AIModel') -> None:
        """
        Save the network weights back to an AIModel.

        Args:
            ai_model: The AIModel to save to.
        """
        from ..networks import NetworkBuilder

        builder = NetworkBuilder()
        ai_model.network_weights = builder.serialize_weights(self.network)
        ai_model.save(update_fields=['network_weights', 'updated_at'])
