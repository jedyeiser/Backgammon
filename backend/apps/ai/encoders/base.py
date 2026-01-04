"""
Base encoder abstraction for game state encoding.

Encoders transform game-specific board states into fixed-size
numerical feature vectors suitable for neural network input.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BaseEncoder(ABC):
    """
    Abstract base class for game state encoders.

    Encoders transform game states into fixed-size feature vectors
    for neural network input. Each game type has its own encoder
    that captures the relevant features of that game.

    Attributes:
        input_size: The size of the output feature vector.
        game_type: The game type this encoder is designed for.

    Example:
        encoder = BackgammonEncoder()
        features = encoder.encode(game_state)  # Returns numpy array
        tensor = encoder.encode_tensor(game_state)  # Returns PyTorch tensor
    """

    input_size: int = 0
    game_type: str = ''

    @abstractmethod
    def encode(self, game_state: Dict[str, Any], player: str = 'white') -> 'np.ndarray':
        """
        Encode a game state into a feature vector.

        Args:
            game_state: The game state dictionary.
            player: The player's perspective ('white' or 'black').
                   Features are encoded from this player's viewpoint.

        Returns:
            A numpy array of shape (input_size,) with float values.
        """
        pass

    def encode_tensor(self, game_state: Dict[str, Any], player: str = 'white') -> 'torch.Tensor':
        """
        Encode a game state into a PyTorch tensor.

        Args:
            game_state: The game state dictionary.
            player: The player's perspective.

        Returns:
            A PyTorch tensor of shape (input_size,).

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for encode_tensor()")

        features = self.encode(game_state, player)
        return torch.from_numpy(features).float()

    def encode_batch(
        self,
        game_states: List[Dict[str, Any]],
        player: str = 'white',
    ) -> 'np.ndarray':
        """
        Encode multiple game states into a batch.

        Args:
            game_states: List of game state dictionaries.
            player: The player's perspective.

        Returns:
            A numpy array of shape (batch_size, input_size).
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for encode_batch()")

        features = [self.encode(state, player) for state in game_states]
        return np.stack(features)

    def encode_batch_tensor(
        self,
        game_states: List[Dict[str, Any]],
        player: str = 'white',
    ) -> 'torch.Tensor':
        """
        Encode multiple game states into a batch tensor.

        Args:
            game_states: List of game state dictionaries.
            player: The player's perspective.

        Returns:
            A PyTorch tensor of shape (batch_size, input_size).
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for encode_batch_tensor()")

        batch = self.encode_batch(game_states, player)
        return torch.from_numpy(batch).float()

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return human-readable names for each feature.

        Useful for debugging and understanding what the network sees.

        Returns:
            List of feature names, length equal to input_size.
        """
        pass

    def describe(self) -> Dict[str, Any]:
        """
        Return a description of this encoder.

        Returns:
            Dictionary with encoder metadata.
        """
        return {
            'game_type': self.game_type,
            'input_size': self.input_size,
            'feature_names': self.get_feature_names(),
        }
