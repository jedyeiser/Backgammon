"""
Backgammon board state encoder.

Implements the classic TD-Gammon 198-feature encoding:
- 24 points × 2 players × 4 features = 192 features
- Plus: bar (2), borne off (2), turn indicator (2) = 6 features
- Total: 198 features

Reference: Tesauro, G. (1995). Temporal difference learning and TD-Gammon.
"""
from typing import Any, Dict, List

try:
    import numpy as np
except ImportError:
    raise ImportError("NumPy is required for BackgammonEncoder")

from .base import BaseEncoder


class BackgammonEncoder(BaseEncoder):
    """
    TD-Gammon style encoder for backgammon positions.

    Encodes the board state into a 198-dimensional feature vector.
    Features are normalized to [0, 1] range for neural network input.

    Feature Layout:
        - Features 0-95: White's checkers on points 1-24 (4 features each)
        - Features 96-191: Black's checkers on points 1-24 (4 features each)
        - Feature 192: White's bar count (normalized)
        - Feature 193: Black's bar count (normalized)
        - Feature 194: White's borne off (normalized)
        - Feature 195: Black's borne off (normalized)
        - Feature 196: Turn indicator (1 if encoding player's turn)
        - Feature 197: Turn indicator (1 if opponent's turn)

    Per-Point Features (4 per point per player):
        - f0: 1 if at least 1 checker on point
        - f1: 1 if at least 2 checkers on point
        - f2: 1 if at least 3 checkers on point
        - f3: (count - 3) / 2 if count > 3, else 0

    Example:
        encoder = BackgammonEncoder()
        features = encoder.encode(game_state, player='white')
        # features is a numpy array of shape (198,)
    """

    input_size = 198
    game_type = 'backgammon'

    # Constants
    NUM_POINTS = 24
    FEATURES_PER_POINT = 4
    CHECKERS_PER_PLAYER = 15

    def encode(self, game_state: Dict[str, Any], player: str = 'white') -> np.ndarray:
        """
        Encode a backgammon position into 198 features.

        The encoding is always from the specified player's perspective,
        making it suitable for neural network evaluation.

        Args:
            game_state: Backgammon game state dictionary with keys:
                       - points: {str: int} point positions
                       - bar: {white: int, black: int}
                       - home: {white: int, black: int}
                       - current_turn: 'white' or 'black'
            player: The player's perspective for encoding.

        Returns:
            Numpy array of shape (198,) with float values in [0, 1].
        """
        features = np.zeros(self.input_size, dtype=np.float32)

        points = game_state.get('points', {})
        bar = game_state.get('bar', {'white': 0, 'black': 0})
        home = game_state.get('home', {'white': 0, 'black': 0})
        current_turn = game_state.get('current_turn', 'white')

        opponent = 'black' if player == 'white' else 'white'

        # Encode points for both players
        for point_num in range(1, self.NUM_POINTS + 1):
            count = points.get(str(point_num), 0)

            # White checkers are positive, black are negative
            white_count = max(0, count)
            black_count = max(0, -count)

            # Determine which count belongs to player vs opponent
            if player == 'white':
                player_count = white_count
                opponent_count = black_count
            else:
                player_count = black_count
                opponent_count = white_count

            # Encode player's checkers (first 96 features)
            base_idx = (point_num - 1) * self.FEATURES_PER_POINT
            self._encode_point_features(features, base_idx, player_count)

            # Encode opponent's checkers (features 96-191)
            opp_base_idx = 96 + (point_num - 1) * self.FEATURES_PER_POINT
            self._encode_point_features(features, opp_base_idx, opponent_count)

        # Bar counts (normalized by max possible = 15)
        features[192] = bar.get(player, 0) / self.CHECKERS_PER_PLAYER
        features[193] = bar.get(opponent, 0) / self.CHECKERS_PER_PLAYER

        # Borne off counts (normalized by 15)
        features[194] = home.get(player, 0) / self.CHECKERS_PER_PLAYER
        features[195] = home.get(opponent, 0) / self.CHECKERS_PER_PLAYER

        # Turn indicator
        features[196] = 1.0 if current_turn == player else 0.0
        features[197] = 1.0 if current_turn == opponent else 0.0

        return features

    def _encode_point_features(
        self,
        features: np.ndarray,
        base_idx: int,
        count: int,
    ) -> None:
        """
        Encode the 4 features for a single point.

        The encoding captures:
        - Whether the point is occupied
        - Whether it's a "made point" (2+ checkers, safe from being hit)
        - Whether it has extra builders
        - Scaled count for stacks > 3

        Args:
            features: The feature array to modify in place.
            base_idx: Starting index for this point's features.
            count: Number of checkers on this point.
        """
        if count >= 1:
            features[base_idx] = 1.0
        if count >= 2:
            features[base_idx + 1] = 1.0
        if count >= 3:
            features[base_idx + 2] = 1.0
        if count > 3:
            # Normalize extra checkers: (count - 3) / 2
            # This caps at ~6 for maximum stack of 15
            features[base_idx + 3] = (count - 3) / 2.0

    def decode_point_features(self, features: np.ndarray, point_num: int, is_player: bool) -> int:
        """
        Decode features back to approximate checker count.

        Useful for debugging and visualization.

        Args:
            features: The full 198-feature array.
            point_num: Point number (1-24).
            is_player: True for player's features, False for opponent's.

        Returns:
            Approximate checker count on that point.
        """
        if is_player:
            base_idx = (point_num - 1) * self.FEATURES_PER_POINT
        else:
            base_idx = 96 + (point_num - 1) * self.FEATURES_PER_POINT

        # Reconstruct count from features
        if features[base_idx] < 0.5:
            return 0
        if features[base_idx + 1] < 0.5:
            return 1
        if features[base_idx + 2] < 0.5:
            return 2

        # At least 3, check for extras
        extra = features[base_idx + 3] * 2.0
        return 3 + int(round(extra))

    def get_feature_names(self) -> List[str]:
        """
        Return human-readable names for all 198 features.

        Returns:
            List of 198 feature names.
        """
        names = []

        # Player's point features
        for point in range(1, 25):
            names.append(f"player_p{point}_has1")
            names.append(f"player_p{point}_has2")
            names.append(f"player_p{point}_has3")
            names.append(f"player_p{point}_extra")

        # Opponent's point features
        for point in range(1, 25):
            names.append(f"opponent_p{point}_has1")
            names.append(f"opponent_p{point}_has2")
            names.append(f"opponent_p{point}_has3")
            names.append(f"opponent_p{point}_extra")

        # Additional features
        names.extend([
            "player_bar",
            "opponent_bar",
            "player_borne_off",
            "opponent_borne_off",
            "is_player_turn",
            "is_opponent_turn",
        ])

        return names

    def encode_with_dice(
        self,
        game_state: Dict[str, Any],
        player: str = 'white',
        include_dice: bool = True,
    ) -> np.ndarray:
        """
        Encode position with optional dice information.

        Extends the base 198 features with dice values.
        Output size: 198 + 12 = 210 if include_dice is True.

        Args:
            game_state: Backgammon game state.
            player: Player perspective.
            include_dice: Whether to include dice features.

        Returns:
            Feature array of shape (198,) or (210,).
        """
        base_features = self.encode(game_state, player)

        if not include_dice:
            return base_features

        # Add dice features (one-hot for each die, 6 values each)
        dice = game_state.get('dice', [])
        dice_features = np.zeros(12, dtype=np.float32)

        if len(dice) >= 1 and 1 <= dice[0] <= 6:
            dice_features[dice[0] - 1] = 1.0
        if len(dice) >= 2 and 1 <= dice[1] <= 6:
            dice_features[6 + dice[1] - 1] = 1.0

        return np.concatenate([base_features, dice_features])
