"""
Tests for board state encoders.

Tests the BackgammonEncoder and encoding utilities for:
- Correct feature dimensions
- Feature range validation [0, 1]
- Encode/decode round-trip consistency
- Edge cases (empty board, all checkers on bar, etc.)
"""
import pytest
import numpy as np

from apps.ai.encoders.backgammon import BackgammonEncoder


class TestBackgammonEncoder:
    """Tests for BackgammonEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return BackgammonEncoder()

    def test_encoder_input_size(self, encoder):
        """Test that encoder has correct input size constant."""
        assert encoder.input_size == 198

    def test_encoder_game_type(self, encoder):
        """Test that encoder has correct game type."""
        assert encoder.game_type == 'backgammon'

    def test_encode_output_shape(self, encoder, initial_backgammon_state):
        """Test that encode returns correct shape."""
        features = encoder.encode(initial_backgammon_state, player='white')
        assert features.shape == (198,)
        assert features.dtype == np.float32

    def test_encode_features_in_range(self, encoder, initial_backgammon_state):
        """Test that all features are in [0, 1] range."""
        features = encoder.encode(initial_backgammon_state, player='white')
        assert np.all(features >= 0.0), "Features should be >= 0"
        assert np.all(features <= 1.0), "Features should be <= 1"

    def test_encode_empty_board(self, encoder, empty_board_state):
        """Test encoding an empty board (all checkers borne off)."""
        features = encoder.encode(empty_board_state, player='white')
        assert features.shape == (198,)

        # All point features should be 0
        assert np.all(features[:192] == 0.0)

        # Bar should be 0
        assert features[192] == 0.0
        assert features[193] == 0.0

        # Borne off should be full (15/15 = 1.0)
        assert features[194] == 1.0  # Player borne off
        assert features[195] == 1.0  # Opponent borne off

    def test_encode_different_perspectives(self, encoder, initial_backgammon_state):
        """Test that white and black perspectives differ."""
        white_features = encoder.encode(initial_backgammon_state, player='white')
        black_features = encoder.encode(initial_backgammon_state, player='black')

        # They should not be identical
        assert not np.allclose(white_features, black_features)

    def test_encode_turn_indicator(self, encoder, initial_backgammon_state):
        """Test turn indicator features."""
        # White's turn
        initial_backgammon_state['current_turn'] = 'white'
        white_features = encoder.encode(initial_backgammon_state, player='white')
        assert white_features[196] == 1.0  # Is player's turn
        assert white_features[197] == 0.0  # Is opponent's turn

        # Black's turn
        initial_backgammon_state['current_turn'] = 'black'
        black_features = encoder.encode(initial_backgammon_state, player='white')
        assert black_features[196] == 0.0
        assert black_features[197] == 1.0

    def test_encode_bar_normalization(self, encoder):
        """Test that bar counts are normalized by 15."""
        state = {
            'points': {},
            'bar': {'white': 3, 'black': 6},
            'home': {'white': 0, 'black': 0},
            'current_turn': 'white',
        }
        features = encoder.encode(state, player='white')

        assert features[192] == pytest.approx(3 / 15)  # Player bar
        assert features[193] == pytest.approx(6 / 15)  # Opponent bar

    def test_encode_point_features_single_checker(self, encoder):
        """Test point encoding with 1 checker (blot)."""
        state = {
            'points': {'1': 1},  # 1 white checker on point 1
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
            'current_turn': 'white',
        }
        features = encoder.encode(state, player='white')

        # Point 1 features for player (first 4 features)
        assert features[0] == 1.0   # has 1+ checker
        assert features[1] == 0.0   # has 2+ checkers
        assert features[2] == 0.0   # has 3+ checkers
        assert features[3] == 0.0   # extra checkers

    def test_encode_point_features_made_point(self, encoder):
        """Test point encoding with 2 checkers (made point)."""
        state = {
            'points': {'1': 2},
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
            'current_turn': 'white',
        }
        features = encoder.encode(state, player='white')

        assert features[0] == 1.0
        assert features[1] == 1.0   # has 2+ checkers
        assert features[2] == 0.0
        assert features[3] == 0.0

    def test_encode_point_features_stacked(self, encoder):
        """Test point encoding with many checkers (stack)."""
        state = {
            'points': {'1': 5},
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
            'current_turn': 'white',
        }
        features = encoder.encode(state, player='white')

        assert features[0] == 1.0
        assert features[1] == 1.0
        assert features[2] == 1.0   # has 3+ checkers
        assert features[3] == pytest.approx((5 - 3) / 2)  # (5-3)/2 = 1.0

    def test_decode_point_features(self, encoder, initial_backgammon_state):
        """Test that decode_point_features inverts encoding."""
        features = encoder.encode(initial_backgammon_state, player='white')

        # Point 6 has 5 white checkers in initial position
        count = encoder.decode_point_features(features, 6, is_player=True)
        assert count == 5

    def test_decode_point_features_empty(self, encoder, empty_board_state):
        """Test decoding empty points."""
        features = encoder.encode(empty_board_state, player='white')

        for point in range(1, 25):
            count = encoder.decode_point_features(features, point, is_player=True)
            assert count == 0

    def test_get_feature_names(self, encoder):
        """Test that feature names list has correct length."""
        names = encoder.get_feature_names()
        assert len(names) == 198

    def test_get_feature_names_content(self, encoder):
        """Test that feature names are descriptive."""
        names = encoder.get_feature_names()

        # Check first player point features
        assert names[0] == 'player_p1_has1'
        assert names[1] == 'player_p1_has2'
        assert names[2] == 'player_p1_has3'
        assert names[3] == 'player_p1_extra'

        # Check opponent features
        assert names[96] == 'opponent_p1_has1'

        # Check special features
        assert names[192] == 'player_bar'
        assert names[193] == 'opponent_bar'
        assert names[196] == 'is_player_turn'
        assert names[197] == 'is_opponent_turn'

    def test_encode_with_dice_without_dice(self, encoder, initial_backgammon_state):
        """Test encode_with_dice without dice returns 198 features."""
        features = encoder.encode_with_dice(
            initial_backgammon_state,
            player='white',
            include_dice=False,
        )
        assert features.shape == (198,)

    def test_encode_with_dice_with_dice(self, encoder, sample_backgammon_state):
        """Test encode_with_dice with dice returns 210 features."""
        features = encoder.encode_with_dice(
            sample_backgammon_state,
            player='white',
            include_dice=True,
        )
        assert features.shape == (210,)

        # Check dice features are one-hot encoded
        dice_features = features[198:]
        assert np.sum(dice_features) == pytest.approx(2.0)  # Two dice

    def test_encode_deterministic(self, encoder, initial_backgammon_state):
        """Test that encoding is deterministic."""
        features1 = encoder.encode(initial_backgammon_state, player='white')
        features2 = encoder.encode(initial_backgammon_state, player='white')
        assert np.allclose(features1, features2)

    def test_encode_symmetric_players(self, encoder):
        """Test that swapped players produce swapped features."""
        # State where white has checkers on point 6, black on point 19
        state = {
            'points': {'6': 3, '19': -3},
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
            'current_turn': 'white',
        }

        white_features = encoder.encode(state, player='white')
        black_features = encoder.encode(state, player='black')

        # For white: point 6 is player's, point 19 is opponent's
        # For black: point 19 is player's, point 6 is opponent's
        # Features should have swapped player/opponent sections
        assert white_features[20] != black_features[20]  # Point 6 features differ
