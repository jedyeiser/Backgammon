"""
Tests for behavioral cloning (player daemon) functionality.

Tests the BehavioralPlayer, BehavioralCloningTrainer, and PlayerMoveDataset for:
- Network architecture creation
- Action scoring and selection
- Training data loading
- Model training and evaluation
"""
import pytest
import torch

from apps.ai.players.behavioral import BehavioralPlayer
from apps.ai.players.registry import PlayerRegistry


class TestBehavioralPlayerArchitecture:
    """Tests for BehavioralPlayer network architecture."""

    def test_create_network_architecture(self):
        """Test creating the behavioral cloning architecture."""
        arch = BehavioralPlayer.create_network_architecture()

        assert arch['input_size'] == 198
        # Output: 6 action types + 26 from points + 26 to points = 58
        assert arch['output_size'] == 58
        assert 'layers' in arch
        assert len(arch['layers']) >= 4

    def test_architecture_buildable(self):
        """Test that the architecture can be built."""
        from apps.ai.networks.builder import NetworkBuilder

        builder = NetworkBuilder()
        arch = BehavioralPlayer.create_network_architecture()
        network = builder.from_json(arch)

        # Test forward pass
        x = torch.randn(1, 198)
        output = network(x)

        assert output.shape == (1, 58)

    def test_architecture_gradient_flow(self):
        """Test gradient flow through the network."""
        from apps.ai.networks.builder import NetworkBuilder

        builder = NetworkBuilder()
        arch = BehavioralPlayer.create_network_architecture()
        network = builder.from_json(arch)
        network.train()

        x = torch.randn(1, 198, requires_grad=True)
        output = network(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestBehavioralPlayer:
    """Tests for BehavioralPlayer."""

    @pytest.fixture
    def network(self):
        """Create a behavioral cloning network."""
        from apps.ai.networks.builder import NetworkBuilder

        builder = NetworkBuilder()
        arch = BehavioralPlayer.create_network_architecture()
        return builder.from_json(arch)

    @pytest.fixture
    def encoder(self):
        """Create a backgammon encoder."""
        from apps.ai.encoders.backgammon import BackgammonEncoder
        return BackgammonEncoder()

    @pytest.fixture
    def player(self, network, encoder):
        """Create a behavioral player."""
        return BehavioralPlayer(
            player_id='test_daemon',
            game_type='backgammon',
            network=network,
            encoder=encoder,
            user_id=1,
            name='Test Daemon',
            temperature=0.0,
        )

    def test_init(self, player):
        """Test player initialization."""
        assert player.player_id == 'test_daemon'
        assert player.game_type == 'backgammon'
        assert player.user_id == 1
        assert player.name == 'Test Daemon'
        assert player.temperature == 0.0

    def test_get_player_type(self, player):
        """Test player type."""
        assert player.get_player_type() == 'behavioral'

    def test_player_registered(self):
        """Test that behavioral player is registered."""
        assert 'behavioral' in PlayerRegistry.list_types()

    def test_select_action_returns_valid(
        self, player, sample_backgammon_state, sample_legal_actions
    ):
        """Test that select_action returns a legal action."""
        action = player.select_action(sample_backgammon_state, sample_legal_actions)
        assert action in sample_legal_actions

    def test_select_action_single(self, player, sample_backgammon_state):
        """Test with single legal action."""
        actions = [{'type': 'move', 'from': 6, 'to': 3}]
        action = player.select_action(sample_backgammon_state, actions)
        assert action == actions[0]

    def test_select_action_empty_raises(self, player, sample_backgammon_state):
        """Test that empty actions raises error."""
        with pytest.raises(ValueError, match="empty action list"):
            player.select_action(sample_backgammon_state, [])

    def test_select_action_deterministic_greedy(
        self, player, sample_backgammon_state, sample_legal_actions
    ):
        """Test greedy selection is deterministic."""
        player.temperature = 0.0
        action1 = player.select_action(sample_backgammon_state, sample_legal_actions)
        action2 = player.select_action(sample_backgammon_state, sample_legal_actions)
        assert action1 == action2

    def test_set_temperature(self, player):
        """Test temperature setting."""
        player.set_temperature(1.0)
        assert player.temperature == 1.0

    def test_get_action_distribution(self, player, sample_backgammon_state):
        """Test getting action distribution."""
        dist = player.get_action_distribution(sample_backgammon_state)

        assert 'action_types' in dist
        assert 'from_points' in dist
        assert 'to_points' in dist

        # Check action types probabilities
        assert len(dist['action_types']) == 6
        assert 'move' in dist['action_types']

        # Check point distributions
        assert len(dist['from_points']) == 26
        assert len(dist['to_points']) == 26

    def test_get_config(self, player):
        """Test configuration retrieval."""
        config = player.get_config()

        assert config['type'] == 'behavioral'
        assert config['user_id'] == 1
        assert config['temperature'] == 0.0

    def test_point_to_index_regular(self, player):
        """Test point to index conversion for regular points."""
        # Points 1-24 map to indices 0-23
        assert player._point_to_index(1) == 0
        assert player._point_to_index(12) == 11
        assert player._point_to_index(24) == 23

    def test_point_to_index_bar(self, player):
        """Test point to index conversion for bar."""
        assert player._point_to_index(0) == 24  # White bar
        assert player._point_to_index(25) == 24  # Black bar

    def test_point_to_index_bearoff(self, player):
        """Test point to index conversion for bear-off."""
        assert player._point_to_index(26) == 25
        assert player._point_to_index(27) == 25


class TestBehavioralPlayerScoring:
    """Tests for action scoring in BehavioralPlayer."""

    @pytest.fixture
    def player(self):
        """Create a behavioral player with known network state."""
        from apps.ai.networks.builder import NetworkBuilder
        from apps.ai.encoders.backgammon import BackgammonEncoder

        builder = NetworkBuilder()
        arch = BehavioralPlayer.create_network_architecture()
        network = builder.from_json(arch)

        return BehavioralPlayer(
            player_id='scorer',
            game_type='backgammon',
            network=network,
            encoder=BackgammonEncoder(),
            temperature=0.0,
        )

    def test_score_move_action(self, player):
        """Test scoring a move action."""
        import torch

        # Create mock logits
        action_type_logits = torch.randn(6)
        from_point_logits = torch.randn(26)
        to_point_logits = torch.randn(26)

        action = {'type': 'move', 'from': 6, 'to': 3}
        score = player._score_action(
            action, action_type_logits, from_point_logits, to_point_logits
        )

        # Score should be a float (log probability)
        assert isinstance(score, float)
        assert score <= 0  # Log probabilities are <= 0

    def test_score_non_move_action(self, player):
        """Test scoring a non-move action."""
        import torch

        action_type_logits = torch.randn(6)
        from_point_logits = torch.randn(26)
        to_point_logits = torch.randn(26)

        action = {'type': 'double'}
        score = player._score_action(
            action, action_type_logits, from_point_logits, to_point_logits
        )

        # Non-move actions only use action type logits
        assert isinstance(score, float)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from apps.ai.training.behavioral_cloning import TrainingConfig

        config = TrainingConfig()

        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.validation_split == 0.1
        assert config.early_stopping_patience == 10
        assert config.min_games == 10
        assert config.device == 'cpu'

    def test_custom_config(self):
        """Test custom configuration."""
        from apps.ai.training.behavioral_cloning import TrainingConfig

        config = TrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=0.0001,
        )

        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 0.0001


class TestTrainingResult:
    """Tests for TrainingResult."""

    def test_training_result_fields(self):
        """Test TrainingResult has all required fields."""
        from apps.ai.training.behavioral_cloning import TrainingResult

        result = TrainingResult(
            epochs_trained=50,
            final_loss=0.5,
            final_accuracy=0.7,
            best_accuracy=0.75,
            validation_loss=0.6,
            validation_accuracy=0.72,
            total_samples=1000,
            training_samples=900,
            validation_samples=100,
        )

        assert result.epochs_trained == 50
        assert result.final_accuracy == 0.7
        assert result.best_accuracy == 0.75
        assert result.total_samples == 1000


class TestPlayerMoveDataset:
    """Tests for PlayerMoveDataset functionality."""

    def test_action_types_defined(self):
        """Test that action types are properly defined."""
        from apps.ai.training.behavioral_cloning import PlayerMoveDataset

        assert len(PlayerMoveDataset.ACTION_TYPES) == 6
        assert 'move' in PlayerMoveDataset.ACTION_TYPES
        assert 'roll' in PlayerMoveDataset.ACTION_TYPES
        assert 'double' in PlayerMoveDataset.ACTION_TYPES

    def test_num_points_defined(self):
        """Test that number of points is correctly defined."""
        from apps.ai.training.behavioral_cloning import PlayerMoveDataset

        assert PlayerMoveDataset.NUM_POINTS == 26


class TestBehavioralCloningIntegration:
    """Integration tests for behavioral cloning."""

    @pytest.fixture
    def mock_network(self):
        """Create a mock network for testing."""
        from apps.ai.networks.builder import NetworkBuilder

        builder = NetworkBuilder()
        arch = BehavioralPlayer.create_network_architecture()
        return builder.from_json(arch)

    def test_full_action_selection_pipeline(
        self, mock_network, sample_backgammon_state, sample_legal_actions
    ):
        """Test the complete action selection pipeline."""
        from apps.ai.encoders.backgammon import BackgammonEncoder

        player = BehavioralPlayer(
            player_id='integration_test',
            game_type='backgammon',
            network=mock_network,
            encoder=BackgammonEncoder(),
            temperature=0.0,
        )

        # Should complete without error
        action = player.select_action(sample_backgammon_state, sample_legal_actions)

        # Should return a valid action
        assert action in sample_legal_actions
        assert 'type' in action

    def test_network_batch_processing(self, mock_network):
        """Test that network can handle batch inputs."""
        batch_size = 16
        x = torch.randn(batch_size, 198)
        output = mock_network(x)

        assert output.shape == (batch_size, 58)

    def test_training_loop_components(self):
        """Test that training loop components work together."""
        import torch.nn as nn

        from apps.ai.networks.builder import NetworkBuilder

        # Create network
        builder = NetworkBuilder()
        arch = BehavioralPlayer.create_network_architecture()
        network = builder.from_json(arch)

        # Create optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create loss function
        loss_fn = nn.CrossEntropyLoss()

        # Simulate one training step
        network.train()
        x = torch.randn(8, 198)
        targets = torch.randint(0, 6, (8,))  # Action type targets

        optimizer.zero_grad()
        output = network(x)
        action_logits = output[:, :6]
        loss = loss_fn(action_logits, targets)
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert loss.item() >= 0
