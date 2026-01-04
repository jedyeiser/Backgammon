"""
Tests for player implementations.

Tests the BasePlayer interface, RandomPlayer, HeuristicPlayer,
NeuralPlayer, and PlayerRegistry for:
- Player interface compliance
- Action selection behavior
- Seed reproducibility
- Model loading/saving
"""
import pytest
from unittest.mock import MagicMock, patch

from apps.ai.players.base import BasePlayer
from apps.ai.players.random_player import RandomPlayer
from apps.ai.players.heuristic import HeuristicPlayer
from apps.ai.players.neural import NeuralPlayer
from apps.ai.players.registry import PlayerRegistry, get_player, register_player


class TestRandomPlayer:
    """Tests for RandomPlayer."""

    @pytest.fixture
    def player(self):
        """Create a random player."""
        return RandomPlayer(
            player_id='random_1',
            game_type='backgammon',
        )

    @pytest.fixture
    def seeded_player(self):
        """Create a seeded random player for reproducibility."""
        return RandomPlayer(
            player_id='random_seeded',
            game_type='backgammon',
            seed=42,
        )

    def test_init(self, player):
        """Test random player initialization."""
        assert player.player_id == 'random_1'
        assert player.game_type == 'backgammon'
        assert player.name == 'Random Player'

    def test_init_custom_name(self):
        """Test random player with custom name."""
        player = RandomPlayer(
            player_id='r1',
            game_type='backgammon',
            name='My Random',
        )
        assert player.name == 'My Random'

    def test_get_player_type(self, player):
        """Test player type."""
        assert player.get_player_type() == 'random'

    def test_select_action_returns_valid(self, player, sample_legal_actions):
        """Test that select_action returns one of the legal actions."""
        action = player.select_action({}, sample_legal_actions)
        assert action in sample_legal_actions

    def test_select_action_empty_raises(self, player):
        """Test that empty actions raises error."""
        with pytest.raises(ValueError, match="empty action list"):
            player.select_action({}, [])

    def test_select_action_single(self, player):
        """Test with single legal action."""
        actions = [{'type': 'move', 'from': 1, 'to': 2}]
        action = player.select_action({}, actions)
        assert action == actions[0]

    def test_seeded_reproducibility(self, seeded_player, sample_legal_actions):
        """Test that seeded player produces reproducible results."""
        # First selection
        action1 = seeded_player.select_action({}, sample_legal_actions)

        # Reset seed and select again
        seeded_player.reset_seed(42)
        action2 = seeded_player.select_action({}, sample_legal_actions)

        assert action1 == action2

    def test_different_seeds_different_results(self, sample_legal_actions):
        """Test that different seeds produce different results (usually)."""
        player1 = RandomPlayer('p1', 'backgammon', seed=1)
        player2 = RandomPlayer('p2', 'backgammon', seed=2)

        # Run multiple selections to check they differ
        actions1 = [player1.select_action({}, sample_legal_actions) for _ in range(10)]
        player1.reset_seed(1)
        player2.reset_seed(2)
        actions2 = [player2.select_action({}, sample_legal_actions) for _ in range(10)]

        # At least some actions should differ (probabilistic but nearly certain)
        assert actions1 != actions2

    def test_reset_seed(self, seeded_player):
        """Test reset_seed method."""
        seeded_player.reset_seed(123)
        assert seeded_player.seed == 123

    def test_reset_seed_none(self, seeded_player):
        """Test reset_seed with None uses system entropy."""
        seeded_player.reset_seed(None)
        assert seeded_player.seed is None

    def test_get_config(self, seeded_player):
        """Test configuration includes seed."""
        config = seeded_player.get_config()
        assert config['seed'] == 42
        assert config['type'] == 'random'

    def test_repr(self, player):
        """Test string representation."""
        assert 'RandomPlayer' in repr(player)
        assert 'random_1' in repr(player)

    def test_str(self, player):
        """Test string output."""
        assert 'random' in str(player).lower()


class TestHeuristicPlayer:
    """Tests for HeuristicPlayer."""

    @pytest.fixture
    def player(self):
        """Create a heuristic player."""
        return HeuristicPlayer(
            player_id='heur_1',
            game_type='backgammon',
        )

    def test_init(self, player):
        """Test heuristic player initialization."""
        assert player.player_id == 'heur_1'
        assert player.game_type == 'backgammon'

    def test_get_player_type(self, player):
        """Test player type."""
        assert player.get_player_type() == 'heuristic'

    def test_select_action_returns_valid(
        self, player, sample_backgammon_state, sample_legal_actions
    ):
        """Test that select_action returns one of the legal actions."""
        action = player.select_action(sample_backgammon_state, sample_legal_actions)
        assert action in sample_legal_actions

    def test_select_action_single(self, player, sample_backgammon_state):
        """Test with single legal action."""
        actions = [{'type': 'move', 'from': 1, 'to': 2}]
        action = player.select_action(sample_backgammon_state, actions)
        assert action == actions[0]

    def test_select_action_deterministic(
        self, player, sample_backgammon_state, sample_legal_actions
    ):
        """Test that heuristic selection is deterministic."""
        action1 = player.select_action(sample_backgammon_state, sample_legal_actions)
        action2 = player.select_action(sample_backgammon_state, sample_legal_actions)
        assert action1 == action2


class TestNeuralPlayer:
    """Tests for NeuralPlayer."""

    @pytest.fixture
    def player(self, minimal_architecture):
        """Create a neural player with minimal network."""
        return NeuralPlayer(
            player_id='neural_1',
            game_type='backgammon',
            architecture=minimal_architecture,
        )

    def test_init(self, player):
        """Test neural player initialization."""
        assert player.player_id == 'neural_1'
        assert player.game_type == 'backgammon'

    def test_get_player_type(self, player):
        """Test player type."""
        assert player.get_player_type() == 'neural'

    def test_select_action_returns_valid(
        self, player, sample_backgammon_state, sample_legal_actions
    ):
        """Test that select_action returns one of the legal actions."""
        action = player.select_action(sample_backgammon_state, sample_legal_actions)
        assert action in sample_legal_actions

    def test_select_action_single(self, player, sample_backgammon_state):
        """Test with single legal action."""
        actions = [{'type': 'move', 'from': 1, 'to': 2}]
        action = player.select_action(sample_backgammon_state, actions)
        assert action == actions[0]

    def test_greedy_selection(self, sample_backgammon_state, minimal_architecture):
        """Test greedy selection with temperature=0."""
        player = NeuralPlayer(
            player_id='greedy',
            game_type='backgammon',
            architecture=minimal_architecture,
            temperature=0.0,
        )

        actions = [
            {'type': 'move', 'from': 1, 'to': 2},
            {'type': 'move', 'from': 1, 'to': 3},
        ]

        # Should be deterministic with temperature=0
        action1 = player.select_action(sample_backgammon_state, actions)
        action2 = player.select_action(sample_backgammon_state, actions)
        assert action1 == action2

    def test_training_mode(self, player):
        """Test setting training mode."""
        player.set_training_mode(True)
        assert player.network.training

        player.set_training_mode(False)
        assert not player.network.training

    def test_evaluate_position(self, player, sample_backgammon_state):
        """Test position evaluation."""
        value = player.evaluate_position(sample_backgammon_state)
        assert 0 <= value <= 1  # Sigmoid output


class TestPlayerRegistry:
    """Tests for PlayerRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry after each test."""
        # Store original registry
        original = PlayerRegistry._registry.copy()
        yield
        # Restore original registry
        PlayerRegistry._registry = original

    def test_builtin_players_registered(self):
        """Test that built-in players are registered."""
        assert 'random' in PlayerRegistry.list_types()
        assert 'heuristic' in PlayerRegistry.list_types()
        assert 'neural' in PlayerRegistry.list_types()

    def test_get_registered_player(self):
        """Test getting a registered player class."""
        cls = PlayerRegistry.get('random')
        assert cls is RandomPlayer

    def test_get_unregistered_returns_none(self):
        """Test getting unregistered player returns None."""
        cls = PlayerRegistry.get('nonexistent')
        assert cls is None

    def test_create_player(self):
        """Test creating a player instance."""
        player = PlayerRegistry.create(
            player_type='random',
            player_id='test_player',
            game_type='backgammon',
        )
        assert isinstance(player, RandomPlayer)
        assert player.player_id == 'test_player'

    def test_create_with_kwargs(self):
        """Test creating player with additional kwargs."""
        player = PlayerRegistry.create(
            player_type='random',
            player_id='seeded',
            game_type='backgammon',
            seed=42,
        )
        assert player.seed == 42

    def test_create_unknown_raises(self):
        """Test that creating unknown player raises error."""
        with pytest.raises(ValueError, match="Unknown player type"):
            PlayerRegistry.create(
                player_type='nonexistent',
                player_id='test',
                game_type='backgammon',
            )

    def test_register_new_player(self):
        """Test registering a new player type."""
        class CustomPlayer(BasePlayer):
            def select_action(self, game_state, legal_actions):
                return legal_actions[0]

            def get_player_type(self):
                return 'custom'

        PlayerRegistry.register('custom', CustomPlayer)
        assert 'custom' in PlayerRegistry.list_types()

    def test_register_duplicate_raises(self):
        """Test that registering duplicate raises error."""
        with pytest.raises(ValueError, match="already registered"):
            PlayerRegistry.register('random', RandomPlayer)

    def test_register_non_player_raises(self):
        """Test that registering non-player raises error."""
        with pytest.raises(TypeError, match="subclass of BasePlayer"):
            PlayerRegistry.register('bad', str)

    def test_unregister(self):
        """Test unregistering a player type."""
        # Register a temporary type
        class TempPlayer(BasePlayer):
            def select_action(self, game_state, legal_actions):
                return legal_actions[0]
            def get_player_type(self):
                return 'temp'

        PlayerRegistry.register('temp', TempPlayer)
        assert 'temp' in PlayerRegistry.list_types()

        PlayerRegistry.unregister('temp')
        assert 'temp' not in PlayerRegistry.list_types()

    def test_list_types(self):
        """Test listing player types."""
        types = PlayerRegistry.list_types()
        assert isinstance(types, list)
        assert len(types) >= 3  # At least the builtins


class TestGetPlayerFunction:
    """Tests for get_player convenience function."""

    def test_get_player(self):
        """Test get_player function."""
        player = get_player(
            player_type='random',
            player_id='test',
            game_type='backgammon',
        )
        assert isinstance(player, RandomPlayer)


class TestRegisterPlayerDecorator:
    """Tests for register_player decorator."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Clean up test registrations."""
        yield
        PlayerRegistry.unregister('decorated_test')

    def test_decorator(self):
        """Test register_player decorator."""
        @register_player('decorated_test')
        class DecoratedPlayer(BasePlayer):
            def select_action(self, game_state, legal_actions):
                return legal_actions[0]
            def get_player_type(self):
                return 'decorated_test'

        assert 'decorated_test' in PlayerRegistry.list_types()
        assert PlayerRegistry.get('decorated_test') is DecoratedPlayer


class TestBasePlayerInterface:
    """Tests for BasePlayer interface compliance."""

    def test_random_player_is_base_player(self):
        """Test RandomPlayer is a BasePlayer."""
        player = RandomPlayer('p1', 'backgammon')
        assert isinstance(player, BasePlayer)

    def test_heuristic_player_is_base_player(self):
        """Test HeuristicPlayer is a BasePlayer."""
        player = HeuristicPlayer('p1', 'backgammon')
        assert isinstance(player, BasePlayer)

    def test_lifecycle_hooks_callable(self):
        """Test lifecycle hooks can be called without error."""
        player = RandomPlayer('p1', 'backgammon')

        # These should not raise
        player.on_game_start({})
        player.on_game_end({}, {'winner': 'p1'})
        player.on_opponent_action({'type': 'move'}, {})
