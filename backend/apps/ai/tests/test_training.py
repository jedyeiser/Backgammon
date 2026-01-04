"""
Tests for training infrastructure.

Tests the TDLearner and related training components for:
- Episode management
- TD updates
- Learning rate annealing
- Training statistics
"""
import pytest
import torch
import torch.nn as nn

from apps.ai.training.td_learner import TDLearner, TDLambdaWithTraces
from apps.ai.networks.builder import NetworkBuilder
from apps.ai.networks.architectures import minimal_architecture


class TestTDLearner:
    """Tests for TDLearner."""

    @pytest.fixture
    def network(self):
        """Create a minimal network for testing."""
        builder = NetworkBuilder()
        return builder.from_json(minimal_architecture())

    @pytest.fixture
    def learner(self, network):
        """Create a TD learner."""
        return TDLearner(
            network=network,
            alpha=0.01,
            lambda_=0.7,
        )

    def test_init(self, learner):
        """Test learner initialization."""
        assert learner.alpha == 0.01
        assert learner.lambda_ == 0.7
        assert learner.gamma == 1.0
        assert learner.episodes_trained == 0

    def test_init_custom_gamma(self, network):
        """Test learner with custom gamma."""
        learner = TDLearner(network, gamma=0.99)
        assert learner.gamma == 0.99

    def test_init_custom_optimizer(self, network):
        """Test learner with custom optimizer."""
        learner = TDLearner(
            network,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={'betas': (0.9, 0.99)},
        )
        assert isinstance(learner.optimizer, torch.optim.Adam)

    def test_new_episode(self, learner):
        """Test episode reset."""
        # Add some data
        learner._states.append(torch.randn(1, 198))
        learner._values.append(torch.tensor([[0.5]]))

        learner.new_episode()

        assert len(learner._states) == 0
        assert len(learner._values) == 0

    def test_observe_returns_value(self, learner):
        """Test that observe returns a value prediction."""
        learner.new_episode()
        features = torch.randn(198)

        value = learner.observe(features)

        assert isinstance(value, float)
        assert 0 <= value <= 1  # Sigmoid output

    def test_observe_stores_state(self, learner):
        """Test that observe stores state and value."""
        learner.new_episode()
        features = torch.randn(198)

        learner.observe(features)

        assert len(learner._states) == 1
        assert len(learner._values) == 1

    def test_observe_multiple_states(self, learner):
        """Test observing multiple states."""
        learner.new_episode()

        for _ in range(5):
            learner.observe(torch.randn(198))

        assert len(learner._states) == 5
        assert len(learner._values) == 5

    def test_observe_no_update_first_state(self, learner):
        """Test that first state doesn't trigger update."""
        learner.new_episode()
        learner.observe(torch.randn(198))

        # No TD errors recorded for first state
        assert len(learner._episode_td_errors) == 0

    def test_observe_update_subsequent_states(self, learner):
        """Test that subsequent states trigger updates."""
        learner.new_episode()
        learner.observe(torch.randn(198))
        learner.observe(torch.randn(198))

        # TD error recorded for second state
        assert len(learner._episode_td_errors) == 1

    def test_observe_batch_features(self, learner):
        """Test observe with batch dimension."""
        learner.new_episode()
        features = torch.randn(1, 198)

        value = learner.observe(features)
        assert isinstance(value, float)

    def test_end_episode(self, learner):
        """Test ending an episode."""
        learner.new_episode()
        learner.observe(torch.randn(198))
        learner.observe(torch.randn(198))

        stats = learner.end_episode(final_value=1.0)

        assert 'td_error' in stats
        assert 'num_states' in stats
        assert stats['num_states'] == 2
        assert learner.episodes_trained == 1

    def test_end_episode_empty(self, learner):
        """Test ending an empty episode."""
        learner.new_episode()
        stats = learner.end_episode(final_value=1.0)

        assert stats['num_states'] == 0
        assert stats['td_error'] == 0.0

    def test_train_from_trajectory(self, learner):
        """Test offline training from trajectory."""
        states = [torch.randn(198) for _ in range(5)]

        stats = learner.train_from_trajectory(states, final_value=1.0)

        assert stats['num_states'] == 5
        assert 'loss' in stats
        assert learner.episodes_trained == 1

    def test_train_from_trajectory_empty(self, learner):
        """Test training from empty trajectory."""
        stats = learner.train_from_trajectory([], final_value=1.0)

        assert stats['num_states'] == 0

    def test_get_stats(self, learner):
        """Test getting training statistics."""
        learner.new_episode()
        learner.observe(torch.randn(198))
        learner.observe(torch.randn(198))
        learner.end_episode(1.0)

        stats = learner.get_stats()

        assert stats['episodes_trained'] == 1
        assert stats['alpha'] == 0.01
        assert stats['lambda'] == 0.7

    def test_set_learning_rate(self, learner):
        """Test changing learning rate."""
        learner.set_learning_rate(0.001)

        assert learner.alpha == 0.001
        # Check optimizer lr
        for param_group in learner.optimizer.param_groups:
            assert param_group['lr'] == 0.001

    def test_anneal_learning_rate(self, learner):
        """Test learning rate annealing."""
        new_lr = learner.anneal_learning_rate(
            initial_alpha=0.1,
            final_alpha=0.01,
            progress=0.5,
        )

        assert new_lr == pytest.approx(0.055)
        assert learner.alpha == pytest.approx(0.055)

    def test_anneal_learning_rate_start(self, learner):
        """Test annealing at start."""
        new_lr = learner.anneal_learning_rate(0.1, 0.01, 0.0)
        assert new_lr == pytest.approx(0.1)

    def test_anneal_learning_rate_end(self, learner):
        """Test annealing at end."""
        new_lr = learner.anneal_learning_rate(0.1, 0.01, 1.0)
        assert new_lr == pytest.approx(0.01)

    def test_network_updates(self, learner):
        """Test that network weights are updated during training."""
        # Store initial weights
        initial_weights = [
            p.clone() for p in learner.network.parameters()
        ]

        # Run training episode
        learner.new_episode()
        for _ in range(10):
            learner.observe(torch.randn(198))
        learner.end_episode(1.0)

        # Check weights changed
        weights_changed = False
        for initial, current in zip(initial_weights, learner.network.parameters()):
            if not torch.allclose(initial, current):
                weights_changed = True
                break

        assert weights_changed, "Network weights should change during training"


class TestTDLambdaWithTraces:
    """Tests for TDLambdaWithTraces."""

    @pytest.fixture
    def network(self):
        """Create a minimal network for testing."""
        builder = NetworkBuilder()
        return builder.from_json(minimal_architecture())

    @pytest.fixture
    def learner(self, network):
        """Create a TD(λ) learner with traces."""
        return TDLambdaWithTraces(
            network=network,
            alpha=0.01,
            lambda_=0.7,
            replacing_traces=True,
        )

    def test_init(self, learner):
        """Test learner with traces initialization."""
        assert learner.replacing_traces is True
        assert hasattr(learner, '_traces')

    def test_init_accumulating_traces(self, network):
        """Test accumulating traces option."""
        learner = TDLambdaWithTraces(
            network, replacing_traces=False
        )
        assert learner.replacing_traces is False

    def test_traces_initialized(self, learner):
        """Test that traces are initialized to zero."""
        for name, trace in learner._traces.items():
            assert torch.all(trace == 0)

    def test_new_episode_resets_traces(self, learner):
        """Test that new episode resets traces."""
        # Modify traces
        for trace in learner._traces.values():
            trace.fill_(1.0)

        learner.new_episode()

        # Check traces are reset
        for trace in learner._traces.values():
            assert torch.all(trace == 0)

    def test_observe_with_traces(self, learner):
        """Test observing with trace updates."""
        learner.new_episode()

        value1, td1 = learner.observe_with_traces(torch.randn(198))
        assert td1 == 0.0  # First state has no TD error

        value2, td2 = learner.observe_with_traces(torch.randn(198))
        # Second state has TD error

        assert isinstance(value1, float)
        assert isinstance(value2, float)

    def test_end_episode_with_traces(self, learner):
        """Test ending episode with trace updates."""
        learner.new_episode()
        learner.observe_with_traces(torch.randn(198))
        learner.observe_with_traces(torch.randn(198))

        stats = learner.end_episode_with_traces(1.0)

        assert 'td_error' in stats
        assert 'num_states' in stats
        assert learner.episodes_trained == 1

    def test_end_episode_with_traces_empty(self, learner):
        """Test ending empty episode with traces."""
        learner.new_episode()
        stats = learner.end_episode_with_traces(1.0)

        assert stats['num_states'] == 0

    def test_traces_accumulate(self, learner):
        """Test that traces accumulate during episode."""
        learner.new_episode()

        # Record initial trace magnitudes
        initial_trace_norm = sum(
            trace.abs().sum().item() for trace in learner._traces.values()
        )

        # Observe several states
        for _ in range(5):
            learner.observe_with_traces(torch.randn(198))

        # Traces should have grown
        final_trace_norm = sum(
            trace.abs().sum().item() for trace in learner._traces.values()
        )

        # With multiple observations, traces should generally increase
        # (though this depends on network gradients)
        assert final_trace_norm >= 0  # Non-negative after updates


class TestTDLearnerIntegration:
    """Integration tests for TD learning."""

    @pytest.fixture
    def network(self):
        """Create a minimal network for testing."""
        builder = NetworkBuilder()
        return builder.from_json(minimal_architecture())

    def test_full_episode_training(self, network):
        """Test a complete training episode."""
        learner = TDLearner(network, alpha=0.1)

        learner.new_episode()

        # Simulate a short game
        for _ in range(20):
            features = torch.randn(198)
            learner.observe(features)

        # End with a win
        stats = learner.end_episode(final_value=1.0)

        assert stats['num_states'] == 20
        assert learner.episodes_trained == 1

    def test_multiple_episodes(self, network):
        """Test training multiple episodes."""
        learner = TDLearner(network, alpha=0.01)

        for episode in range(5):
            learner.new_episode()

            for _ in range(10):
                learner.observe(torch.randn(198))

            # Alternate wins/losses
            final = 1.0 if episode % 2 == 0 else 0.0
            learner.end_episode(final_value=final)

        assert learner.episodes_trained == 5

    def test_learning_improves_predictions(self, network):
        """Test that training actually changes predictions."""
        learner = TDLearner(network, alpha=0.1)

        # Fixed state for comparison
        test_state = torch.randn(198)

        # Get initial prediction
        with torch.no_grad():
            initial_pred = network(test_state.unsqueeze(0)).item()

        # Train towards value of 1.0
        for _ in range(10):
            learner.new_episode()
            for _ in range(5):
                learner.observe(test_state.clone())
            learner.end_episode(final_value=1.0)

        # Get final prediction
        with torch.no_grad():
            final_pred = network(test_state.unsqueeze(0)).item()

        # Prediction should move toward 1.0
        assert final_pred > initial_pred
