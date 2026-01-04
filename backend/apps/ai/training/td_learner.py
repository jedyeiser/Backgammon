"""
Temporal Difference (TD) learning implementation.

TD(λ) is the classic reinforcement learning algorithm used by TD-Gammon.
It learns to predict the value of game positions by bootstrapping
from subsequent position predictions.

Key concepts:
- TD error: The difference between consecutive value predictions
- Eligibility traces: Credit assignment for past states
- λ (lambda): Trade-off between TD(0) and Monte Carlo methods
"""
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class TDLearner:
    """
    TD(λ) learner for training neural network position evaluators.

    Implements the TD(λ) algorithm with eligibility traces, suitable
    for training game-playing neural networks through self-play.

    The algorithm updates network weights based on temporal difference
    errors, using eligibility traces to assign credit to past states.

    Attributes:
        network: The neural network to train.
        optimizer: PyTorch optimizer for weight updates.
        alpha: Learning rate (step size).
        lambda_: Eligibility trace decay parameter (0-1).
        gamma: Discount factor for future rewards (typically 1.0 for games).

    Example:
        learner = TDLearner(
            network=network,
            alpha=0.01,
            lambda_=0.7,
        )

        # During a game, record states
        learner.new_episode()
        learner.observe(state1_features)
        learner.observe(state2_features)
        ...

        # At game end, provide final outcome
        learner.end_episode(winner_value=1.0)  # 1.0 = win, 0.0 = loss

    Reference:
        Sutton, R. S. (1988). Learning to predict by the methods of
        temporal differences. Machine Learning, 3(1), 9-44.

        Tesauro, G. (1995). Temporal difference learning and TD-Gammon.
        Communications of the ACM, 38(3), 58-68.
    """

    def __init__(
        self,
        network: 'nn.Module',
        alpha: float = 0.01,
        lambda_: float = 0.7,
        gamma: float = 1.0,
        optimizer_class: Optional[type] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the TD learner.

        Args:
            network: Neural network to train.
            alpha: Learning rate (0.001 to 0.1 typical).
            lambda_: Eligibility trace decay (0.0 to 1.0).
                    0.0 = TD(0), only immediate transitions
                    1.0 = Monte Carlo, full episode credit
                    0.7 = Common choice, good balance
            gamma: Discount factor (1.0 for finite games).
            optimizer_class: Optional optimizer class (default: SGD).
            optimizer_kwargs: Additional optimizer arguments.
        """
        import torch.optim as optim

        self.network = network
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma

        # Set up optimizer
        optimizer_class = optimizer_class or optim.SGD
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_class(
            network.parameters(),
            lr=alpha,
            **optimizer_kwargs,
        )

        # Episode state
        self._states: List['torch.Tensor'] = []
        self._values: List['torch.Tensor'] = []
        self._eligibility_traces: Optional[List['torch.Tensor']] = None

        # Training statistics
        self.episodes_trained = 0
        self.total_td_errors = 0.0
        self._episode_td_errors: List[float] = []

    def new_episode(self) -> None:
        """
        Start a new episode (game).

        Clears the state history and resets eligibility traces.
        Must be called before each new game.
        """
        self._states = []
        self._values = []
        self._eligibility_traces = None
        self._episode_td_errors = []

    def observe(
        self,
        features: 'torch.Tensor',
        update_online: bool = True,
    ) -> float:
        """
        Observe a new game state during an episode.

        Records the state and optionally performs an online TD update
        if this is not the first state.

        Args:
            features: Encoded game state as a tensor.
            update_online: Whether to perform immediate TD updates.

        Returns:
            The network's value prediction for this state.
        """
        import torch

        # Ensure network is in training mode
        self.network.train()

        # Get value prediction
        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.enable_grad():
            value = self.network(features)

        # Store state and value
        self._states.append(features.detach())
        self._values.append(value)

        # Perform online TD update if we have a previous state
        if update_online and len(self._values) > 1:
            td_error = self._td_update(
                prev_value=self._values[-2],
                curr_value=value.detach(),
            )
            self._episode_td_errors.append(td_error)

        return value.item()

    def end_episode(
        self,
        final_value: float,
        update_remaining: bool = True,
    ) -> Dict[str, float]:
        """
        End the current episode with the final outcome.

        For games, final_value is typically:
        - 1.0 if the player won
        - 0.0 if the player lost
        - 0.5 for a draw

        Args:
            final_value: The true value at episode end.
            update_remaining: Whether to update from final states.

        Returns:
            Dictionary with episode statistics.
        """
        import torch

        if not self._values:
            return {'td_error': 0.0, 'num_states': 0}

        # Final TD update from last state to actual outcome
        if update_remaining and self._values:
            final_tensor = torch.tensor(
                [[final_value]],
                dtype=self._values[-1].dtype,
                device=self._values[-1].device,
            )
            td_error = self._td_update(
                prev_value=self._values[-1],
                curr_value=final_tensor,
                is_terminal=True,
            )
            self._episode_td_errors.append(td_error)

        # Update statistics
        self.episodes_trained += 1
        avg_td_error = (
            sum(abs(e) for e in self._episode_td_errors) /
            len(self._episode_td_errors)
            if self._episode_td_errors else 0.0
        )
        self.total_td_errors += avg_td_error

        return {
            'td_error': avg_td_error,
            'num_states': len(self._states),
            'final_value': final_value,
        }

    def _td_update(
        self,
        prev_value: 'torch.Tensor',
        curr_value: 'torch.Tensor',
        is_terminal: bool = False,
    ) -> float:
        """
        Perform a single TD update step.

        Computes the TD error and updates network weights using
        backpropagation through the value prediction.

        Args:
            prev_value: Value prediction at previous state (with grad).
            curr_value: Value prediction at current state (no grad).
            is_terminal: Whether this is the final transition.

        Returns:
            The TD error magnitude.
        """
        import torch

        # TD error: δ = r + γ*V(s') - V(s)
        # For games during play, r=0; at terminal, r is the outcome
        # We use V(s') directly as the target

        if is_terminal:
            target = curr_value
        else:
            target = self.gamma * curr_value

        td_error = (target - prev_value).item()

        # Backpropagate through the previous value prediction
        self.optimizer.zero_grad()

        # Loss: move prediction toward target
        # Using MSE-style update: gradient of (target - pred)^2
        loss = (prev_value - target.detach()).pow(2).mean()
        loss.backward()

        self.optimizer.step()

        return td_error

    def train_from_trajectory(
        self,
        states: List['torch.Tensor'],
        final_value: float,
    ) -> Dict[str, float]:
        """
        Train from a complete trajectory (offline).

        Alternative to online training: process entire episode at once.
        Can use accumulated eligibility traces for credit assignment.

        Args:
            states: List of encoded states from the episode.
            final_value: The final outcome (1.0 win, 0.0 loss).

        Returns:
            Training statistics.
        """
        import torch

        if not states:
            return {'td_error': 0.0, 'num_states': 0}

        self.network.train()
        total_loss = 0.0
        num_updates = 0

        # Forward pass: get all value predictions
        values = []
        for state in states:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            values.append(self.network(state))

        # Backward TD updates with eligibility traces
        # Start from end and work backward
        traces = [
            torch.zeros_like(p) for p in self.network.parameters()
        ]

        # Target for last state is the final value
        targets = [None] * len(values)
        targets[-1] = torch.tensor(
            [[final_value]],
            dtype=values[-1].dtype,
            device=values[-1].device,
        )

        # Propagate targets backward
        for i in range(len(values) - 2, -1, -1):
            targets[i] = self.gamma * values[i + 1].detach()

        # Compute total loss
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)

        for i, (value, target) in enumerate(zip(values, targets)):
            # Weight earlier states less (implicit trace decay)
            weight = self.lambda_ ** (len(values) - 1 - i)
            step_loss = weight * (value - target).pow(2).mean()
            loss = loss + step_loss

        loss.backward()
        self.optimizer.step()

        self.episodes_trained += 1

        return {
            'loss': loss.item(),
            'td_error': loss.item() / len(states),
            'num_states': len(states),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'episodes_trained': self.episodes_trained,
            'avg_td_error': (
                self.total_td_errors / self.episodes_trained
                if self.episodes_trained > 0 else 0.0
            ),
            'alpha': self.alpha,
            'lambda': self.lambda_,
        }

    def set_learning_rate(self, alpha: float) -> None:
        """Update the learning rate."""
        self.alpha = alpha
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = alpha

    def anneal_learning_rate(
        self,
        initial_alpha: float,
        final_alpha: float,
        progress: float,
    ) -> float:
        """
        Anneal learning rate based on training progress.

        Args:
            initial_alpha: Starting learning rate.
            final_alpha: Ending learning rate.
            progress: Training progress (0.0 to 1.0).

        Returns:
            The new learning rate.
        """
        new_alpha = initial_alpha + (final_alpha - initial_alpha) * progress
        self.set_learning_rate(new_alpha)
        return new_alpha


class TDLambdaWithTraces(TDLearner):
    """
    TD(λ) learner with explicit eligibility traces.

    This variant maintains explicit eligibility traces for each
    network parameter, providing more faithful implementation of
    the original TD(λ) algorithm.

    Eligibility traces remember which weights were responsible for
    recent predictions and give them more credit/blame for TD errors.
    """

    def __init__(
        self,
        network: 'nn.Module',
        alpha: float = 0.01,
        lambda_: float = 0.7,
        gamma: float = 1.0,
        replacing_traces: bool = True,
    ):
        """
        Initialize with eligibility traces.

        Args:
            network: Neural network to train.
            alpha: Learning rate.
            lambda_: Trace decay parameter.
            gamma: Discount factor.
            replacing_traces: Use replacing traces (vs accumulating).
                            Replacing traces are more stable.
        """
        super().__init__(network, alpha, lambda_, gamma)
        self.replacing_traces = replacing_traces
        self._init_traces()

    def _init_traces(self) -> None:
        """Initialize eligibility traces to zero."""
        import torch
        self._traces = {
            name: torch.zeros_like(param)
            for name, param in self.network.named_parameters()
        }

    def new_episode(self) -> None:
        """Reset traces at episode start."""
        super().new_episode()
        self._init_traces()

    def observe_with_traces(
        self,
        features: 'torch.Tensor',
    ) -> Tuple[float, float]:
        """
        Observe state and update with eligibility traces.

        Args:
            features: Encoded game state.

        Returns:
            Tuple of (value prediction, td_error or 0.0 if first state).
        """
        import torch

        self.network.train()

        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Get value with gradients
        value = self.network(features)

        # Store for later
        self._states.append(features.detach())
        self._values.append(value)

        td_error = 0.0

        if len(self._values) > 1:
            # Compute TD error
            prev_value = self._values[-2]
            curr_value = value.detach()

            td_error = (self.gamma * curr_value - prev_value).item()

            # Update traces: e = γλe + ∇V(s)
            # First, get gradients of previous value prediction
            self.optimizer.zero_grad()
            prev_value.backward(retain_graph=True)

            # Update traces and apply TD update
            with torch.no_grad():
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        # Decay old trace and add new gradient
                        self._traces[name] = (
                            self.gamma * self.lambda_ * self._traces[name]
                        )

                        if self.replacing_traces:
                            # Replacing traces: set to gradient magnitude
                            self._traces[name] = torch.max(
                                self._traces[name].abs(),
                                param.grad.abs(),
                            ) * param.grad.sign()
                        else:
                            # Accumulating traces: add gradient
                            self._traces[name] = (
                                self._traces[name] + param.grad
                            )

                        # Apply TD update: Δw = α * δ * e
                        param.data += self.alpha * td_error * self._traces[name]

            self._episode_td_errors.append(td_error)

        return value.item(), td_error

    def end_episode_with_traces(
        self,
        final_value: float,
    ) -> Dict[str, float]:
        """
        End episode and apply final TD update with traces.

        Args:
            final_value: Terminal reward (1.0 win, 0.0 loss).

        Returns:
            Episode statistics.
        """
        import torch

        if not self._values:
            return {'td_error': 0.0, 'num_states': 0}

        # Final TD update
        prev_value = self._values[-1]
        final_tensor = torch.tensor([[final_value]], dtype=prev_value.dtype)
        td_error = (final_tensor - prev_value).item()

        # Get gradients and apply final update
        self.optimizer.zero_grad()
        prev_value.backward()

        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    # Final trace update
                    self._traces[name] = (
                        self.gamma * self.lambda_ * self._traces[name] +
                        param.grad
                    )
                    # Apply update
                    param.data += self.alpha * td_error * self._traces[name]

        self._episode_td_errors.append(td_error)

        # Statistics
        self.episodes_trained += 1
        avg_td_error = (
            sum(abs(e) for e in self._episode_td_errors) /
            len(self._episode_td_errors)
        )

        return {
            'td_error': avg_td_error,
            'num_states': len(self._states),
            'final_value': final_value,
        }
