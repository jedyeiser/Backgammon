"""
Tests for match runner and ELO rating system.

Tests the MatchRunner, EloCalculator, and related utilities for:
- ELO calculation correctness
- K-factor logic
- Match execution
- Rating updates
"""
import pytest
import math

from apps.ai.matches.elo import (
    EloCalculator,
    EloResult,
    rating_probability,
    rating_difference_to_expected,
    expected_score_to_rating_difference,
)
from apps.ai.matches.runner import MatchRunner, GameResult, MatchResult


class TestEloCalculator:
    """Tests for EloCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create ELO calculator with default settings."""
        return EloCalculator()

    @pytest.fixture
    def fixed_k_calculator(self):
        """Create ELO calculator with fixed K-factor."""
        return EloCalculator(k_factor=32, dynamic_k=False)

    def test_expected_score_equal_ratings(self, calculator):
        """Test expected score for equal ratings."""
        expected = calculator.expected_score(1500, 1500)
        assert expected == pytest.approx(0.5)

    def test_expected_score_higher_rating(self, calculator):
        """Test expected score when player A is higher rated."""
        expected = calculator.expected_score(1600, 1400)
        assert expected > 0.5
        assert expected < 1.0

    def test_expected_score_lower_rating(self, calculator):
        """Test expected score when player A is lower rated."""
        expected = calculator.expected_score(1400, 1600)
        assert expected < 0.5
        assert expected > 0.0

    def test_expected_score_400_difference(self, calculator):
        """Test expected score with 400 point difference."""
        expected = calculator.expected_score(1800, 1400)
        # 400 points difference should give ~91% expected
        assert expected == pytest.approx(0.909, abs=0.01)

    def test_expected_scores_sum_to_one(self, calculator):
        """Test that expected scores for both players sum to 1."""
        expected_a = calculator.expected_score(1500, 1400)
        expected_b = calculator.expected_score(1400, 1500)
        assert expected_a + expected_b == pytest.approx(1.0)

    def test_get_k_factor_provisional(self, calculator):
        """Test K-factor for new players."""
        k = calculator.get_k_factor(1200, games_played=10)
        assert k == EloCalculator.K_PROVISIONAL

    def test_get_k_factor_standard(self, calculator):
        """Test K-factor for standard players."""
        k = calculator.get_k_factor(1600, games_played=100)
        assert k == EloCalculator.K_STANDARD

    def test_get_k_factor_established(self, calculator):
        """Test K-factor for established players."""
        k = calculator.get_k_factor(2100, games_played=100)
        assert k == EloCalculator.K_ESTABLISHED

    def test_get_k_factor_master(self, calculator):
        """Test K-factor for master players."""
        k = calculator.get_k_factor(2500, games_played=100)
        assert k == EloCalculator.K_MASTER

    def test_get_k_factor_fixed(self, fixed_k_calculator):
        """Test fixed K-factor ignores rating and games."""
        k = fixed_k_calculator.get_k_factor(2500, games_played=10)
        assert k == 32

    def test_update_ratings_win(self, calculator):
        """Test rating update after a win."""
        result_a, result_b = calculator.update_ratings(
            rating_a=1500,
            rating_b=1500,
            score_a=1.0,  # A wins
        )

        assert result_a.new_rating > result_a.old_rating
        assert result_b.new_rating < result_b.old_rating
        assert result_a.change > 0
        assert result_b.change < 0

    def test_update_ratings_loss(self, calculator):
        """Test rating update after a loss."""
        result_a, result_b = calculator.update_ratings(
            rating_a=1500,
            rating_b=1500,
            score_a=0.0,  # A loses
        )

        assert result_a.new_rating < result_a.old_rating
        assert result_b.new_rating > result_b.old_rating

    def test_update_ratings_draw(self, calculator):
        """Test rating update after a draw with equal ratings."""
        result_a, result_b = calculator.update_ratings(
            rating_a=1500,
            rating_b=1500,
            score_a=0.5,  # Draw
        )

        # No change when equal ratings draw
        assert result_a.change == 0
        assert result_b.change == 0

    def test_update_ratings_upset(self, calculator):
        """Test large rating gain for upset."""
        result_a, result_b = calculator.update_ratings(
            rating_a=1300,
            rating_b=1700,
            score_a=1.0,  # Lower rated player wins
        )

        # Large gain for lower rated player
        assert result_a.change > 20
        # Large loss for higher rated player
        assert result_b.change < -20

    def test_update_ratings_floor(self, calculator):
        """Test rating floor is respected."""
        result_a, result_b = calculator.update_ratings(
            rating_a=100,  # Near floor
            rating_b=200,
            score_a=0.0,  # A loses
        )

        assert result_a.new_rating >= calculator.floor_rating

    def test_update_ratings_returns_elo_result(self, calculator):
        """Test that update returns EloResult objects."""
        result_a, result_b = calculator.update_ratings(1500, 1500, 1.0)

        assert isinstance(result_a, EloResult)
        assert isinstance(result_b, EloResult)
        assert result_a.expected_score is not None
        assert result_a.actual_score == 1.0
        assert result_b.actual_score == 0.0

    def test_update_from_match(self, calculator):
        """Test rating update from match result."""
        new_a, new_b = calculator.update_from_match(
            rating_a=1500,
            rating_b=1500,
            wins_a=3,
            wins_b=1,
        )

        assert new_a > 1500  # A won majority
        assert new_b < 1500

    def test_update_from_match_with_draws(self, calculator):
        """Test rating update with draws."""
        new_a, new_b = calculator.update_from_match(
            rating_a=1500,
            rating_b=1500,
            wins_a=1,
            wins_b=1,
            draws=2,
        )

        # Equal performance should yield little change
        assert abs(new_a - 1500) <= 5
        assert abs(new_b - 1500) <= 5

    def test_update_from_match_no_games(self, calculator):
        """Test rating update with no games."""
        new_a, new_b = calculator.update_from_match(
            rating_a=1500,
            rating_b=1500,
            wins_a=0,
            wins_b=0,
            draws=0,
        )

        assert new_a == 1500
        assert new_b == 1500


class TestEloUtilityFunctions:
    """Tests for ELO utility functions."""

    def test_rating_probability_equal(self):
        """Test probability distribution for equal ratings."""
        win_a, draw, win_b = rating_probability(1500, 1500)

        assert win_a == pytest.approx(win_b, abs=0.01)
        assert draw == pytest.approx(0.1)  # 10% draw rate

    def test_rating_probability_stronger(self):
        """Test probability when A is stronger."""
        win_a, draw, win_b = rating_probability(1700, 1300)

        assert win_a > win_b
        assert win_a + draw + win_b == pytest.approx(1.0)

    def test_rating_difference_to_expected_positive(self):
        """Test expected score from positive difference."""
        expected = rating_difference_to_expected(200)
        assert expected > 0.5

    def test_rating_difference_to_expected_negative(self):
        """Test expected score from negative difference."""
        expected = rating_difference_to_expected(-200)
        assert expected < 0.5

    def test_rating_difference_to_expected_zero(self):
        """Test expected score from zero difference."""
        expected = rating_difference_to_expected(0)
        assert expected == pytest.approx(0.5)

    def test_expected_to_difference_round_trip(self):
        """Test round-trip conversion expected <-> difference."""
        for diff in [-400, -200, 0, 200, 400]:
            expected = rating_difference_to_expected(diff)
            recovered = expected_score_to_rating_difference(expected)
            assert recovered == pytest.approx(diff, abs=1)

    def test_expected_to_difference_boundary(self):
        """Test boundary cases for expected score."""
        # Extreme values should return 0
        assert expected_score_to_rating_difference(0.0) == 0
        assert expected_score_to_rating_difference(1.0) == 0


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_game_result_defaults(self):
        """Test default values."""
        result = GameResult()

        assert result.winner is None
        assert result.loser is None
        assert result.is_draw is False
        assert result.num_moves == 0
        assert result.game_history == []

    def test_game_result_with_winner(self):
        """Test game result with winner."""
        result = GameResult(
            winner='player_1',
            loser='player_2',
            num_moves=42,
        )

        assert result.winner == 'player_1'
        assert result.loser == 'player_2'
        assert result.num_moves == 42


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_match_result_defaults(self):
        """Test default values."""
        result = MatchResult(
            player_a_id='player_a',
            player_b_id='player_b',
        )

        assert result.player_a_wins == 0
        assert result.player_b_wins == 0
        assert result.draws == 0
        assert result.total_games == 0

    def test_match_result_total_games(self):
        """Test total games calculation."""
        result = MatchResult(
            player_a_id='a',
            player_b_id='b',
            player_a_wins=3,
            player_b_wins=2,
            draws=1,
        )

        assert result.total_games == 6

    def test_match_result_scores(self):
        """Test score calculations."""
        result = MatchResult(
            player_a_id='a',
            player_b_id='b',
            player_a_wins=3,
            player_b_wins=1,
            draws=2,
        )

        assert result.player_a_score == 4.0  # 3 + 0.5*2
        assert result.player_b_score == 2.0  # 1 + 0.5*2


class TestMatchRunner:
    """Tests for MatchRunner."""

    @pytest.fixture
    def runner(self):
        """Create match runner."""
        return MatchRunner(
            game_type='backgammon',
            update_ratings=False,
            max_moves_per_game=100,  # Low for testing
        )

    def test_runner_init(self, runner):
        """Test runner initialization."""
        assert runner.game_type == 'backgammon'
        assert runner.update_ratings is False
        assert runner.max_moves_per_game == 100

    def test_runner_with_ratings(self):
        """Test runner with rating updates enabled."""
        runner = MatchRunner(update_ratings=True)
        assert runner.update_ratings is True
        assert runner.elo_calculator is not None

    def test_runner_unknown_game_type(self):
        """Test runner with unknown game type."""
        with pytest.raises(ValueError, match="Unknown game type"):
            MatchRunner(game_type='unknown_game')

    def test_runner_has_elo_calculator(self, runner):
        """Test runner has ELO calculator."""
        assert hasattr(runner, 'elo_calculator')
        assert isinstance(runner.elo_calculator, EloCalculator)
