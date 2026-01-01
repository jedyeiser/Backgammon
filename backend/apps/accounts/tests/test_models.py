"""
Tests for accounts models.

Tests the User model functionality including:
- User creation
- Win rate calculation
- Stats updates
- ELO rating updates
"""
import pytest
from .factories import UserFactory


@pytest.mark.django_db
class TestUserModel:
    """Tests for the User model."""

    def test_create_user(self):
        """Test user creation with default values."""
        user = UserFactory()
        assert user.pk is not None
        assert user.elo_rating == 1200
        assert user.games_played == 0

    def test_str_representation(self):
        """Test user string representation."""
        user = UserFactory(username='testplayer')
        assert str(user) == 'testplayer'

    def test_win_rate_no_games(self):
        """Test win rate returns 0 when no games played."""
        user = UserFactory()
        assert user.win_rate == 0.0

    def test_win_rate_with_games(self):
        """Test win rate calculation with games played."""
        user = UserFactory(games_played=10, games_won=7, games_lost=3)
        assert user.win_rate == 70.0

    def test_win_rate_all_wins(self):
        """Test win rate with 100% wins."""
        user = UserFactory(games_played=5, games_won=5, games_lost=0)
        assert user.win_rate == 100.0

    def test_update_stats_win(self):
        """Test stats update after a win."""
        user = UserFactory()
        user.update_stats(won=True)

        assert user.games_played == 1
        assert user.games_won == 1
        assert user.games_lost == 0

    def test_update_stats_loss(self):
        """Test stats update after a loss."""
        user = UserFactory()
        user.update_stats(won=False)

        assert user.games_played == 1
        assert user.games_won == 0
        assert user.games_lost == 1

    def test_update_stats_multiple(self):
        """Test multiple stats updates."""
        user = UserFactory()
        user.update_stats(won=True)
        user.update_stats(won=True)
        user.update_stats(won=False)

        assert user.games_played == 3
        assert user.games_won == 2
        assert user.games_lost == 1

    def test_update_elo_increase(self):
        """Test ELO update with new high score."""
        user = UserFactory(elo_rating=1200, highest_elo=1200)
        user.update_elo(1300)

        assert user.elo_rating == 1300
        assert user.highest_elo == 1300

    def test_update_elo_decrease(self):
        """Test ELO update with decrease (highest unchanged)."""
        user = UserFactory(elo_rating=1300, highest_elo=1350)
        user.update_elo(1250)

        assert user.elo_rating == 1250
        assert user.highest_elo == 1350  # Unchanged

    def test_update_elo_new_high(self):
        """Test ELO update surpassing previous highest."""
        user = UserFactory(elo_rating=1300, highest_elo=1350)
        user.update_elo(1400)

        assert user.elo_rating == 1400
        assert user.highest_elo == 1400

    def test_default_preferred_color(self):
        """Test default preferred color is random."""
        user = UserFactory()
        assert user.preferred_color == 'random'
