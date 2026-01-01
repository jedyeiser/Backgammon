"""
Factory Boy factories for the accounts app.

These factories create test instances of User model with
sensible defaults for testing.
"""
import factory
from django.contrib.auth import get_user_model

User = get_user_model()


class UserFactory(factory.django.DjangoModelFactory):
    """Factory for creating User instances."""

    class Meta:
        model = User

    username = factory.Sequence(lambda n: f'testuser_{n}')
    email = factory.LazyAttribute(lambda obj: f'{obj.username}@example.com')
    password = factory.PostGenerationMethodCall('set_password', 'testpass123')
    elo_rating = 1200
    highest_elo = 1200
    games_played = 0
    games_won = 0
    games_lost = 0


class ExperiencedUserFactory(UserFactory):
    """Factory for users with game history."""

    games_played = 50
    games_won = 30
    games_lost = 20
    elo_rating = 1450
    highest_elo = 1500


class LeaderboardUserFactory(UserFactory):
    """Factory for users eligible for leaderboard (10+ games)."""

    games_played = 15
    games_won = 10
    games_lost = 5
    elo_rating = factory.Sequence(lambda n: 1500 + n * 10)
