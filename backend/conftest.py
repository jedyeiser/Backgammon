"""
Pytest configuration and shared fixtures for the Backgammon project.

This module provides fixtures for:
- Database access
- API client setup (authenticated and unauthenticated)
- Common test data factories
- Game state fixtures
"""
import pytest
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken


@pytest.fixture
def api_client():
    """Return an unauthenticated API client."""
    return APIClient()


@pytest.fixture
def user(db):
    """Create and return a test user."""
    from apps.accounts.tests.factories import UserFactory
    return UserFactory()


@pytest.fixture
def authenticated_client(user):
    """Return an API client authenticated as the test user."""
    client = APIClient()
    refresh = RefreshToken.for_user(user)
    client.credentials(HTTP_AUTHORIZATION=f'Bearer {refresh.access_token}')
    return client


@pytest.fixture
def user_with_client(db):
    """Create a user and return both user and authenticated client."""
    from apps.accounts.tests.factories import UserFactory
    user = UserFactory()
    client = APIClient()
    refresh = RefreshToken.for_user(user)
    client.credentials(HTTP_AUTHORIZATION=f'Bearer {refresh.access_token}')
    return user, client


@pytest.fixture
def two_players(db):
    """Create two test users for game testing."""
    from apps.accounts.tests.factories import UserFactory
    return UserFactory(), UserFactory()


@pytest.fixture
def active_game(two_players):
    """Create an active game with two players."""
    from apps.game.tests.factories import GameFactory
    white, black = two_players
    game = GameFactory(white_player=white, black_player=black)
    game.start_game()
    return game


@pytest.fixture
def waiting_game(user):
    """Create a game waiting for a second player."""
    from apps.game.tests.factories import GameFactory
    return GameFactory(white_player=user)
