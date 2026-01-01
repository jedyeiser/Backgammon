"""
Tests for game models.

Tests the Game, Move, and GameInvite models including:
- Game creation and initialization
- Board state management
- Game lifecycle (start, complete)
- Win type determination
- Invite acceptance/decline
"""
import pytest
from apps.accounts.tests.factories import UserFactory
from apps.game.models import Game
from .factories import GameFactory, ActiveGameFactory, GameInviteFactory


@pytest.mark.django_db
class TestGameModel:
    """Tests for the Game model."""

    def test_create_game(self):
        """Test game creation with default values."""
        game = GameFactory()
        assert game.pk is not None
        assert game.status == Game.Status.WAITING
        assert game.current_turn == 'white'
        assert game.cube_value == 1

    def test_str_representation(self):
        """Test game string representation."""
        white = UserFactory(username='alice')
        game = GameFactory(white_player=white)
        assert 'alice' in str(game)
        assert 'Waiting' in str(game)

    def test_initial_board_state(self):
        """Test initial board setup is correct."""
        board = Game.get_initial_board()

        # White positions
        assert board['points']['1'] == 2
        assert board['points']['12'] == 5
        assert board['points']['17'] == 3
        assert board['points']['19'] == 5

        # Black positions (negative)
        assert board['points']['6'] == -5
        assert board['points']['8'] == -3
        assert board['points']['13'] == -5
        assert board['points']['24'] == -2

        # Bar and home empty
        assert board['bar']['white'] == 0
        assert board['bar']['black'] == 0
        assert board['home']['white'] == 0
        assert board['home']['black'] == 0

    def test_initialize_board(self):
        """Test board initialization sets correct state."""
        game = GameFactory(board_state={})
        game.initialize_board()

        assert game.board_state['points']['1'] == 2
        assert game.board_state['bar']['white'] == 0

    def test_start_game(self):
        """Test starting a game with both players."""
        white = UserFactory()
        black = UserFactory()
        game = GameFactory(white_player=white)
        game.black_player = black
        game.start_game()

        assert game.status == Game.Status.ACTIVE
        assert game.started_at is not None
        assert game.board_state is not None

    def test_start_game_requires_both_players(self):
        """Test game doesn't start with only one player."""
        game = GameFactory()
        game.start_game()

        # Status should remain WAITING
        assert game.status == Game.Status.WAITING

    def test_complete_game_normal(self):
        """Test completing a game with normal win."""
        game = ActiveGameFactory()
        game.complete_game(game.white_player, Game.WinType.NORMAL)

        assert game.status == Game.Status.COMPLETED
        assert game.winner == game.white_player
        assert game.win_type == Game.WinType.NORMAL
        assert game.points_won == 1
        assert game.completed_at is not None

    def test_complete_game_gammon(self):
        """Test completing a game with gammon (2x points)."""
        game = ActiveGameFactory(cube_value=2)
        game.complete_game(game.white_player, Game.WinType.GAMMON)

        assert game.points_won == 4  # 2 (cube) * 2 (gammon)

    def test_complete_game_backgammon(self):
        """Test completing a game with backgammon (3x points)."""
        game = ActiveGameFactory(cube_value=2)
        game.complete_game(game.black_player, Game.WinType.BACKGAMMON)

        assert game.points_won == 6  # 2 (cube) * 3 (backgammon)
        assert game.winner == game.black_player

    def test_get_player_color_white(self):
        """Test getting player color for white player."""
        game = ActiveGameFactory()
        assert game.get_player_color(game.white_player) == 'white'

    def test_get_player_color_black(self):
        """Test getting player color for black player."""
        game = ActiveGameFactory()
        assert game.get_player_color(game.black_player) == 'black'

    def test_get_player_color_not_player(self):
        """Test getting player color for non-player."""
        game = ActiveGameFactory()
        other_user = UserFactory()
        assert game.get_player_color(other_user) is None

    def test_get_opponent_white(self):
        """Test getting opponent for white player."""
        game = ActiveGameFactory()
        assert game.get_opponent(game.white_player) == game.black_player

    def test_get_opponent_black(self):
        """Test getting opponent for black player."""
        game = ActiveGameFactory()
        assert game.get_opponent(game.black_player) == game.white_player

    def test_get_opponent_not_player(self):
        """Test getting opponent for non-player."""
        game = ActiveGameFactory()
        other_user = UserFactory()
        assert game.get_opponent(other_user) is None

    def test_version_increments_on_save(self):
        """Test that version increments on each save."""
        game = GameFactory()
        initial_version = game.version

        game.current_turn = 'black'
        game.save()

        assert game.version == initial_version + 1


@pytest.mark.django_db
class TestGameInviteModel:
    """Tests for the GameInvite model."""

    def test_create_invite(self):
        """Test invite creation."""
        invite = GameInviteFactory()
        assert invite.pk is not None
        assert invite.status == 'pending'
        assert invite.from_user != invite.to_user

    def test_str_representation(self):
        """Test invite string representation."""
        from_user = UserFactory(username='alice')
        to_user = UserFactory(username='bob')
        invite = GameInviteFactory(from_user=from_user, to_user=to_user)
        assert 'alice' in str(invite)
        assert 'bob' in str(invite)

    def test_accept_invite(self):
        """Test accepting an invite creates a game."""
        invite = GameInviteFactory()
        game = invite.accept()

        assert invite.status == 'accepted'
        assert invite.game == game
        assert invite.responded_at is not None
        assert game.white_player == invite.from_user
        assert game.black_player == invite.to_user
        assert game.status == Game.Status.ACTIVE

    def test_decline_invite(self):
        """Test declining an invite."""
        invite = GameInviteFactory()
        invite.decline()

        assert invite.status == 'declined'
        assert invite.responded_at is not None
        assert invite.game is None
