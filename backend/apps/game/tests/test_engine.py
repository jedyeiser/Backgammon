"""
Tests for the game engine.

Tests the GameEngine class including:
- Dice rolling
- Legal move calculation
- Move execution
- Bearing off
- Hitting blots
- Doubling cube
- Win conditions
"""
import pytest
from unittest.mock import patch
from apps.game.services.game_engine import GameEngine
from apps.game.models import Game, Move
from apps.accounts.tests.factories import UserFactory
from .factories import ActiveGameFactory


@pytest.mark.django_db
class TestDiceRolling:
    """Tests for dice rolling functionality."""

    def test_roll_dice_returns_two_values(self):
        """Test that rolling returns two dice values."""
        game = ActiveGameFactory()
        engine = GameEngine(game)

        with patch('random.randint', side_effect=[3, 5]):
            result = engine.roll_dice()

        assert result['dice'] == [3, 5]
        assert len(result['moves_remaining']) == 2

    def test_roll_dice_updates_game_state(self):
        """Test that dice roll updates game state."""
        game = ActiveGameFactory()
        engine = GameEngine(game)

        with patch('random.randint', side_effect=[4, 2]):
            engine.roll_dice()

        assert game.dice == [4, 2]
        assert game.moves_remaining == [4, 2]

    def test_roll_doubles_gives_four_moves(self):
        """Test that rolling doubles gives four moves."""
        game = ActiveGameFactory()
        engine = GameEngine(game)

        with patch('random.randint', side_effect=[6, 6]):
            result = engine.roll_dice()

        assert result['dice'] == [6, 6]
        assert result['moves_remaining'] == [6, 6, 6, 6]

    def test_roll_dice_creates_move_record(self):
        """Test that dice roll creates a Move record."""
        game = ActiveGameFactory()
        engine = GameEngine(game)
        initial_count = Move.objects.filter(game=game).count()

        with patch('random.randint', side_effect=[3, 4]):
            engine.roll_dice()

        assert Move.objects.filter(game=game).count() == initial_count + 1
        move = Move.objects.filter(game=game).last()
        assert move.move_type == Move.MoveType.ROLL
        assert move.dice_values == [3, 4]


@pytest.mark.django_db
class TestLegalMoves:
    """Tests for legal move calculation."""

    def test_get_legal_moves_initial_position(self):
        """Test legal moves from initial position."""
        game = ActiveGameFactory()
        game.dice = [3, 5]
        game.moves_remaining = [3, 5]
        game.save()

        engine = GameEngine(game)
        moves = engine.get_legal_moves()

        # Should have legal moves from white's starting positions
        assert len(moves) > 0

    def test_no_legal_moves_without_dice(self):
        """Test no legal moves when dice haven't been rolled."""
        game = ActiveGameFactory()
        game.dice = []
        game.moves_remaining = []
        game.save()

        engine = GameEngine(game)
        moves = engine.get_legal_moves()

        assert moves == []

    def test_bar_entry_required_first(self):
        """Test that bar entry is required before other moves."""
        game = ActiveGameFactory()
        game.board_state['bar']['white'] = 1
        game.dice = [3, 5]
        game.moves_remaining = [3, 5]
        game.save()

        engine = GameEngine(game)
        moves = engine.get_legal_moves()

        # All moves should be from bar (point 0)
        for from_point, to_point in moves:
            assert from_point == 0


@pytest.mark.django_db
class TestMoveExecution:
    """Tests for move execution."""

    def test_make_moves_wrong_turn(self):
        """Test making moves when it's not your turn fails."""
        game = ActiveGameFactory()
        game.current_turn = 'white'
        game.dice = [3, 5]
        game.moves_remaining = [3, 5]
        game.save()

        engine = GameEngine(game)

        with pytest.raises(ValueError, match="not your turn"):
            engine.make_moves(game.black_player, [[19, 24]])

    def test_move_switches_turn_when_complete(self):
        """Test that turn switches after all moves are used."""
        game = ActiveGameFactory()
        game.current_turn = 'white'
        game.dice = [1]
        game.moves_remaining = [1]
        game.board_state = {
            'points': {'19': 5},
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
        }
        game.save()

        engine = GameEngine(game)
        engine.make_moves(game.white_player, [[19, 20]])

        assert game.current_turn == 'black'
        assert game.dice == []
        assert game.moves_remaining == []


@pytest.mark.django_db
class TestBearingOff:
    """Tests for bearing off mechanics."""

    def test_can_bear_off_all_in_home(self):
        """Test can bear off when all checkers are in home board."""
        game = ActiveGameFactory()
        game.board_state = {
            'points': {
                '19': 3, '20': 4, '21': 3, '22': 2, '23': 2, '24': 1
            },
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
        }
        game.save()

        engine = GameEngine(game)
        assert engine._can_bear_off('white') is True

    def test_cannot_bear_off_checker_outside(self):
        """Test cannot bear off with checker outside home board."""
        game = ActiveGameFactory()
        game.board_state = {
            'points': {
                '12': 5,  # Outside home board
                '19': 3, '20': 4, '21': 3
            },
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
        }
        game.save()

        engine = GameEngine(game)
        assert engine._can_bear_off('white') is False

    def test_cannot_bear_off_with_bar_checker(self):
        """Test cannot bear off with checker on bar."""
        game = ActiveGameFactory()
        game.board_state = {
            'points': {'19': 3, '20': 4, '21': 3, '22': 2, '23': 2},
            'bar': {'white': 1, 'black': 0},
            'home': {'white': 0, 'black': 0},
        }
        game.save()

        engine = GameEngine(game)
        assert engine._can_bear_off('white') is False


@pytest.mark.django_db
class TestHittingBlots:
    """Tests for hitting opponent blots."""

    def test_hit_blot_sends_to_bar(self):
        """Test hitting a blot sends opponent to bar."""
        game = ActiveGameFactory()
        game.board_state = {
            'points': {
                '19': 5,
                '20': -1,  # Black blot
            },
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
        }
        game.current_turn = 'white'
        game.dice = [1]
        game.moves_remaining = [1]
        game.save()

        engine = GameEngine(game)
        engine._execute_move(19, 20, 'white')

        assert game.board_state['points']['20'] == 1  # White now there
        assert game.board_state['bar']['black'] == 1  # Black on bar


@pytest.mark.django_db
class TestDoublingCube:
    """Tests for doubling cube functionality."""

    def test_offer_double(self):
        """Test offering a double."""
        game = ActiveGameFactory()
        engine = GameEngine(game)

        result = engine.offer_double(game.white_player)

        assert result['double_offered'] is True
        assert result['cube_value'] == 2
        assert game.double_offered is True

    def test_offer_double_when_already_offered(self):
        """Test cannot offer double when one is pending."""
        game = ActiveGameFactory()
        game.double_offered = True
        game.save()

        engine = GameEngine(game)

        with pytest.raises(ValueError, match="already been offered"):
            engine.offer_double(game.white_player)

    def test_accept_double(self):
        """Test accepting a double."""
        game = ActiveGameFactory()
        game.double_offered = True
        game.current_turn = 'white'
        game.save()

        engine = GameEngine(game)
        result = engine.accept_double(game.black_player)

        assert result['accepted'] is True
        assert game.cube_value == 2
        assert game.cube_owner == 'black'
        assert game.double_offered is False

    def test_reject_double_ends_game(self):
        """Test rejecting a double ends the game."""
        game = ActiveGameFactory()
        game.double_offered = True
        game.current_turn = 'white'
        game.save()

        engine = GameEngine(game)
        result = engine.reject_double(game.black_player)

        assert result['rejected'] is True
        assert game.status == Game.Status.COMPLETED
        assert game.winner == game.white_player


@pytest.mark.django_db
class TestResignation:
    """Tests for resignation."""

    def test_resign_ends_game(self):
        """Test resignation ends the game."""
        game = ActiveGameFactory()
        engine = GameEngine(game)

        result = engine.resign(game.white_player)

        assert result['resigned'] is True
        assert result['winner'] == 'black'
        assert game.status == Game.Status.COMPLETED
        assert game.winner == game.black_player
        assert game.win_type == Game.WinType.RESIGN


@pytest.mark.django_db
class TestWinConditions:
    """Tests for win condition checking."""

    def test_check_winner_white(self):
        """Test detecting white winner."""
        game = ActiveGameFactory()
        game.board_state = {
            'points': {},
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 15, 'black': 0},
        }
        game.save()

        engine = GameEngine(game)
        winner = engine._check_winner()

        assert winner == 'white'

    def test_check_winner_black(self):
        """Test detecting black winner."""
        game = ActiveGameFactory()
        game.board_state = {
            'points': {},
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 15},
        }
        game.save()

        engine = GameEngine(game)
        winner = engine._check_winner()

        assert winner == 'black'

    def test_check_winner_none(self):
        """Test no winner when game in progress."""
        game = ActiveGameFactory()
        engine = GameEngine(game)
        winner = engine._check_winner()

        assert winner is None

    def test_determine_win_type_gammon(self):
        """Test gammon detection (opponent has no checkers off)."""
        game = ActiveGameFactory()
        game.board_state = {
            'points': {'1': -5, '6': -5, '12': -5},  # 15 black checkers
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 15, 'black': 0},
        }
        game.save()

        engine = GameEngine(game)
        win_type = engine._determine_win_type('white')

        assert win_type == Game.WinType.GAMMON
