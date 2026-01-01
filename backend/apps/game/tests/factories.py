"""
Factory Boy factories for the game app.

These factories create test instances of Game, Move, and GameInvite
models with sensible defaults for testing.
"""
import factory
from apps.accounts.tests.factories import UserFactory
from apps.game.models import Game, Move, GameInvite


class GameFactory(factory.django.DjangoModelFactory):
    """Factory for creating Game instances."""

    class Meta:
        model = Game

    white_player = factory.SubFactory(UserFactory)
    status = Game.Status.WAITING
    board_state = factory.LazyFunction(Game.get_initial_board)
    current_turn = 'white'
    cube_value = 1
    cube_owner = 'center'
    double_offered = False


class ActiveGameFactory(GameFactory):
    """Factory for active games with both players."""

    black_player = factory.SubFactory(UserFactory)
    status = Game.Status.ACTIVE


class CompletedGameFactory(GameFactory):
    """Factory for completed games."""

    black_player = factory.SubFactory(UserFactory)
    status = Game.Status.COMPLETED
    winner = factory.LazyAttribute(lambda obj: obj.white_player)
    win_type = Game.WinType.NORMAL
    points_won = 1


class MoveFactory(factory.django.DjangoModelFactory):
    """Factory for creating Move instances."""

    class Meta:
        model = Move

    game = factory.SubFactory(ActiveGameFactory)
    player = factory.LazyAttribute(lambda obj: obj.game.white_player)
    move_number = factory.Sequence(lambda n: n + 1)
    move_type = Move.MoveType.ROLL
    dice_values = [3, 5]
    board_state_after = factory.LazyAttribute(lambda obj: obj.game.board_state)


class GameInviteFactory(factory.django.DjangoModelFactory):
    """Factory for creating GameInvite instances."""

    class Meta:
        model = GameInvite

    from_user = factory.SubFactory(UserFactory)
    to_user = factory.SubFactory(UserFactory)
    status = GameInvite.Status.PENDING
