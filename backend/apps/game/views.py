"""Views for the game app."""
from django.contrib.auth import get_user_model
from django.db.models import Q
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Game, Move, GameInvite
from .serializers import (
    GameListSerializer,
    GameDetailSerializer,
    GameCreateSerializer,
    GameJoinSerializer,
    MakeMoveSerializer,
    MoveSerializer,
    GameInviteSerializer,
    CreateInviteSerializer,
)
from .services.game_engine import GameEngine

User = get_user_model()


class GameListView(generics.ListAPIView):
    """List all games for the current user."""

    serializer_class = GameListSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return games where the user is a player."""
        user = self.request.user
        return Game.objects.filter(
            Q(white_player=user) | Q(black_player=user)
        ).select_related('white_player', 'black_player', 'winner')


class ActiveGamesView(generics.ListAPIView):
    """List active games for the current user."""

    serializer_class = GameListSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return active games where the user is a player."""
        user = self.request.user
        return Game.objects.filter(
            Q(white_player=user) | Q(black_player=user),
            status=Game.Status.ACTIVE
        ).select_related('white_player', 'black_player')


class OpenGamesView(generics.ListAPIView):
    """List games waiting for a second player."""

    serializer_class = GameListSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return games waiting for players, excluding user's own games."""
        user = self.request.user
        return Game.objects.filter(
            status=Game.Status.WAITING
        ).exclude(
            white_player=user
        ).select_related('white_player')


class GameDetailView(generics.RetrieveAPIView):
    """View detailed game state."""

    serializer_class = GameDetailSerializer
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = 'id'

    def get_queryset(self):
        """Return games where the user is a player."""
        user = self.request.user
        return Game.objects.filter(
            Q(white_player=user) | Q(black_player=user)
        ).select_related(
            'white_player', 'black_player', 'winner'
        ).prefetch_related('moves')


class GameCreateView(generics.CreateAPIView):
    """Create a new game."""

    serializer_class = GameCreateSerializer
    permission_classes = [permissions.IsAuthenticated]


class GameJoinView(APIView):
    """Join an existing game."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, game_id):
        """Join a game as the black player."""
        serializer = GameJoinSerializer(
            data={'game_id': game_id},
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)

        game = Game.objects.get(id=game_id)
        game.black_player = request.user
        game.start_game()

        return Response(
            GameDetailSerializer(game).data,
            status=status.HTTP_200_OK
        )


class MakeMoveView(APIView):
    """Make a move in a game."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, game_id):
        """Execute a move."""
        try:
            game = Game.objects.get(
                id=game_id,
                status=Game.Status.ACTIVE
            )
        except Game.DoesNotExist:
            return Response(
                {'error': 'Game not found or not active.'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Check if user is a player
        user = request.user
        if user not in [game.white_player, game.black_player]:
            return Response(
                {'error': 'You are not a player in this game.'},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = MakeMoveSerializer(
            data=request.data,
            context={'request': request, 'game': game}
        )
        serializer.is_valid(raise_exception=True)

        # Process the move using the game engine
        engine = GameEngine(game)
        move_type = serializer.validated_data['move_type']

        try:
            if move_type == Move.MoveType.ROLL:
                result = engine.roll_dice()
            elif move_type == Move.MoveType.MOVE:
                checker_moves = serializer.validated_data.get('checker_moves', [])
                result = engine.make_moves(user, checker_moves)
            elif move_type == Move.MoveType.DOUBLE:
                result = engine.offer_double(user)
            elif move_type == Move.MoveType.ACCEPT_DOUBLE:
                result = engine.accept_double(user)
            elif move_type == Move.MoveType.REJECT_DOUBLE:
                result = engine.reject_double(user)
            elif move_type == Move.MoveType.RESIGN:
                result = engine.resign(user)
            else:
                return Response(
                    {'error': 'Invalid move type.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Refresh game state
        game.refresh_from_db()

        return Response({
            'game': GameDetailSerializer(game).data,
            'result': result
        })


class GameMoveHistoryView(generics.ListAPIView):
    """View move history for a game."""

    serializer_class = MoveSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return moves for the specified game."""
        game_id = self.kwargs['game_id']
        user = self.request.user

        # Verify user is a player
        game = Game.objects.filter(
            id=game_id
        ).filter(
            Q(white_player=user) | Q(black_player=user)
        ).first()

        if not game:
            return Move.objects.none()

        return Move.objects.filter(game=game).select_related('player')


class ResignGameView(APIView):
    """Resign from a game."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, game_id):
        """Resign from the game."""
        try:
            game = Game.objects.get(id=game_id, status=Game.Status.ACTIVE)
        except Game.DoesNotExist:
            return Response(
                {'error': 'Game not found or not active.'},
                status=status.HTTP_404_NOT_FOUND
            )

        user = request.user
        if user not in [game.white_player, game.black_player]:
            return Response(
                {'error': 'You are not a player in this game.'},
                status=status.HTTP_403_FORBIDDEN
            )

        engine = GameEngine(game)
        engine.resign(user)

        return Response(
            GameDetailSerializer(game).data,
            status=status.HTTP_200_OK
        )


# Game Invite Views

class InviteListView(generics.ListAPIView):
    """List received invites."""

    serializer_class = GameInviteSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return pending invites for the user."""
        return GameInvite.objects.filter(
            to_user=self.request.user,
            status=GameInvite.Status.PENDING
        ).select_related('from_user', 'to_user')


class SentInvitesView(generics.ListAPIView):
    """List sent invites."""

    serializer_class = GameInviteSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return invites sent by the user."""
        return GameInvite.objects.filter(
            from_user=self.request.user
        ).select_related('from_user', 'to_user')


class CreateInviteView(APIView):
    """Create a game invite."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Send a game invite."""
        serializer = CreateInviteSerializer(
            data=request.data,
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)

        to_user = User.objects.get(
            username=serializer.validated_data['to_username']
        )

        invite = GameInvite.objects.create(
            from_user=request.user,
            to_user=to_user
        )

        return Response(
            GameInviteSerializer(invite).data,
            status=status.HTTP_201_CREATED
        )


class RespondInviteView(APIView):
    """Accept or decline an invite."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, invite_id):
        """Respond to an invite."""
        try:
            invite = GameInvite.objects.get(
                id=invite_id,
                to_user=request.user,
                status=GameInvite.Status.PENDING
            )
        except GameInvite.DoesNotExist:
            return Response(
                {'error': 'Invite not found.'},
                status=status.HTTP_404_NOT_FOUND
            )

        action = request.data.get('action')

        if action == 'accept':
            game = invite.accept()
            return Response({
                'invite': GameInviteSerializer(invite).data,
                'game': GameDetailSerializer(game).data
            })
        elif action == 'decline':
            invite.decline()
            return Response(GameInviteSerializer(invite).data)
        else:
            return Response(
                {'error': "Action must be 'accept' or 'decline'."},
                status=status.HTTP_400_BAD_REQUEST
            )
