"""
ViewSets for the game app.

This module provides ViewSet-based API endpoints for game management,
using DRF routers for automatic URL generation.
"""
from django.contrib.auth import get_user_model
from django.db.models import Q
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter

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


@extend_schema_view(
    list=extend_schema(
        summary="List user's games",
        description="Returns all games where the authenticated user is a player.",
        tags=['games'],
    ),
    retrieve=extend_schema(
        summary="Get game details",
        description="Returns detailed game state including board position and move history.",
        tags=['games'],
    ),
    create=extend_schema(
        summary="Create a new game",
        description="Creates a new game with the authenticated user as white player.",
        tags=['games'],
    ),
)
class GameViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Game model.

    Provides CRUD operations plus custom actions for gameplay:
    - list: Get all games for the current user
    - create: Start a new game
    - retrieve: Get detailed game state
    - active: List active games only
    - open: List games waiting for players
    - join: Join an existing game as black player
    - move: Make a move in the game
    - moves: Get move history
    - resign: Resign from the game
    """

    permission_classes = [IsAuthenticated]
    lookup_field = 'id'

    def get_queryset(self):
        """Return games where the user is a player."""
        user = self.request.user
        return Game.objects.filter(
            Q(white_player=user) | Q(black_player=user)
        ).select_related('white_player', 'black_player', 'winner')

    def get_serializer_class(self):
        """Return appropriate serializer based on action."""
        if self.action == 'list':
            return GameListSerializer
        elif self.action == 'create':
            return GameCreateSerializer
        elif self.action in ['active', 'open']:
            return GameListSerializer
        return GameDetailSerializer

    @extend_schema(
        summary="List active games",
        description="Returns only active (in-progress) games for the current user.",
        tags=['games'],
    )
    @action(detail=False, methods=['get'])
    def active(self, request):
        """List active games for the current user."""
        games = self.get_queryset().filter(status=Game.Status.ACTIVE)
        serializer = GameListSerializer(games, many=True)
        return Response(serializer.data)

    @extend_schema(
        summary="List open games",
        description="Returns games waiting for a second player to join.",
        tags=['games'],
    )
    @action(detail=False, methods=['get'])
    def open(self, request):
        """List games waiting for a second player."""
        games = Game.objects.filter(
            status=Game.Status.WAITING
        ).exclude(
            white_player=request.user
        ).select_related('white_player')
        serializer = GameListSerializer(games, many=True)
        return Response(serializer.data)

    @extend_schema(
        summary="Join a game",
        description="Join an existing game as the black player.",
        tags=['games'],
        responses={200: GameDetailSerializer},
    )
    @action(detail=True, methods=['post'])
    def join(self, request, id=None):
        """Join an existing game as the black player."""
        serializer = GameJoinSerializer(
            data={'game_id': id},
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)

        game = Game.objects.get(id=id)
        game.black_player = request.user
        game.start_game()

        return Response(GameDetailSerializer(game).data)

    @extend_schema(
        summary="Make a move",
        description="Execute a move in the game (roll dice, move checkers, double, etc.).",
        tags=['games'],
        request=MakeMoveSerializer,
    )
    @action(detail=True, methods=['post'])
    def move(self, request, id=None):
        """Execute a move in the game."""
        try:
            game = Game.objects.get(id=id, status=Game.Status.ACTIVE)
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

        serializer = MakeMoveSerializer(
            data=request.data,
            context={'request': request, 'game': game}
        )
        serializer.is_valid(raise_exception=True)

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

        game.refresh_from_db()

        return Response({
            'game': GameDetailSerializer(game).data,
            'result': result
        })

    @extend_schema(
        summary="Get move history",
        description="Returns the complete move history for a game.",
        tags=['games'],
        responses={200: MoveSerializer(many=True)},
    )
    @action(detail=True, methods=['get'])
    def moves(self, request, id=None):
        """Get move history for a game."""
        game = self.get_object()
        moves = Move.objects.filter(game=game).select_related('player')
        serializer = MoveSerializer(moves, many=True)
        return Response(serializer.data)

    @extend_schema(
        summary="Resign from game",
        description="Resign from the game, forfeiting to the opponent.",
        tags=['games'],
        responses={200: GameDetailSerializer},
    )
    @action(detail=True, methods=['post'])
    def resign(self, request, id=None):
        """Resign from the game."""
        try:
            game = Game.objects.get(id=id, status=Game.Status.ACTIVE)
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

        return Response(GameDetailSerializer(game).data)


@extend_schema_view(
    list=extend_schema(
        summary="List all invites",
        description="Returns all game invites sent or received by the user.",
        tags=['invites'],
    ),
    create=extend_schema(
        summary="Create an invite",
        description="Send a game invitation to another user.",
        tags=['invites'],
    ),
)
class GameInviteViewSet(viewsets.ModelViewSet):
    """
    ViewSet for GameInvite model.

    Provides endpoints for managing game invitations:
    - list: Get all invites for the current user
    - create: Send a new invite
    - received: List pending received invites
    - sent: List sent invites
    - respond: Accept or decline an invite
    """

    permission_classes = [IsAuthenticated]
    serializer_class = GameInviteSerializer

    def get_queryset(self):
        """Return invites involving the current user."""
        user = self.request.user
        return GameInvite.objects.filter(
            Q(from_user=user) | Q(to_user=user)
        ).select_related('from_user', 'to_user', 'game')

    def get_serializer_class(self):
        """Return appropriate serializer based on action."""
        if self.action == 'create':
            return CreateInviteSerializer
        return GameInviteSerializer

    def create(self, request, *args, **kwargs):
        """Create a new game invite."""
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

    @extend_schema(
        summary="List received invites",
        description="Returns pending invites received by the current user.",
        tags=['invites'],
    )
    @action(detail=False, methods=['get'])
    def received(self, request):
        """List pending received invites."""
        invites = GameInvite.objects.filter(
            to_user=request.user,
            status=GameInvite.Status.PENDING
        ).select_related('from_user', 'to_user')
        return Response(GameInviteSerializer(invites, many=True).data)

    @extend_schema(
        summary="List sent invites",
        description="Returns all invites sent by the current user.",
        tags=['invites'],
    )
    @action(detail=False, methods=['get'])
    def sent(self, request):
        """List sent invites."""
        invites = GameInvite.objects.filter(
            from_user=request.user
        ).select_related('from_user', 'to_user')
        return Response(GameInviteSerializer(invites, many=True).data)

    @extend_schema(
        summary="Respond to invite",
        description="Accept or decline a game invitation.",
        tags=['invites'],
        parameters=[
            OpenApiParameter(
                name='action',
                description="Response action: 'accept' or 'decline'",
                required=True,
                type=str,
            ),
        ],
    )
    @action(detail=True, methods=['post'])
    def respond(self, request, pk=None):
        """Accept or decline an invite."""
        try:
            invite = GameInvite.objects.get(
                id=pk,
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
