# Django Backend Research for Backgammon Game

This document contains comprehensive research on Django best practices for building a backgammon game backend. It covers async support, real-time WebSockets, database patterns, authentication, and game-specific considerations.

---

## Table of Contents

1. [Django 5.x Best Practices for Game Backends](#1-django-5x-best-practices-for-game-backends)
2. [Django REST Framework Patterns](#2-django-rest-framework-patterns)
3. [Authentication & Authorization](#3-authentication--authorization)
4. [Real-time Considerations](#4-real-time-considerations)
5. [Database Design for Games](#5-database-design-for-games)
6. [Implementation Recommendations](#6-implementation-recommendations)

---

## 1. Django 5.x Best Practices for Game Backends

### 1.1 Latest Async Support

Django 5.x has mature async support, building on the foundation laid in Django 4.1+. Understanding when to use async is critical for game backends.

#### Async Views Declaration

```python
# Function-based async view
async def get_game_state(request, game_id):
    game = await Game.objects.aget(id=game_id)
    return JsonResponse(await serialize_game(game))

# Class-based async view
class GameStateView(View):
    async def get(self, request, game_id):
        game = await Game.objects.aget(id=game_id)
        return JsonResponse(await serialize_game(game))
```

#### Async ORM Methods

All query-triggering methods have `a`-prefixed async variants:

```python
# Async queryset operations
game = await Game.objects.aget(id=game_id)
games = [g async for g in Game.objects.filter(status='active')]
count = await Game.objects.filter(player=user).acount()
exists = await Game.objects.filter(id=game_id).aexists()

# Async model operations
game = Game(player_white=user)
await game.asave()
await game.adelete()

# Async related object operations
await game.moves.acreate(from_point=6, to_point=5)
await game.players.aset([player1, player2])
```

#### When to Use Async in a Game Backend

| Use Case | Sync or Async | Reasoning |
|----------|---------------|-----------|
| Simple CRUD operations | Sync | No performance benefit, adds complexity |
| External API calls (AI service) | Async | Can wait on multiple services concurrently |
| WebSocket consumers | Async | Required for Django Channels |
| Database transactions | Sync | Transactions don't work in async mode |
| Long-polling endpoints | Async | Efficient connection handling |
| Concurrent game state checks | Async | Can parallelize multiple queries |

#### Critical Limitation: Transactions

**Transactions do NOT work in async mode.** This is crucial for game state updates where atomicity matters.

```python
# WRONG - Will raise SynchronousOnlyOperation
async def make_move(request, game_id):
    async with transaction.atomic():  # This doesn't work!
        game = await Game.objects.aget(id=game_id)
        await game.asave()

# CORRECT - Wrap transaction logic in sync function
from asgiref.sync import sync_to_async

@sync_to_async
def _make_move_transaction(game_id, move_data, user):
    """Execute move within a transaction."""
    with transaction.atomic():
        game = Game.objects.select_for_update().get(id=game_id)
        if game.current_player != user:
            raise PermissionError("Not your turn")

        move = Move.objects.create(game=game, **move_data)
        game.apply_move(move)
        game.save()
        return game, move

async def make_move(request, game_id):
    game, move = await _make_move_transaction(
        game_id,
        request.data,
        request.user
    )
    return JsonResponse(serialize_game(game))
```

#### Deployment Requirements

- **ASGI Server Required**: Use Uvicorn, Daphne, or Hypercorn for true async benefits
- **Middleware Consideration**: All middleware must be async-compatible for full benefits
- **WSGI Fallback**: Async views work under WSGI but with ~1ms overhead per request

```python
# settings.py for ASGI
ASGI_APPLICATION = 'config.asgi.application'

# Use async-compatible middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',  # Async-compatible
    'django.contrib.sessions.middleware.SessionMiddleware',  # Async-compatible
    'django.middleware.common.CommonMiddleware',  # Async-compatible
    'django.middleware.csrf.CsrfViewMiddleware',  # Async-compatible
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Async-compatible
    'django.contrib.messages.middleware.MessageMiddleware',  # Async-compatible
]
```

### 1.2 Django Channels for WebSockets

Django Channels extends Django to handle WebSockets, which is essential for real-time game updates.

#### Architecture Overview

```
                    ┌─────────────────┐
                    │   HTTP Client   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ASGI Server    │
                    │ (Uvicorn/Daphne)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
    │ HTTP Protocol  │ │ WebSocket │ │   Other     │
    │    Handler     │ │  Handler  │ │  Protocols  │
    └─────────┬──────┘ └─────┬─────┘ └─────────────┘
              │              │
    ┌─────────▼──────┐ ┌─────▼─────┐
    │  Django Views  │ │ Consumers │
    └────────────────┘ └─────┬─────┘
                             │
                    ┌────────▼────────┐
                    │  Channel Layer  │
                    │     (Redis)     │
                    └─────────────────┘
```

#### Installation and Setup

```bash
pip install channels channels-redis
```

```python
# settings.py
INSTALLED_APPS = [
    'daphne',  # Must be before django.contrib.staticfiles
    'channels',
    # ... other apps
]

ASGI_APPLICATION = 'config.asgi.application'

# Channel layer with Redis
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [('127.0.0.1', 6379)],
            'capacity': 1500,  # Max messages in channel before dropping
            'expiry': 10,  # Message expiry in seconds
        },
    },
}

# Development fallback (in-memory, single process only)
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
    },
}
```

```python
# config/asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

from apps.game.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})
```

#### Game Consumer Implementation

```python
# apps/game/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.exceptions import PermissionDenied

class GameConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time game updates.

    Handles:
    - Game state synchronization
    - Move notifications
    - Player presence (connected/disconnected)
    - Spectator support
    """

    async def connect(self):
        """Handle WebSocket connection."""
        self.game_id = self.scope['url_route']['kwargs']['game_id']
        self.game_group_name = f'game_{self.game_id}'
        self.user = self.scope['user']

        # Verify user can access this game
        try:
            self.game = await self.get_game()
            self.is_player = await self.check_is_player()
        except PermissionDenied:
            await self.close(code=4003)
            return

        # Join game group
        await self.channel_layer.group_add(
            self.game_group_name,
            self.channel_name
        )

        await self.accept()

        # Send current game state
        await self.send_game_state()

        # Notify others of connection
        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'player_connected',
                'user_id': self.user.id,
                'username': self.user.username,
            }
        )

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        # Leave game group
        await self.channel_layer.group_discard(
            self.game_group_name,
            self.channel_name
        )

        # Notify others
        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'player_disconnected',
                'user_id': self.user.id,
                'username': self.user.username,
            }
        )

    async def receive(self, text_data):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            handlers = {
                'move': self.handle_move,
                'roll_dice': self.handle_roll_dice,
                'resign': self.handle_resign,
                'offer_double': self.handle_offer_double,
                'accept_double': self.handle_accept_double,
                'decline_double': self.handle_decline_double,
                'ping': self.handle_ping,
            }

            handler = handlers.get(message_type)
            if handler:
                await handler(data)
            else:
                await self.send_error(f'Unknown message type: {message_type}')

        except json.JSONDecodeError:
            await self.send_error('Invalid JSON')
        except Exception as e:
            await self.send_error(str(e))

    async def handle_move(self, data):
        """Process a move from a player."""
        if not self.is_player:
            await self.send_error('Spectators cannot make moves')
            return

        try:
            move_data = data.get('move')
            game, move = await self.make_move(move_data)

            # Broadcast move to all connected clients
            await self.channel_layer.group_send(
                self.game_group_name,
                {
                    'type': 'game_move',
                    'move': move.to_dict(),
                    'game_state': game.to_dict(),
                    'player_id': self.user.id,
                }
            )
        except Exception as e:
            await self.send_error(f'Invalid move: {str(e)}')

    async def handle_ping(self, data):
        """Respond to ping with pong."""
        await self.send(text_data=json.dumps({
            'type': 'pong',
            'timestamp': data.get('timestamp'),
        }))

    # Group message handlers (called when group_send is used)

    async def game_move(self, event):
        """Send move update to WebSocket."""
        await self.send(text_data=json.dumps({
            'type': 'move',
            'move': event['move'],
            'game_state': event['game_state'],
            'player_id': event['player_id'],
        }))

    async def player_connected(self, event):
        """Send player connected notification."""
        await self.send(text_data=json.dumps({
            'type': 'player_connected',
            'user_id': event['user_id'],
            'username': event['username'],
        }))

    async def player_disconnected(self, event):
        """Send player disconnected notification."""
        await self.send(text_data=json.dumps({
            'type': 'player_disconnected',
            'user_id': event['user_id'],
            'username': event['username'],
        }))

    async def game_state_update(self, event):
        """Send full game state update."""
        await self.send(text_data=json.dumps({
            'type': 'state_update',
            'game_state': event['game_state'],
            'reason': event.get('reason', 'update'),
        }))

    # Database operations (must use sync_to_async)

    @database_sync_to_async
    def get_game(self):
        """Fetch game from database."""
        from apps.game.models import Game
        return Game.objects.get(id=self.game_id)

    @database_sync_to_async
    def check_is_player(self):
        """Check if current user is a player in this game."""
        return (
            self.game.player_white_id == self.user.id or
            self.game.player_black_id == self.user.id
        )

    @database_sync_to_async
    def make_move(self, move_data):
        """Execute a move within a transaction."""
        from django.db import transaction
        from apps.game.models import Game, Move

        with transaction.atomic():
            game = Game.objects.select_for_update().get(id=self.game_id)

            if game.current_player_id != self.user.id:
                raise PermissionError("Not your turn")

            move = Move.objects.create(
                game=game,
                player=self.user,
                **move_data
            )
            game.apply_move(move)
            game.save()

            return game, move

    # Helper methods

    async def send_game_state(self):
        """Send current game state to this connection."""
        game = await self.get_game()
        await self.send(text_data=json.dumps({
            'type': 'state_sync',
            'game_state': await database_sync_to_async(game.to_dict)(),
        }))

    async def send_error(self, message):
        """Send error message to this connection."""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': message,
        }))
```

```python
# apps/game/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/game/(?P<game_id>\w+)/$', consumers.GameConsumer.as_asgi()),
]
```

### 1.3 Database Transaction Patterns for Game State

Game state updates require careful transaction management to ensure consistency.

#### Basic Atomic Transaction

```python
from django.db import transaction

def make_move(game_id, player, move_data):
    """
    Apply a move to a game atomically.

    If any part fails, the entire transaction rolls back.
    """
    with transaction.atomic():
        game = Game.objects.select_for_update().get(id=game_id)

        # Validate the move
        if not game.is_valid_move(move_data):
            raise InvalidMoveError("Invalid move")

        # Create move record
        move = Move.objects.create(
            game=game,
            player=player,
            from_point=move_data['from'],
            to_point=move_data['to'],
            move_number=game.moves.count() + 1,
        )

        # Update game state
        game.apply_move(move)
        game.version += 1  # For optimistic locking
        game.save()

        return game, move
```

#### Savepoints for Partial Rollback

```python
from django.db import transaction

def make_turn(game_id, player, moves):
    """
    Apply multiple moves in a turn with savepoints.

    If a move fails, roll back only that move and continue.
    """
    with transaction.atomic():
        game = Game.objects.select_for_update().get(id=game_id)
        applied_moves = []

        for move_data in moves:
            # Create savepoint
            sid = transaction.savepoint()

            try:
                move = apply_single_move(game, player, move_data)
                applied_moves.append(move)
                transaction.savepoint_commit(sid)
            except InvalidMoveError as e:
                # Roll back just this move
                transaction.savepoint_rollback(sid)
                # Continue with remaining moves or raise
                raise

        # All moves applied successfully
        game.end_turn()
        game.save()

        return game, applied_moves
```

#### Durable Transactions for Critical Operations

```python
from django.db import transaction

def complete_game(game_id, winner_id):
    """
    Complete a game with durable transaction.

    Durable=True ensures data is written to disk before returning.
    Use for critical operations like game completion where data loss
    would be catastrophic.
    """
    with transaction.atomic(durable=True):
        game = Game.objects.select_for_update().get(id=game_id)
        game.status = 'completed'
        game.winner_id = winner_id
        game.finished_at = timezone.now()
        game.save()

        # Update player statistics
        winner = game.winner
        winner.games_won += 1
        winner.save()

        loser = game.player_white if winner == game.player_black else game.player_black
        loser.games_lost += 1
        loser.save()

        return game
```

### 1.4 Optimistic vs Pessimistic Locking Strategies

#### Pessimistic Locking (select_for_update)

**How it works**: Locks the row in the database, blocking other transactions.

**Pros**:
- Guarantees no conflicts
- Simple to reason about
- Good for high-contention scenarios

**Cons**:
- Can cause deadlocks
- Holds database connections
- Reduces concurrency

```python
from django.db import transaction

def pessimistic_move(game_id, player, move_data):
    """
    Use pessimistic locking with select_for_update.

    Best for:
    - High contention (many players trying to modify same game)
    - Critical operations that must not fail
    - When conflicts are expected to be common
    """
    with transaction.atomic():
        # NOWAIT raises DatabaseError if row is locked
        # skip_locked skips locked rows (not useful for single game)
        game = Game.objects.select_for_update(nowait=True).get(id=game_id)

        if game.current_player != player:
            raise PermissionError("Not your turn")

        game.apply_move(move_data)
        game.save()
        return game
```

```python
# Handle lock contention
from django.db import OperationalError

def safe_pessimistic_move(game_id, player, move_data, max_retries=3):
    """Pessimistic locking with retry logic."""
    for attempt in range(max_retries):
        try:
            return pessimistic_move(game_id, player, move_data)
        except OperationalError:  # Row is locked
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
```

#### Optimistic Locking (Version Field)

**How it works**: Track a version number; fail if it changed since read.

**Pros**:
- No database locks held
- Higher concurrency
- Better for read-heavy workloads

**Cons**:
- Requires retry logic
- Can fail in high-contention scenarios
- More complex error handling

```python
class Game(models.Model):
    version = models.PositiveIntegerField(default=0)
    # ... other fields

class ConcurrencyError(Exception):
    """Raised when optimistic lock fails."""
    pass

def optimistic_move(game_id, player, move_data, expected_version):
    """
    Use optimistic locking with version checking.

    Best for:
    - Low contention (turn-based games where only one player acts)
    - Read-heavy workloads
    - When conflicts are expected to be rare
    """
    with transaction.atomic():
        game = Game.objects.get(id=game_id)

        # Check version hasn't changed
        if game.version != expected_version:
            raise ConcurrencyError(
                f"Game state changed. Expected version {expected_version}, "
                f"got {game.version}. Please refresh and retry."
            )

        if game.current_player != player:
            raise PermissionError("Not your turn")

        game.apply_move(move_data)
        game.version += 1  # Increment version
        game.save()

        return game
```

```python
# Using Django's update() for atomic version check
def atomic_optimistic_move(game_id, player_id, move_data, expected_version):
    """
    More robust optimistic locking using UPDATE with WHERE clause.

    This is a single atomic operation at the database level.
    """
    with transaction.atomic():
        # First, try to update with version check
        updated = Game.objects.filter(
            id=game_id,
            version=expected_version,
            current_player_id=player_id,
        ).update(
            version=F('version') + 1,
            # Note: Can't do complex state updates in UPDATE
            # This only works for simple field updates
        )

        if updated == 0:
            # Either version changed, or not player's turn
            game = Game.objects.get(id=game_id)
            if game.version != expected_version:
                raise ConcurrencyError("Game state changed")
            if game.current_player_id != player_id:
                raise PermissionError("Not your turn")
            raise RuntimeError("Unknown update failure")

        # For complex state updates, fetch and save
        game = Game.objects.get(id=game_id)
        game.apply_move(move_data)
        game.save()

        return game
```

#### Recommendation for Backgammon

**Use Optimistic Locking** because:
1. Turn-based game = low contention (only one player can act at a time)
2. Version field helps with client state synchronization
3. Frontend can show "state changed, please refresh" messages gracefully
4. Simpler to implement with WebSocket state sync

```python
# Recommended pattern for backgammon
class Game(models.Model):
    version = models.PositiveIntegerField(default=0)

    class Meta:
        # Add constraint to prevent negative version
        constraints = [
            models.CheckConstraint(
                check=models.Q(version__gte=0),
                name='version_non_negative'
            )
        ]

    def save(self, *args, **kwargs):
        if not self._state.adding:  # Not a new object
            self.version += 1
        super().save(*args, **kwargs)
```

---

## 2. Django REST Framework Patterns

### 2.1 ViewSets vs APIViews for Game Actions

#### When to Use ViewSets

**ViewSets** are ideal when you have standard CRUD operations on a resource.

```python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

class GameViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Game CRUD operations.

    Provides:
    - list: GET /games/
    - create: POST /games/
    - retrieve: GET /games/{id}/
    - update: PUT /games/{id}/
    - partial_update: PATCH /games/{id}/
    - destroy: DELETE /games/{id}/
    """
    queryset = Game.objects.all()
    serializer_class = GameSerializer
    permission_classes = [IsAuthenticated, IsGameParticipant]

    def get_queryset(self):
        """Filter to only games the user is involved in."""
        user = self.request.user
        return Game.objects.filter(
            models.Q(player_white=user) |
            models.Q(player_black=user)
        ).select_related('player_white', 'player_black')

    def perform_create(self, serializer):
        """Set the creating user as player_white."""
        serializer.save(player_white=self.request.user)

    # Custom actions for game-specific operations
    @action(detail=True, methods=['post'])
    def roll_dice(self, request, pk=None):
        """Roll dice for the current turn."""
        game = self.get_object()

        if game.current_player != request.user:
            return Response(
                {'error': 'Not your turn'},
                status=status.HTTP_403_FORBIDDEN
            )

        dice = game.roll_dice()
        return Response({
            'dice': dice,
            'valid_moves': game.get_valid_moves(dice),
        })

    @action(detail=True, methods=['post'])
    def make_move(self, request, pk=None):
        """Make a move in the game."""
        game = self.get_object()
        serializer = MoveSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            move = game.apply_move(
                player=request.user,
                **serializer.validated_data
            )
            return Response(MoveSerializer(move).data)
        except InvalidMoveError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    @action(detail=True, methods=['post'])
    def resign(self, request, pk=None):
        """Resign from the game."""
        game = self.get_object()
        game.resign(request.user)
        return Response({'status': 'resigned'})

    @action(detail=True, methods=['get'])
    def history(self, request, pk=None):
        """Get move history for the game."""
        game = self.get_object()
        moves = game.moves.all().order_by('move_number')
        return Response(MoveSerializer(moves, many=True).data)
```

#### When to Use APIViews

**APIViews** are better for:
- Non-resource-oriented endpoints
- Complex logic that doesn't fit CRUD
- Endpoints with multiple HTTP methods with different logic
- One-off operations

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class QuickMatchView(APIView):
    """
    Find or create a quick match game.

    This doesn't fit the standard CRUD pattern because:
    - It might return an existing game
    - It might create a new game
    - It might add the user to a waiting queue
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Check for existing waiting games
        waiting_game = Game.objects.filter(
            status='waiting',
            player_black__isnull=True,
        ).exclude(
            player_white=request.user
        ).first()

        if waiting_game:
            # Join existing game
            waiting_game.player_black = request.user
            waiting_game.status = 'playing'
            waiting_game.save()
            return Response(
                GameSerializer(waiting_game).data,
                status=status.HTTP_200_OK
            )

        # Create new waiting game
        game = Game.objects.create(
            player_white=request.user,
            status='waiting',
        )
        return Response(
            GameSerializer(game).data,
            status=status.HTTP_201_CREATED
        )


class GameAnalysisView(APIView):
    """
    Analyze a game position using AI.

    Separate from GameViewSet because it's a specialized operation
    that might call external services.
    """
    permission_classes = [IsAuthenticated]
    throttle_classes = [AIAnalysisThrottle]  # Rate limit expensive operations

    def post(self, request, game_id):
        game = get_object_or_404(Game, id=game_id)

        # Verify user can access this game
        if not game.can_view(request.user):
            return Response(status=status.HTTP_403_FORBIDDEN)

        # Run analysis
        analysis = analyze_position(game.board_state)

        return Response({
            'evaluation': analysis.score,
            'best_moves': analysis.best_moves,
            'explanation': analysis.explanation,
        })
```

#### Hybrid Approach (Recommended)

```python
# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'games', GameViewSet, basename='game')

urlpatterns = [
    path('', include(router.urls)),
    # Non-CRUD endpoints outside the router
    path('quick-match/', QuickMatchView.as_view(), name='quick-match'),
    path('games/<int:game_id>/analyze/', GameAnalysisView.as_view(), name='game-analyze'),
    path('leaderboard/', LeaderboardView.as_view(), name='leaderboard'),
]
```

### 2.2 Serializer Patterns for Complex Game State

#### Base Game Serializer

```python
from rest_framework import serializers
from apps.game.models import Game, Move, Player

class PlayerSerializer(serializers.ModelSerializer):
    """Serializer for player info in game context."""

    class Meta:
        model = Player
        fields = ['id', 'username', 'rating', 'avatar_url']


class MoveSerializer(serializers.ModelSerializer):
    """Serializer for individual moves."""
    player = PlayerSerializer(read_only=True)

    class Meta:
        model = Move
        fields = [
            'id', 'move_number', 'player',
            'from_point', 'to_point', 'is_hit',
            'dice_used', 'timestamp'
        ]
        read_only_fields = ['id', 'move_number', 'player', 'is_hit', 'timestamp']
```

#### Nested vs Flat Serializers

**Nested Serializers** - Include related objects inline

```python
class GameDetailSerializer(serializers.ModelSerializer):
    """
    Nested serializer for full game details.

    Pros:
    - Single request gets all data
    - Natural object structure
    - Good for "detail" views

    Cons:
    - Larger payload
    - Can't update nested objects easily
    - May include unnecessary data
    """
    player_white = PlayerSerializer(read_only=True)
    player_black = PlayerSerializer(read_only=True)
    moves = MoveSerializer(many=True, read_only=True)
    board_state = serializers.SerializerMethodField()

    class Meta:
        model = Game
        fields = [
            'id', 'status', 'player_white', 'player_black',
            'current_player', 'board_state', 'moves',
            'created_at', 'updated_at', 'version'
        ]

    def get_board_state(self, obj):
        """Return structured board representation."""
        return {
            'points': obj.get_points_state(),  # List of 24 points
            'bar': {
                'white': obj.bar_white,
                'black': obj.bar_black,
            },
            'borne_off': {
                'white': obj.borne_off_white,
                'black': obj.borne_off_black,
            },
            'dice': obj.current_dice,
            'doubling_cube': obj.doubling_cube_value,
            'cube_owner': obj.cube_owner,
        }
```

**Flat Serializers** - Use IDs for related objects

```python
class GameListSerializer(serializers.ModelSerializer):
    """
    Flat serializer for game lists.

    Pros:
    - Smaller payload
    - Faster serialization
    - Good for list views

    Cons:
    - Requires additional requests for related data
    - Less convenient for frontend
    """
    player_white_id = serializers.IntegerField(source='player_white.id')
    player_white_username = serializers.CharField(source='player_white.username')
    player_black_id = serializers.IntegerField(source='player_black.id', allow_null=True)
    player_black_username = serializers.CharField(source='player_black.username', allow_null=True)

    class Meta:
        model = Game
        fields = [
            'id', 'status',
            'player_white_id', 'player_white_username',
            'player_black_id', 'player_black_username',
            'created_at', 'move_count'
        ]
```

#### Dynamic Serializer Depth

```python
class GameSerializer(serializers.ModelSerializer):
    """
    Serializer with configurable depth based on context.
    """
    player_white = serializers.SerializerMethodField()
    player_black = serializers.SerializerMethodField()

    class Meta:
        model = Game
        fields = ['id', 'status', 'player_white', 'player_black', 'version']

    def get_player_white(self, obj):
        if self.context.get('expand_players'):
            return PlayerSerializer(obj.player_white).data
        return obj.player_white_id

    def get_player_black(self, obj):
        if self.context.get('expand_players'):
            return PlayerSerializer(obj.player_black).data if obj.player_black else None
        return obj.player_black_id

# Usage in view
class GameViewSet(viewsets.ModelViewSet):
    def get_serializer_context(self):
        context = super().get_serializer_context()
        # Expand players for detail view, not for list
        context['expand_players'] = self.action == 'retrieve'
        return context
```

#### Write Serializers (Input Validation)

```python
class CreateGameSerializer(serializers.Serializer):
    """Serializer for creating a new game."""
    opponent_id = serializers.IntegerField(required=False, allow_null=True)
    time_control = serializers.ChoiceField(
        choices=['none', 'rapid', 'blitz'],
        default='none'
    )
    stake = serializers.IntegerField(min_value=1, max_value=64, default=1)

    def validate_opponent_id(self, value):
        if value is not None:
            try:
                User.objects.get(id=value)
            except User.DoesNotExist:
                raise serializers.ValidationError("Opponent not found")
        return value

    def create(self, validated_data):
        user = self.context['request'].user
        opponent_id = validated_data.pop('opponent_id', None)

        game = Game.objects.create(
            player_white=user,
            player_black_id=opponent_id,
            status='waiting' if opponent_id is None else 'playing',
            **validated_data
        )
        return game


class MakeMoveSerializer(serializers.Serializer):
    """Serializer for validating move input."""
    from_point = serializers.IntegerField(min_value=0, max_value=25)
    to_point = serializers.IntegerField(min_value=0, max_value=25)
    game_version = serializers.IntegerField()  # For optimistic locking

    def validate(self, data):
        if data['from_point'] == data['to_point']:
            raise serializers.ValidationError("From and to points must be different")

        # Additional move validation can go here
        return data
```

### 2.3 Action Decorators for Game-Specific Endpoints

```python
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status

class GameViewSet(viewsets.ModelViewSet):
    # ... base configuration ...

    @action(
        detail=True,  # Operates on a single game
        methods=['post'],
        url_path='roll',
        url_name='roll-dice',
        permission_classes=[IsAuthenticated, IsCurrentPlayer],
    )
    def roll_dice(self, request, pk=None):
        """
        Roll dice for the current turn.

        POST /games/{id}/roll/
        """
        game = self.get_object()

        if game.dice_rolled:
            return Response(
                {'error': 'Dice already rolled this turn'},
                status=status.HTTP_400_BAD_REQUEST
            )

        dice = game.roll_dice()
        game.save()

        return Response({
            'dice': dice,
            'possible_moves': game.calculate_possible_moves(),
            'game_version': game.version,
        })

    @action(
        detail=True,
        methods=['post'],
        url_path='move',
        url_name='make-move',
        serializer_class=MakeMoveSerializer,
    )
    def make_move(self, request, pk=None):
        """
        Make a move in the game.

        POST /games/{id}/move/
        {
            "from_point": 6,
            "to_point": 4,
            "game_version": 15
        }
        """
        game = self.get_object()
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Optimistic locking check
        if game.version != serializer.validated_data['game_version']:
            return Response(
                {
                    'error': 'Game state has changed',
                    'current_version': game.version,
                },
                status=status.HTTP_409_CONFLICT
            )

        try:
            move = game.apply_move(
                player=request.user,
                from_point=serializer.validated_data['from_point'],
                to_point=serializer.validated_data['to_point'],
            )
            return Response({
                'move': MoveSerializer(move).data,
                'game_state': GameStateSerializer(game).data,
            })
        except InvalidMoveError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    @action(detail=True, methods=['post'], url_path='double')
    def offer_double(self, request, pk=None):
        """Offer to double the stakes."""
        game = self.get_object()

        if not game.can_offer_double(request.user):
            return Response(
                {'error': 'Cannot offer double'},
                status=status.HTTP_400_BAD_REQUEST
            )

        game.offer_double(request.user)
        game.save()

        return Response({'status': 'double_offered'})

    @action(detail=True, methods=['post'], url_path='accept-double')
    def accept_double(self, request, pk=None):
        """Accept a doubling offer."""
        game = self.get_object()

        if game.pending_double_from == request.user:
            return Response(
                {'error': 'Cannot accept your own double'},
                status=status.HTTP_400_BAD_REQUEST
            )

        game.accept_double()
        game.save()

        return Response({
            'status': 'double_accepted',
            'new_stake': game.stake,
        })

    @action(detail=True, methods=['post'], url_path='undo')
    def undo_move(self, request, pk=None):
        """
        Undo the last move (only in casual games).

        This demonstrates how actions can have different
        permission requirements.
        """
        game = self.get_object()

        if game.game_type != 'casual':
            return Response(
                {'error': 'Undo only available in casual games'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if game.last_move_player != request.user:
            return Response(
                {'error': 'Can only undo your own moves'},
                status=status.HTTP_403_FORBIDDEN
            )

        game.undo_last_move()
        game.save()

        return Response(GameStateSerializer(game).data)

    @action(detail=False, methods=['get'], url_path='active')
    def active_games(self, request):
        """
        Get all active games for the current user.

        GET /games/active/

        detail=False means this operates on the collection,
        not a single game.
        """
        games = self.get_queryset().filter(status='playing')
        serializer = self.get_serializer(games, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get game statistics for the current user.

        GET /games/stats/
        """
        user = request.user
        stats = {
            'total_games': Game.objects.filter(
                Q(player_white=user) | Q(player_black=user)
            ).count(),
            'wins': Game.objects.filter(winner=user).count(),
            'active_games': Game.objects.filter(
                Q(player_white=user) | Q(player_black=user),
                status='playing'
            ).count(),
        }
        stats['losses'] = stats['total_games'] - stats['wins'] - stats['active_games']
        stats['win_rate'] = (
            stats['wins'] / (stats['wins'] + stats['losses'])
            if stats['wins'] + stats['losses'] > 0
            else 0
        )
        return Response(stats)
```

---

## 3. Authentication & Authorization

### 3.1 django-rest-framework-simplejwt Setup

#### Installation

```bash
pip install djangorestframework-simplejwt
```

#### Configuration

```python
# settings.py
from datetime import timedelta

INSTALLED_APPS = [
    # ...
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',  # For logout/token revocation
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',  # For browsable API
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

SIMPLE_JWT = {
    # Token lifetimes
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,  # Issue new refresh token with each refresh
    'BLACKLIST_AFTER_ROTATION': True,  # Blacklist old refresh tokens

    # Token settings
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',

    # Claims
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'TOKEN_TYPE_CLAIM': 'token_type',

    # Custom claims
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),

    # Sliding token configuration (alternative to refresh tokens)
    'SLIDING_TOKEN_LIFETIME': timedelta(minutes=30),
    'SLIDING_TOKEN_REFRESH_LIFETIME': timedelta(days=1),
}
```

#### Custom Token Claims

```python
# apps/accounts/serializers.py
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Add custom claims to JWT tokens."""

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['username'] = user.username
        token['email'] = user.email
        token['rating'] = user.profile.rating if hasattr(user, 'profile') else 1500

        return token

    def validate(self, attrs):
        data = super().validate(attrs)

        # Add extra response data
        data['user'] = {
            'id': self.user.id,
            'username': self.user.username,
            'email': self.user.email,
        }

        return data


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer
```

#### URL Configuration

```python
# apps/accounts/urls.py
from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView, TokenVerifyView
from .views import CustomTokenObtainPairView, LogoutView, RegisterView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('logout/', LogoutView.as_view(), name='logout'),
]
```

#### Logout with Token Blacklist

```python
# apps/accounts/views.py
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError

class LogoutView(APIView):
    """Blacklist the refresh token to logout."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')
            if refresh_token:
                token = RefreshToken(refresh_token)
                token.blacklist()
            return Response(status=status.HTTP_205_RESET_CONTENT)
        except TokenError:
            return Response(status=status.HTTP_400_BAD_REQUEST)
```

### 3.2 Permission Classes for Game Access

```python
# apps/game/permissions.py
from rest_framework import permissions

class IsGameParticipant(permissions.BasePermission):
    """
    Allow access only to players in the game.

    For list views: Filter happens in get_queryset
    For detail views: Check if user is a participant
    """
    message = "You are not a participant in this game."

    def has_object_permission(self, request, view, obj):
        # obj is a Game instance
        return (
            obj.player_white == request.user or
            obj.player_black == request.user
        )


class IsCurrentPlayer(permissions.BasePermission):
    """
    Allow access only to the player whose turn it is.
    """
    message = "It's not your turn."

    def has_object_permission(self, request, view, obj):
        return obj.current_player == request.user


class IsGameParticipantOrSpectator(permissions.BasePermission):
    """
    Allow read access to anyone (for spectators).
    Allow write access only to participants.
    """
    def has_object_permission(self, request, view, obj):
        # Allow read access to anyone
        if request.method in permissions.SAFE_METHODS:
            return True

        # Write access only for participants
        return (
            obj.player_white == request.user or
            obj.player_black == request.user
        )


class CanModifyGame(permissions.BasePermission):
    """
    Complex permission checking for game modifications.
    """
    def has_object_permission(self, request, view, obj):
        user = request.user

        # Game must be in playing state
        if obj.status != 'playing':
            self.message = "Game is not active."
            return False

        # User must be a participant
        if user not in [obj.player_white, obj.player_black]:
            self.message = "You are not a participant."
            return False

        # For move actions, must be current player
        if view.action in ['make_move', 'roll_dice']:
            if obj.current_player != user:
                self.message = "It's not your turn."
                return False

        return True


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Object-level permission to only allow owners to edit.
    Used for user profiles, etc.
    """
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.user == request.user
```

#### Using Permissions in Views

```python
# apps/game/views.py
class GameViewSet(viewsets.ModelViewSet):
    queryset = Game.objects.all()
    serializer_class = GameSerializer

    def get_permissions(self):
        """
        Return different permissions based on action.
        """
        if self.action == 'create':
            permission_classes = [permissions.IsAuthenticated]
        elif self.action in ['update', 'partial_update', 'destroy']:
            permission_classes = [permissions.IsAuthenticated, IsGameParticipant]
        elif self.action in ['make_move', 'roll_dice']:
            permission_classes = [permissions.IsAuthenticated, CanModifyGame]
        elif self.action == 'retrieve':
            permission_classes = [permissions.IsAuthenticated, IsGameParticipantOrSpectator]
        else:
            permission_classes = [permissions.IsAuthenticated]
        return [permission() for permission in permission_classes]
```

### 3.3 Guest User Support Patterns

#### Option 1: Anonymous Guest with Session

```python
# apps/accounts/models.py
from django.contrib.auth.models import AbstractUser
import uuid

class User(AbstractUser):
    is_guest = models.BooleanField(default=False)
    guest_id = models.UUIDField(null=True, blank=True)

    # Guest accounts can be converted to full accounts
    converted_at = models.DateTimeField(null=True, blank=True)


# apps/accounts/views.py
class GuestLoginView(APIView):
    """Create a temporary guest account."""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        guest_id = uuid.uuid4()
        username = f"Guest_{guest_id.hex[:8]}"

        # Create guest user
        user = User.objects.create_user(
            username=username,
            is_guest=True,
            guest_id=guest_id,
        )

        # Generate tokens
        refresh = RefreshToken.for_user(user)

        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': {
                'id': user.id,
                'username': user.username,
                'is_guest': True,
            }
        })


class ConvertGuestView(APIView):
    """Convert a guest account to a full account."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user = request.user

        if not user.is_guest:
            return Response(
                {'error': 'Not a guest account'},
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = ConvertGuestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Update user
        user.username = serializer.validated_data['username']
        user.email = serializer.validated_data['email']
        user.set_password(serializer.validated_data['password'])
        user.is_guest = False
        user.converted_at = timezone.now()
        user.save()

        return Response({
            'status': 'converted',
            'user': UserSerializer(user).data,
        })
```

#### Option 2: Device-based Guest (No Account)

```python
# apps/accounts/authentication.py
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
import uuid

class DeviceAuthentication(BaseAuthentication):
    """
    Authenticate based on device ID header.
    Falls back to anonymous user with device tracking.
    """

    def authenticate(self, request):
        device_id = request.META.get('HTTP_X_DEVICE_ID')

        if not device_id:
            return None  # Fall through to other authentication

        try:
            # Validate device ID format
            uuid.UUID(device_id)
        except ValueError:
            raise AuthenticationFailed('Invalid device ID')

        # Get or create device record
        device, _ = Device.objects.get_or_create(
            device_id=device_id,
            defaults={'last_seen': timezone.now()}
        )
        device.last_seen = timezone.now()
        device.save(update_fields=['last_seen'])

        # Return anonymous user with device attached
        return (AnonymousDeviceUser(device), device_id)


class AnonymousDeviceUser:
    """Anonymous user with device tracking."""
    is_authenticated = True  # For permission checks
    is_anonymous = True
    is_guest = True

    def __init__(self, device):
        self.device = device
        self.id = f"device_{device.id}"
        self.username = f"Guest_{device.device_id[:8]}"
```

#### Cleanup Task for Guest Accounts

```python
# apps/accounts/tasks.py (using Celery)
from celery import shared_task
from django.utils import timezone
from datetime import timedelta

@shared_task
def cleanup_guest_accounts():
    """
    Delete guest accounts that:
    - Haven't been used in 30 days
    - Have no active games
    - Have no significant game history
    """
    cutoff = timezone.now() - timedelta(days=30)

    stale_guests = User.objects.filter(
        is_guest=True,
        last_login__lt=cutoff,
    ).exclude(
        # Keep guests with active games
        games_as_white__status='playing'
    ).exclude(
        games_as_black__status='playing'
    ).annotate(
        game_count=Count('games_as_white') + Count('games_as_black')
    ).filter(
        game_count__lt=5  # Only delete if few games played
    )

    deleted_count = stale_guests.count()
    stale_guests.delete()

    return f"Deleted {deleted_count} stale guest accounts"
```

---

## 4. Real-time Considerations

### 4.1 WebSocket Authentication

#### JWT Authentication for WebSockets

```python
# apps/game/middleware.py
from channels.middleware import BaseMiddleware
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework_simplejwt.exceptions import TokenError
from django.contrib.auth import get_user_model

User = get_user_model()

class JWTAuthMiddleware(BaseMiddleware):
    """
    JWT authentication middleware for WebSocket connections.

    Token can be passed as:
    1. Query parameter: ws://host/ws/game/123/?token=xxx
    2. Cookie: 'access_token' cookie
    """

    async def __call__(self, scope, receive, send):
        # Get token from query string or cookie
        token = self.get_token_from_scope(scope)

        if token:
            scope['user'] = await self.get_user_from_token(token)
        else:
            scope['user'] = AnonymousUser()

        return await super().__call__(scope, receive, send)

    def get_token_from_scope(self, scope):
        """Extract JWT token from query string or cookies."""
        # Try query string first
        query_string = scope.get('query_string', b'').decode()
        params = dict(
            param.split('=')
            for param in query_string.split('&')
            if '=' in param
        )
        if 'token' in params:
            return params['token']

        # Try cookies
        headers = dict(scope.get('headers', []))
        cookie_header = headers.get(b'cookie', b'').decode()
        cookies = dict(
            cookie.strip().split('=', 1)
            for cookie in cookie_header.split(';')
            if '=' in cookie
        )
        return cookies.get('access_token')

    @database_sync_to_async
    def get_user_from_token(self, token):
        """Validate JWT and return user."""
        try:
            access_token = AccessToken(token)
            user_id = access_token['user_id']
            return User.objects.get(id=user_id)
        except (TokenError, User.DoesNotExist):
            return AnonymousUser()


# config/asgi.py
from channels.routing import ProtocolTypeRouter, URLRouter
from apps.game.middleware import JWTAuthMiddleware
from apps.game.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': JWTAuthMiddleware(
        URLRouter(websocket_urlpatterns)
    ),
})
```

#### Handling Authentication in Consumer

```python
# apps/game/consumers.py
class GameConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.user = self.scope['user']

        # Reject unauthenticated connections
        if not self.user.is_authenticated:
            await self.close(code=4001)  # Custom close code for auth failure
            return

        # Continue with connection setup...
        self.game_id = self.scope['url_route']['kwargs']['game_id']

        # Verify user can access this game
        can_access = await self.check_game_access()
        if not can_access:
            await self.close(code=4003)  # Custom close code for access denied
            return

        await self.channel_layer.group_add(
            self.game_group_name,
            self.channel_name
        )
        await self.accept()
```

### 4.2 Channel Layers with Redis

#### Redis Configuration Options

```python
# settings.py

# Basic Redis configuration
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [('localhost', 6379)],
        },
    },
}

# Production with connection pooling
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [os.environ.get('REDIS_URL', 'redis://localhost:6379')],
            'capacity': 1500,  # Max messages per channel
            'expiry': 10,  # Message expiry in seconds
            'group_expiry': 86400,  # Group expiry (24 hours)
            'symmetric_encryption_keys': [SECRET_KEY],  # Encrypt messages
        },
    },
}

# Redis Sentinel for high availability
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [(
                'mymaster',  # Sentinel master name
                [('sentinel1', 26379), ('sentinel2', 26379)],
            )],
        },
    },
}
```

#### Channel Layer Usage Patterns

```python
# apps/game/consumers.py
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

class GameConsumer(AsyncWebsocketConsumer):

    async def broadcast_game_update(self, game):
        """Broadcast game state to all connected clients."""
        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'game.update',  # Calls game_update method
                'game_state': await self.serialize_game(game),
                'sender_channel': self.channel_name,
            }
        )

    async def game_update(self, event):
        """Handle game update messages from the group."""
        # Optionally skip sending to the sender
        if event.get('sender_channel') == self.channel_name:
            return

        await self.send(text_data=json.dumps({
            'type': 'game_update',
            'game_state': event['game_state'],
        }))


# Sending from outside a consumer (e.g., from a view or task)
def send_game_notification(game_id, message):
    """Send notification to all clients in a game room."""
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f'game_{game_id}',
        {
            'type': 'notification',
            'message': message,
        }
    )


# Using in an async view
async def process_ai_move(game_id):
    """Process AI move and broadcast update."""
    game = await get_game(game_id)
    move = await calculate_ai_move(game)
    game = await apply_move(game, move)

    # Broadcast to connected clients
    channel_layer = get_channel_layer()
    await channel_layer.group_send(
        f'game_{game_id}',
        {
            'type': 'game.move',
            'move': move.to_dict(),
            'game_state': game.to_dict(),
            'player': 'AI',
        }
    )
```

### 4.3 Handling Reconnection and State Sync

```python
# apps/game/consumers.py
class GameConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        """Handle initial connection and state sync."""
        # ... authentication checks ...

        await self.channel_layer.group_add(
            self.game_group_name,
            self.channel_name
        )
        await self.accept()

        # Send full state on connect (handles reconnection)
        await self.send_full_state()

    async def send_full_state(self):
        """Send complete game state for synchronization."""
        game = await self.get_game()
        state = await database_sync_to_async(game.to_dict)()

        await self.send(text_data=json.dumps({
            'type': 'state_sync',
            'game_state': state,
            'your_color': await self.get_player_color(),
            'connected_players': await self.get_connected_players(),
        }))

    async def receive(self, text_data):
        """Handle incoming messages including sync requests."""
        data = json.loads(text_data)

        if data['type'] == 'request_sync':
            # Client requesting state sync (after reconnection)
            client_version = data.get('version', 0)

            game = await self.get_game()

            if game.version > client_version:
                # Client is behind, send full state
                await self.send_full_state()
            else:
                # Client is up to date
                await self.send(text_data=json.dumps({
                    'type': 'sync_confirmed',
                    'version': game.version,
                }))

        elif data['type'] == 'move':
            await self.handle_move(data)
        # ... other message types ...

    @database_sync_to_async
    def get_connected_players(self):
        """Get list of currently connected players."""
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync

        # This is a simplified version
        # In production, track connections in Redis or database
        return []


# Tracking player presence with Redis
class GameConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        # ... other setup ...

        # Track player presence
        await self.update_presence(connected=True)

        # Notify others
        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'presence.update',
                'user_id': self.user.id,
                'username': self.user.username,
                'status': 'connected',
            }
        )

    async def disconnect(self, close_code):
        await self.update_presence(connected=False)

        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'presence.update',
                'user_id': self.user.id,
                'username': self.user.username,
                'status': 'disconnected',
            }
        )

        await self.channel_layer.group_discard(
            self.game_group_name,
            self.channel_name
        )

    async def update_presence(self, connected):
        """Track presence in Redis."""
        import aioredis

        redis = await aioredis.from_url('redis://localhost')
        key = f'game:{self.game_id}:presence'

        if connected:
            await redis.hset(key, self.user.id, json.dumps({
                'username': self.user.username,
                'connected_at': timezone.now().isoformat(),
                'channel': self.channel_name,
            }))
        else:
            await redis.hdel(key, self.user.id)

        await redis.close()
```

### 4.4 Broadcasting Game Events to Spectators

```python
# apps/game/consumers.py
class GameConsumer(AsyncWebsocketConsumer):
    """
    Enhanced consumer supporting spectators.

    Groups:
    - game_{id}: All connected clients (players + spectators)
    - game_{id}_players: Only players (for private messages)
    """

    async def connect(self):
        self.game_id = self.scope['url_route']['kwargs']['game_id']
        self.game_group_name = f'game_{self.game_id}'
        self.players_group_name = f'game_{self.game_id}_players'
        self.user = self.scope['user']

        # Determine if user is player or spectator
        self.is_player = await self.check_is_player()
        self.is_spectator = not self.is_player

        # Everyone joins the main group
        await self.channel_layer.group_add(
            self.game_group_name,
            self.channel_name
        )

        # Only players join the players group
        if self.is_player:
            await self.channel_layer.group_add(
                self.players_group_name,
                self.channel_name
            )

        await self.accept()

        # Send appropriate state based on role
        await self.send_state_for_role()

        # Announce connection
        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'user.joined',
                'user_id': self.user.id if self.user.is_authenticated else None,
                'username': self.get_display_name(),
                'role': 'player' if self.is_player else 'spectator',
            }
        )

    async def send_state_for_role(self):
        """Send state appropriate for the user's role."""
        game = await self.get_game()

        if self.is_player:
            # Players get full state including private info
            state = await database_sync_to_async(game.to_player_dict)(self.user)
        else:
            # Spectators get public state only
            state = await database_sync_to_async(game.to_spectator_dict)()

        await self.send(text_data=json.dumps({
            'type': 'state_sync',
            'game_state': state,
            'role': 'player' if self.is_player else 'spectator',
        }))

    async def receive(self, text_data):
        """Handle incoming messages with role-based restrictions."""
        data = json.loads(text_data)

        # Spectators can only send certain message types
        spectator_allowed = ['request_sync', 'ping', 'chat']

        if self.is_spectator and data['type'] not in spectator_allowed:
            await self.send_error('Spectators cannot perform this action')
            return

        # Process message...

    async def broadcast_move(self, move, game):
        """Broadcast a move to all connected clients."""
        # Full state to players
        await self.channel_layer.group_send(
            self.players_group_name,
            {
                'type': 'game.move.player',
                'move': move.to_dict(),
                'game_state': game.to_player_dict(),
            }
        )

        # Sanitized state to spectators (via main group)
        # Spectators are in main group but not players group
        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'game.move.spectator',
                'move': move.to_public_dict(),
                'game_state': game.to_spectator_dict(),
            }
        )

    async def game_move_player(self, event):
        """Send move to players with full details."""
        if not self.is_player:
            return  # Skip for spectators

        await self.send(text_data=json.dumps({
            'type': 'move',
            'move': event['move'],
            'game_state': event['game_state'],
        }))

    async def game_move_spectator(self, event):
        """Send move to spectators with public details only."""
        if self.is_player:
            return  # Skip for players (they got the full message)

        await self.send(text_data=json.dumps({
            'type': 'move',
            'move': event['move'],
            'game_state': event['game_state'],
        }))

    # Chat support
    async def handle_chat(self, data):
        """Handle chat messages (allowed for all)."""
        message = data.get('message', '').strip()[:500]  # Limit length

        if not message:
            return

        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'chat.message',
                'user_id': self.user.id if self.user.is_authenticated else None,
                'username': self.get_display_name(),
                'message': message,
                'role': 'player' if self.is_player else 'spectator',
                'timestamp': timezone.now().isoformat(),
            }
        )

    async def chat_message(self, event):
        """Forward chat message to client."""
        await self.send(text_data=json.dumps({
            'type': 'chat',
            **event,
        }))
```

---

## 5. Database Design for Games

### 5.1 Storing Game State: JSON Field vs Normalized Tables

#### Option 1: JSON Field (Denormalized)

**Pros:**
- Simpler queries for full state
- Atomic updates
- Flexible schema
- Faster reads for complete state

**Cons:**
- Harder to query specific positions
- No referential integrity
- Larger storage per game
- Can't use database constraints

```python
from django.db import models
from django.contrib.postgres.fields import ArrayField

class Game(models.Model):
    """Game model using JSON for board state."""

    # Players
    player_white = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games_as_white')
    player_black = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games_as_black', null=True)
    current_player = models.ForeignKey(User, on_delete=models.CASCADE, related_name='current_games')

    # Game status
    status = models.CharField(max_length=20, choices=[
        ('waiting', 'Waiting'),
        ('playing', 'Playing'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
    ], default='waiting')
    winner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='won_games')

    # Board state as JSON
    board_state = models.JSONField(default=dict)
    """
    Structure:
    {
        "points": [
            {"count": 2, "color": "white"},  # Point 1
            {"count": 0, "color": null},      # Point 2
            ...  # 24 points
        ],
        "bar": {"white": 0, "black": 0},
        "borne_off": {"white": 0, "black": 0},
        "dice": [3, 5],
        "dice_used": [false, false],
        "doubling_cube": 1,
        "cube_owner": null
    }
    """

    # Optimistic locking
    version = models.PositiveIntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_initial_board_state(self):
        """Return standard backgammon starting position."""
        return {
            "points": self._get_initial_points(),
            "bar": {"white": 0, "black": 0},
            "borne_off": {"white": 0, "black": 0},
            "dice": [],
            "dice_used": [],
            "doubling_cube": 1,
            "cube_owner": None,
        }

    def _get_initial_points(self):
        """Standard backgammon starting positions."""
        points = [{"count": 0, "color": None} for _ in range(24)]

        # White pieces (moving towards point 24)
        points[0] = {"count": 2, "color": "white"}   # Point 1
        points[11] = {"count": 5, "color": "white"}  # Point 12
        points[16] = {"count": 3, "color": "white"}  # Point 17
        points[18] = {"count": 5, "color": "white"}  # Point 19

        # Black pieces (moving towards point 1)
        points[23] = {"count": 2, "color": "black"}  # Point 24
        points[12] = {"count": 5, "color": "black"}  # Point 13
        points[7] = {"count": 3, "color": "black"}   # Point 8
        points[5] = {"count": 5, "color": "black"}   # Point 6

        return points
```

#### Option 2: Normalized Tables

**Pros:**
- Queryable positions
- Database constraints ensure validity
- Smaller individual records
- Better for analytics

**Cons:**
- More complex queries
- More joins needed
- Multiple updates per move

```python
class Game(models.Model):
    """Game model with normalized state."""
    player_white = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games_as_white')
    player_black = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games_as_black', null=True)
    current_player = models.ForeignKey(User, on_delete=models.CASCADE, related_name='current_games')

    status = models.CharField(max_length=20, default='waiting')
    winner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='won_games')

    # Game rules state
    doubling_cube = models.PositiveIntegerField(default=1)
    cube_owner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='+')

    # Current dice
    die_1 = models.PositiveSmallIntegerField(null=True)
    die_2 = models.PositiveSmallIntegerField(null=True)
    die_1_used = models.BooleanField(default=False)
    die_2_used = models.BooleanField(default=False)

    version = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class BoardPoint(models.Model):
    """Individual point on the board."""
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='points')
    point_number = models.PositiveSmallIntegerField()  # 1-24
    checker_count = models.PositiveSmallIntegerField(default=0)
    checker_color = models.CharField(max_length=5, choices=[
        ('white', 'White'),
        ('black', 'Black'),
    ], null=True)

    class Meta:
        unique_together = ['game', 'point_number']
        ordering = ['point_number']
        constraints = [
            models.CheckConstraint(
                check=models.Q(point_number__gte=1, point_number__lte=24),
                name='valid_point_number'
            ),
            models.CheckConstraint(
                check=(
                    models.Q(checker_count=0, checker_color__isnull=True) |
                    models.Q(checker_count__gt=0, checker_color__isnull=False)
                ),
                name='color_requires_checkers'
            ),
        ]


class GameBar(models.Model):
    """Checkers on the bar."""
    game = models.OneToOneField(Game, on_delete=models.CASCADE, related_name='bar')
    white_count = models.PositiveSmallIntegerField(default=0)
    black_count = models.PositiveSmallIntegerField(default=0)


class GameBorneOff(models.Model):
    """Checkers borne off."""
    game = models.OneToOneField(Game, on_delete=models.CASCADE, related_name='borne_off')
    white_count = models.PositiveSmallIntegerField(default=0)
    black_count = models.PositiveSmallIntegerField(default=0)
```

#### Recommendation: Hybrid Approach

Use JSON for the board state but normalized tables for moves and game metadata.

```python
class Game(models.Model):
    """Hybrid approach: JSON state + normalized relations."""

    # Normalized: Players and game metadata
    player_white = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games_as_white')
    player_black = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games_as_black', null=True)
    current_player = models.ForeignKey(User, on_delete=models.CASCADE, related_name='current_games')
    status = models.CharField(max_length=20, default='waiting')
    winner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='won_games')

    # Denormalized: Board state as JSON (single source of truth)
    board_state = models.JSONField(default=dict)

    # Normalized: Game settings
    time_control = models.CharField(max_length=20, default='none')
    stake = models.PositiveIntegerField(default=1)
    match_length = models.PositiveIntegerField(default=1)

    # Locking and timestamps
    version = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True)

    class Meta:
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['player_white', 'status']),
            models.Index(fields=['player_black', 'status']),
        ]
```

### 5.2 Move History: Event Sourcing vs Snapshot

#### Event Sourcing (Store All Moves)

**Concept:** Store every move as an event. The current state is derived by replaying all events.

**Pros:**
- Complete history for replays
- Can reconstruct any past state
- Natural audit trail
- Good for analysis and ML training

**Cons:**
- Slower to get current state (needs replay)
- More storage for long games
- Migration complexity

```python
class Move(models.Model):
    """
    Event sourced move record.

    Each move is immutable and represents a single event in the game.
    """
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='moves')
    player = models.ForeignKey(User, on_delete=models.CASCADE)
    move_number = models.PositiveIntegerField()

    # Move details
    from_point = models.PositiveSmallIntegerField()  # 0=bar, 25=bear-off
    to_point = models.PositiveSmallIntegerField()
    dice_value = models.PositiveSmallIntegerField()  # Which die was used

    # Derived/computed at move time
    is_hit = models.BooleanField(default=False)
    is_bear_off = models.BooleanField(default=False)

    # For event sourcing: store the resulting state
    resulting_board_state = models.JSONField(null=True)  # Optional snapshot

    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['game', 'move_number']
        ordering = ['move_number']
        indexes = [
            models.Index(fields=['game', 'move_number']),
            models.Index(fields=['game', 'player']),
        ]

    def __str__(self):
        return f"Move {self.move_number}: {self.from_point} -> {self.to_point}"


class Turn(models.Model):
    """
    Groups moves within a single turn.

    A turn contains:
    - Dice roll
    - 1-4 moves (depending on dice and available moves)
    - Optional doubling decision
    """
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='turns')
    player = models.ForeignKey(User, on_delete=models.CASCADE)
    turn_number = models.PositiveIntegerField()

    dice_roll = ArrayField(models.PositiveSmallIntegerField(), size=2)
    is_doubles = models.BooleanField(default=False)

    # Doubling decisions
    double_offered = models.BooleanField(default=False)
    double_accepted = models.BooleanField(null=True)

    # State before this turn (for quick replay)
    board_state_before = models.JSONField()

    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True)

    class Meta:
        unique_together = ['game', 'turn_number']
        ordering = ['turn_number']


# Service to replay game state
class GameReplayService:
    """Service to reconstruct game state from moves."""

    @staticmethod
    def get_state_at_move(game, move_number):
        """Reconstruct board state at a specific move."""
        # Start from initial state
        state = Game.get_initial_board_state()

        # Apply all moves up to move_number
        moves = game.moves.filter(move_number__lte=move_number).order_by('move_number')

        for move in moves:
            state = GameReplayService.apply_move(state, move)

        return state

    @staticmethod
    def apply_move(state, move):
        """Apply a single move to a state."""
        points = state['points']

        # Handle move from bar
        if move.from_point == 0:
            if move.player.color == 'white':
                state['bar']['white'] -= 1
            else:
                state['bar']['black'] -= 1
        else:
            # Remove from source point
            from_point = points[move.from_point - 1]
            from_point['count'] -= 1
            if from_point['count'] == 0:
                from_point['color'] = None

        # Handle bear-off
        if move.to_point == 25:
            if move.player.color == 'white':
                state['borne_off']['white'] += 1
            else:
                state['borne_off']['black'] += 1
        else:
            # Add to destination point
            to_point = points[move.to_point - 1]

            # Check for hit
            if move.is_hit:
                # Send opponent to bar
                opponent_color = 'black' if move.player.color == 'white' else 'white'
                state['bar'][opponent_color] += 1
                to_point['count'] = 0

            to_point['count'] += 1
            to_point['color'] = move.player.color

        return state
```

#### Snapshot Approach

**Concept:** Store the current state directly. Optionally store periodic snapshots for history.

**Pros:**
- Fast current state retrieval
- Simple implementation
- Less storage

**Cons:**
- No automatic history
- Need explicit snapshot mechanism for replay
- Harder to debug issues

```python
class Game(models.Model):
    # Current state is always in board_state
    board_state = models.JSONField(default=dict)

    # Store snapshots periodically for replay
    def save_snapshot(self):
        """Save current state as a snapshot."""
        Snapshot.objects.create(
            game=self,
            move_number=self.moves.count(),
            state=self.board_state,
        )

    def get_snapshot_interval(self):
        """Determine how often to snapshot (every N moves)."""
        return 10  # Snapshot every 10 moves


class Snapshot(models.Model):
    """Periodic game state snapshot for replay."""
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='snapshots')
    move_number = models.PositiveIntegerField()
    state = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['game', 'move_number']
        ordering = ['move_number']
```

#### Recommendation: Event Sourcing with Materialized State

Best of both worlds: store all moves for history, but also maintain current state for fast access.

```python
class Game(models.Model):
    # Materialized current state (updated after each move)
    board_state = models.JSONField(default=dict)
    version = models.PositiveIntegerField(default=0)

    def apply_and_record_move(self, player, from_point, to_point, dice_value):
        """Apply a move and record it as an event."""
        with transaction.atomic():
            # Validate move
            self.validate_move(from_point, to_point, dice_value)

            # Create move record (event)
            move = Move.objects.create(
                game=self,
                player=player,
                move_number=self.moves.count() + 1,
                from_point=from_point,
                to_point=to_point,
                dice_value=dice_value,
                is_hit=self.would_hit(to_point),
                is_bear_off=(to_point == 25),
            )

            # Update materialized state
            self.update_board_state(move)
            self.version += 1
            self.save()

            # Periodic snapshot for long games
            if move.move_number % 20 == 0:
                self.save_snapshot(move.move_number)

            return move
```

### 5.3 PostgreSQL-Specific Features

#### JSONB Querying

```python
from django.db.models import F
from django.contrib.postgres.fields.jsonb import KeyTextTransform

# Query games where white has pieces on the bar
games_with_white_on_bar = Game.objects.filter(
    board_state__bar__white__gt=0
)

# Query games where doubling cube is high
high_stake_games = Game.objects.filter(
    board_state__doubling_cube__gte=8
)

# Using KeyTextTransform for complex queries
from django.db.models.functions import Cast
from django.db.models import IntegerField

games = Game.objects.annotate(
    cube_value=Cast(
        KeyTextTransform('doubling_cube', 'board_state'),
        output_field=IntegerField()
    )
).filter(cube_value__gte=4)

# Count checkers on specific points using JSON path
# PostgreSQL 12+ supports jsonpath
from django.db.models import Func

class JsonbArrayElement(Func):
    function = 'jsonb_array_element'
    template = "(%(expressions)s)[%(index)s]"

    def __init__(self, expression, index, **extra):
        super().__init__(expression, index=index, **extra)
```

#### JSONB Indexes

```python
# migrations/00XX_add_jsonb_indexes.py
from django.db import migrations

class Migration(migrations.Migration):
    operations = [
        # GIN index for general JSONB queries
        migrations.RunSQL(
            sql='''
                CREATE INDEX game_board_state_gin
                ON game_game USING GIN (board_state);
            ''',
            reverse_sql='DROP INDEX game_board_state_gin;'
        ),

        # Expression index for specific field
        migrations.RunSQL(
            sql='''
                CREATE INDEX game_doubling_cube
                ON game_game ((board_state->>'doubling_cube')::int);
            ''',
            reverse_sql='DROP INDEX game_doubling_cube;'
        ),
    ]
```

#### ArrayField for Dice

```python
from django.contrib.postgres.fields import ArrayField

class Turn(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE)

    # Dice as array
    dice = ArrayField(
        models.PositiveSmallIntegerField(),
        size=2,
        default=list
    )

    # Which dice have been used
    dice_used = ArrayField(
        models.BooleanField(),
        size=4,  # Up to 4 for doubles
        default=list
    )


# Query games where doubles were rolled
doubles_turns = Turn.objects.filter(
    dice__0=F('dice__1')  # First die equals second die
)

# Using array operators
from django.contrib.postgres.fields import ArrayField
from django.db.models import Q

# Find turns with specific dice
Turn.objects.filter(dice__contains=[6])  # Contains a 6
Turn.objects.filter(dice__contained_by=[1, 2, 3, 4, 5, 6])  # Valid dice
Turn.objects.filter(dice__overlap=[5, 6])  # Has 5 or 6
```

#### Full-Text Search for Game Analysis

```python
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank

class GameAnnotation(models.Model):
    """Annotations and commentary on games."""
    game = models.ForeignKey(Game, on_delete=models.CASCADE)
    move_number = models.PositiveIntegerField()
    comment = models.TextField()

    # Search vector field (updated via trigger)
    search_vector = models.GeneratedField(
        expression=SearchVector('comment'),
        output_field=models.TextField(),
        db_persist=True,
    )


# Search annotations
query = SearchQuery('blitz')
GameAnnotation.objects.annotate(
    rank=SearchRank(F('search_vector'), query)
).filter(
    search_vector=query
).order_by('-rank')
```

#### Database Triggers for State Validation

```sql
-- migrations/00XX_add_triggers.sql

-- Trigger to validate board state
CREATE OR REPLACE FUNCTION validate_board_state()
RETURNS TRIGGER AS $$
DECLARE
    white_count INTEGER;
    black_count INTEGER;
BEGIN
    -- Count white checkers
    SELECT
        COALESCE(SUM((p->>'count')::int), 0) +
        (NEW.board_state->'bar'->>'white')::int +
        (NEW.board_state->'borne_off'->>'white')::int
    INTO white_count
    FROM jsonb_array_elements(NEW.board_state->'points') AS p
    WHERE p->>'color' = 'white';

    -- Count black checkers
    SELECT
        COALESCE(SUM((p->>'count')::int), 0) +
        (NEW.board_state->'bar'->>'black')::int +
        (NEW.board_state->'borne_off'->>'black')::int
    INTO black_count
    FROM jsonb_array_elements(NEW.board_state->'points') AS p
    WHERE p->>'color' = 'black';

    -- Each player should have exactly 15 checkers
    IF white_count != 15 OR black_count != 15 THEN
        RAISE EXCEPTION 'Invalid board state: checker count mismatch (white=%, black=%)',
            white_count, black_count;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_game_state
BEFORE UPDATE ON game_game
FOR EACH ROW
EXECUTE FUNCTION validate_board_state();
```

---

## 6. Implementation Recommendations

Based on the research above, here are specific recommendations for the Backgammon project:

### 6.1 Project Architecture

```
backend/
├── config/
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── asgi.py
│   ├── wsgi.py
│   └── urls.py
├── apps/
│   ├── accounts/
│   │   ├── models.py          # Custom User model
│   │   ├── serializers.py     # Auth serializers
│   │   ├── views.py           # Login, register, guest
│   │   └── urls.py
│   ├── game/
│   │   ├── models.py          # Game, Move, Turn models
│   │   ├── serializers.py     # Game state serializers
│   │   ├── views.py           # GameViewSet
│   │   ├── consumers.py       # WebSocket consumers
│   │   ├── routing.py         # WebSocket routes
│   │   ├── permissions.py     # Game-specific permissions
│   │   └── services/
│   │       ├── game_logic.py  # Core game rules
│   │       ├── move_validator.py
│   │       └── board_state.py
│   └── ai/
│       ├── models.py          # AI configurations
│       ├── players/           # AI player implementations
│       └── training/          # ML training code
└── requirements/
    ├── base.txt
    ├── development.txt
    └── production.txt
```

### 6.2 Key Dependencies

```txt
# requirements/base.txt
Django>=5.0
djangorestframework>=3.14
djangorestframework-simplejwt>=5.3
channels>=4.0
channels-redis>=4.2
psycopg[binary]>=3.1
redis>=5.0

# requirements/development.txt
-r base.txt
django-debug-toolbar
pytest-django
pytest-asyncio
factory-boy
```

### 6.3 Implementation Priority

1. **Phase 1: Core API**
   - User authentication (JWT)
   - Game CRUD operations
   - Move validation and application
   - Basic game state management

2. **Phase 2: Real-time**
   - WebSocket setup with Channels
   - Live game state updates
   - Player presence tracking

3. **Phase 3: Advanced Features**
   - Guest user support
   - Spectator mode
   - Game history and replay
   - Matchmaking

4. **Phase 4: AI Integration**
   - AI player interface
   - Async AI move calculation
   - Multiple AI difficulty levels

### 6.4 Key Design Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Locking Strategy | Optimistic | Turn-based = low contention |
| Board State Storage | JSON field | Atomic updates, flexible |
| Move History | Event sourcing | ML training, replay |
| Real-time | Django Channels | Native Django integration |
| Authentication | JWT | Stateless, mobile-friendly |
| Database | PostgreSQL | JSONB, performance |

---

## References

- [Django 5.0 Async Documentation](https://docs.djangoproject.com/en/5.0/topics/async/)
- [Django Channels Documentation](https://channels.readthedocs.io/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [Simple JWT Documentation](https://django-rest-framework-simplejwt.readthedocs.io/)
- [PostgreSQL JSONB](https://www.postgresql.org/docs/current/datatype-json.html)

---

*Last updated: December 2024*
*For: Backgammon AI Sandbox Project*
