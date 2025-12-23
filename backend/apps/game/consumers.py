"""
WebSocket consumers for real-time game updates.
"""
import json
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser

from .models import Game, Move
from .serializers import GameDetailSerializer, MoveSerializer
from .services.game_engine import GameEngine


class GameConsumer(AsyncJsonWebsocketConsumer):
    """
    WebSocket consumer for real-time game communication.

    Handles:
    - Game state updates
    - Move notifications
    - Chat messages
    - Spectator mode
    """

    async def connect(self):
        """Handle WebSocket connection."""
        # Get game ID from URL
        self.game_id = self.scope['url_route']['kwargs']['game_id']
        self.game_group_name = f'game_{self.game_id}'
        self.user = self.scope.get('user', AnonymousUser())

        # Reject anonymous connections
        if isinstance(self.user, AnonymousUser):
            await self.close(code=4001)
            return

        # Verify user has access to this game
        game = await self.get_game()
        if not game:
            await self.close(code=4004)
            return

        # Check if user is a player or can spectate
        self.is_player = await self.is_game_player(game)
        self.player_color = await self.get_player_color(game)

        # Join game group
        await self.channel_layer.group_add(
            self.game_group_name,
            self.channel_name
        )

        await self.accept()

        # Send current game state
        game_data = await self.get_game_state()
        await self.send_json({
            'type': 'game_state',
            'data': game_data
        })

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        # Leave game group
        await self.channel_layer.group_discard(
            self.game_group_name,
            self.channel_name
        )

    async def receive_json(self, content):
        """Handle incoming WebSocket messages."""
        message_type = content.get('type')

        if message_type == 'roll_dice':
            await self.handle_roll_dice()
        elif message_type == 'make_move':
            await self.handle_make_move(content)
        elif message_type == 'offer_double':
            await self.handle_offer_double()
        elif message_type == 'respond_double':
            await self.handle_respond_double(content)
        elif message_type == 'resign':
            await self.handle_resign()
        elif message_type == 'chat':
            await self.handle_chat(content)

    async def handle_roll_dice(self):
        """Handle dice roll request."""
        if not self.is_player:
            await self.send_error("You are not a player in this game.")
            return

        game = await self.get_game()
        if not game or game.status != Game.Status.ACTIVE:
            await self.send_error("Game is not active.")
            return

        if game.current_turn != self.player_color:
            await self.send_error("It's not your turn.")
            return

        # Roll dice
        result = await self.roll_dice(game)

        # Broadcast to all clients
        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'game_update',
                'event': 'dice_rolled',
                'data': result
            }
        )

    async def handle_make_move(self, content):
        """Handle move request."""
        if not self.is_player:
            await self.send_error("You are not a player in this game.")
            return

        checker_moves = content.get('moves', [])
        version = content.get('version')

        game = await self.get_game()
        if not game or game.status != Game.Status.ACTIVE:
            await self.send_error("Game is not active.")
            return

        if game.current_turn != self.player_color:
            await self.send_error("It's not your turn.")
            return

        if version != game.version:
            await self.send_error("Game state changed. Please refresh.")
            return

        try:
            result = await self.make_moves(game, checker_moves)

            # Broadcast to all clients
            await self.channel_layer.group_send(
                self.game_group_name,
                {
                    'type': 'game_update',
                    'event': 'move_made',
                    'data': result
                }
            )
        except ValueError as e:
            await self.send_error(str(e))

    async def handle_offer_double(self):
        """Handle double offer."""
        if not self.is_player:
            await self.send_error("You are not a player in this game.")
            return

        game = await self.get_game()

        try:
            result = await self.offer_double(game)

            await self.channel_layer.group_send(
                self.game_group_name,
                {
                    'type': 'game_update',
                    'event': 'double_offered',
                    'data': result
                }
            )
        except ValueError as e:
            await self.send_error(str(e))

    async def handle_respond_double(self, content):
        """Handle response to double."""
        if not self.is_player:
            await self.send_error("You are not a player in this game.")
            return

        accept = content.get('accept', False)
        game = await self.get_game()

        try:
            if accept:
                result = await self.accept_double(game)
                event = 'double_accepted'
            else:
                result = await self.reject_double(game)
                event = 'double_rejected'

            await self.channel_layer.group_send(
                self.game_group_name,
                {
                    'type': 'game_update',
                    'event': event,
                    'data': result
                }
            )
        except ValueError as e:
            await self.send_error(str(e))

    async def handle_resign(self):
        """Handle resignation."""
        if not self.is_player:
            await self.send_error("You are not a player in this game.")
            return

        game = await self.get_game()

        try:
            result = await self.resign(game)

            await self.channel_layer.group_send(
                self.game_group_name,
                {
                    'type': 'game_update',
                    'event': 'player_resigned',
                    'data': result
                }
            )
        except ValueError as e:
            await self.send_error(str(e))

    async def handle_chat(self, content):
        """Handle chat message."""
        message = content.get('message', '').strip()
        if not message:
            return

        # Limit message length
        message = message[:500]

        await self.channel_layer.group_send(
            self.game_group_name,
            {
                'type': 'chat_message',
                'username': self.user.username,
                'message': message
            }
        )

    # Event handlers for channel layer messages

    async def game_update(self, event):
        """Send game update to client."""
        game_data = await self.get_game_state()

        await self.send_json({
            'type': 'game_update',
            'event': event['event'],
            'data': event.get('data', {}),
            'game': game_data
        })

    async def chat_message(self, event):
        """Send chat message to client."""
        await self.send_json({
            'type': 'chat',
            'username': event['username'],
            'message': event['message']
        })

    # Helper methods

    async def send_error(self, message: str):
        """Send error message to client."""
        await self.send_json({
            'type': 'error',
            'message': message
        })

    @database_sync_to_async
    def get_game(self):
        """Get the game instance."""
        try:
            return Game.objects.select_related(
                'white_player', 'black_player'
            ).get(id=self.game_id)
        except Game.DoesNotExist:
            return None

    @database_sync_to_async
    def get_game_state(self):
        """Get serialized game state."""
        game = Game.objects.select_related(
            'white_player', 'black_player', 'winner'
        ).prefetch_related('moves').get(id=self.game_id)
        return GameDetailSerializer(game).data

    @database_sync_to_async
    def is_game_player(self, game) -> bool:
        """Check if the user is a player in the game."""
        return self.user in [game.white_player, game.black_player]

    @database_sync_to_async
    def get_player_color(self, game) -> str:
        """Get the player's color in the game."""
        return game.get_player_color(self.user)

    @database_sync_to_async
    def roll_dice(self, game):
        """Roll dice for the game."""
        engine = GameEngine(game)
        return engine.roll_dice()

    @database_sync_to_async
    def make_moves(self, game, checker_moves):
        """Execute moves in the game."""
        engine = GameEngine(game)
        return engine.make_moves(self.user, checker_moves)

    @database_sync_to_async
    def offer_double(self, game):
        """Offer to double the stakes."""
        engine = GameEngine(game)
        return engine.offer_double(self.user)

    @database_sync_to_async
    def accept_double(self, game):
        """Accept the offered double."""
        engine = GameEngine(game)
        return engine.accept_double(self.user)

    @database_sync_to_async
    def reject_double(self, game):
        """Reject the offered double."""
        engine = GameEngine(game)
        return engine.reject_double(self.user)

    @database_sync_to_async
    def resign(self, game):
        """Resign from the game."""
        engine = GameEngine(game)
        return engine.resign(self.user)


class LobbyConsumer(AsyncJsonWebsocketConsumer):
    """
    WebSocket consumer for the game lobby.

    Handles:
    - New game notifications
    - Player online status
    - Game invitations
    """

    async def connect(self):
        """Handle connection to lobby."""
        self.user = self.scope.get('user', AnonymousUser())

        if isinstance(self.user, AnonymousUser):
            await self.close(code=4001)
            return

        self.lobby_group = 'lobby'
        self.user_group = f'user_{self.user.id}'

        # Join lobby and personal groups
        await self.channel_layer.group_add(self.lobby_group, self.channel_name)
        await self.channel_layer.group_add(self.user_group, self.channel_name)

        await self.accept()

        # Notify others that user is online
        await self.channel_layer.group_send(
            self.lobby_group,
            {
                'type': 'user_status',
                'user_id': str(self.user.id),
                'username': self.user.username,
                'status': 'online'
            }
        )

    async def disconnect(self, close_code):
        """Handle disconnection from lobby."""
        # Notify others that user is offline
        await self.channel_layer.group_send(
            self.lobby_group,
            {
                'type': 'user_status',
                'user_id': str(self.user.id),
                'username': self.user.username,
                'status': 'offline'
            }
        )

        await self.channel_layer.group_discard(self.lobby_group, self.channel_name)
        await self.channel_layer.group_discard(self.user_group, self.channel_name)

    async def receive_json(self, content):
        """Handle incoming messages."""
        message_type = content.get('type')

        if message_type == 'invite':
            await self.handle_invite(content)

    async def handle_invite(self, content):
        """Handle game invitation."""
        to_user_id = content.get('to_user_id')

        # Send invite to specific user
        await self.channel_layer.group_send(
            f'user_{to_user_id}',
            {
                'type': 'game_invite',
                'from_user_id': str(self.user.id),
                'from_username': self.user.username
            }
        )

    # Event handlers

    async def user_status(self, event):
        """Broadcast user status change."""
        await self.send_json({
            'type': 'user_status',
            'user_id': event['user_id'],
            'username': event['username'],
            'status': event['status']
        })

    async def game_invite(self, event):
        """Send game invitation to user."""
        await self.send_json({
            'type': 'game_invite',
            'from_user_id': event['from_user_id'],
            'from_username': event['from_username']
        })

    async def new_game(self, event):
        """Broadcast new game available."""
        await self.send_json({
            'type': 'new_game',
            'game': event['game']
        })
