"""
Serializers for the game app.

This module contains DRF serializers for:
- Game list/detail views
- Move history
- Game invitations
- Player information

All serializers are documented for OpenAPI schema generation via drf-spectacular.
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model

from .models import Game, Move, GameInvite

User = get_user_model()


class PlayerSerializer(serializers.ModelSerializer):
    """
    Minimal player information for game context.

    Used for embedding player data in game responses without
    exposing full user profile details.
    """

    class Meta:
        model = User
        fields = ['id', 'username', 'avatar', 'elo_rating']


class MoveSerializer(serializers.ModelSerializer):
    """
    Serializer for game moves.

    Represents a single action in a game (dice roll, checker move, double, etc.).
    Used for move history and replay functionality.
    """

    player = PlayerSerializer(read_only=True, help_text="Player who made this move")

    class Meta:
        model = Move
        fields = [
            'id', 'player', 'move_number', 'move_type',
            'dice_values', 'checker_moves', 'board_state_after', 'created_at'
        ]
        read_only_fields = ['id', 'move_number', 'board_state_after', 'created_at']


class GameListSerializer(serializers.ModelSerializer):
    """
    Serializer for game list view.

    Provides summary information for displaying games in lists.
    Does not include full board state to minimize payload size.
    """

    white_player = PlayerSerializer(read_only=True)
    black_player = PlayerSerializer(read_only=True)
    winner = PlayerSerializer(read_only=True)

    class Meta:
        model = Game
        fields = [
            'id', 'white_player', 'black_player', 'status',
            'current_turn', 'winner', 'win_type', 'points_won',
            'created_at', 'started_at', 'completed_at'
        ]


class GameDetailSerializer(serializers.ModelSerializer):
    """
    Serializer for game detail view with full state.

    Includes complete board state, dice values, doubling cube state,
    and full move history. Used for the game play interface.
    """

    white_player = PlayerSerializer(read_only=True, help_text="Player controlling white checkers")
    black_player = PlayerSerializer(read_only=True, help_text="Player controlling black checkers")
    winner = PlayerSerializer(read_only=True, help_text="Winner of the game (null if ongoing)")
    moves = MoveSerializer(many=True, read_only=True, help_text="Complete move history")

    class Meta:
        model = Game
        fields = [
            'id', 'white_player', 'black_player', 'status',
            'board_state', 'current_turn', 'dice', 'moves_remaining',
            'cube_value', 'cube_owner', 'double_offered',
            'winner', 'win_type', 'points_won',
            'created_at', 'started_at', 'completed_at', 'updated_at',
            'version', 'moves'
        ]


class GameCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new game."""

    class Meta:
        model = Game
        fields = ['id']
        read_only_fields = ['id']

    def create(self, validated_data):
        """Create a new game with the current user as white."""
        user = self.context['request'].user
        game = Game.objects.create(white_player=user)
        return game


class GameJoinSerializer(serializers.Serializer):
    """Serializer for joining a game."""

    game_id = serializers.UUIDField()

    def validate_game_id(self, value):
        """Validate that the game exists and can be joined."""
        try:
            game = Game.objects.get(id=value)
        except Game.DoesNotExist:
            raise serializers.ValidationError("Game not found.")

        if game.status != Game.Status.WAITING:
            raise serializers.ValidationError("This game is not accepting players.")

        if game.black_player is not None:
            raise serializers.ValidationError("This game is already full.")

        user = self.context['request'].user
        if game.white_player == user:
            raise serializers.ValidationError("You cannot join your own game.")

        return value


class MakeMoveSerializer(serializers.Serializer):
    """
    Serializer for making a move.

    Handles validation for all move types including dice rolls,
    checker movements, doubling, and resignation. Uses optimistic
    locking via version field to prevent race conditions.
    """

    move_type = serializers.ChoiceField(
        choices=Move.MoveType.choices,
        help_text="Type of move: roll, move, double, accept_double, reject_double, resign"
    )
    checker_moves = serializers.ListField(
        child=serializers.ListField(
            child=serializers.IntegerField(),
            min_length=2,
            max_length=2
        ),
        required=False,
        help_text="List of [from_point, to_point] pairs for checker moves"
    )
    version = serializers.IntegerField(
        required=True,
        help_text="Game version for optimistic locking (prevents race conditions)"
    )

    def validate(self, attrs):
        """Validate the move is legal."""
        game = self.context['game']
        user = self.context['request'].user

        # Check if it's the player's turn
        player_color = game.get_player_color(user)
        if player_color != game.current_turn:
            raise serializers.ValidationError("It's not your turn.")

        # Check version for optimistic locking
        if attrs['version'] != game.version:
            raise serializers.ValidationError(
                "Game state has changed. Please refresh and try again."
            )

        return attrs


class GameInviteSerializer(serializers.ModelSerializer):
    """Serializer for game invites."""

    from_user = PlayerSerializer(read_only=True)
    to_user = PlayerSerializer(read_only=True)

    class Meta:
        model = GameInvite
        fields = [
            'id', 'from_user', 'to_user', 'status',
            'game', 'created_at', 'responded_at'
        ]
        read_only_fields = ['id', 'from_user', 'status', 'game', 'created_at', 'responded_at']


class CreateInviteSerializer(serializers.Serializer):
    """Serializer for creating a game invite."""

    to_username = serializers.CharField()

    def validate_to_username(self, value):
        """Validate the invited user exists and is not the sender."""
        try:
            to_user = User.objects.get(username=value)
        except User.DoesNotExist:
            raise serializers.ValidationError("User not found.")

        from_user = self.context['request'].user
        if to_user == from_user:
            raise serializers.ValidationError("You cannot invite yourself.")

        # Check for existing pending invite
        existing = GameInvite.objects.filter(
            from_user=from_user,
            to_user=to_user,
            status=GameInvite.Status.PENDING
        ).exists()
        if existing:
            raise serializers.ValidationError(
                "You already have a pending invite to this user."
            )

        return value
