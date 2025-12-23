"""Serializers for the game app."""
from rest_framework import serializers
from django.contrib.auth import get_user_model

from .models import Game, Move, GameInvite

User = get_user_model()


class PlayerSerializer(serializers.ModelSerializer):
    """Minimal player info for game context."""

    class Meta:
        model = User
        fields = ['id', 'username', 'avatar', 'elo_rating']


class MoveSerializer(serializers.ModelSerializer):
    """Serializer for game moves."""

    player = PlayerSerializer(read_only=True)

    class Meta:
        model = Move
        fields = [
            'id', 'player', 'move_number', 'move_type',
            'dice_values', 'checker_moves', 'board_state_after', 'created_at'
        ]
        read_only_fields = ['id', 'move_number', 'board_state_after', 'created_at']


class GameListSerializer(serializers.ModelSerializer):
    """Serializer for game list view."""

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
    """Serializer for game detail view with full state."""

    white_player = PlayerSerializer(read_only=True)
    black_player = PlayerSerializer(read_only=True)
    winner = PlayerSerializer(read_only=True)
    moves = MoveSerializer(many=True, read_only=True)

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
    """Serializer for making a move."""

    move_type = serializers.ChoiceField(choices=Move.MoveType.choices)
    checker_moves = serializers.ListField(
        child=serializers.ListField(
            child=serializers.IntegerField(),
            min_length=2,
            max_length=2
        ),
        required=False
    )
    version = serializers.IntegerField(required=True)

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
