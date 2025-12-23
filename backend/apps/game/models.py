"""Models for the game app."""
import uuid
from django.conf import settings
from django.db import models
from django.utils import timezone


class Game(models.Model):
    """
    Represents a backgammon game between two players.

    The board state is stored as JSON representing checker positions.
    """

    class Status(models.TextChoices):
        WAITING = 'waiting', 'Waiting for Player'
        ACTIVE = 'active', 'Active'
        COMPLETED = 'completed', 'Completed'
        ABANDONED = 'abandoned', 'Abandoned'

    class WinType(models.TextChoices):
        NORMAL = 'normal', 'Normal'
        GAMMON = 'gammon', 'Gammon'
        BACKGAMMON = 'backgammon', 'Backgammon'
        RESIGN = 'resign', 'Resignation'
        TIMEOUT = 'timeout', 'Timeout'

    # Identifiers
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Players
    white_player = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='games_as_white',
        null=True,
        blank=True
    )
    black_player = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='games_as_black',
        null=True,
        blank=True
    )

    # Game state
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.WAITING
    )
    board_state = models.JSONField(default=dict)
    current_turn = models.CharField(
        max_length=5,
        choices=[('white', 'White'), ('black', 'Black')],
        default='white'
    )
    dice = models.JSONField(default=list)  # Current dice values
    moves_remaining = models.JSONField(default=list)  # Available moves from dice

    # Doubling cube
    cube_value = models.PositiveIntegerField(default=1)
    cube_owner = models.CharField(
        max_length=5,
        choices=[('white', 'White'), ('black', 'Black'), ('center', 'Center')],
        default='center'
    )
    double_offered = models.BooleanField(default=False)

    # Game result
    winner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name='games_won',
        null=True,
        blank=True
    )
    win_type = models.CharField(
        max_length=20,
        choices=WinType.choices,
        null=True,
        blank=True
    )
    points_won = models.PositiveIntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optimistic locking
    version = models.PositiveIntegerField(default=0)

    class Meta:
        db_table = 'games'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['white_player', 'status']),
            models.Index(fields=['black_player', 'status']),
        ]

    def __str__(self):
        white = self.white_player.username if self.white_player else 'Waiting'
        black = self.black_player.username if self.black_player else 'Waiting'
        return f"Game {self.id}: {white} vs {black}"

    def save(self, *args, **kwargs):
        """Increment version on every save for optimistic locking."""
        self.version += 1
        super().save(*args, **kwargs)

    @classmethod
    def get_initial_board(cls) -> dict:
        """
        Return the initial board state for a new game.

        Board representation:
        - Points 1-24 represent the board positions
        - Point 0 is white's bar
        - Point 25 is black's bar
        - Point 26 is white's home (bearoff)
        - Point 27 is black's home (bearoff)

        Positive numbers = white checkers
        Negative numbers = black checkers
        """
        return {
            'points': {
                '1': 2,     # White starts with 2 on point 1
                '6': -5,    # Black starts with 5 on point 6
                '8': -3,    # Black starts with 3 on point 8
                '12': 5,    # White starts with 5 on point 12
                '13': -5,   # Black starts with 5 on point 13
                '17': 3,    # White starts with 3 on point 17
                '19': 5,    # White starts with 5 on point 19
                '24': -2,   # Black starts with 2 on point 24
            },
            'bar': {'white': 0, 'black': 0},
            'home': {'white': 0, 'black': 0},
        }

    def initialize_board(self) -> None:
        """Set up the initial board state."""
        self.board_state = self.get_initial_board()
        self.save(update_fields=['board_state', 'version'])

    def start_game(self) -> None:
        """Start the game when both players are present."""
        if self.white_player and self.black_player:
            self.status = self.Status.ACTIVE
            self.started_at = timezone.now()
            self.initialize_board()
            self.save(update_fields=['status', 'started_at', 'version'])

    def complete_game(self, winner, win_type: str = WinType.NORMAL) -> None:
        """Mark the game as completed."""
        self.status = self.Status.COMPLETED
        self.winner = winner
        self.win_type = win_type
        self.completed_at = timezone.now()

        # Calculate points won based on win type and cube
        multiplier = {'normal': 1, 'gammon': 2, 'backgammon': 3, 'resign': 1, 'timeout': 1}
        self.points_won = self.cube_value * multiplier.get(win_type, 1)

        self.save(update_fields=[
            'status', 'winner', 'win_type', 'completed_at', 'points_won', 'version'
        ])

    def get_opponent(self, user):
        """Get the opponent of the given user."""
        if user == self.white_player:
            return self.black_player
        elif user == self.black_player:
            return self.white_player
        return None

    def get_player_color(self, user) -> str | None:
        """Get the color of the given user in this game."""
        if user == self.white_player:
            return 'white'
        elif user == self.black_player:
            return 'black'
        return None


class Move(models.Model):
    """
    Represents a single move in a game.

    Stores moves as events for replay and analysis.
    """

    class MoveType(models.TextChoices):
        ROLL = 'roll', 'Dice Roll'
        MOVE = 'move', 'Move Checker'
        DOUBLE = 'double', 'Offer Double'
        ACCEPT_DOUBLE = 'accept_double', 'Accept Double'
        REJECT_DOUBLE = 'reject_double', 'Reject Double'
        RESIGN = 'resign', 'Resign'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    game = models.ForeignKey(
        Game,
        on_delete=models.CASCADE,
        related_name='moves'
    )
    player = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='moves'
    )
    move_number = models.PositiveIntegerField()

    move_type = models.CharField(
        max_length=20,
        choices=MoveType.choices
    )

    # For dice rolls
    dice_values = models.JSONField(null=True, blank=True)

    # For checker moves: list of (from_point, to_point) tuples
    checker_moves = models.JSONField(null=True, blank=True)

    # Board state after this move (for easy replay)
    board_state_after = models.JSONField(default=dict)

    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'moves'
        ordering = ['game', 'move_number']
        unique_together = ['game', 'move_number']
        indexes = [
            models.Index(fields=['game', 'move_number']),
        ]

    def __str__(self):
        return f"Move {self.move_number} in {self.game_id}: {self.move_type}"


class GameInvite(models.Model):
    """
    Represents an invitation to play a game.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        ACCEPTED = 'accepted', 'Accepted'
        DECLINED = 'declined', 'Declined'
        EXPIRED = 'expired', 'Expired'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    from_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='sent_invites'
    )
    to_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='received_invites'
    )

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )

    game = models.OneToOneField(
        Game,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='invite'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    responded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'game_invites'
        ordering = ['-created_at']

    def __str__(self):
        return f"Invite from {self.from_user} to {self.to_user}"

    def accept(self) -> Game:
        """Accept the invite and create a game."""
        game = Game.objects.create(
            white_player=self.from_user,
            black_player=self.to_user,
        )
        game.start_game()

        self.status = self.Status.ACCEPTED
        self.game = game
        self.responded_at = timezone.now()
        self.save()

        return game

    def decline(self) -> None:
        """Decline the invite."""
        self.status = self.Status.DECLINED
        self.responded_at = timezone.now()
        self.save()
