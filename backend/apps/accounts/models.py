"""Custom User model for Backgammon project."""
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Custom User model extending Django's AbstractUser.

    Adds game-related fields for tracking player statistics
    and preferences.
    """

    # Profile fields
    avatar = models.URLField(blank=True, null=True)
    bio = models.TextField(max_length=500, blank=True)

    # Game statistics
    games_played = models.PositiveIntegerField(default=0)
    games_won = models.PositiveIntegerField(default=0)
    games_lost = models.PositiveIntegerField(default=0)

    # ELO rating for matchmaking
    elo_rating = models.IntegerField(default=1200)
    highest_elo = models.IntegerField(default=1200)

    # Preferences
    preferred_color = models.CharField(
        max_length=10,
        choices=[('white', 'White'), ('black', 'Black'), ('random', 'Random')],
        default='random'
    )

    # Timestamps
    last_game_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return self.username

    @property
    def win_rate(self) -> float:
        """Calculate win rate as a percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.games_won / self.games_played) * 100

    def update_stats(self, won: bool) -> None:
        """Update user statistics after a game."""
        self.games_played += 1
        if won:
            self.games_won += 1
        else:
            self.games_lost += 1
        self.save(update_fields=['games_played', 'games_won', 'games_lost'])

    def update_elo(self, new_elo: int) -> None:
        """Update ELO rating and track highest."""
        self.elo_rating = new_elo
        if new_elo > self.highest_elo:
            self.highest_elo = new_elo
        self.save(update_fields=['elo_rating', 'highest_elo'])
