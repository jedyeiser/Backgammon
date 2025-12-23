"""Game app configuration."""
from django.apps import AppConfig


class GameConfig(AppConfig):
    """Configuration for the game app."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.game'
    verbose_name = 'Game'
