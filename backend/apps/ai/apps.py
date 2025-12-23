"""AI app configuration."""
from django.apps import AppConfig


class AiConfig(AppConfig):
    """Configuration for the AI app."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.ai'
    verbose_name = 'AI'
