"""Admin configuration for the accounts app."""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Custom User admin with game statistics."""

    list_display = [
        'username', 'email', 'elo_rating', 'games_played',
        'games_won', 'games_lost', 'is_staff', 'date_joined'
    ]
    list_filter = ['is_staff', 'is_superuser', 'is_active', 'date_joined']
    search_fields = ['username', 'email']
    ordering = ['-elo_rating']

    fieldsets = BaseUserAdmin.fieldsets + (
        ('Game Statistics', {
            'fields': (
                'games_played', 'games_won', 'games_lost',
                'elo_rating', 'highest_elo', 'last_game_at'
            )
        }),
        ('Profile', {
            'fields': ('avatar', 'bio', 'preferred_color')
        }),
    )

    readonly_fields = [
        'games_played', 'games_won', 'games_lost',
        'elo_rating', 'highest_elo', 'last_game_at'
    ]
