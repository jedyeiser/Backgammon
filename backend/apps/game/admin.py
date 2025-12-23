"""Admin configuration for the game app."""
from django.contrib import admin
from django.utils.html import format_html

from .models import Game, Move, GameInvite


@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    """Admin for Game model."""

    list_display = [
        'id_short', 'white_player', 'black_player', 'status',
        'current_turn', 'winner', 'win_type', 'created_at'
    ]
    list_filter = ['status', 'win_type', 'created_at']
    search_fields = [
        'white_player__username', 'black_player__username',
        'id'
    ]
    readonly_fields = [
        'id', 'board_state', 'dice', 'moves_remaining',
        'version', 'created_at', 'started_at', 'completed_at', 'updated_at'
    ]
    ordering = ['-created_at']

    fieldsets = (
        ('Players', {
            'fields': ('white_player', 'black_player')
        }),
        ('Game State', {
            'fields': (
                'status', 'current_turn', 'board_state',
                'dice', 'moves_remaining'
            )
        }),
        ('Doubling Cube', {
            'fields': ('cube_value', 'cube_owner', 'double_offered')
        }),
        ('Result', {
            'fields': ('winner', 'win_type', 'points_won')
        }),
        ('Metadata', {
            'fields': (
                'id', 'version',
                'created_at', 'started_at', 'completed_at', 'updated_at'
            )
        }),
    )

    def id_short(self, obj):
        """Display shortened UUID."""
        return str(obj.id)[:8]
    id_short.short_description = 'ID'


@admin.register(Move)
class MoveAdmin(admin.ModelAdmin):
    """Admin for Move model."""

    list_display = [
        'id_short', 'game_short', 'player', 'move_number',
        'move_type', 'dice_values', 'created_at'
    ]
    list_filter = ['move_type', 'created_at']
    search_fields = ['game__id', 'player__username']
    readonly_fields = [
        'id', 'game', 'player', 'move_number', 'move_type',
        'dice_values', 'checker_moves', 'board_state_after', 'created_at'
    ]
    ordering = ['game', 'move_number']

    def id_short(self, obj):
        """Display shortened UUID."""
        return str(obj.id)[:8]
    id_short.short_description = 'ID'

    def game_short(self, obj):
        """Display shortened game UUID."""
        return str(obj.game_id)[:8]
    game_short.short_description = 'Game'


@admin.register(GameInvite)
class GameInviteAdmin(admin.ModelAdmin):
    """Admin for GameInvite model."""

    list_display = [
        'id_short', 'from_user', 'to_user', 'status',
        'game_link', 'created_at', 'responded_at'
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['from_user__username', 'to_user__username']
    readonly_fields = ['id', 'created_at', 'responded_at']
    ordering = ['-created_at']

    def id_short(self, obj):
        """Display shortened UUID."""
        return str(obj.id)[:8]
    id_short.short_description = 'ID'

    def game_link(self, obj):
        """Display link to game if exists."""
        if obj.game:
            return format_html(
                '<a href="/admin/game/game/{}/change/">{}</a>',
                obj.game.id,
                str(obj.game.id)[:8]
            )
        return '-'
    game_link.short_description = 'Game'
