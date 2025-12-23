"""Admin configuration for the AI app."""
from django.contrib import admin

from .models import AIModel, TrainingSession


@admin.register(AIModel)
class AIModelAdmin(admin.ModelAdmin):
    """Admin for AIModel."""

    list_display = [
        'name', 'model_type', 'training_games', 'win_rate_vs_random',
        'is_active', 'created_at'
    ]
    list_filter = ['model_type', 'is_active', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['id', 'created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    """Admin for TrainingSession."""

    list_display = [
        'id_short', 'model', 'status', 'games_completed', 'num_games',
        'final_loss', 'created_at'
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['model__name']
    readonly_fields = [
        'id', 'games_completed', 'current_loss', 'final_loss',
        'training_log', 'started_at', 'completed_at', 'created_at'
    ]
    ordering = ['-created_at']

    def id_short(self, obj):
        """Display shortened UUID."""
        return str(obj.id)[:8]
    id_short.short_description = 'ID'
