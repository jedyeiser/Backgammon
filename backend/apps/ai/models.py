"""Models for the AI app."""
import uuid
from django.db import models


class AIModel(models.Model):
    """
    Represents a trained AI model for playing backgammon.

    Stores model metadata and training history.
    """

    class ModelType(models.TextChoices):
        TD_GAMMON = 'td_gammon', 'TD-Gammon'
        RANDOM = 'random', 'Random'
        HEURISTIC = 'heuristic', 'Heuristic'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    model_type = models.CharField(
        max_length=20,
        choices=ModelType.choices,
        default=ModelType.RANDOM
    )
    description = models.TextField(blank=True)

    # Model file path
    weights_path = models.CharField(max_length=255, blank=True)

    # Training metadata
    training_games = models.PositiveIntegerField(default=0)
    training_epochs = models.PositiveIntegerField(default=0)

    # Performance metrics
    win_rate_vs_random = models.FloatField(null=True, blank=True)
    win_rate_vs_self = models.FloatField(null=True, blank=True)

    # Hyperparameters (stored as JSON)
    hyperparameters = models.JSONField(default=dict)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Active model flag
    is_active = models.BooleanField(default=False)

    class Meta:
        db_table = 'ai_models'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.model_type})"


class TrainingSession(models.Model):
    """
    Represents a training session for an AI model.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        CANCELLED = 'cancelled', 'Cancelled'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(
        AIModel,
        on_delete=models.CASCADE,
        related_name='training_sessions'
    )

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )

    # Training configuration
    num_games = models.PositiveIntegerField(default=1000)
    learning_rate = models.FloatField(default=0.1)
    lambda_value = models.FloatField(default=0.7)  # TD(lambda)

    # Progress tracking
    games_completed = models.PositiveIntegerField(default=0)
    current_loss = models.FloatField(null=True, blank=True)

    # Results
    final_loss = models.FloatField(null=True, blank=True)
    training_log = models.JSONField(default=list)

    # Timestamps
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'training_sessions'
        ordering = ['-created_at']

    def __str__(self):
        return f"Training {self.model.name} - {self.status}"
