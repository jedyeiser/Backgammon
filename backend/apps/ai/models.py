"""
Models for the AI app.

This module contains database models for:
- AIModel: Trained AI models with network architecture and weights
- TrainingSession: Training run tracking
- PlayerRating: Per-game ELO ratings for humans and AI
- EvolutionSession: Neuroevolution experiment tracking
- EvolutionLineage: Parent-child relationships in evolution
"""
import uuid
from django.conf import settings
from django.db import models


class AIModel(models.Model):
    """
    Represents a trained AI model for playing games.

    Stores model metadata, network architecture (as JSON), and serialized weights.
    Supports evolution tracking through parent_model relationship.
    """

    class ModelType(models.TextChoices):
        RANDOM = 'random', 'Random'
        HEURISTIC = 'heuristic', 'Heuristic'
        TD_GAMMON = 'td_gammon', 'TD-Gammon'
        NEURAL = 'neural', 'Neural Network'
        EVOLVED = 'evolved', 'Evolved Network'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    model_type = models.CharField(
        max_length=20,
        choices=ModelType.choices,
        default=ModelType.RANDOM
    )
    description = models.TextField(blank=True)

    # Game type this model is trained for
    game_type = models.ForeignKey(
        'game.GameType',
        on_delete=models.PROTECT,
        related_name='ai_models',
        null=True,
        blank=True,
        help_text="The game type this model plays"
    )

    # Network architecture stored as JSON
    # Format: {"input_size": 198, "layers": [...], "output_size": 1}
    network_architecture = models.JSONField(
        default=dict,
        help_text="JSON representation of network architecture"
    )

    # Serialized network weights (gzip compressed pickle)
    network_weights = models.BinaryField(
        null=True,
        blank=True,
        help_text="Compressed serialized network weights"
    )

    # Legacy: file path for older models
    weights_path = models.CharField(max_length=255, blank=True)

    # Evolution lineage
    parent_model = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='children',
        help_text="Parent model in evolution tree"
    )
    generation = models.PositiveIntegerField(
        default=0,
        help_text="Generation number in evolution"
    )
    mutation_history = models.JSONField(
        default=list,
        help_text="List of mutations applied to create this model"
    )

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
        indexes = [
            models.Index(fields=['game_type', 'model_type']),
            models.Index(fields=['game_type', 'is_active']),
        ]

    def __str__(self):
        game = self.game_type_id or 'any'
        return f"{self.name} ({self.model_type}) - {game}"


class TrainingSession(models.Model):
    """
    Represents a training session for an AI model.

    Tracks training progress, configuration, and results.
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


class PlayerRating(models.Model):
    """
    Per-game ELO rating for any player (human or AI).

    Each player has a separate rating for each game type they play.
    Polymorphic: either user OR ai_model is set, not both.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Polymorphic player reference - exactly one should be set
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='game_ratings',
        help_text="Human player (null if AI)"
    )
    ai_model = models.ForeignKey(
        AIModel,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='game_ratings',
        help_text="AI player (null if human)"
    )

    # Game type for this rating
    game_type = models.ForeignKey(
        'game.GameType',
        on_delete=models.CASCADE,
        related_name='player_ratings'
    )

    # Rating data
    elo = models.IntegerField(
        default=1000,
        help_text="Current ELO rating"
    )
    games_played = models.PositiveIntegerField(default=0)
    games_won = models.PositiveIntegerField(default=0)
    games_lost = models.PositiveIntegerField(default=0)
    games_drawn = models.PositiveIntegerField(default=0)

    # Peak rating tracking
    peak_elo = models.IntegerField(default=1000)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'player_ratings'
        ordering = ['-elo']
        constraints = [
            # Ensure exactly one of user or ai_model is set
            models.CheckConstraint(
                check=(
                    models.Q(user__isnull=False, ai_model__isnull=True) |
                    models.Q(user__isnull=True, ai_model__isnull=False)
                ),
                name='player_rating_one_player_type'
            ),
        ]
        indexes = [
            models.Index(fields=['game_type', 'elo']),
            models.Index(fields=['user', 'game_type']),
            models.Index(fields=['ai_model', 'game_type']),
        ]
        # Unique per player per game type
        unique_together = []  # Handled by constraints below

    def __str__(self):
        player = self.user.username if self.user else self.ai_model.name
        return f"{player} - {self.game_type_id}: {self.elo}"

    def save(self, *args, **kwargs):
        """Update peak_elo if current elo exceeds it."""
        if self.elo > self.peak_elo:
            self.peak_elo = self.elo
        super().save(*args, **kwargs)

    @property
    def player_name(self) -> str:
        """Get the player's display name."""
        if self.user:
            return self.user.username
        return self.ai_model.name

    @property
    def is_human(self) -> bool:
        """Check if this rating belongs to a human player."""
        return self.user is not None

    @property
    def win_rate(self) -> float:
        """Calculate win rate as a percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.games_won / self.games_played) * 100

    def update_after_game(self, won: bool, drawn: bool = False) -> None:
        """Update stats after a game."""
        self.games_played += 1
        if drawn:
            self.games_drawn += 1
        elif won:
            self.games_won += 1
        else:
            self.games_lost += 1
        self.save()


class EvolutionSession(models.Model):
    """
    Tracks a neuroevolution experiment.

    Manages population, configuration, and progress of an evolution run.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        PAUSED = 'paused', 'Paused'
        COMPLETED = 'completed', 'Completed'
        CANCELLED = 'cancelled', 'Cancelled'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    game_type = models.ForeignKey(
        'game.GameType',
        on_delete=models.CASCADE,
        related_name='evolution_sessions'
    )

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )

    # Evolution configuration
    population_size = models.PositiveIntegerField(
        default=50,
        help_text="Number of individuals in each generation"
    )
    mutation_rate = models.FloatField(
        default=0.3,
        help_text="Probability of mutation per individual"
    )
    crossover_rate = models.FloatField(
        default=0.5,
        help_text="Probability of crossover between parents"
    )
    weight_mutation_sigma = models.FloatField(
        default=0.1,
        help_text="Standard deviation for Gaussian weight perturbation"
    )
    topology_mutation_rate = models.FloatField(
        default=0.1,
        help_text="Probability of topology mutation (add node/connection)"
    )
    training_games_per_eval = models.PositiveIntegerField(
        default=100,
        help_text="Games to play before evaluating fitness"
    )
    elitism_count = models.PositiveIntegerField(
        default=5,
        help_text="Number of top individuals to preserve unchanged"
    )

    # Progress tracking
    current_generation = models.PositiveIntegerField(default=0)
    target_generations = models.PositiveIntegerField(
        default=100,
        help_text="Stop after this many generations"
    )
    best_fitness = models.FloatField(null=True, blank=True)
    best_model = models.ForeignKey(
        AIModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='best_in_sessions',
        help_text="Best model found so far"
    )

    # Population reference (models in this session)
    # Accessed via AIModel.evolution_session

    # Evolution log
    generation_log = models.JSONField(
        default=list,
        help_text="Log of fitness stats per generation"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'evolution_sessions'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - Gen {self.current_generation}/{self.target_generations}"


class EvolutionLineage(models.Model):
    """
    Tracks parent-child relationships in evolution.

    Records the mutation or crossover that created each new model.
    """

    class MutationType(models.TextChoices):
        WEIGHT = 'weight', 'Weight Perturbation'
        ADD_NODE = 'add_node', 'Add Node'
        ADD_CONNECTION = 'add_conn', 'Add Connection'
        REMOVE_CONNECTION = 'rm_conn', 'Remove Connection'
        CROSSOVER = 'crossover', 'Crossover'
        CLONE = 'clone', 'Clone (Elitism)'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    session = models.ForeignKey(
        EvolutionSession,
        on_delete=models.CASCADE,
        related_name='lineage_records'
    )

    # For crossover, parent2 is used. For mutations, only parent1.
    parent1 = models.ForeignKey(
        AIModel,
        on_delete=models.CASCADE,
        related_name='offspring_as_parent1'
    )
    parent2 = models.ForeignKey(
        AIModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='offspring_as_parent2',
        help_text="Second parent (for crossover only)"
    )
    child = models.ForeignKey(
        AIModel,
        on_delete=models.CASCADE,
        related_name='lineage_records'
    )

    mutation_type = models.CharField(
        max_length=20,
        choices=MutationType.choices
    )
    generation = models.PositiveIntegerField()

    # Details about the mutation
    mutation_details = models.JSONField(
        default=dict,
        help_text="Specific details about the mutation applied"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'evolution_lineage'
        ordering = ['session', 'generation', 'created_at']
        indexes = [
            models.Index(fields=['session', 'generation']),
            models.Index(fields=['child']),
        ]

    def __str__(self):
        return f"Gen {self.generation}: {self.parent1.name} -> {self.child.name} ({self.mutation_type})"
