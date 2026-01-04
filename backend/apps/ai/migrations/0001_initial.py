# Initial migration for AI app
import uuid
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('game', '0001_initial'),
    ]

    operations = [
        # AIModel
        migrations.CreateModel(
            name='AIModel',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100)),
                ('model_type', models.CharField(
                    choices=[
                        ('random', 'Random'),
                        ('heuristic', 'Heuristic'),
                        ('td_gammon', 'TD-Gammon'),
                        ('neural', 'Neural Network'),
                        ('evolved', 'Evolved Network'),
                    ],
                    default='random',
                    max_length=20
                )),
                ('description', models.TextField(blank=True)),
                ('network_architecture', models.JSONField(default=dict, help_text='JSON representation of network architecture')),
                ('network_weights', models.BinaryField(blank=True, help_text='Compressed serialized network weights', null=True)),
                ('weights_path', models.CharField(blank=True, max_length=255)),
                ('generation', models.PositiveIntegerField(default=0, help_text='Generation number in evolution')),
                ('mutation_history', models.JSONField(default=list, help_text='List of mutations applied to create this model')),
                ('training_games', models.PositiveIntegerField(default=0)),
                ('training_epochs', models.PositiveIntegerField(default=0)),
                ('win_rate_vs_random', models.FloatField(blank=True, null=True)),
                ('win_rate_vs_self', models.FloatField(blank=True, null=True)),
                ('hyperparameters', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('is_active', models.BooleanField(default=False)),
                ('game_type', models.ForeignKey(
                    blank=True,
                    help_text='The game type this model plays',
                    null=True,
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='ai_models',
                    to='game.gametype'
                )),
                ('parent_model', models.ForeignKey(
                    blank=True,
                    help_text='Parent model in evolution tree',
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='children',
                    to='ai.aimodel'
                )),
            ],
            options={
                'db_table': 'ai_models',
                'ordering': ['-created_at'],
            },
        ),

        # TrainingSession
        migrations.CreateModel(
            name='TrainingSession',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('status', models.CharField(
                    choices=[
                        ('pending', 'Pending'),
                        ('running', 'Running'),
                        ('completed', 'Completed'),
                        ('failed', 'Failed'),
                        ('cancelled', 'Cancelled'),
                    ],
                    default='pending',
                    max_length=20
                )),
                ('num_games', models.PositiveIntegerField(default=1000)),
                ('learning_rate', models.FloatField(default=0.1)),
                ('lambda_value', models.FloatField(default=0.7)),
                ('games_completed', models.PositiveIntegerField(default=0)),
                ('current_loss', models.FloatField(blank=True, null=True)),
                ('final_loss', models.FloatField(blank=True, null=True)),
                ('training_log', models.JSONField(default=list)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('model', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='training_sessions',
                    to='ai.aimodel'
                )),
            ],
            options={
                'db_table': 'training_sessions',
                'ordering': ['-created_at'],
            },
        ),

        # EvolutionSession
        migrations.CreateModel(
            name='EvolutionSession',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField(blank=True)),
                ('status', models.CharField(
                    choices=[
                        ('pending', 'Pending'),
                        ('running', 'Running'),
                        ('paused', 'Paused'),
                        ('completed', 'Completed'),
                        ('cancelled', 'Cancelled'),
                    ],
                    default='pending',
                    max_length=20
                )),
                ('population_size', models.PositiveIntegerField(default=50, help_text='Number of individuals in each generation')),
                ('mutation_rate', models.FloatField(default=0.3, help_text='Probability of mutation per individual')),
                ('crossover_rate', models.FloatField(default=0.5, help_text='Probability of crossover between parents')),
                ('weight_mutation_sigma', models.FloatField(default=0.1, help_text='Standard deviation for Gaussian weight perturbation')),
                ('topology_mutation_rate', models.FloatField(default=0.1, help_text='Probability of topology mutation (add node/connection)')),
                ('training_games_per_eval', models.PositiveIntegerField(default=100, help_text='Games to play before evaluating fitness')),
                ('elitism_count', models.PositiveIntegerField(default=5, help_text='Number of top individuals to preserve unchanged')),
                ('current_generation', models.PositiveIntegerField(default=0)),
                ('target_generations', models.PositiveIntegerField(default=100, help_text='Stop after this many generations')),
                ('best_fitness', models.FloatField(blank=True, null=True)),
                ('generation_log', models.JSONField(default=list, help_text='Log of fitness stats per generation')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('game_type', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='evolution_sessions',
                    to='game.gametype'
                )),
                ('best_model', models.ForeignKey(
                    blank=True,
                    help_text='Best model found so far',
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='best_in_sessions',
                    to='ai.aimodel'
                )),
            ],
            options={
                'db_table': 'evolution_sessions',
                'ordering': ['-created_at'],
            },
        ),

        # PlayerRating
        migrations.CreateModel(
            name='PlayerRating',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('elo', models.IntegerField(default=1000, help_text='Current ELO rating')),
                ('games_played', models.PositiveIntegerField(default=0)),
                ('games_won', models.PositiveIntegerField(default=0)),
                ('games_lost', models.PositiveIntegerField(default=0)),
                ('games_drawn', models.PositiveIntegerField(default=0)),
                ('peak_elo', models.IntegerField(default=1000)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('ai_model', models.ForeignKey(
                    blank=True,
                    help_text='AI player (null if human)',
                    null=True,
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='game_ratings',
                    to='ai.aimodel'
                )),
                ('game_type', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='player_ratings',
                    to='game.gametype'
                )),
                ('user', models.ForeignKey(
                    blank=True,
                    help_text='Human player (null if AI)',
                    null=True,
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='game_ratings',
                    to=settings.AUTH_USER_MODEL
                )),
            ],
            options={
                'db_table': 'player_ratings',
                'ordering': ['-elo'],
            },
        ),

        # EvolutionLineage
        migrations.CreateModel(
            name='EvolutionLineage',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('mutation_type', models.CharField(
                    choices=[
                        ('weight', 'Weight Perturbation'),
                        ('add_node', 'Add Node'),
                        ('add_conn', 'Add Connection'),
                        ('rm_conn', 'Remove Connection'),
                        ('crossover', 'Crossover'),
                        ('clone', 'Clone (Elitism)'),
                    ],
                    max_length=20
                )),
                ('generation', models.PositiveIntegerField()),
                ('mutation_details', models.JSONField(default=dict, help_text='Specific details about the mutation applied')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('child', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='lineage_records',
                    to='ai.aimodel'
                )),
                ('parent1', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='offspring_as_parent1',
                    to='ai.aimodel'
                )),
                ('parent2', models.ForeignKey(
                    blank=True,
                    help_text='Second parent (for crossover only)',
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='offspring_as_parent2',
                    to='ai.aimodel'
                )),
                ('session', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='lineage_records',
                    to='ai.evolutionsession'
                )),
            ],
            options={
                'db_table': 'evolution_lineage',
                'ordering': ['session', 'generation', 'created_at'],
            },
        ),

        # Add indexes
        migrations.AddIndex(
            model_name='aimodel',
            index=models.Index(fields=['game_type', 'model_type'], name='ai_models_game_type_idx'),
        ),
        migrations.AddIndex(
            model_name='aimodel',
            index=models.Index(fields=['game_type', 'is_active'], name='ai_models_active_idx'),
        ),
        migrations.AddIndex(
            model_name='playerrating',
            index=models.Index(fields=['game_type', 'elo'], name='player_ratings_elo_idx'),
        ),
        migrations.AddIndex(
            model_name='playerrating',
            index=models.Index(fields=['user', 'game_type'], name='player_ratings_user_idx'),
        ),
        migrations.AddIndex(
            model_name='playerrating',
            index=models.Index(fields=['ai_model', 'game_type'], name='player_ratings_ai_idx'),
        ),
        migrations.AddIndex(
            model_name='evolutionlineage',
            index=models.Index(fields=['session', 'generation'], name='evo_lineage_session_idx'),
        ),
        migrations.AddIndex(
            model_name='evolutionlineage',
            index=models.Index(fields=['child'], name='evo_lineage_child_idx'),
        ),

        # Add constraint for PlayerRating (exactly one of user or ai_model)
        migrations.AddConstraint(
            model_name='playerrating',
            constraint=models.CheckConstraint(
                check=models.Q(user__isnull=False, ai_model__isnull=True) | models.Q(user__isnull=True, ai_model__isnull=False),
                name='player_rating_one_player_type',
            ),
        ),
    ]
