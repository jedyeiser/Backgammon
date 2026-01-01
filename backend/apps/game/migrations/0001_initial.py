# Generated manually for initial setup
import uuid
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # Create GameType first (no dependencies)
        migrations.CreateModel(
            name='GameType',
            fields=[
                ('code', models.CharField(help_text="Unique identifier for the game type (e.g., 'backgammon', 'catan')", max_length=50, primary_key=True, serialize=False)),
                ('name', models.CharField(help_text='Display name for the game type', max_length=100)),
                ('description', models.TextField(blank=True, help_text='Description of the game for UI display')),
                ('min_players', models.PositiveIntegerField(default=2, help_text='Minimum number of players required')),
                ('max_players', models.PositiveIntegerField(default=2, help_text='Maximum number of players allowed')),
                ('requires_dice', models.BooleanField(default=False, help_text='Whether this game uses dice')),
                ('is_active', models.BooleanField(default=True, help_text='Whether this game type is available for play')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'game_types',
                'ordering': ['name'],
            },
        ),
        # Create Game model
        migrations.CreateModel(
            name='Game',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('status', models.CharField(choices=[('waiting', 'Waiting for Player'), ('active', 'Active'), ('completed', 'Completed'), ('abandoned', 'Abandoned')], default='waiting', max_length=20)),
                ('board_state', models.JSONField(default=dict)),
                ('current_turn', models.CharField(choices=[('white', 'White'), ('black', 'Black')], default='white', max_length=5)),
                ('dice', models.JSONField(default=list)),
                ('moves_remaining', models.JSONField(default=list)),
                ('cube_value', models.PositiveIntegerField(default=1)),
                ('cube_owner', models.CharField(choices=[('white', 'White'), ('black', 'Black'), ('center', 'Center')], default='center', max_length=5)),
                ('double_offered', models.BooleanField(default=False)),
                ('win_type', models.CharField(blank=True, choices=[('normal', 'Normal'), ('gammon', 'Gammon'), ('backgammon', 'Backgammon'), ('resign', 'Resignation'), ('timeout', 'Timeout')], max_length=20, null=True)),
                ('points_won', models.PositiveIntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('version', models.PositiveIntegerField(default=0)),
                ('game_type', models.ForeignKey(default='backgammon', help_text='The type of game being played', on_delete=django.db.models.deletion.PROTECT, related_name='games', to='game.gametype')),
                ('black_player', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='games_as_black', to=settings.AUTH_USER_MODEL)),
                ('white_player', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='games_as_white', to=settings.AUTH_USER_MODEL)),
                ('winner', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='games_won', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'games',
                'ordering': ['-created_at'],
            },
        ),
        # Create Move model
        migrations.CreateModel(
            name='Move',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('move_number', models.PositiveIntegerField()),
                ('move_type', models.CharField(choices=[('roll', 'Dice Roll'), ('move', 'Move Checker'), ('double', 'Offer Double'), ('accept_double', 'Accept Double'), ('reject_double', 'Reject Double'), ('resign', 'Resign')], max_length=20)),
                ('dice_values', models.JSONField(blank=True, null=True)),
                ('checker_moves', models.JSONField(blank=True, null=True)),
                ('board_state_after', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('game', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='moves', to='game.game')),
                ('player', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='moves', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'moves',
                'ordering': ['game', 'move_number'],
            },
        ),
        # Create GameInvite model
        migrations.CreateModel(
            name='GameInvite',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('accepted', 'Accepted'), ('declined', 'Declined'), ('expired', 'Expired')], default='pending', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('responded_at', models.DateTimeField(blank=True, null=True)),
                ('from_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='sent_invites', to=settings.AUTH_USER_MODEL)),
                ('to_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='received_invites', to=settings.AUTH_USER_MODEL)),
                ('game_type', models.ForeignKey(default='backgammon', help_text='The type of game being invited to', on_delete=django.db.models.deletion.PROTECT, related_name='invites', to='game.gametype')),
                ('game', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='invite', to='game.game')),
            ],
            options={
                'db_table': 'game_invites',
                'ordering': ['-created_at'],
            },
        ),
        # Add indexes
        migrations.AddIndex(
            model_name='game',
            index=models.Index(fields=['status'], name='games_status_idx'),
        ),
        migrations.AddIndex(
            model_name='game',
            index=models.Index(fields=['game_type', 'status'], name='games_type_status_idx'),
        ),
        migrations.AddIndex(
            model_name='game',
            index=models.Index(fields=['white_player', 'status'], name='games_white_status_idx'),
        ),
        migrations.AddIndex(
            model_name='game',
            index=models.Index(fields=['black_player', 'status'], name='games_black_status_idx'),
        ),
        migrations.AddIndex(
            model_name='move',
            index=models.Index(fields=['game', 'move_number'], name='moves_game_num_idx'),
        ),
        # Add unique constraint
        migrations.AlterUniqueTogether(
            name='move',
            unique_together={('game', 'move_number')},
        ),
    ]
