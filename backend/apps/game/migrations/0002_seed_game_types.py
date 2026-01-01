# Data migration to seed default game types
from django.db import migrations


def seed_game_types(apps, schema_editor):
    """Seed the default game types."""
    GameType = apps.get_model('game', 'GameType')

    game_types = [
        {
            'code': 'backgammon',
            'name': 'Backgammon',
            'description': 'Classic backgammon - a two-player board game where players move '
                           'checkers based on dice rolls, aiming to bear off all checkers first.',
            'min_players': 2,
            'max_players': 2,
            'requires_dice': True,
            'is_active': True,
        },
        {
            'code': 'catan',
            'name': 'Settlers of Catan',
            'description': 'Build settlements, cities, and roads on a hex-based island. '
                           'Gather resources, trade with other players, and be first to 10 victory points.',
            'min_players': 3,
            'max_players': 4,
            'requires_dice': True,
            'is_active': False,  # Placeholder implementation
        },
    ]

    for gt in game_types:
        GameType.objects.get_or_create(code=gt['code'], defaults=gt)


def reverse_seed_game_types(apps, schema_editor):
    """Remove seeded game types."""
    GameType = apps.get_model('game', 'GameType')
    GameType.objects.filter(code__in=['backgammon', 'catan']).delete()


class Migration(migrations.Migration):
    dependencies = [
        ('game', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(seed_game_types, reverse_seed_game_types),
    ]
