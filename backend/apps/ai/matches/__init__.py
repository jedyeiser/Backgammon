"""
Match and tournament infrastructure for game AI.

Provides tools for running matches between players, managing
tournaments, and updating ELO ratings.

Components:
- EloCalculator: Standard ELO rating calculations
- MatchRunner: Run games between any players
- RoundRobinTournament: Everyone plays everyone
- SwissTournament: Pair players by score

Example usage:
    from apps.ai.matches import MatchRunner, RoundRobinTournament
    from apps.ai.players import get_player

    # Create players
    neural = get_player('neural', 'p1', 'backgammon', network=network)
    random = get_player('random', 'p2', 'backgammon')

    # Run a match
    runner = MatchRunner(game_type='backgammon')
    result = runner.run_match(neural, random, num_games=10)
    print(f"Neural wins: {result.player_a_wins}")

    # Run a tournament
    players = [neural, random, heuristic]
    tournament = RoundRobinTournament(players, game_type='backgammon')
    result = tournament.run(games_per_match=2)
    print(format_standings(result.standings))
"""
from .elo import (
    EloCalculator,
    EloResult,
    rating_probability,
    rating_difference_to_expected,
    expected_score_to_rating_difference,
)
from .runner import (
    MatchRunner,
    GameResult,
    MatchResult,
    run_benchmark,
)
from .tournament import (
    TournamentPlayer,
    TournamentResult,
    RoundRobinTournament,
    SwissTournament,
    format_standings,
)

__all__ = [
    # ELO
    'EloCalculator',
    'EloResult',
    'rating_probability',
    'rating_difference_to_expected',
    'expected_score_to_rating_difference',

    # Match runner
    'MatchRunner',
    'GameResult',
    'MatchResult',
    'run_benchmark',

    # Tournaments
    'TournamentPlayer',
    'TournamentResult',
    'RoundRobinTournament',
    'SwissTournament',
    'format_standings',
]
