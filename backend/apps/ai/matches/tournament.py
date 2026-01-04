"""
Tournament systems for multi-player competitions.

Implements various tournament formats:
- Round Robin: Everyone plays everyone
- Swiss: Pair players with similar scores
- Elimination: Single/double elimination brackets

Tournaments update ELO ratings and produce rankings.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .runner import MatchRunner, MatchResult
from .elo import EloCalculator

if TYPE_CHECKING:
    from ..players.base import BasePlayer


@dataclass
class TournamentPlayer:
    """Player in a tournament with score tracking."""
    player: 'BasePlayer'
    rating: int = 1000
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score: float = 0.0  # 1 per win, 0.5 per draw
    buchholz: float = 0.0  # Tiebreaker: sum of opponents' scores

    @property
    def player_id(self) -> str:
        return self.player.player_id


@dataclass
class TournamentResult:
    """Result of a tournament."""
    format: str
    rounds_played: int = 0
    total_games: int = 0
    standings: List[TournamentPlayer] = field(default_factory=list)
    match_history: List[MatchResult] = field(default_factory=list)
    rating_changes: Dict[str, int] = field(default_factory=dict)


class RoundRobinTournament:
    """
    Round Robin tournament: everyone plays everyone.

    Each player plays one match against every other player.
    Rankings are by total score (1 for win, 0.5 for draw).
    Tiebreaker is Buchholz score (sum of opponents' scores).

    Example:
        players = [player1, player2, player3, player4]
        tournament = RoundRobinTournament(players, game_type='backgammon')
        result = tournament.run(games_per_match=2)
    """

    def __init__(
        self,
        players: List['BasePlayer'],
        game_type: str = 'backgammon',
        initial_ratings: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize round robin tournament.

        Args:
            players: List of players.
            game_type: Game type for matches.
            initial_ratings: Optional dict of player_id -> rating.
        """
        self.game_type = game_type
        initial_ratings = initial_ratings or {}

        self.participants = [
            TournamentPlayer(
                player=p,
                rating=initial_ratings.get(p.player_id, 1000),
            )
            for p in players
        ]

        self.runner = MatchRunner(game_type=game_type, update_ratings=False)
        self.elo_calculator = EloCalculator()

    def run(
        self,
        games_per_match: int = 1,
        progress_callback: Optional[callable] = None,
    ) -> TournamentResult:
        """
        Run the tournament.

        Args:
            games_per_match: Number of games per match.
            progress_callback: Called with (match_num, total) after each match.

        Returns:
            TournamentResult with standings.
        """
        result = TournamentResult(format='round_robin')
        n = len(self.participants)

        # Generate all pairings
        pairings = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
        ]

        total_matches = len(pairings)

        for match_num, (i, j) in enumerate(pairings, 1):
            p1 = self.participants[i]
            p2 = self.participants[j]

            # Run match
            match_result = self.runner.run_match(
                player_a=p1.player,
                player_b=p2.player,
                num_games=games_per_match,
                rating_a=p1.rating,
                rating_b=p2.rating,
            )

            # Update statistics
            p1.games_played += games_per_match
            p2.games_played += games_per_match
            p1.wins += match_result.player_a_wins
            p2.wins += match_result.player_b_wins
            p1.losses += match_result.player_b_wins
            p2.losses += match_result.player_a_wins
            p1.draws += match_result.draws
            p2.draws += match_result.draws
            p1.score += match_result.player_a_score
            p2.score += match_result.player_b_score

            # Update ratings
            new_a, new_b = self.elo_calculator.update_from_match(
                rating_a=p1.rating,
                rating_b=p2.rating,
                wins_a=match_result.player_a_wins,
                wins_b=match_result.player_b_wins,
                draws=match_result.draws,
            )

            result.rating_changes[p1.player_id] = (
                result.rating_changes.get(p1.player_id, 0) + (new_a - p1.rating)
            )
            result.rating_changes[p2.player_id] = (
                result.rating_changes.get(p2.player_id, 0) + (new_b - p2.rating)
            )

            p1.rating = new_a
            p2.rating = new_b

            result.match_history.append(match_result)
            result.total_games += games_per_match

            if progress_callback:
                progress_callback(match_num, total_matches)

        # Calculate Buchholz tiebreaker
        self._calculate_buchholz(result.match_history)

        # Sort standings
        result.standings = sorted(
            self.participants,
            key=lambda p: (p.score, p.buchholz, p.rating),
            reverse=True,
        )

        result.rounds_played = 1

        return result

    def _calculate_buchholz(self, matches: List[MatchResult]) -> None:
        """Calculate Buchholz tiebreaker scores."""
        # Build opponent map
        opponents = {p.player_id: [] for p in self.participants}
        id_to_player = {p.player_id: p for p in self.participants}

        for match in matches:
            opponents[match.player_a_id].append(match.player_b_id)
            opponents[match.player_b_id].append(match.player_a_id)

        # Calculate Buchholz (sum of opponents' scores)
        for player in self.participants:
            player.buchholz = sum(
                id_to_player[opp_id].score
                for opp_id in opponents[player.player_id]
            )


class SwissTournament:
    """
    Swiss system tournament.

    Players are paired with opponents of similar score.
    Good for large tournaments - determines rankings
    without requiring everyone to play everyone.

    Pairing rules:
    1. Players with same score play each other
    2. Avoid repeat matchups
    3. Balance colors (each player alternates)
    """

    def __init__(
        self,
        players: List['BasePlayer'],
        game_type: str = 'backgammon',
        num_rounds: Optional[int] = None,
        initial_ratings: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize Swiss tournament.

        Args:
            players: List of players.
            game_type: Game type.
            num_rounds: Number of rounds (default: ceil(log2(n))).
            initial_ratings: Optional dict of player_id -> rating.
        """
        import math

        self.game_type = game_type
        initial_ratings = initial_ratings or {}

        self.participants = [
            TournamentPlayer(
                player=p,
                rating=initial_ratings.get(p.player_id, 1000),
            )
            for p in players
        ]

        n = len(players)
        self.num_rounds = num_rounds or max(1, math.ceil(math.log2(n)))

        self.runner = MatchRunner(game_type=game_type, update_ratings=False)
        self.elo_calculator = EloCalculator()

        # Track previous matchups
        self.played_pairs: set = set()

    def run(
        self,
        games_per_match: int = 1,
        progress_callback: Optional[callable] = None,
    ) -> TournamentResult:
        """
        Run the Swiss tournament.

        Args:
            games_per_match: Games per match.
            progress_callback: Progress callback.

        Returns:
            TournamentResult.
        """
        result = TournamentResult(format='swiss')

        for round_num in range(1, self.num_rounds + 1):
            # Generate pairings for this round
            pairings = self._generate_pairings()

            for p1, p2 in pairings:
                # Run match
                match_result = self.runner.run_match(
                    player_a=p1.player,
                    player_b=p2.player,
                    num_games=games_per_match,
                    rating_a=p1.rating,
                    rating_b=p2.rating,
                )

                # Update statistics
                self._update_stats(p1, p2, match_result)

                # Track pairing
                pair_key = tuple(sorted([p1.player_id, p2.player_id]))
                self.played_pairs.add(pair_key)

                result.match_history.append(match_result)
                result.total_games += games_per_match

            result.rounds_played = round_num

            if progress_callback:
                progress_callback(round_num, self.num_rounds)

        # Final standings
        result.standings = sorted(
            self.participants,
            key=lambda p: (p.score, p.buchholz, p.rating),
            reverse=True,
        )

        # Calculate rating changes
        for p in self.participants:
            original = 1000  # Default
            result.rating_changes[p.player_id] = p.rating - original

        return result

    def _generate_pairings(self) -> List[Tuple[TournamentPlayer, TournamentPlayer]]:
        """Generate pairings for a round."""
        # Sort by score (and rating as tiebreaker)
        sorted_players = sorted(
            self.participants,
            key=lambda p: (p.score, p.rating),
            reverse=True,
        )

        pairings = []
        used = set()

        for p1 in sorted_players:
            if p1.player_id in used:
                continue

            # Find best opponent
            best_opponent = None
            for p2 in sorted_players:
                if p2.player_id == p1.player_id:
                    continue
                if p2.player_id in used:
                    continue

                pair_key = tuple(sorted([p1.player_id, p2.player_id]))
                if pair_key in self.played_pairs:
                    continue

                best_opponent = p2
                break

            if best_opponent:
                pairings.append((p1, best_opponent))
                used.add(p1.player_id)
                used.add(best_opponent.player_id)

        return pairings

    def _update_stats(
        self,
        p1: TournamentPlayer,
        p2: TournamentPlayer,
        match: MatchResult,
    ) -> None:
        """Update player statistics after a match."""
        p1.games_played += match.total_games
        p2.games_played += match.total_games
        p1.wins += match.player_a_wins
        p2.wins += match.player_b_wins
        p1.losses += match.player_b_wins
        p2.losses += match.player_a_wins
        p1.draws += match.draws
        p2.draws += match.draws
        p1.score += match.player_a_score
        p2.score += match.player_b_score

        # Update ratings
        new_a, new_b = self.elo_calculator.update_from_match(
            rating_a=p1.rating,
            rating_b=p2.rating,
            wins_a=match.player_a_wins,
            wins_b=match.player_b_wins,
            draws=match.draws,
        )
        p1.rating = new_a
        p2.rating = new_b


def format_standings(standings: List[TournamentPlayer]) -> str:
    """Format tournament standings as text."""
    lines = [
        "=" * 60,
        "TOURNAMENT STANDINGS",
        "=" * 60,
        f"{'Rank':<6}{'Player':<20}{'Score':<10}{'W-L-D':<15}{'Rating':<10}",
        "-" * 60,
    ]

    for i, p in enumerate(standings, 1):
        wld = f"{p.wins}-{p.losses}-{p.draws}"
        lines.append(
            f"{i:<6}{p.player_id:<20}{p.score:<10.1f}{wld:<15}{p.rating:<10}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
