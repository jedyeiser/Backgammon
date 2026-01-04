"""
ELO rating system implementation.

Provides accurate skill ratings for players (human or AI) using
the ELO rating system. Supports per-game-type ratings and
various ELO calculation methods.

Reference:
    Elo, A. E. (1978). The rating of chessplayers, past and present.
    Arco Publishing.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EloResult:
    """Result of an ELO calculation."""
    player_id: str
    old_rating: int
    new_rating: int
    change: int
    expected_score: float
    actual_score: float


class EloCalculator:
    """
    Calculate ELO rating updates.

    Standard ELO formula:
    - Expected score: E = 1 / (1 + 10^((R_opponent - R_player) / 400))
    - New rating: R' = R + K * (S - E)

    Where:
    - K = development coefficient (32 typical, higher = faster adjustment)
    - S = actual score (1 = win, 0.5 = draw, 0 = loss)
    - E = expected score

    Attributes:
        k_factor: Base K-factor for rating adjustments.
        dynamic_k: If True, use rating-dependent K-factors.
        floor_rating: Minimum possible rating.

    Example:
        calc = EloCalculator(k_factor=32)

        # Update after a game
        results = calc.update_ratings(
            rating_a=1500,
            rating_b=1400,
            score_a=1.0,  # Player A won
        )

        # results[0] = EloResult for player A
        # results[1] = EloResult for player B
    """

    # Standard K-factors for different contexts
    K_PROVISIONAL = 40  # New players
    K_STANDARD = 32     # Normal games
    K_ESTABLISHED = 24  # Experienced players
    K_MASTER = 16       # High-rated players

    def __init__(
        self,
        k_factor: int = 32,
        dynamic_k: bool = True,
        floor_rating: int = 100,
        provisional_games: int = 30,
    ):
        """
        Initialize the ELO calculator.

        Args:
            k_factor: Base K-factor for updates.
            dynamic_k: If True, adjust K based on rating and games.
            floor_rating: Minimum rating allowed.
            provisional_games: Games before player is "established".
        """
        self.k_factor = k_factor
        self.dynamic_k = dynamic_k
        self.floor_rating = floor_rating
        self.provisional_games = provisional_games

    def expected_score(
        self,
        rating_a: int,
        rating_b: int,
    ) -> float:
        """
        Calculate expected score for player A.

        Args:
            rating_a: Player A's rating.
            rating_b: Player B's rating.

        Returns:
            Expected score (0-1) for player A.
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def get_k_factor(
        self,
        rating: int,
        games_played: int,
    ) -> int:
        """
        Get appropriate K-factor for a player.

        Higher K for new/low-rated players (faster adjustment).
        Lower K for established/high-rated players (stability).

        Args:
            rating: Player's current rating.
            games_played: Number of games played.

        Returns:
            K-factor to use for this player.
        """
        if not self.dynamic_k:
            return self.k_factor

        # Provisional period
        if games_played < self.provisional_games:
            return self.K_PROVISIONAL

        # Rating-based K
        if rating >= 2400:
            return self.K_MASTER
        elif rating >= 2000:
            return self.K_ESTABLISHED
        else:
            return self.K_STANDARD

    def update_ratings(
        self,
        rating_a: int,
        rating_b: int,
        score_a: float,
        games_a: int = 100,
        games_b: int = 100,
        player_id_a: str = 'player_a',
        player_id_b: str = 'player_b',
    ) -> Tuple[EloResult, EloResult]:
        """
        Calculate new ratings after a game.

        Args:
            rating_a: Player A's current rating.
            rating_b: Player B's current rating.
            score_a: Player A's score (1.0 = win, 0.5 = draw, 0.0 = loss).
            games_a: Games played by A (for K-factor).
            games_b: Games played by B (for K-factor).
            player_id_a: ID for player A.
            player_id_b: ID for player B.

        Returns:
            Tuple of (EloResult for A, EloResult for B).
        """
        # Get K-factors
        k_a = self.get_k_factor(rating_a, games_a)
        k_b = self.get_k_factor(rating_b, games_b)

        # Expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        # Actual scores
        score_b = 1.0 - score_a

        # Calculate changes
        change_a = round(k_a * (score_a - expected_a))
        change_b = round(k_b * (score_b - expected_b))

        # Apply changes with floor
        new_rating_a = max(self.floor_rating, rating_a + change_a)
        new_rating_b = max(self.floor_rating, rating_b + change_b)

        return (
            EloResult(
                player_id=player_id_a,
                old_rating=rating_a,
                new_rating=new_rating_a,
                change=new_rating_a - rating_a,
                expected_score=expected_a,
                actual_score=score_a,
            ),
            EloResult(
                player_id=player_id_b,
                old_rating=rating_b,
                new_rating=new_rating_b,
                change=new_rating_b - rating_b,
                expected_score=expected_b,
                actual_score=score_b,
            ),
        )

    def update_from_match(
        self,
        rating_a: int,
        rating_b: int,
        wins_a: int,
        wins_b: int,
        draws: int = 0,
        games_a: int = 100,
        games_b: int = 100,
    ) -> Tuple[int, int]:
        """
        Update ratings after a multi-game match.

        Args:
            rating_a: Player A's rating.
            rating_b: Player B's rating.
            wins_a: Number of games A won.
            wins_b: Number of games B won.
            draws: Number of drawn games.
            games_a: Prior games for A.
            games_b: Prior games for B.

        Returns:
            Tuple of (new rating A, new rating B).
        """
        total_games = wins_a + wins_b + draws
        if total_games == 0:
            return rating_a, rating_b

        # Calculate total score
        score_a = (wins_a + 0.5 * draws) / total_games

        # Update using average game
        result_a, result_b = self.update_ratings(
            rating_a=rating_a,
            rating_b=rating_b,
            score_a=score_a,
            games_a=games_a,
            games_b=games_b,
        )

        # Scale change by number of games (approximately)
        scale = min(total_games, 5)  # Cap at 5 games worth of change
        change_a = int(result_a.change * scale / 1.5)
        change_b = int(result_b.change * scale / 1.5)

        return (
            max(self.floor_rating, rating_a + change_a),
            max(self.floor_rating, rating_b + change_b),
        )


def rating_probability(
    rating_a: int,
    rating_b: int,
) -> Tuple[float, float, float]:
    """
    Calculate win/draw/loss probabilities from ratings.

    Simple model based on ELO expected scores.

    Args:
        rating_a: Player A's rating.
        rating_b: Player B's rating.

    Returns:
        Tuple of (P(A wins), P(draw), P(B wins)).
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    # Simple model: win prob proportional to expected score
    # Assume ~10% draw rate for most games
    draw_prob = 0.1
    win_a = expected_a * (1 - draw_prob)
    win_b = (1 - expected_a) * (1 - draw_prob)

    return win_a, draw_prob, win_b


def rating_difference_to_expected(diff: int) -> float:
    """
    Convert rating difference to expected score.

    Args:
        diff: Rating difference (positive = player is stronger).

    Returns:
        Expected score (0-1).
    """
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def expected_score_to_rating_difference(expected: float) -> int:
    """
    Convert expected score to rating difference.

    Args:
        expected: Expected score (0-1).

    Returns:
        Rating difference (positive = player is stronger).
    """
    import math
    if expected <= 0 or expected >= 1:
        return 0

    return int(-400 * math.log10(1.0 / expected - 1))
