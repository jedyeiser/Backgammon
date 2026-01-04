"""
Backgammon position evaluation functions.

These heuristics capture important aspects of backgammon positions:
- Pip count (race position)
- Blot exposure (vulnerability)
- Made points (blocking)
- Home board strength (bearing off readiness)
- Prime structures (blocking opponent)

Higher values are better for the specified player.
"""
from typing import Any, Dict, List, Tuple


def evaluate_position(
    game_state: Dict[str, Any],
    player: str = 'white',
    weights: Dict[str, float] = None,
) -> float:
    """
    Evaluate a backgammon position using weighted heuristics.

    Combines multiple evaluation factors into a single score.
    Higher scores are better for the specified player.

    Args:
        game_state: Backgammon game state dictionary.
        player: The player to evaluate for.
        weights: Optional custom weights for each factor.
                Default weights are tuned for reasonable play.

    Returns:
        Position evaluation score (higher is better).
    """
    default_weights = {
        'pip_count': -0.01,      # Lower pip count is better (negative weight)
        'blot_penalty': -0.1,    # Fewer blots is better
        'made_points': 0.15,     # More made points is better
        'home_board': 0.2,       # Stronger home board is better
        'borne_off': 0.5,        # More borne off is better
        'opponent_bar': 0.3,     # Opponent on bar is good
        'prime_bonus': 0.25,     # Primes are valuable
    }
    w = weights or default_weights

    opponent = 'black' if player == 'white' else 'white'
    points = game_state.get('points', {})
    bar = game_state.get('bar', {'white': 0, 'black': 0})
    home = game_state.get('home', {'white': 0, 'black': 0})

    # Calculate individual factors
    player_pip = pip_count(game_state, player)
    opponent_pip = pip_count(game_state, opponent)
    pip_diff = opponent_pip - player_pip  # Positive means we're ahead

    player_blots = blot_count(game_state, player)
    opponent_blots = blot_count(game_state, opponent)

    player_made = made_points_count(game_state, player)
    player_home_strength = home_board_strength(game_state, player)

    player_borne = home.get(player, 0)
    opponent_on_bar = bar.get(opponent, 0)

    prime_length = longest_prime(game_state, player)

    # Combine factors
    score = 0.0
    score += w['pip_count'] * player_pip
    score += w['blot_penalty'] * player_blots
    score += w['made_points'] * player_made
    score += w['home_board'] * player_home_strength
    score += w['borne_off'] * player_borne
    score += w['opponent_bar'] * opponent_on_bar
    score += w['prime_bonus'] * max(0, prime_length - 2)  # Bonus for 3+ primes

    return score


def pip_count(game_state: Dict[str, Any], player: str) -> int:
    """
    Calculate the pip count for a player.

    Pip count is the total number of pips (points) a player must
    move to bear off all checkers. Lower is better.

    Args:
        game_state: Backgammon game state.
        player: 'white' or 'black'.

    Returns:
        Total pip count for the player.
    """
    points = game_state.get('points', {})
    bar = game_state.get('bar', {'white': 0, 'black': 0})

    total = 0

    # Add pips from bar (25 pips for white, 25 for black from their perspective)
    bar_count = bar.get(player, 0)
    total += bar_count * 25

    # Add pips from each point
    for point_str, count in points.items():
        point = int(point_str)

        if player == 'white' and count > 0:
            # White moves from point toward 24, then off
            # Pip distance = 25 - point
            total += count * (25 - point)
        elif player == 'black' and count < 0:
            # Black moves from point toward 1, then off
            # Pip distance = point
            total += abs(count) * point

    return total


def blot_count(game_state: Dict[str, Any], player: str) -> int:
    """
    Count the number of blots (single checkers) for a player.

    Blots can be hit by the opponent, so fewer is generally better.

    Args:
        game_state: Backgammon game state.
        player: 'white' or 'black'.

    Returns:
        Number of blots.
    """
    points = game_state.get('points', {})
    blots = 0

    for point_str, count in points.items():
        if player == 'white' and count == 1:
            blots += 1
        elif player == 'black' and count == -1:
            blots += 1

    return blots


def made_points_count(game_state: Dict[str, Any], player: str) -> int:
    """
    Count the number of made points (2+ checkers) for a player.

    Made points block the opponent and are safe from being hit.

    Args:
        game_state: Backgammon game state.
        player: 'white' or 'black'.

    Returns:
        Number of made points.
    """
    points = game_state.get('points', {})
    made = 0

    for point_str, count in points.items():
        if player == 'white' and count >= 2:
            made += 1
        elif player == 'black' and count <= -2:
            made += 1

    return made


def home_board_strength(game_state: Dict[str, Any], player: str) -> int:
    """
    Calculate home board strength (made points in home board).

    A strong home board makes it harder for opponent to re-enter
    from the bar.

    Args:
        game_state: Backgammon game state.
        player: 'white' or 'black'.

    Returns:
        Number of made points in home board (0-6).
    """
    points = game_state.get('points', {})
    strength = 0

    # Home board points
    if player == 'white':
        home_points = range(19, 25)  # Points 19-24
    else:
        home_points = range(1, 7)    # Points 1-6

    for point in home_points:
        count = points.get(str(point), 0)
        if player == 'white' and count >= 2:
            strength += 1
        elif player == 'black' and count <= -2:
            strength += 1

    return strength


def longest_prime(game_state: Dict[str, Any], player: str) -> int:
    """
    Find the longest prime (consecutive made points) for a player.

    A 6-prime (all 6 consecutive points made) completely blocks
    the opponent's checkers behind it.

    Args:
        game_state: Backgammon game state.
        player: 'white' or 'black'.

    Returns:
        Length of longest consecutive made points (0-6).
    """
    points = game_state.get('points', {})

    # Create a list of whether each point is made
    made = [False] * 24
    for i in range(24):
        point = i + 1
        count = points.get(str(point), 0)
        if player == 'white' and count >= 2:
            made[i] = True
        elif player == 'black' and count <= -2:
            made[i] = True

    # Find longest consecutive run
    max_length = 0
    current_length = 0

    for is_made in made:
        if is_made:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    return max_length


def race_position(game_state: Dict[str, Any], player: str) -> bool:
    """
    Determine if the position is a "race" (no contact).

    A race occurs when neither player can hit the other,
    so it's purely about rolling dice and bearing off.

    Args:
        game_state: Backgammon game state.
        player: 'white' or 'black'.

    Returns:
        True if no contact is possible.
    """
    points = game_state.get('points', {})
    bar = game_state.get('bar', {'white': 0, 'black': 0})

    # If anyone is on the bar, not a race
    if bar.get('white', 0) > 0 or bar.get('black', 0) > 0:
        return False

    # Find frontmost white and rearmost black
    white_front = 0  # Highest point white occupies
    black_rear = 25  # Lowest point black occupies

    for point_str, count in points.items():
        point = int(point_str)
        if count > 0:  # White
            white_front = max(white_front, point)
        elif count < 0:  # Black
            black_rear = min(black_rear, point)

    # If white's frontmost is behind black's rearmost, it's a race
    # (White moves toward 24, black moves toward 1)
    return white_front < black_rear


def blot_exposure(game_state: Dict[str, Any], player: str) -> float:
    """
    Calculate weighted blot exposure based on position.

    Blots closer to opponent's checkers are more vulnerable.
    This is a more sophisticated measure than simple blot count.

    Args:
        game_state: Backgammon game state.
        player: 'white' or 'black'.

    Returns:
        Weighted exposure score (higher = more vulnerable).
    """
    points = game_state.get('points', {})
    opponent = 'black' if player == 'white' else 'white'

    # Find opponent's checker positions
    opponent_positions = []
    for point_str, count in points.items():
        point = int(point_str)
        if opponent == 'white' and count > 0:
            opponent_positions.append(point)
        elif opponent == 'black' and count < 0:
            opponent_positions.append(point)

    if not opponent_positions:
        return 0.0

    # Calculate exposure for each blot
    exposure = 0.0

    for point_str, count in points.items():
        point = int(point_str)
        is_blot = (player == 'white' and count == 1) or (player == 'black' and count == -1)

        if is_blot:
            # Calculate probability of being hit
            for opp_point in opponent_positions:
                if player == 'white':
                    # Opponent (black) moves from higher to lower
                    distance = opp_point - point
                else:
                    # Opponent (white) moves from lower to higher
                    distance = point - opp_point

                if 1 <= distance <= 24:
                    # Direct shots (can be hit with single die or combination)
                    hit_prob = _direct_hit_probability(distance)
                    exposure += hit_prob

    return exposure


def _direct_hit_probability(distance: int) -> float:
    """
    Approximate probability of hitting a blot at given distance.

    Based on backgammon hitting probabilities.
    """
    # Approximate hitting chances for each distance
    # These are rough approximations
    hit_probs = {
        1: 0.306,   # 11/36
        2: 0.333,   # 12/36
        3: 0.389,   # 14/36
        4: 0.417,   # 15/36
        5: 0.417,   # 15/36
        6: 0.472,   # 17/36
        7: 0.167,   # 6/36 (only 1-6, 6-1, 2-5, 5-2, 3-4, 4-3)
        8: 0.167,   # 6/36
        9: 0.139,   # 5/36
        10: 0.083,  # 3/36
        11: 0.056,  # 2/36
        12: 0.083,  # 3/36 (double 6)
        15: 0.028,  # 1/36 (5-5)
        16: 0.028,  # 1/36 (4-4)
        18: 0.028,  # 1/36 (6-6)
        20: 0.028,  # 1/36 (5-5)
        24: 0.028,  # 1/36 (6-6)
    }
    return hit_probs.get(distance, 0.0)


def generate_all_moves(
    game_state: Dict[str, Any],
    player: str,
) -> List[Dict[str, Any]]:
    """
    Generate all legal moves for a player given the current state.

    This generates individual moves (single checker movements), not
    complete turn sequences. For training, we process moves one at a time.

    Args:
        game_state: Current backgammon state with points, bar, dice, etc.
        player: 'white' or 'black'.

    Returns:
        List of legal move dictionaries with 'from', 'to', 'die_used'.
    """
    points = game_state.get('points', {})
    bar = game_state.get('bar', {'white': 0, 'black': 0})
    home = game_state.get('home', {'white': 0, 'black': 0})
    moves_remaining = game_state.get('moves_remaining', [])

    if not moves_remaining:
        return []

    legal_moves = []
    is_white = player == 'white'
    sign = 1 if is_white else -1

    # Get unique dice values to avoid duplicate move generation
    unique_dice = set(moves_remaining)

    # If player has checkers on the bar, must move them first
    player_bar = bar.get(player, 0)

    if player_bar > 0:
        # Must enter from bar
        for die in unique_dice:
            if is_white:
                # White enters on opponent's home board (points 1-6)
                to_point = die  # Die roll of 1 enters on point 1, etc.
                from_point = 0  # Bar
            else:
                # Black enters on opponent's home board (points 19-24)
                to_point = 25 - die  # Die roll of 1 enters on point 24
                from_point = 25  # Bar

            # Check if landing point is valid
            if _can_land(points, to_point, is_white):
                legal_moves.append({
                    'type': 'move',
                    'from': from_point,
                    'to': to_point,
                    'die_used': die,
                })

        return legal_moves

    # Check if player can bear off
    can_bear_off = _can_bear_off(game_state, player)

    # Generate moves from each point
    for point in range(1, 25):
        point_count = points.get(str(point), 0)

        # Check if player has checkers on this point
        if is_white and point_count <= 0:
            continue
        if not is_white and point_count >= 0:
            continue

        for die in unique_dice:
            if is_white:
                to_point = point + die
            else:
                to_point = point - die

            # Bearing off
            if is_white and to_point > 24:
                if can_bear_off:
                    # Can bear off if exact or highest point
                    if to_point == 25 or _is_highest_point(points, point, is_white):
                        legal_moves.append({
                            'type': 'move',
                            'from': point,
                            'to': 26,  # White bear off
                            'die_used': die,
                        })
            elif not is_white and to_point < 1:
                if can_bear_off:
                    if to_point == 0 or _is_highest_point(points, point, is_white):
                        legal_moves.append({
                            'type': 'move',
                            'from': point,
                            'to': 27,  # Black bear off
                            'die_used': die,
                        })
            elif 1 <= to_point <= 24:
                # Regular move
                if _can_land(points, to_point, is_white):
                    legal_moves.append({
                        'type': 'move',
                        'from': point,
                        'to': to_point,
                        'die_used': die,
                    })

    return legal_moves


def _can_land(points: Dict[str, int], point: int, is_white: bool) -> bool:
    """Check if a player can land on a point."""
    count = points.get(str(point), 0)

    if is_white:
        # White can land if empty, has white checkers, or single black (hit)
        return count >= -1
    else:
        # Black can land if empty, has black checkers, or single white (hit)
        return count <= 1


def _can_bear_off(game_state: Dict[str, Any], player: str) -> bool:
    """Check if a player can bear off (all checkers in home board)."""
    points = game_state.get('points', {})
    bar = game_state.get('bar', {'white': 0, 'black': 0})

    # Cannot bear off if any checker on bar
    if bar.get(player, 0) > 0:
        return False

    is_white = player == 'white'

    for point in range(1, 25):
        count = points.get(str(point), 0)

        if is_white and count > 0:
            # White checker not in home board (19-24)
            if point < 19:
                return False
        elif not is_white and count < 0:
            # Black checker not in home board (1-6)
            if point > 6:
                return False

    return True


def _is_highest_point(points: Dict[str, int], point: int, is_white: bool) -> bool:
    """Check if this is the highest occupied point for bearing off."""
    if is_white:
        # Check if any white checkers on higher points
        for p in range(point + 1, 25):
            if points.get(str(p), 0) > 0:
                return False
    else:
        # Check if any black checkers on lower points
        for p in range(1, point):
            if points.get(str(p), 0) < 0:
                return False
    return True
