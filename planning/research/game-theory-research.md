# Backgammon AI: Game Theory and Artificial Intelligence Research

## Table of Contents
1. [Backgammon as a Game Theory Problem](#1-backgammon-as-a-game-theory-problem)
2. [Classical AI Approaches](#2-classical-ai-approaches)
3. [Position Evaluation Concepts](#3-position-evaluation-concepts)
4. [TD-Gammon Deep Dive](#4-td-gammon-deep-dive)
5. [Modern AI Approaches](#5-modern-ai-approaches)
6. [Doubling Cube Theory](#6-doubling-cube-theory)

---

## 1. Backgammon as a Game Theory Problem

### 1.1 Formal Classification

Backgammon occupies a unique position in game theory, classified as:

| Property | Classification | Explanation |
|----------|---------------|-------------|
| **Information** | Perfect Information | Both players see the complete board state |
| **Determinism** | Stochastic (Non-deterministic) | Dice rolls introduce randomness |
| **Sum Type** | Zero-Sum | One player's gain equals the other's loss |
| **Players** | Two-Player | Strictly adversarial |
| **Horizon** | Finite | Games eventually terminate |

#### Formal Definition

A backgammon game can be modeled as a tuple:

```
G = (S, A, T, P, R, γ)

Where:
- S: State space (board configurations + cube state + player to move)
- A: Action space (legal moves given current dice roll)
- T: Transition function T(s, a, d) → s' where d is dice outcome
- P: Probability distribution over dice outcomes (36 combinations)
- R: Reward function (win/loss, potentially with gammon/backgammon multipliers)
- γ: Discount factor (typically 1 for episodic games)
```

### 1.2 Comparison with Other Classic Games

| Aspect | Backgammon | Chess | Go | Poker |
|--------|------------|-------|----|----- |
| **Information** | Perfect | Perfect | Perfect | Imperfect |
| **Determinism** | Stochastic | Deterministic | Deterministic | Stochastic |
| **Branching Factor** | ~400 (avg) | ~35 | ~250 | Variable |
| **Game Length** | ~50-100 moves | ~40-80 moves | ~150-200 moves | Variable |
| **State Space** | ~10^20 | ~10^47 | ~10^170 | ~10^14 |
| **Luck Component** | Significant | None | None | Significant |
| **Solved Status** | No | Opening books | No | Limit variants solved |

#### Why Backgammon is Uniquely Challenging

1. **Chance Nodes**: Unlike chess/Go, the game tree includes chance nodes (dice rolls)
2. **High Branching**: With 21 distinct dice outcomes and multiple legal moves per roll, effective branching factor is enormous
3. **Evaluation Complexity**: Position value is an *expected value* over possible dice sequences
4. **Cube Decisions**: The doubling cube adds a meta-game layer absent in other classics

### 1.3 The Role of Luck vs. Skill

#### Mathematical Framework

The outcome of a backgammon game can be decomposed:

```
Result = Skill_Differential + Luck_Component + ε

Where:
- Skill_Differential: Expected edge from better play
- Luck_Component: Variance from dice (zero mean, high variance)
- ε: Interaction effects
```

#### Quantifying Skill vs. Luck

**Short-term (Single Game)**:
- Luck dominates: Even a world champion loses ~35-40% of games against intermediate players
- Standard deviation of a single game result: ~1.0 to 1.2 points (including gammons)

**Medium-term (Match/Session)**:
- In a 7-point match, the better player wins ~60-65% against significantly weaker opponents
- Luck standard deviation decreases as √n where n = number of decisions

**Long-term (Many Games)**:
- Skill completely dominates: Over 1000+ games, superior players consistently outperform
- The "luck component" averages to zero; only skill differential remains

#### The Luck Quantification Formula

Researchers have developed methods to measure luck in individual games:

```
Luck(game) = Σ [P(win|actual_roll) - P(win|expected_roll)] for each roll

Where:
- P(win|actual_roll): Win probability after the roll that occurred
- P(win|expected_roll): Expected win probability averaged over all possible rolls
```

This allows post-game analysis showing how much luck affected the outcome.

### 1.4 Expected Value Calculations with Dice

#### Dice Probability Distribution

There are 36 possible dice outcomes from two six-sided dice:

| Type | Count | Probability | Examples |
|------|-------|-------------|----------|
| Doubles | 6 | 1/36 each | 1-1, 2-2, 3-3, 4-4, 5-5, 6-6 |
| Non-doubles | 30 | 2/36 each (order doesn't matter) | 1-2, 1-3, ..., 5-6 |
| **Distinct outcomes** | 21 | Varies | - |

#### Expected Value of a Position

For any position P, the true value is:

```
V(P) = Σ P(dice=d) × max_a∈A(d) [V(successor(P, a))]
       d∈D

Where:
- D: Set of all 21 distinct dice outcomes
- A(d): Legal actions given dice d
- successor(P, a): Resulting position after action a
```

#### Probability Calculations for Common Events

**Hitting a Blot (single checker)**:

The probability of hitting a blot n points away:

| Distance | Direct Hits | Combinations | Probability |
|----------|-------------|--------------|-------------|
| 1 | 1, 1-x | 11 ways | 11/36 = 30.6% |
| 2 | 2, 1-1, 2-x | 12 ways | 12/36 = 33.3% |
| 3 | 3, 1-2, 3-x | 14 ways | 14/36 = 38.9% |
| 4 | 4, 1-3, 2-2, 4-x | 15 ways | 15/36 = 41.7% |
| 5 | 5, 1-4, 2-3, 5-x | 15 ways | 15/36 = 41.7% |
| 6 | 6, 1-5, 2-4, 3-3, 6-x | 17 ways | 17/36 = 47.2% |
| 7 | 1-6, 2-5, 3-4 | 6 ways | 6/36 = 16.7% |
| 8 | 2-6, 3-5, 4-4 | 6 ways | 6/36 = 16.7% |
| 9 | 3-6, 4-5 | 5 ways | 5/36 = 13.9% |
| 10 | 4-6, 5-5 | 4 ways | 4/36 = 11.1% |
| 11 | 5-6 | 2 ways | 2/36 = 5.6% |
| 12 | 6-6, 4-4, 3-3 | 3 ways | 3/36 = 8.3% |

**Entering from the Bar**:

With n points blocked in opponent's home board:

```
P(entering) = 1 - (n/6)^2

n=1: 97.2%
n=2: 88.9%
n=3: 75.0%
n=4: 55.6%
n=5: 30.6%
n=6: 0% (closed out)
```

**Expected Pip Count from a Roll**:

```
E[pips moved] = E[sum of dice] × multiplier

For non-doubles: E[sum] = 7 (average of d1 + d2)
For doubles: E[sum] = 14 (4 × average die = 4 × 3.5)

P(doubles) = 1/6
P(non-doubles) = 5/6

E[pips per roll] = (1/6 × 14) + (5/6 × 7) × factor
                 = 2.33 + 5.83 × factor
                 ≈ 8.17 pips (without blocked points)
```

---

## 2. Classical AI Approaches

### 2.1 Expectimax Algorithm

#### Theory and Foundation

Expectimax extends minimax to handle stochastic games by introducing **chance nodes** into the game tree:

```
              MAX (Our move)
             /    |    \
          Chance Chance Chance  (Dice outcomes)
         /  |  \
       MIN MIN MIN  (Opponent's response to our move)
      / | \
  Chance...  (Opponent's dice)
```

#### Algorithm Definition

```python
def expectimax(state, depth, maximizing_player):
    if depth == 0 or is_terminal(state):
        return evaluate(state)

    if is_chance_node(state):
        # Average over all dice outcomes
        value = 0
        for dice, prob in dice_probabilities():
            next_state = apply_dice(state, dice)
            value += prob * expectimax(next_state, depth, maximizing_player)
        return value

    elif maximizing_player:
        value = -infinity
        for move in get_legal_moves(state):
            next_state = apply_move(state, move)
            value = max(value, expectimax(next_state, depth-1, False))
        return value

    else:  # Minimizing player
        value = +infinity
        for move in get_legal_moves(state):
            next_state = apply_move(state, move)
            value = min(value, expectimax(next_state, depth-1, True))
        return value
```

#### Complexity Analysis

At depth d (counting half-moves):

```
Nodes = (B_move × B_dice)^(d/2)

Where:
- B_move ≈ 20 (average legal moves per dice roll)
- B_dice = 21 (distinct dice outcomes)
- Effective branching factor ≈ 420

For d=4 (2-ply lookahead):
Nodes ≈ 420^2 = 176,400

For d=6 (3-ply lookahead):
Nodes ≈ 420^3 = 74,088,000
```

### 2.2 Handling the Branching Factor

#### Problem: 21 Dice Outcomes × Many Moves

The combinatorial explosion is severe:

1. **Dice Combinations**: 21 distinct outcomes per turn
2. **Move Combinations**: Given dice (a, b):
   - For non-doubles: up to 4 partial moves (a, b, or combined)
   - For doubles: up to 8 partial moves (4 × die value)
3. **Permutations**: Order matters in some positions

#### Solutions and Optimizations

**1. Move Generation Pruning**
```python
def generate_moves(position, dice):
    """Generate only legal, non-equivalent moves"""
    moves = []

    # Prune equivalent moves (same end state)
    seen_states = set()

    for move_sequence in all_move_sequences(dice):
        if is_legal(move_sequence, position):
            end_state = apply_moves(position, move_sequence)
            state_hash = hash(end_state)

            if state_hash not in seen_states:
                seen_states.add(state_hash)
                moves.append(move_sequence)

    return moves
```

**2. Dice Outcome Grouping**
- Group similar dice outcomes that lead to similar evaluations
- Use importance sampling for rare but impactful rolls

**3. Progressive Deepening**
```python
def iterative_deepening_expectimax(state, time_limit):
    best_move = None
    depth = 1

    while time_remaining(time_limit):
        try:
            best_move = expectimax_search(state, depth)
            depth += 1
        except TimeoutException:
            break

    return best_move
```

### 2.3 Pruning Strategies for Stochastic Games

#### Star1 and Star2 Algorithms

Unlike deterministic games where alpha-beta pruning is straightforward, stochastic games require specialized pruning:

**Star1 Algorithm (Ballard, 1983)**:
- Bounds expected values using optimistic/pessimistic assumptions
- If upper bound of subtree < current alpha, prune

```python
def star1(state, depth, alpha, beta, is_max):
    if depth == 0:
        return evaluate(state)

    if is_chance_node(state):
        # Compute bounds on expected value
        outcomes = get_dice_outcomes()

        # Optimistic bound: assume best dice for current player
        upper_bound = compute_upper_bound(state, outcomes, depth)

        if upper_bound <= alpha:
            return upper_bound  # Prune

        # Compute actual expected value
        return expected_value_search(state, outcomes, depth, alpha, beta)
```

**Star2 Algorithm (Hauk, 2004)**:
- Improves on Star1 with tighter bounds
- Uses probe searches to estimate subtree values

**Monte Carlo Pruning**:
- Sample dice outcomes instead of exhaustive enumeration
- Statistically valid with confidence bounds

```python
def monte_carlo_expectimax(state, depth, num_samples):
    if depth == 0:
        return evaluate(state)

    if is_chance_node(state):
        samples = random.choices(dice_outcomes,
                                  weights=dice_probabilities,
                                  k=num_samples)

        total = sum(expectimax(apply_dice(state, d), depth-1)
                    for d in samples)
        return total / num_samples
```

### 2.4 Evaluation Function Design

#### Components of a Classical Evaluation Function

```python
def evaluate(position):
    """
    Classical hand-crafted evaluation function
    Returns: float in [-1, 1] representing expected game outcome
    """

    # 1. Material/Racing Advantage
    pip_diff = pip_count(opponent) - pip_count(player)
    racing_score = sigmoid(pip_diff / 30)  # Normalize

    # 2. Positional Factors
    blot_penalty = -0.02 * count_blots(player)
    blot_threat = 0.015 * count_threatened_blots(opponent)

    # 3. Structure
    prime_bonus = 0.05 * prime_length(player)
    anchor_bonus = evaluate_anchors(player)

    # 4. Board Coverage
    point_control = 0.01 * points_controlled(player)

    # 5. Bearing Off
    if is_bearing_off(player):
        bearoff_efficiency = evaluate_bearoff(player)
    else:
        bearoff_efficiency = 0

    # 6. Checker Distribution
    distribution_score = evaluate_distribution(player)

    # 7. Timing (relative race progress)
    timing_score = evaluate_timing(position)

    # Combine with learned or tuned weights
    score = (
        0.3 * racing_score +
        0.15 * blot_penalty +
        0.1 * blot_threat +
        0.15 * prime_bonus +
        0.1 * anchor_bonus +
        0.05 * point_control +
        0.1 * bearoff_efficiency +
        0.05 * distribution_score
    )

    return tanh(score)  # Squash to [-1, 1]
```

#### Feature Engineering

**Pip Count Features**:
```python
def pip_count(player):
    """Total distance checkers must travel to bear off"""
    total = 0
    for point in range(1, 25):
        checkers = count_checkers(player, point)
        total += checkers * point_distance(point, player)

    # Add bar penalty
    total += bar_checkers(player) * 25  # Must re-enter first

    return total
```

**Blot Exposure Calculation**:
```python
def blot_danger(position, blot_point):
    """Calculate probability blot gets hit"""
    danger = 0

    for distance in range(1, 25):
        if opponent_can_reach(blot_point, distance):
            # Factor in number of opponent checkers that can hit
            hitters = count_hitters(opponent, blot_point, distance)
            hit_prob = probability_of_roll(distance)

            # Weight by cost of being hit
            cost = estimate_hit_cost(position, blot_point)

            danger += hitters * hit_prob * cost

    return danger
```

---

## 3. Position Evaluation Concepts

### 3.1 Pip Count and Racing Positions

#### Basic Pip Count

The pip count is the fundamental racing metric:

```
Pip Count = Σ (distance_to_bearoff × checkers_at_point)

For each player, sum over all 15 checkers:
- Point 1 (closest to bearoff) = 1 pip per checker
- Point 24 (farthest) = 24 pips per checker
- Bar = 25 pips per checker (must re-enter)
- Borne off = 0 pips
```

**Starting Position Pip Count**: Each player starts with 167 pips.

#### Adjusted Pip Count

Raw pip count is insufficient; adjustments improve accuracy:

```python
def adjusted_pip_count(position, player):
    raw = pip_count(position, player)

    # Wastage: checkers too close to bear off
    # Having checkers on 1-point with high dice wastes pips
    wastage = calculate_wastage(position, player)

    # Crossover penalty: checkers in outer boards
    # Each "crossover" between quadrants costs ~1 pip
    crossovers = count_crossovers(position, player)

    # Gap penalty: missing points create inefficiency
    gaps = count_gaps_in_bearoff(position, player)

    # Stacking penalty: too many on one point
    stacking = calculate_stacking_penalty(position, player)

    return raw + wastage + 0.5 * crossovers + 0.5 * gaps + stacking
```

#### Keith Count (Practical Racing Formula)

A practical formula developed by Tom Keith:

```
Keith Count = Pip Count + Adjustments

Adjustments:
+2 for each checker more than 1 on the 1-point
+1 for each checker more than 1 on the 2-point
+1 for each checker more than 3 on the 3-point
+1 for each empty space on 4, 5, or 6-point
+1 for each extra crossover (>8)
```

#### Race Decision Thresholds

When to simplify to pure race (empirically derived):

```
If pip_lead >= 8% of total pips:
    Strong racing advantage, favor simplification

If pip_lead < 4% and have back anchor:
    Consider holding/back game

If trailing by > 15 pips:
    Seek contact, blots, complications
```

### 3.2 Blot Exposure and Safety

#### Direct Shot Analysis

A checker is "directly exposed" if opponent can hit with a single die:

```python
def direct_shots(blot_point, opponent_position):
    """Count number of direct hitting rolls"""
    shots = 0

    for dist in range(1, 7):  # Direct shots only
        source_point = blot_point + dist  # Opponent's perspective

        if opponent_has_checker(source_point):
            shots += direct_roll_count(dist)

    return shots

def direct_roll_count(distance):
    """How many of 36 rolls hit at exactly this distance"""
    if distance <= 6:
        return 11 if distance == 1 else 11 + (distance == 6)
        # Actually: varies by distance, see table in Section 1.4
```

#### Indirect (Combination) Shot Analysis

Shots requiring both dice:

```python
def indirect_shots(blot_point, opponent_position):
    """Count combination shots (7-24 distance)"""
    shots = 0

    for dist in range(7, 25):
        if opponent_has_checker_at_distance(blot_point, dist):
            # Must check intermediate points are clear
            shots += count_combination_rolls(dist, opponent_position, blot_point)

    return shots
```

#### Blot Risk Assessment Framework

```python
def evaluate_blot_risk(position, blot_point, player):
    """
    Comprehensive blot risk analysis
    Returns: tuple (hit_probability, expected_loss_if_hit)
    """

    # Phase 1: Count shots
    direct = count_direct_shots(blot_point, opponent)
    indirect = count_indirect_shots(blot_point, opponent)
    total_shots = direct + indirect
    hit_prob = total_shots / 36

    # Phase 2: Assess cost if hit
    # Where will hit checker go?
    bar_entry_difficulty = blocked_home_points(opponent)

    # Timing loss
    timing_loss = estimate_timing_loss(position, blot_point)

    # Losing key point
    point_value = evaluate_point_importance(blot_point)

    # Positional damage
    position_damage = (
        timing_loss * 0.4 +
        bar_entry_difficulty * 0.3 +
        point_value * 0.3
    )

    expected_loss = hit_prob * position_damage

    return hit_prob, expected_loss
```

### 3.3 Priming and Blocking

#### Prime Definition and Value

A **prime** is a sequence of consecutive occupied points blocking opponent checkers:

```
Prime Length | Blocking Power
-------------|---------------
2 points     | Weak (6+ or specific rolls escape)
3 points     | Moderate (limits movement)
4 points     | Strong (very hard to escape)
5 points     | Very strong (2 numbers escape)
6 points     | Perfect (no escape possible)
```

#### Prime Evaluation

```python
def evaluate_prime(position, player):
    """
    Assess prime quality and impact
    """
    prime_points = find_prime_points(position, player)
    prime_length = len(prime_points)

    if prime_length < 3:
        return 0  # Not a real prime

    # Base value by length
    base_value = {3: 0.1, 4: 0.25, 5: 0.5, 6: 1.0}[prime_length]

    # Adjust for location
    # Primes in front of opponent back checkers are more valuable
    blocked_checkers = count_blocked_opponent_checkers(prime_points, opponent)
    location_bonus = 0.1 * blocked_checkers

    # Adjust for timing
    # Having a prime but running out of moves = bad
    timing = assess_timing(position, player)
    timing_factor = 1.0 if timing > 0 else 0.5

    # Adjust for prime mobility
    # Can we advance the prime?
    mobility = can_advance_prime(position, player)
    mobility_factor = 1.1 if mobility else 0.9

    return base_value * (1 + location_bonus) * timing_factor * mobility_factor
```

#### Blocking Strategy Principles

1. **Prime Building Priority**: 5, 4, bar points most valuable initially
2. **Timing Maintenance**: Keep builders available to roll prime forward
3. **Escape Prevention**: Hold prime until opponent checkers escape
4. **Conversion**: Know when to release prime and race

### 3.4 Anchor Strategies

#### Anchor Types and Values

An **anchor** is two or more checkers on a point in opponent's home board:

```
Anchor Point | Name          | Strategic Value
-------------|---------------|------------------
20 (bar)     | Bar anchor    | Best: blocks and threatens
21           | Advanced      | Excellent: good balance
22           | Mid-anchor    | Good: flexibility
23           | Deep          | Defensive: last resort
24           | Ace-point     | Defensive: limited options
```

#### Anchor Strategy Framework

```python
def evaluate_anchor_strategy(position, player):
    """
    Assess quality of anchor-based game
    """
    anchors = find_anchors(position, player)

    if not anchors:
        return 0  # No back game

    # Best anchor value
    best_anchor = min(anchors)  # Lower = further advanced
    anchor_quality = {
        20: 1.0,  # Bar point
        21: 0.9,
        22: 0.7,
        23: 0.4,
        24: 0.2   # Ace point
    }.get(best_anchor, 0)

    # Double anchor bonus
    if len(anchors) >= 2:
        anchor_quality *= 1.3

    # Timing assessment
    # Back games need good timing (spare checkers, not too advanced)
    timing = assess_timing_for_backgame(position, player)

    # Too many checkers back = poor timing
    checkers_back = count_checkers_in_opponent_home(player)
    if checkers_back > 4:
        anchor_quality *= 0.7  # Overcommitted

    return anchor_quality * timing
```

#### Back Game Theory

A **back game** involves holding deep anchors and hoping to hit:

**Requirements for Successful Back Game**:
1. **Two anchors** (preferably including 20, 21, or 22 point)
2. **Good timing** (not forced to break anchors prematurely)
3. **Opponent with blots** (something to hit)

**Back Game Equity**:
```
Back game win% ≈ P(hit) × P(win|hit) + P(no_hit) × P(win|no_hit)

Typically:
- P(hit) depends on # of blots opponent must leave
- P(win|hit) ≈ 60-70% (depending on timing)
- P(win|no_hit) ≈ 5-10%
```

### 3.5 Doubling Cube Theory (Position Evaluation Context)

When evaluating a position, cube access matters:

```python
def position_value_with_cube(position, cube_owner):
    """
    Position value depends on who owns the cube
    """
    base_equity = evaluate_position(position)

    if cube_owner == 'centered':
        # Both can double - some added value to player ahead
        return base_equity * 1.05

    elif cube_owner == current_player:
        # We own cube - can cash out or play on
        if base_equity > 0.7:
            # Strong position - cube ownership worth more
            return base_equity * 1.1
        else:
            return base_equity * 1.02

    else:  # Opponent owns cube
        # We can't double - position worth slightly less
        return base_equity * 0.98
```

---

## 4. TD-Gammon Deep Dive

### 4.1 How Tesauro's Approach Worked

#### Historical Context

Gerald Tesauro developed TD-Gammon at IBM Research in 1992, marking a watershed moment in both game AI and reinforcement learning.

#### The TD(λ) Algorithm

TD-Gammon used **Temporal Difference Learning**, specifically TD(λ):

```python
def td_lambda_update(network, trajectory, lambda_param, alpha):
    """
    TD(λ) learning update

    trajectory: list of (state, reward) pairs from a game
    lambda_param: eligibility trace decay (typically 0.7)
    alpha: learning rate
    """
    T = len(trajectory)
    eligibility = initialize_zero_traces()

    for t in range(T - 1):
        state_t, reward_t = trajectory[t]
        state_t1, reward_t1 = trajectory[t + 1]

        # Current and next value estimates
        V_t = network.forward(state_t)
        V_t1 = network.forward(state_t1)

        # TD error
        delta = reward_t + V_t1 - V_t

        # Update eligibility traces
        gradient = network.compute_gradient(state_t)
        eligibility = lambda_param * eligibility + gradient

        # Update weights
        network.weights += alpha * delta * eligibility
```

#### Self-Play Training Loop

```python
def train_td_gammon(num_games=1_500_000):
    network = create_neural_network()

    for game in range(num_games):
        # Initialize game
        board = initial_position()
        trajectory = []

        while not game_over(board):
            # Roll dice
            dice = roll_dice()

            # Get all legal moves
            moves = generate_legal_moves(board, dice)

            # Evaluate each move using current network
            best_move = None
            best_value = -infinity

            for move in moves:
                new_board = apply_move(board, move)
                # Average over opponent's dice responses
                value = expected_value(network, new_board)

                if value > best_value:
                    best_value = value
                    best_move = move

            # Apply best move
            board = apply_move(board, best_move)
            trajectory.append((encode_state(board), 0))

            # Switch player
            board = switch_player(board)

        # Game ended - assign final reward
        final_reward = get_game_result(board)  # +1, 0, -1
        trajectory[-1] = (trajectory[-1][0], final_reward)

        # TD(λ) update
        td_lambda_update(network, trajectory, lambda_param=0.7, alpha=0.1)

        # Anneal learning rate
        if game % 100000 == 0:
            alpha *= 0.9
```

### 4.2 Why It Was Revolutionary

#### Key Innovations

1. **Learning from Self-Play**: No human expert games needed
2. **Temporal Difference**: Learned from game outcomes via backpropagation through time
3. **Emergent Expert-Level Play**: Discovered known and novel strategies independently
4. **Generalization**: Neural network generalized across similar positions

#### Comparison with Prior Approaches

| Approach | Knowledge Source | Playing Strength |
|----------|------------------|------------------|
| Hand-coded (BKG 1.0) | Human expertise | Strong amateur |
| Neurogammon | Supervised on expert games | Intermediate |
| TD-Gammon 1.0 | Self-play TD learning | Strong intermediate |
| TD-Gammon 2.1 | Self-play + features | World-class |

#### Why Backgammon was Ideal for TD Learning

1. **Stochastic Nature**: Dice add exploration naturally
2. **Quick Games**: Many training games possible
3. **Smooth Evaluation Landscape**: Win probability changes gradually
4. **No Positional Loops**: Games always progress (unlike chess)

### 4.3 Key Hyperparameters and Architecture

#### Network Architecture Evolution

**TD-Gammon 1.0 (1992)**:
```
Input Layer:  198 units (raw board encoding)
Hidden Layer: 40 units, sigmoid activation
Output Layer: 4 units (P(white wins), P(white gammons),
                       P(black wins), P(black gammons))
```

**TD-Gammon 2.1 (1995)**:
```
Input Layer:  198 + additional hand-crafted features ≈ 250 units
Hidden Layer: 80 units, sigmoid activation
Output Layer: 4 units (same as above)

Additional features included:
- Pip count
- Number of blots
- Presence of primes
- Anchor information
- Contact/racing indicators
```

#### Input Encoding

```python
def encode_board_state(board):
    """
    Tesauro's raw board encoding
    198 inputs total
    """
    features = []

    # For each of 24 points
    for point in range(24):
        for player in [WHITE, BLACK]:
            count = checkers_on_point(board, point, player)

            # Encoding: 4 units per point per player
            # Unit 1: 1 if at least 1 checker
            # Unit 2: 1 if at least 2 checkers
            # Unit 3: 1 if at least 3 checkers
            # Unit 4: (n-3)/2 if n > 3, else 0

            features.append(1 if count >= 1 else 0)
            features.append(1 if count >= 2 else 0)
            features.append(1 if count >= 3 else 0)
            features.append((count - 3) / 2 if count > 3 else 0)

    # Bar checkers (2 players × 4 units each)
    for player in [WHITE, BLACK]:
        count = bar_checkers(board, player)
        features.extend([
            1 if count >= 1 else 0,
            (count - 1) / 2 if count > 1 else 0,
            0, 0  # Padding
        ])

    # Borne off checkers (2 players × 4 units each)
    for player in [WHITE, BLACK]:
        count = borne_off(board, player)
        features.extend([
            count / 15,  # Fraction borne off
            0, 0, 0  # Padding
        ])

    # Side to move (2 units)
    features.extend([1, 0] if board.to_move == WHITE else [0, 1])

    return np.array(features)  # 24×2×4 + 2×4 + 2×4 + 2 = 198
```

#### Training Hyperparameters

```python
HYPERPARAMETERS = {
    'learning_rate_initial': 0.1,
    'learning_rate_final': 0.01,
    'lambda': 0.7,  # Eligibility trace decay
    'hidden_units': 80,  # TD-Gammon 2.1
    'training_games': 1_500_000,
    'annealing_schedule': 'exponential',
    'weight_initialization': 'small_random',  # [-0.5, 0.5]
    'activation': 'sigmoid',
}
```

### 4.4 Strategies TD-Gammon Discovered

#### Novel Strategic Insights

1. **Slot and Build**:
   - TD-Gammon frequently placed blots on the 5-point early
   - Higher risk than human conventional wisdom
   - Led to revision of opening theory

2. **Running vs. Priming Trade-offs**:
   - Discovered nuanced positions where running was preferred
   - Contradicted human "always build a prime" instincts

3. **Duplication Concepts**:
   - Implicitly learned that having the same number accomplish multiple tasks is defensive
   - Example: If opponent needs 6 to both escape and hit, duplicating that 6

4. **Deep Anchor Reassessment**:
   - TD-Gammon revealed when ace-point anchors were viable
   - Modified understanding of back game timing

5. **Cube Handling (TD-Gammon 2.1+)**:
   - Learned aggressive cube actions in certain positions
   - Better understood gammon rates and their cube implications

#### Opening Move Preferences

TD-Gammon's opening preferences often differed from human consensus:

| Roll | Human Preference | TD-Gammon | Outcome |
|------|------------------|-----------|---------|
| 2-1 | 13/11, 6/5 | 13/11, 24/23 (split) | Human revised toward TD |
| 4-1 | 13/9, 6/5 | 24/23, 13/9 (split) | Still debated |
| 5-2 | 13/11, 13/8 | 24/22, 13/8 | Human revised |
| 3-2 | 13/11, 13/10 | 24/21, 13/11 | Human revised |

These "controversial" moves were later validated by rollout analysis.

---

## 5. Modern AI Approaches

### 5.1 Neural Network Architectures for Backgammon

#### Fully Connected Networks (Post TD-Gammon)

Modern improvements on the TD-Gammon architecture:

```python
class ModernBackgammonNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Richer input encoding (~250-500 features)
        self.input_size = 450

        # Deeper network with residual connections
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)

        # Multi-head output
        self.value_head = nn.Linear(64, 5)  # Win, gammon, backgammon (each side)
        self.cube_head = nn.Linear(64, 3)   # Double, take, pass

        # Residual connections
        self.skip1 = nn.Linear(self.input_size, 256)
        self.skip2 = nn.Linear(256, 128)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Layer 1 with residual
        h1 = self.activation(self.fc1(x)) + self.skip1(x)
        h1 = self.dropout(h1)

        # Layer 2
        h2 = self.activation(self.fc2(h1)) + h1
        h2 = self.dropout(h2)

        # Layer 3 with residual
        h3 = self.activation(self.fc3(h2)) + self.skip2(h2)

        # Layer 4
        h4 = self.activation(self.fc4(h3))

        # Outputs
        value = torch.sigmoid(self.value_head(h4))
        cube = torch.softmax(self.cube_head(h4), dim=-1)

        return value, cube
```

#### Convolutional Approaches

Treating the board as a 1D or pseudo-2D image:

```python
class ConvBackgammonNet(nn.Module):
    """
    1D Convolutional network treating 24 points as sequence
    """
    def __init__(self):
        super().__init__()

        # Input: 24 points × channels (checker counts, ownership, etc.)
        self.conv1 = nn.Conv1d(8, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        # Global features (bar, borne off, cube)
        self.global_fc = nn.Linear(16, 64)

        # Combined processing
        self.fc1 = nn.Linear(64 * 24 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 5)

    def forward(self, point_features, global_features):
        # Point-wise convolutions
        x = F.relu(self.conv1(point_features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten point features
        x = x.view(x.size(0), -1)

        # Process global features
        g = F.relu(self.global_fc(global_features))

        # Combine
        combined = torch.cat([x, g], dim=1)

        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))

        return torch.sigmoid(self.output(h))
```

#### Attention-Based Architectures

Transformer-style attention for positional relationships:

```python
class AttentionBackgammonNet(nn.Module):
    """
    Self-attention to capture point relationships
    """
    def __init__(self, d_model=64, n_heads=4, n_layers=3):
        super().__init__()

        # Point embeddings
        self.point_embed = nn.Linear(8, d_model)  # 8 features per point

        # Positional encoding (point positions 1-24)
        self.pos_encoding = nn.Parameter(torch.randn(24, d_model))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Global pooling and output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model + 16, 128)  # +16 for global features
        self.output = nn.Linear(128, 5)

    def forward(self, point_features, global_features):
        # Embed points
        x = self.point_embed(point_features)  # (batch, 24, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer (expects seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # (batch, d_model, 24)

        # Pool across points
        x = self.pool(x).squeeze(-1)  # (batch, d_model)

        # Combine with global features
        combined = torch.cat([x, global_features], dim=1)

        h = F.relu(self.fc(combined))
        return torch.sigmoid(self.output(h))
```

### 5.2 Monte Carlo Tree Search for Stochastic Games

#### Standard MCTS Review

```
MCTS operates in four phases:
1. Selection: Navigate tree using UCB1
2. Expansion: Add new node(s)
3. Simulation: Random playout to terminal
4. Backpropagation: Update statistics
```

#### MCTS Adaptations for Stochastic Games

**Challenge**: Chance nodes (dice) change the tree structure.

**Solution 1: Explicit Chance Nodes**

```python
class MCTSNode:
    def __init__(self, state, parent=None, node_type='decision'):
        self.state = state
        self.parent = parent
        self.node_type = node_type  # 'decision' or 'chance'
        self.children = {}
        self.visits = 0
        self.value_sum = 0

    def ucb1_select(self, c=1.414):
        if self.node_type == 'chance':
            # For chance nodes, sample according to probability
            return self._sample_chance_child()
        else:
            # For decision nodes, use UCB1
            best_child = None
            best_ucb = -infinity

            for action, child in self.children.items():
                if child.visits == 0:
                    return child

                exploitation = child.value_sum / child.visits
                exploration = c * sqrt(log(self.visits) / child.visits)
                ucb = exploitation + exploration

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            return best_child

    def _sample_chance_child(self):
        """Sample dice outcome according to true probabilities"""
        outcomes = list(self.children.keys())
        probs = [dice_probability(outcome) for outcome in outcomes]

        # Ensure all outcomes exist
        for dice_roll in ALL_DICE_ROLLS:
            if dice_roll not in self.children:
                # Create new chance outcome node
                new_state = apply_dice(self.state, dice_roll)
                self.children[dice_roll] = MCTSNode(
                    new_state, self, 'decision'
                )

        chosen = random.choices(outcomes, weights=probs)[0]
        return self.children[chosen]
```

**Solution 2: Double Progressive Widening**

Limit branching by progressively adding children:

```python
def progressive_widening_select(node, c_pw=2.0):
    """
    Add children gradually as visits increase
    """
    max_children = int(c_pw * (node.visits ** 0.5))

    if len(node.children) < max_children:
        # Expand a new child
        unexplored = get_unexplored_actions(node)
        if unexplored:
            action = select_expansion_action(unexplored, node)
            return expand(node, action)

    # Select among existing children
    return ucb1_select(node)
```

**Solution 3: Information Set MCTS (IS-MCTS)**

For handling uncertainty, though less relevant for perfect information:

```python
def determinized_mcts(state, num_determinizations=50):
    """
    Sample dice sequences and average MCTS results
    """
    action_values = defaultdict(list)

    for _ in range(num_determinizations):
        # Fix a dice sequence for this determinization
        dice_sequence = sample_dice_sequence(max_length=20)

        # Run MCTS with this fixed sequence
        root = MCTSNode(state)

        for _ in range(simulations_per_det):
            node = root
            det_index = 0

            while not node.is_terminal():
                if node.is_chance:
                    # Use predetermined dice
                    dice = dice_sequence[det_index % len(dice_sequence)]
                    det_index += 1
                    node = node.children.get(dice, expand(node, dice))
                else:
                    node = select(node)

            backpropagate(node, evaluate(node.state))

        # Record action values from this determinization
        for action, child in root.children.items():
            if child.visits > 0:
                action_values[action].append(child.value_sum / child.visits)

    # Average across determinizations
    best_action = max(action_values.keys(),
                      key=lambda a: np.mean(action_values[a]))
    return best_action
```

### 5.3 Comparison with AlphaGo/AlphaZero Techniques

#### AlphaGo/AlphaZero Architecture

```
Core components:
1. Policy Network: P(action|state) - suggests promising moves
2. Value Network: V(state) - evaluates positions
3. MCTS: Combines policy and value for search
4. Self-play: Generates training data
```

#### Adapting AlphaZero for Backgammon

**Key Differences**:

| Aspect | AlphaGo/Zero | Backgammon Adaptation |
|--------|-------------|----------------------|
| Game tree | Deterministic | Stochastic (chance nodes) |
| Action space | Place stone | Move checkers (compound) |
| Board representation | 19×19 grid | 24 points + bar |
| Policy output | Distribution over 361+ moves | Distribution over legal moves |
| Value output | Single scalar | Multiple outcomes (gammon, etc.) |

**Stochastic AlphaZero Implementation**:

```python
class StochasticAlphaZero:
    def __init__(self, network):
        self.network = network  # Outputs (policy, value)
        self.mcts_simulations = 800
        self.c_puct = 1.5

    def search(self, root_state):
        root = MCTSNode(root_state)

        for _ in range(self.mcts_simulations):
            node = root
            path = [node]

            # Selection with chance nodes
            while node.is_expanded():
                if node.is_chance_node:
                    # Sample dice according to probability
                    dice = sample_dice()
                    child_state = apply_dice(node.state, dice)
                    node = node.get_or_create_child(dice, child_state)
                else:
                    # PUCT selection
                    node = self.puct_select(node)
                path.append(node)

            # Expansion and evaluation
            if not node.is_terminal():
                policy, value = self.network(encode(node.state))
                node.expand(policy)
            else:
                value = get_terminal_value(node.state)

            # Backpropagation
            self.backpropagate(path, value)

        return self.get_action_probs(root)

    def puct_select(self, node):
        """PUCT formula for action selection"""
        best_action = None
        best_puct = -infinity

        total_visits = sum(c.visits for c in node.children.values())

        for action, child in node.children.items():
            prior = node.policy[action]

            if child.visits == 0:
                q_value = 0
            else:
                q_value = child.value_sum / child.visits

            # PUCT formula
            exploration = self.c_puct * prior * sqrt(total_visits) / (1 + child.visits)
            puct = q_value + exploration

            if puct > best_puct:
                best_puct = puct
                best_action = action

        return node.children[best_action]
```

#### Why AlphaZero Hasn't Been Applied Extensively to Backgammon

1. **Problem Already "Solved"**: Strong neural network bots (GNU Backgammon, XG) exist
2. **Lower Commercial Interest**: Less prestigious than Go/Chess
3. **TD Learning Effectiveness**: Original TD approach works well for backgammon
4. **Dice Complexity**: MCTS with chance nodes is computationally expensive

### 5.4 N-Ply Rollouts for Evaluation

#### Rollout Theory

A **rollout** is a Monte Carlo evaluation method:

```
Rollout(position, policy, num_trials):
    For each trial:
        Play out game from position using policy
        Record outcome (win/loss/gammon/backgammon)
    Return: average outcome
```

#### Cubeless vs. Cubeful Rollouts

**Cubeless Rollout**:
- Ignores doubling cube
- Evaluates pure game equity
- Faster computation
- Used for move analysis

**Cubeful Rollout**:
- Includes optimal cube decisions
- More accurate for real play
- Computationally expensive
- Critical for cube decision analysis

#### N-Ply Rollout Explained

```python
def n_ply_rollout(position, n, num_trials=1296):
    """
    N-ply rollout evaluation

    n=0: Immediate neural net evaluation (0-ply)
    n=1: Evaluate after opponent's response (1-ply)
    n=2: Evaluate after next move cycle (2-ply)
    """
    if n == 0:
        return neural_net_evaluate(position)

    total_equity = 0

    for trial in range(num_trials):
        current = position

        # Execute n plies of best play
        for ply in range(n):
            dice = sample_dice()
            moves = generate_legal_moves(current, dice)

            # Choose best move via (n-1)-ply lookahead
            best_move = None
            best_value = -infinity

            for move in moves:
                new_pos = apply_move(current, move)

                if ply == n - 1:
                    # Evaluate with neural net at final ply
                    value = neural_net_evaluate(new_pos)
                else:
                    # Recursive lookahead
                    value = lookahead_value(new_pos, n - ply - 1)

                if value > best_value:
                    best_value = value
                    best_move = move

            current = apply_move(current, best_move)
            current = switch_player(current)

        # Evaluate terminal position of this rollout
        total_equity += neural_net_evaluate(current)

    return total_equity / num_trials
```

#### Truncated Rollouts with Neural Network

```python
def truncated_rollout(position, depth, neural_net, num_trials=1000):
    """
    Roll out for 'depth' plies, then use neural net
    """
    results = []

    for _ in range(num_trials):
        current = copy(position)

        # Play 'depth' plies
        for ply in range(depth):
            if is_terminal(current):
                break

            dice = roll_dice()
            moves = generate_legal_moves(current, dice)

            # Use neural net to select move
            best_move = max(moves,
                           key=lambda m: neural_net(apply_move(current, m)))

            current = apply_move(current, best_move)
            current = switch_player(current)

        # Evaluate terminal state
        if is_terminal(current):
            results.append(terminal_value(current))
        else:
            results.append(neural_net.evaluate(current))

    return np.mean(results), np.std(results) / sqrt(num_trials)
```

#### Rollout Variance Reduction

**Variance Reduced Rollouts (VR)**:

```python
def variance_reduced_rollout(position, num_trials):
    """
    Use same dice sequence for both moves being compared
    """
    # Pre-generate dice sequences
    dice_sequences = [
        [roll_dice() for _ in range(MAX_GAME_LENGTH)]
        for _ in range(num_trials)
    ]

    candidate_moves = generate_legal_moves(position, current_dice)
    move_equities = {move: [] for move in candidate_moves}

    for dice_seq in dice_sequences:
        for move in candidate_moves:
            # Apply candidate move
            new_pos = apply_move(position, move)

            # Complete game with shared dice sequence
            equity = complete_game_with_dice(new_pos, dice_seq)
            move_equities[move].append(equity)

    # Variance is reduced because same luck applies to all moves
    return {move: np.mean(eqs) for move, eqs in move_equities.items()}
```

---

## 6. Doubling Cube Theory

### 6.1 Fundamental Cube Concepts

#### The Doubling Cube Mechanism

- Cube starts at 1, centered (either player can double)
- Doubling raises stakes and gives opponent the cube
- Opponent must take (accept) or pass (drop)
- Holder of cube has exclusive doubling rights

#### Equity with Cube

```
Without cube: Equity = P(win) - P(lose) = 2×P(win) - 1

With cube at N:
- If passed: Opponent loses N points
- If taken: Game continues at 2N stakes
```

### 6.2 Take Points and Drop Points

#### Mathematical Derivation

Let:
- P = probability of winning after taking
- G = gammon rate (prob of losing gammon if lose)
- B = backgammon rate
- Cube = N (being doubled to 2N)

**Simple Case (no gammons)**:

If you **pass**: You lose N points
If you **take**:
- Win prob P: You gain 2N
- Lose prob (1-P): You lose 2N

Expected value of taking:
```
E[take] = P × (2N) + (1-P) × (-2N)
        = 2N × (2P - 1)
```

Taking is correct when:
```
E[take] > E[pass]
2N × (2P - 1) > -N
2P - 1 > -0.5
P > 0.25
```

**Therefore, the basic take point is 25% winning chances.**

#### Take Points with Gammons

With gammon probability G (when losing):

```
E[take] = P × (2N) + (1-P)(1-G) × (-2N) + (1-P)G × (-4N)
        = 2N × [P - (1-P)(1 + G)]
```

Setting E[take] = -N (pass value):
```
P - (1-P)(1 + G) = -0.5
P = 0.5(1 + G) / (2 + G)
```

| Gammon Rate | Take Point |
|-------------|------------|
| 0% | 25.0% |
| 10% | 26.2% |
| 20% | 27.3% |
| 30% | 28.3% |
| 40% | 29.2% |
| 50% | 30.0% |

### 6.3 Doubling Window Theory

#### When to Double

The optimal doubling window depends on:
1. Current win probability
2. Volatility (how fast equity can change)
3. Recube potential (opponent can redouble)

**Dead Cube Model** (no recube):

```
Double when: P(win) ≥ 50%
(Any lead should be doubled to lock in value)
```

**Live Cube Model** (with recube):

```
Optimal doubling point ≈ 70-76% winning chances

Reasoning:
- If you double at 70%, opponent takes (loses equity)
- But opponent now owns cube
- If opponent reaches 70%, they can redouble
- Must account for recube vig
```

#### Market Losing Equity

If equity can become too good to double (opponent will pass), you've "lost your market":

```python
def market_window(position):
    """
    Calculate doubling market window
    Returns: (double_point, too_good_point)
    """
    # Analyze volatility
    one_roll_outcomes = []

    for dice in all_dice_rolls():
        best_move = find_best_move(position, dice)
        new_pos = apply_move(position, best_move)
        new_equity = evaluate(new_pos)
        one_roll_outcomes.append((dice_probability(dice), new_equity))

    # Calculate probability of losing market
    too_good_threshold = 0.75  # Opponent will pass
    prob_too_good = sum(p for p, eq in one_roll_outcomes if eq > too_good_threshold)

    # Adjust double point based on market losing risk
    if prob_too_good > 0.2:
        return (0.65, 0.75)  # Must double earlier
    else:
        return (0.70, 0.82)  # Can wait
```

### 6.4 Gammon Rates and Backgammon Rates

#### Definitions

```
Gammon Rate = P(win gammon | win) + P(lose gammon | lose)
Backgammon Rate = P(win backgammon | win) + P(lose backgammon | lose)

Adjusted gammon value:
Gammon Value = GV = (Gammons won - Gammons lost) / Games played
```

#### Impact on Cube Decisions

```python
def cubeful_equity(position):
    """
    Calculate equity considering gammons and cube
    """
    prob_win = evaluate_win_probability(position)
    prob_win_gammon = evaluate_win_gammon_probability(position)
    prob_win_backgammon = evaluate_win_backgammon_probability(position)

    prob_lose_gammon = evaluate_lose_gammon_probability(position)
    prob_lose_backgammon = evaluate_lose_backgammon_probability(position)

    # Cubeless equity
    equity = (
        prob_win +
        prob_win_gammon +    # Extra point for gammon
        2 * prob_win_backgammon -  # Extra 2 points for backgammon
        (1 - prob_win) -
        prob_lose_gammon -
        2 * prob_lose_backgammon
    )

    return equity
```

#### Gammon Rate Estimation

```python
def estimate_gammon_rate(position, player):
    """
    Heuristic gammon rate estimation
    """
    opponent_checkers_home = count_checkers_in_home(opponent(player))
    opponent_pips = pip_count(position, opponent(player))
    player_pips = pip_count(position, player)

    # Factors increasing gammon probability:
    # 1. Opponent has checkers far from home
    # 2. Player is significantly ahead in race
    # 3. Opponent has checkers on bar or closed out

    race_lead = opponent_pips - player_pips

    if race_lead > 100:
        base_gammon = 0.4
    elif race_lead > 70:
        base_gammon = 0.3
    elif race_lead > 40:
        base_gammon = 0.2
    else:
        base_gammon = 0.1

    # Adjust for back checkers
    opponent_back = count_checkers_back(opponent(player))
    if opponent_back >= 4:
        base_gammon *= 1.5

    # Adjust for closed board
    closed_points = count_consecutive_points(position, player, home_board=True)
    if closed_points >= 4:
        base_gammon *= 1.3

    return min(base_gammon, 0.6)  # Cap at 60%
```

### 6.5 Match Equity Tables (MET)

#### Match Play vs. Money Play

In match play (first to N points), cube decisions differ because:
1. Point values are non-linear
2. Being at match point changes dynamics
3. Crawford rule affects strategy

#### Match Equity Table Concept

A MET gives P(win match) for each score:

```
Match Equity Table (7-point match, simplified):

     Away: 1    2    3    4    5    6    7
You:
  1      50   70   75   81   84   87   89
  2      30   50   60   68   74   79   83
  3      25   40   50   59   66   72   77
  4      19   32   41   50   58   65   71
  5      16   26   34   42   50   57   64
  6      13   21   28   35   43   50   57
  7      11   17   23   29   36   43   50
```

(Values are approximate percentages)

#### Using MET for Cube Decisions

```python
def match_take_point(my_score, opp_score, match_length, cube_value):
    """
    Calculate take point in match play using MET
    """
    # Current match equity
    me_current = MET[match_length - my_score][match_length - opp_score]

    # If I pass
    new_opp_score = opp_score + cube_value
    if new_opp_score >= match_length:
        me_pass = 0  # I lose match
    else:
        me_pass = MET[match_length - my_score][match_length - new_opp_score]

    # If I take and win
    new_my_score_win = my_score + 2 * cube_value
    if new_my_score_win >= match_length:
        me_take_win = 100  # I win match
    else:
        me_take_win = MET[match_length - new_my_score_win][match_length - opp_score]

    # If I take and lose
    new_opp_score_lose = opp_score + 2 * cube_value
    if new_opp_score_lose >= match_length:
        me_take_lose = 0  # I lose match
    else:
        me_take_lose = MET[match_length - my_score][match_length - new_opp_score_lose]

    # Calculate take point
    # me_pass = P(win) * me_take_win + (1-P) * me_take_lose
    # Solve for P

    equity_diff_take = me_take_win - me_take_lose
    needed_equity = me_pass - me_take_lose

    take_point = needed_equity / equity_diff_take

    return take_point
```

### 6.6 Cube Decisions as Separate AI Problem

#### Two-Stage Decision Making

Modern backgammon AI treats cube and checker play separately:

```python
class BackgammonAI:
    def __init__(self):
        self.position_net = PositionEvaluationNetwork()
        self.cube_net = CubeDecisionNetwork()

    def decide_cube_action(self, position, cube_state, match_score):
        """
        Determine: should we double? Should we take?
        """
        # Get cubeless equity
        equity = self.position_net.evaluate(position)
        gammon_probs = self.position_net.get_gammon_probs(position)

        # Features for cube decision
        cube_features = {
            'cubeless_equity': equity,
            'win_gammon_prob': gammon_probs['wg'],
            'lose_gammon_prob': gammon_probs['lg'],
            'race_status': is_race(position),
            'cube_value': cube_state.value,
            'cube_owner': cube_state.owner,
            'match_score': match_score,
            'volatility': estimate_volatility(position),
        }

        # Cube neural network decision
        action = self.cube_net.decide(cube_features)
        # Returns: 'no_double', 'double', 'take', 'pass', 'beaver'

        return action

    def make_move(self, position, dice):
        """
        Choose best checker move given dice roll
        """
        moves = generate_legal_moves(position, dice)

        best_move = None
        best_equity = -infinity

        for move in moves:
            new_pos = apply_move(position, move)

            # Consider cube-adjusted equity
            cubeful_eq = self.cubeful_evaluate(new_pos)

            if cubeful_eq > best_equity:
                best_equity = cubeful_eq
                best_move = move

        return best_move
```

#### Specialized Cube Network Architecture

```python
class CubeDecisionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: equity, gammon probs, position features, match context
        self.input_size = 50

        # Separate pathways for different contexts
        self.money_path = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.match_path = nn.Sequential(
            nn.Linear(self.input_size + 10, 64),  # +10 for match score
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Output heads
        self.double_head = nn.Linear(32, 1)  # P(should double)
        self.take_head = nn.Linear(32, 1)    # P(should take)

    def forward(self, features, is_match=False, match_info=None):
        if is_match and match_info is not None:
            combined = torch.cat([features, match_info], dim=1)
            hidden = self.match_path(combined)
        else:
            hidden = self.money_path(features)

        double_prob = torch.sigmoid(self.double_head(hidden))
        take_prob = torch.sigmoid(self.take_head(hidden))

        return double_prob, take_prob
```

#### Cube Error Classification

```
Cube decisions are classified as:

1. Double/No Double Decision
   - Missed double: Should have doubled, didn't
   - Wrong double: Doubled when shouldn't have

2. Take/Pass Decision
   - Wrong take: Took when should have passed
   - Wrong pass: Passed when should have taken

3. Error Magnitude (in equity points)
   - Blunder: > 0.1 equity (huge error)
   - Error: 0.05 - 0.1 equity
   - Inaccuracy: 0.02 - 0.05 equity
   - Minor: < 0.02 equity
```

---

## Summary: Key Takeaways

### For Implementation

1. **Position Evaluation**: Neural networks with ~200-500 features work well
2. **Training Method**: TD(λ) self-play remains effective; can incorporate modern techniques
3. **Search**: 2-ply lookahead with neural evaluation is strong; deeper search helps
4. **Cube Handling**: Requires separate analysis; match play adds complexity
5. **Move Generation**: Prune equivalent moves; focus computation on distinct outcomes

### Game-Theoretic Insights

1. **Stochastic Nature**: Dice add exploration but require expected value thinking
2. **Skill Expression**: Accumulates over many decisions despite single-game variance
3. **Optimal Play**: Well-defined but computationally intensive to achieve
4. **Cube Mastery**: Often separates strong players; requires distinct training

### Historical Significance

TD-Gammon demonstrated that:
1. Neural networks can learn complex games through self-play
2. Temporal difference learning discovers expert-level strategies
3. Machine learning can challenge and refine human domain knowledge
4. These techniques presaged AlphaGo by two decades

---

## References and Further Reading

### Original Papers
- Tesauro, G. (1995). "Temporal Difference Learning and TD-Gammon"
- Sutton, R. (1988). "Learning to Predict by the Methods of Temporal Differences"
- Silver, D. et al. (2017). "Mastering the Game of Go without Human Knowledge"

### Backgammon Theory
- Robertie, B. "Modern Backgammon" (positional concepts)
- Woolsey, K. "How to Play Tournament Backgammon" (cube theory)
- Trice, W. "Backgammon Boot Camp" (technical foundations)

### Software References
- GNU Backgammon (open source, research grade)
- eXtreme Gammon (commercial, highest strength)
- Backgammon NJ (neural network based)

---

*Document Version: 1.0*
*Last Updated: December 2024*
*For: Backgammon AI Development Project*
