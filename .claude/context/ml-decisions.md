# ML/AI Design Decisions

This document records key machine learning decisions for the game AI platform.

---

## Why TD-Gammon Approach for Backgammon?

TD-Gammon (Tesauro, 1992) is the gold standard for backgammon AI:

1. **Proven effectiveness** - Achieved expert-level play with just 40-80 hidden units
2. **Self-play works** - No need for expert training data
3. **Well-documented** - Extensively studied, clear implementation path
4. **Educational value** - Simple architecture, great for learning TD learning

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden units | 40-80 | Original TD-Gammon used this range |
| λ (trace decay) | 0.7 | Empirically optimal for backgammon |
| α (learning rate) | 0.1 → decay | Start high, anneal over training |
| Activation | Sigmoid | Original choice, works well |

---

## Why 198-Feature Encoding for Backgammon?

The encoding captures board state compactly:

```
24 points × 2 players × 4 features = 192
+ bar (2) + borne_off (2) + turn (2) = 198
```

### Per-Point Features (4 per player)

| Feature | Value | Purpose |
|---------|-------|---------|
| f1 | 1 if ≥1 checker | Point occupied |
| f2 | 1 if ≥2 checkers | Made point |
| f3 | 1 if ≥3 checkers | Stacked |
| f4 | (n-3)/2 if n>3 | Extra checkers (normalized) |

This encoding is:
- **Sparse** - Many zeros, efficient
- **Symmetric** - Same features for both players
- **Normalized** - Values in [0, 1] range

---

## Why Graph Neural Networks for Catan?

Catan has variable topology (especially with custom boards):

| Approach | Pros | Cons |
|----------|------|------|
| Fixed embedding | Simple | Can't handle variable boards |
| Padding to max | Simple | Wasteful, loses structure |
| **GNN** | Handles any graph | More complex |
| Transformer | Flexible | Overkill for this |

### GNN Benefits

1. **Natural for graphs** - Hexes, vertices, edges form a graph
2. **Variable size** - Same network works for any board size
3. **Spatial structure** - Message passing captures local relationships
4. **Modern approach** - Great learning opportunity

### Architecture

```
Node features → Embedding → GNN layers → Global pooling → Value head
```

---

## Why Both NEAT and Weight Perturbation?

Each approach has different strengths:

### Weight Perturbation

- **Simple** - Just add Gaussian noise
- **Fast** - No topology changes
- **Good for refinement** - Fine-tune existing architecture

```python
for param in network.parameters():
    param.data += torch.randn_like(param) * sigma
```

### NEAT-Style Topology Evolution

- **Discovers architecture** - Finds optimal structure
- **Innovation** - Can add complexity over time
- **Historical markings** - Track innovation numbers

### When to Use Each

| Situation | Approach |
|-----------|----------|
| Fixed architecture | Weight perturbation |
| Architecture search | NEAT topology |
| Hybrid evolution | Both (configurable) |

---

## Why Per-Game ELO (not Global)?

Different games require different skills:

1. **Fair comparison** - Backgammon skill ≠ Catan skill
2. **Matchmaking** - Better matches within game type
3. **Progress tracking** - See improvement per game

### Implementation

`PlayerRating` model links (player, game_type) → rating:
```python
PlayerRating.objects.get(user=user, game_type='backgammon')
PlayerRating.objects.get(ai_model=model, game_type='catan')
```

---

## Why JSON for Network Architecture?

Alternatives considered:

| Format | Pros | Cons |
|--------|------|------|
| **JSON** | Human-readable, DB-friendly | Verbose |
| Pickle | Native Python | Not portable |
| ONNX | Standard ML format | Overkill |
| Protobuf | Compact | Extra dependency |

### JSON Advantages

1. **Inspectable** - See architecture without code
2. **DB storage** - JSONField in Django
3. **Editable** - Modify architecture by hand
4. **Evolvable** - Easy to mutate structure

---

## Why Binary for Weights?

Network weights can be large (millions of floats):

```python
# Stored as:
gzip(pickle({layer_id: numpy_array}))
```

### Compression Results

| Method | Size | Notes |
|--------|------|-------|
| Raw JSON | 100% | Very verbose |
| Pickle | ~40% | Native Python |
| **gzip(pickle)** | ~15% | Good compression |

---

## Training Considerations

### Self-Play vs External Opponents

| Method | Pros | Cons |
|--------|------|------|
| **Self-play** | No external data needed | Can develop blind spots |
| Expert games | Learn from best | Need expert data |
| Mixed | Best of both | More complex |

**Decision**: Start with self-play (TD-Gammon approach), add evaluation vs random as baseline.

### Evaluation Metrics

1. **Win rate vs random** - Basic sanity check (should be >90%)
2. **Win rate vs previous version** - Track improvement
3. **ELO rating** - Comparable across models
4. **Position evaluation error** - If we have reference evaluations

---

## Future Considerations

### Potential Enhancements

1. **Policy gradient methods** - Actor-critic for action selection
2. **MCTS integration** - Monte Carlo Tree Search for planning
3. **Transfer learning** - Pre-train on one game, fine-tune on another
4. **Ensemble methods** - Combine multiple networks

### Not Planned

- **AlphaZero approach** - Too resource-intensive for learning project
- **GPU training** - Focus on clarity, not speed
- **Distributed training** - Complexity not worth it here
