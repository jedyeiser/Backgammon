# AI Architecture

This document describes the ML experimentation platform architecture for the game server.

## Design Principles

1. **Clarity over performance** - Code should be educational and readable
2. **Extensibility** - Support any game implementing `BaseRuleSet`
3. **Observability** - Visualize training, evolution, and network structure
4. **Persistence** - Save/load networks, track lineage, resume training

---

## Player Abstraction

All players (human and AI) share a common interface defined in `backend/apps/ai/players/base.py`.

### BasePlayer Interface

```python
class BasePlayer(ABC):
    player_id: str
    game_type: str
    name: str

    @abstractmethod
    def select_action(self, game_state: Dict, legal_actions: List[Dict]) -> Dict:
        """Choose one action from legal_actions based on game_state."""
        pass

    @abstractmethod
    def get_player_type(self) -> str:
        """Return player type identifier."""
        pass
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `select_action()` | Core decision method - returns chosen action |
| `get_player_type()` | Returns type string for registry/serialization |
| `on_game_start()` | Called at game start (optional) |
| `on_game_end()` | Called at game end for learning (optional) |
| `on_opponent_action()` | Observe opponent moves (optional) |

### Player Registry

Players are registered and created via `PlayerRegistry`:

```python
from apps.ai.players import PlayerRegistry, get_player

# Register custom player
PlayerRegistry.register('my_ai', MyAIPlayer)

# Create player instance
player = get_player('random', player_id='p1', game_type='backgammon')
```

---

## Network Serialization Format

Neural networks are stored in two parts:
1. **Architecture** (JSON) - Layer definitions
2. **Weights** (Binary) - Compressed tensor data

### Architecture JSON Schema

```json
{
    "input_size": 198,
    "output_size": 1,
    "layers": [
        {"id": "layer_0", "type": "linear", "in": 198, "out": 80},
        {"id": "act_0", "type": "activation", "fn": "sigmoid"},
        {"id": "layer_1", "type": "linear", "in": 80, "out": 40},
        {"id": "act_1", "type": "activation", "fn": "sigmoid"},
        {"id": "layer_2", "type": "linear", "in": 40, "out": 1},
        {"id": "output", "type": "activation", "fn": "sigmoid"}
    ]
}
```

### Supported Layer Types

| Type | Parameters | Description |
|------|------------|-------------|
| `linear` | `in`, `out` | Fully connected layer |
| `activation` | `fn` | Activation function (sigmoid, relu, tanh) |
| `batchnorm` | `features` | Batch normalization |
| `dropout` | `p` | Dropout layer |

### Weight Serialization

Weights stored in `AIModel.network_weights` as:
```
gzip(pickle({layer_id: numpy_array, ...}))
```

---

## Per-Game ELO System

Each player has separate ratings per game type via `PlayerRating` model.

### Rating Structure

```python
class PlayerRating:
    # Polymorphic - one of these is set
    user = FK(User, null=True)
    ai_model = FK(AIModel, null=True)

    game_type = FK(GameType)

    elo = IntegerField(default=1000)
    games_played = PositiveIntegerField()
    games_won = PositiveIntegerField()
    peak_elo = IntegerField()
```

### ELO Calculation

Standard ELO formula with K=32:
```
expected = 1 / (1 + 10^((rating_b - rating_a) / 400))
new_rating = rating + K * (actual - expected)
```

---

## Evolution Tracking

Evolution is tracked through three models:

### EvolutionSession

Tracks an entire evolution experiment:
- Population size and parameters
- Current generation progress
- Best model found

### EvolutionLineage

Tracks parent-child relationships:
- Parent(s) of each model
- Mutation type applied
- Generation number

### Mutation Types

| Type | Description |
|------|-------------|
| `weight` | Gaussian noise added to weights |
| `add_node` | Insert node into connection (NEAT) |
| `add_conn` | Add new connection between nodes |
| `rm_conn` | Remove a connection |
| `crossover` | Combine two parent networks |
| `clone` | Exact copy (elitism) |

---

## Database Models

Located in `backend/apps/ai/models.py`:

| Model | Purpose |
|-------|---------|
| `AIModel` | Trained model with architecture & weights |
| `TrainingSession` | Training run tracking |
| `PlayerRating` | Per-game ELO for humans and AI |
| `EvolutionSession` | Evolution experiment tracking |
| `EvolutionLineage` | Parent-child relationships |

---

## Directory Structure

```
backend/apps/ai/
├── models.py           # Database models
├── players/
│   ├── base.py         # BasePlayer ABC
│   ├── random_player.py
│   └── registry.py     # Player factory
├── encoders/           # Board state encoders (Phase 2)
├── networks/           # Neural network builders (Phase 3)
├── training/           # TD learning, self-play (Phase 4)
├── evolution/          # NEAT, mutations (Phase 5)
├── matches/            # Match runner, ELO calc (Phase 8)
└── visualization/      # Training plots (Phase 7)
```
