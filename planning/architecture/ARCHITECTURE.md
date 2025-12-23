# Backgammon Project Architecture

> This document synthesizes research findings into concrete architecture decisions.
> Updated as research completes and decisions are made.

## Status: Draft (Awaiting Research Completion)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Vite + React + TypeScript                     │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │    │
│  │  │   Game    │  │   Lobby   │  │   Auth    │  │    AI     │     │    │
│  │  │   Board   │  │   Views   │  │   Pages   │  │ Training  │     │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │    │
│  │                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────┐│    │
│  │  │              Zustand Stores + React Query                    ││    │
│  │  │   gameStore | userStore | uiStore | Server State Cache      ││    │
│  │  └─────────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   REST API + WebSocket │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────────┐
│                              BACKEND                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                  Django + Django REST Framework                   │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │    │
│  │  │ Accounts  │  │   Game    │  │    AI     │  │  Channels │     │    │
│  │  │   App     │  │   App     │  │   App     │  │ Consumers │     │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │    │
│  │                                                                   │    │
│  │  ┌───────────────────┐  ┌───────────────────┐                    │    │
│  │  │   Game Engine     │  │   AI Engine       │                    │    │
│  │  │ (Pure Python)     │  │ (PyTorch)         │                    │    │
│  │  └───────────────────┘  └───────────────────┘                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Data Layer                                   │    │
│  │  ┌─────────────────┐      ┌─────────────────┐                    │    │
│  │  │   PostgreSQL    │      │     Redis       │                    │    │
│  │  │   (Primary DB)  │      │  (Cache/WS)     │                    │    │
│  │  └─────────────────┘      └─────────────────┘                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Backend Architecture

### 2.1 Django App Structure

```
backend/
├── config/                    # Django project config
│   ├── settings/
│   │   ├── base.py           # Shared settings
│   │   ├── dev.py            # Development
│   │   └── prod.py           # Production
│   ├── urls.py
│   ├── asgi.py               # ASGI for Channels
│   └── wsgi.py
├── apps/
│   ├── accounts/             # User management
│   │   ├── models.py         # User model extensions
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── urls.py
│   ├── game/                 # Core game logic
│   │   ├── models.py         # Game, Move models
│   │   ├── serializers.py
│   │   ├── views.py          # REST endpoints
│   │   ├── consumers.py      # WebSocket consumers
│   │   ├── routing.py        # WS routing
│   │   └── engine/           # Pure game logic
│   │       ├── __init__.py
│   │       ├── board.py      # Board representation
│   │       ├── moves.py      # Move generation
│   │       ├── rules.py      # Game rules
│   │       └── notation.py   # Position notation
│   └── ai/                   # AI players
│       ├── models.py         # AIPlayer, TrainingRun
│       ├── players/
│       │   ├── random.py
│       │   ├── heuristic.py
│       │   ├── td_learner.py
│       │   └── neural.py
│       ├── training/
│       │   ├── self_play.py
│       │   └── evaluation.py
│       └── management/
│           └── commands/
│               └── train_ai.py
├── requirements/
└── manage.py
```

### 2.2 Key Models

```python
# apps/game/models.py

class Game(models.Model):
    """A single backgammon game."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    player1 = models.ForeignKey(User, related_name='games_as_p1')
    player2 = models.ForeignKey(User, related_name='games_as_p2', null=True)

    status = models.CharField(choices=STATUS_CHOICES, default='waiting')
    current_player = models.IntegerField(default=1)  # 1 or 2

    # Board state as JSON (denormalized for easy access)
    board_state = models.JSONField(default=initial_board_state)

    # Dice state
    dice = models.JSONField(null=True)  # [3, 5] or null
    remaining_dice = models.JSONField(null=True)  # Dice not yet used

    # Doubling cube
    cube_value = models.IntegerField(default=1)
    cube_owner = models.IntegerField(null=True)  # null = centered

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True)
    finished_at = models.DateTimeField(null=True)

    # Optimistic locking
    version = models.PositiveIntegerField(default=0)

    class Meta:
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['player1', 'status']),
            models.Index(fields=['player2', 'status']),
        ]


class Move(models.Model):
    """A single checker movement within a turn."""
    game = models.ForeignKey(Game, related_name='moves')
    player = models.IntegerField()  # 1 or 2
    move_number = models.PositiveIntegerField()

    from_point = models.IntegerField()  # 0-23, -1=bar
    to_point = models.IntegerField()    # 0-23, 24=bear-off
    die_value = models.IntegerField()
    hits = models.BooleanField(default=False)

    # State snapshot (for replay/undo)
    board_before = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['game', 'move_number']
        unique_together = ['game', 'move_number']
```

### 2.3 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register/` | Create account |
| POST | `/api/auth/login/` | Get JWT tokens |
| POST | `/api/auth/refresh/` | Refresh token |
| GET | `/api/games/` | List user's games |
| POST | `/api/games/` | Create new game |
| GET | `/api/games/{id}/` | Get game state |
| POST | `/api/games/{id}/join/` | Join a game |
| POST | `/api/games/{id}/roll/` | Roll dice |
| POST | `/api/games/{id}/move/` | Make a move |
| POST | `/api/games/{id}/end-turn/` | End turn (pass if can't move) |
| POST | `/api/games/{id}/resign/` | Resign game |
| WS | `/ws/game/{id}/` | Real-time game updates |

### 2.4 WebSocket Protocol

```javascript
// Client -> Server messages
{ type: "roll_dice" }
{ type: "make_move", from: 23, to: 20 }
{ type: "end_turn" }
{ type: "resign" }

// Server -> Client messages
{ type: "game_state", data: { ...fullGameState } }
{ type: "dice_rolled", dice: [3, 5], player: 1 }
{ type: "move_made", move: { from: 23, to: 20, die: 3, hits: false } }
{ type: "turn_ended", nextPlayer: 2 }
{ type: "game_over", winner: 1, reason: "bearoff" }
{ type: "error", message: "Invalid move" }
```

---

## 3. Frontend Architecture

### 3.1 Directory Structure

```
frontend/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── routes.tsx
│   ├── components/
│   │   ├── game/
│   │   │   ├── Board/
│   │   │   │   ├── Board.tsx
│   │   │   │   ├── Point.tsx
│   │   │   │   ├── Checker.tsx
│   │   │   │   ├── Bar.tsx
│   │   │   │   └── BearOff.tsx
│   │   │   ├── Dice/
│   │   │   ├── Controls/
│   │   │   └── GameStatus/
│   │   ├── lobby/
│   │   ├── layout/
│   │   └── ui/
│   ├── stores/
│   │   ├── gameStore.ts
│   │   ├── userStore.ts
│   │   └── uiStore.ts
│   ├── hooks/
│   │   ├── useGame.ts
│   │   ├── useLegalMoves.ts
│   │   ├── useGameSocket.ts
│   │   └── useAuth.ts
│   ├── api/
│   │   ├── client.ts
│   │   ├── auth.ts
│   │   └── games.ts
│   ├── types/
│   │   ├── game.ts
│   │   └── api.ts
│   └── utils/
│       ├── board.ts
│       └── notation.ts
├── public/
├── index.html
└── vite.config.ts
```

### 3.2 Zustand Store Design

```typescript
// stores/gameStore.ts

interface GameState {
  // Current game data
  gameId: string | null;
  board: Board;
  currentPlayer: 1 | 2;
  dice: [number, number] | null;
  remainingDice: number[];
  phase: GamePhase;

  // UI state
  selectedPoint: number | null;
  legalMoves: Move[];
  pendingMoves: Move[];  // Moves not yet confirmed

  // Actions
  setGameState: (state: ServerGameState) => void;
  selectPoint: (point: number | null) => void;
  makeMove: (move: Move) => void;
  confirmTurn: () => void;
  undoMove: () => void;
}
```

### 3.3 Component Hierarchy

```
App
├── AuthProvider
│   └── Router
│       ├── Layout
│       │   ├── Header
│       │   └── Routes
│       │       ├── HomePage
│       │       ├── LobbyPage
│       │       │   ├── GameList
│       │       │   └── CreateGame
│       │       └── GamePage
│       │           ├── GameBoard
│       │           │   ├── Board
│       │           │   │   ├── Point (x24)
│       │           │   │   │   └── Checker (x n)
│       │           │   │   ├── Bar
│       │           │   │   └── BearOff
│       │           │   └── DiceDisplay
│       │           ├── GameControls
│       │           │   ├── RollButton
│       │           │   ├── UndoButton
│       │           │   └── EndTurnButton
│       │           └── GameInfo
│       │               ├── PipCount
│       │               └── MoveHistory
```

---

## 4. Game Engine Architecture

### 4.1 Board Representation

**Decision**: Array-based representation for simplicity and learning clarity.

```python
@dataclass(frozen=True)
class BoardState:
    """Immutable board state."""
    # Points 0-23, positive = player 1, negative = player 2
    points: Tuple[int, ...]  # 24 elements

    # Bar (checkers hit and waiting to re-enter)
    bar: Tuple[int, int]  # (player1_bar, player2_bar)

    # Borne off (checkers removed from game)
    borne_off: Tuple[int, int]  # (player1_off, player2_off)

    def __post_init__(self):
        assert len(self.points) == 24
        assert sum(abs(p) for p in self.points) + sum(self.bar) + sum(self.borne_off) == 30
```

### 4.2 Move Generation

```python
def get_legal_moves(board: BoardState, dice: Tuple[int, int], player: int) -> List[List[Move]]:
    """
    Generate all legal move sequences for the given dice roll.

    Returns a list of complete move sequences (each sequence uses all possible dice).
    """
    if is_on_bar(board, player):
        return _generate_bar_moves(board, dice, player)

    # Generate all ways to use the dice
    sequences = []
    _generate_move_sequences(board, list(dice), player, [], sequences)

    # Filter to only maximal sequences (use most dice possible)
    max_len = max(len(seq) for seq in sequences) if sequences else 0
    return [seq for seq in sequences if len(seq) == max_len]
```

---

## 5. AI Architecture

### 5.1 Player Interface

```python
from abc import ABC, abstractmethod

class AIPlayer(ABC):
    """Base class for AI players."""

    @abstractmethod
    def select_move(
        self,
        board: BoardState,
        dice: Tuple[int, int],
        player: int
    ) -> List[Move]:
        """Select the best move sequence for the given position."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this AI."""
        pass
```

### 5.2 AI Progression

| Level | Name | Description | Purpose |
|-------|------|-------------|---------|
| 1 | RandomPlayer | Picks random legal move | Baseline |
| 2 | HeuristicPlayer | Hand-crafted evaluation | Learn evaluation concepts |
| 3 | ExpectimaxPlayer | Tree search with eval | Learn game tree search |
| 4 | TDPlayer | TD(λ) learned evaluation | Learn TD-Learning |
| 5 | NeuralPlayer | Neural network evaluation | Learn deep learning |
| 6 | MCTSPlayer | Monte Carlo Tree Search | Learn MCTS |

### 5.3 TD-Learning Setup

```python
class TDNetwork(nn.Module):
    """Neural network for position evaluation."""

    def __init__(self, input_size: int = 198, hidden_size: int = 80):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
```

---

## 6. Data Flow

### 6.1 Making a Move

```
User clicks checker on point 23
            │
            ▼
    React: selectPoint(23)
            │
            ▼
    Zustand: Calculate legal destinations
            │
            ▼
    UI: Highlight points 20, 22 (for dice 3,1)
            │
            ▼
    User clicks point 20
            │
            ▼
    React: makeMove({from: 23, to: 20, die: 3})
            │
            ▼
    Zustand: Add to pendingMoves, update local board
            │
            ▼
    (Optimistic UI update)
            │
            ▼
    WebSocket: send({type: "make_move", from: 23, to: 20})
            │
            ▼
    Django: Validate move, update database
            │
            ▼
    WebSocket: broadcast({type: "move_made", ...})
            │
            ▼
    Both clients: Reconcile state
```

---

## 7. Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State management | Zustand + React Query | Zustand for UI state, RQ for server state |
| Board representation | Array-based | Simpler to understand for learning |
| WebSocket library | Django Channels | Native Django integration |
| Animation | Framer Motion | Best DX for React animations |
| Drag & drop | @dnd-kit | Modern, accessible, touch support |
| Testing | pytest + Vitest | Best in class for each stack |
| AI framework | PyTorch | Most popular for research/learning |

---

## 8. Open Questions

> To be resolved through research and experimentation

1. **Board orientation**: How to handle player 1 vs player 2 view?
2. **Move validation**: Server-side only or client-side preview?
3. **Game history**: Event sourcing vs state snapshots?
4. **AI training**: Separate process or Django management command?
5. **Spectator mode**: Dedicated system or reuse player WebSocket?

---

## 9. Research Documents

- [Django Backend Research](../research/django-backend-research.md) - *In Progress*
- [Data Structures Research](../research/data-structures-research.md) - *In Progress*
- [Game Theory Research](../research/game-theory-research.md) - *In Progress*
- [React Frontend Research](../research/react-frontend-research.md) - *In Progress*
- [Zustand State Research](../research/zustand-state-research.md) - *In Progress*
- [ML/AI Research](../research/ml-ai-research.md) - *In Progress*
