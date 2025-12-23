# Zustand State Management Research for Backgammon

**Date**: December 2024
**Purpose**: Comprehensive research on Zustand for implementing state management in the Backgammon AI Sandbox

---

## Table of Contents

1. [Zustand Fundamentals and Best Practices](#1-zustand-fundamentals-and-best-practices)
2. [Selector Patterns](#2-selector-patterns)
3. [Zustand vs Alternatives](#3-zustand-vs-alternatives)
4. [Game State Modeling](#4-game-state-modeling)
5. [Server Synchronization](#5-server-synchronization)
6. [Persistence and Hydration](#6-persistence-and-hydration)
7. [Testing Zustand Stores](#7-testing-zustand-stores)
8. [Recommendations for Backgammon Project](#8-recommendations-for-backgammon-project)

---

## 1. Zustand Fundamentals and Best Practices

### 1.1 Core Concepts

Zustand is a small, fast, and scalable state management solution. Unlike Redux, it doesn't require providers, actions, or reducers. The store is a hook that components can subscribe to directly.

**Key Characteristics:**
- ~1KB bundle size (minified + gzipped)
- No boilerplate required
- React concurrent mode compatible
- Works outside React components
- Built-in devtools support

### 1.2 Store Creation Patterns

#### Basic Store Creation

```typescript
// stores/gameStore.ts
import { create } from 'zustand';

interface GameState {
  board: number[][];
  currentPlayer: 'white' | 'black';
  dice: [number, number] | null;

  // Actions
  rollDice: () => void;
  makeMove: (from: number, to: number) => void;
  resetGame: () => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  board: createInitialBoard(),
  currentPlayer: 'white',
  dice: null,

  rollDice: () => {
    const die1 = Math.floor(Math.random() * 6) + 1;
    const die2 = Math.floor(Math.random() * 6) + 1;
    set({ dice: [die1, die2] });
  },

  makeMove: (from, to) => {
    const { board, currentPlayer } = get();
    const newBoard = applyMove(board, from, to, currentPlayer);
    set({ board: newBoard });
  },

  resetGame: () => set({
    board: createInitialBoard(),
    currentPlayer: 'white',
    dice: null,
  }),
}));
```

#### Store with Explicit Types (Recommended for Complex Stores)

```typescript
// types/game.ts
export interface Point {
  count: number;
  player: 'white' | 'black' | null;
}

export interface BoardState {
  points: Point[];  // 24 points
  bar: { white: number; black: number };
  bearOff: { white: number; black: number };
}

export interface GameState {
  board: BoardState;
  currentPlayer: 'white' | 'black';
  dice: [number, number] | null;
  usedDice: number[];
  gameStatus: 'waiting' | 'playing' | 'finished';
  winner: 'white' | 'black' | null;
  doublingCube: number;
  cubeOwner: 'white' | 'black' | 'center';
}

export interface GameActions {
  rollDice: () => void;
  makeMove: (from: number, to: number) => boolean;
  endTurn: () => void;
  offerDouble: () => void;
  acceptDouble: () => void;
  declineDouble: () => void;
  resetGame: () => void;
}

export type GameStore = GameState & GameActions;
```

```typescript
// stores/gameStore.ts
import { create } from 'zustand';
import type { GameStore } from '../types/game';

export const useGameStore = create<GameStore>()((set, get) => ({
  // State
  board: createInitialBoard(),
  currentPlayer: 'white',
  dice: null,
  usedDice: [],
  gameStatus: 'waiting',
  winner: null,
  doublingCube: 1,
  cubeOwner: 'center',

  // Actions
  rollDice: () => {
    const die1 = Math.floor(Math.random() * 6) + 1;
    const die2 = Math.floor(Math.random() * 6) + 1;
    set({
      dice: [die1, die2],
      usedDice: [],
      gameStatus: 'playing',
    });
  },

  makeMove: (from, to) => {
    const state = get();
    const validMove = validateMove(state, from, to);
    if (!validMove) return false;

    const newBoard = applyMove(state.board, from, to, state.currentPlayer);
    const dieUsed = Math.abs(to - from);

    set({
      board: newBoard,
      usedDice: [...state.usedDice, dieUsed],
    });

    return true;
  },

  endTurn: () => {
    const { currentPlayer, board } = get();
    const winner = checkWinner(board);

    if (winner) {
      set({ winner, gameStatus: 'finished' });
    } else {
      set({
        currentPlayer: currentPlayer === 'white' ? 'black' : 'white',
        dice: null,
        usedDice: [],
      });
    }
  },

  offerDouble: () => {
    // Implementation for doubling cube
  },

  acceptDouble: () => {
    const { doublingCube, currentPlayer } = get();
    set({
      doublingCube: doublingCube * 2,
      cubeOwner: currentPlayer,
    });
  },

  declineDouble: () => {
    const { currentPlayer } = get();
    set({
      winner: currentPlayer,
      gameStatus: 'finished',
    });
  },

  resetGame: () => set({
    board: createInitialBoard(),
    currentPlayer: 'white',
    dice: null,
    usedDice: [],
    gameStatus: 'waiting',
    winner: null,
    doublingCube: 1,
    cubeOwner: 'center',
  }),
}));
```

### 1.3 Middleware

#### DevTools Middleware

```typescript
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

export const useGameStore = create<GameStore>()(
  devtools(
    (set, get) => ({
      // ... store implementation
    }),
    {
      name: 'backgammon-game',
      enabled: process.env.NODE_ENV === 'development',
    }
  )
);
```

**DevTools Features:**
- Time-travel debugging
- Action history
- State diff visualization
- State import/export

#### Persist Middleware

```typescript
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export const useGameStore = create<GameStore>()(
  persist(
    (set, get) => ({
      // ... store implementation
    }),
    {
      name: 'backgammon-game-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Only persist certain fields
        board: state.board,
        currentPlayer: state.currentPlayer,
        doublingCube: state.doublingCube,
        cubeOwner: state.cubeOwner,
      }),
    }
  )
);
```

#### Immer Middleware

Enables mutable-style updates that produce immutable state:

```typescript
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface GameStore {
  board: BoardState;
  moveChecker: (from: number, to: number) => void;
}

export const useGameStore = create<GameStore>()(
  immer((set) => ({
    board: createInitialBoard(),

    moveChecker: (from, to) => set((state) => {
      // Direct mutations are allowed with immer!
      const piece = state.board.points[from];
      state.board.points[from] = { count: 0, player: null };
      state.board.points[to] = piece;
    }),
  }))
);
```

#### Combining Multiple Middleware

```typescript
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

export const useGameStore = create<GameStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Store implementation with all middleware benefits
      })),
      { name: 'backgammon-storage' }
    ),
    { name: 'backgammon-game' }
  )
);
```

**Middleware Order Matters:**
- Outermost: devtools (for debugging the raw store)
- Middle: persist (serialization)
- Innermost: immer (mutations)

### 1.4 Slice Pattern for Large Stores

For the backgammon project, we should split state into logical slices:

```typescript
// stores/slices/boardSlice.ts
import { StateCreator } from 'zustand';

export interface BoardSlice {
  points: Point[];
  bar: { white: number; black: number };
  bearOff: { white: number; black: number };

  initializeBoard: () => void;
  moveChecker: (from: number, to: number) => void;
  hitChecker: (point: number) => void;
  bearOffChecker: (point: number) => void;
}

export const createBoardSlice: StateCreator<
  BoardSlice & DiceSlice & TurnSlice,  // Full store type
  [],
  [],
  BoardSlice
> = (set, get) => ({
  points: createInitialPoints(),
  bar: { white: 0, black: 0 },
  bearOff: { white: 0, black: 0 },

  initializeBoard: () => set({
    points: createInitialPoints(),
    bar: { white: 0, black: 0 },
    bearOff: { white: 0, black: 0 },
  }),

  moveChecker: (from, to) => {
    const { points, bar, currentPlayer } = get();
    // Implementation
  },

  hitChecker: (point) => {
    // Move opponent's checker to bar
  },

  bearOffChecker: (point) => {
    // Bear off logic
  },
});
```

```typescript
// stores/slices/diceSlice.ts
export interface DiceSlice {
  dice: [number, number] | null;
  usedDice: number[];
  isDoubles: boolean;

  rollDice: () => void;
  useDie: (value: number) => void;
  resetDice: () => void;
}

export const createDiceSlice: StateCreator<
  BoardSlice & DiceSlice & TurnSlice,
  [],
  [],
  DiceSlice
> = (set, get) => ({
  dice: null,
  usedDice: [],
  isDoubles: false,

  rollDice: () => {
    const die1 = Math.floor(Math.random() * 6) + 1;
    const die2 = Math.floor(Math.random() * 6) + 1;
    set({
      dice: [die1, die2],
      usedDice: [],
      isDoubles: die1 === die2,
    });
  },

  useDie: (value) => {
    set((state) => ({
      usedDice: [...state.usedDice, value],
    }));
  },

  resetDice: () => set({ dice: null, usedDice: [], isDoubles: false }),
});
```

```typescript
// stores/slices/turnSlice.ts
export interface TurnSlice {
  currentPlayer: 'white' | 'black';
  turnNumber: number;
  gamePhase: 'rolling' | 'moving' | 'waiting';

  switchPlayer: () => void;
  setGamePhase: (phase: 'rolling' | 'moving' | 'waiting') => void;
}

export const createTurnSlice: StateCreator<
  BoardSlice & DiceSlice & TurnSlice,
  [],
  [],
  TurnSlice
> = (set, get) => ({
  currentPlayer: 'white',
  turnNumber: 1,
  gamePhase: 'rolling',

  switchPlayer: () => {
    const { currentPlayer, turnNumber } = get();
    set({
      currentPlayer: currentPlayer === 'white' ? 'black' : 'white',
      turnNumber: turnNumber + 1,
      gamePhase: 'rolling',
    });
    get().resetDice();
  },

  setGamePhase: (phase) => set({ gamePhase: phase }),
});
```

```typescript
// stores/gameStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { createBoardSlice, BoardSlice } from './slices/boardSlice';
import { createDiceSlice, DiceSlice } from './slices/diceSlice';
import { createTurnSlice, TurnSlice } from './slices/turnSlice';

type GameStore = BoardSlice & DiceSlice & TurnSlice;

export const useGameStore = create<GameStore>()(
  devtools(
    persist(
      (...args) => ({
        ...createBoardSlice(...args),
        ...createDiceSlice(...args),
        ...createTurnSlice(...args),
      }),
      {
        name: 'backgammon-storage',
        partialize: (state) => ({
          points: state.points,
          bar: state.bar,
          bearOff: state.bearOff,
          currentPlayer: state.currentPlayer,
          turnNumber: state.turnNumber,
        }),
      }
    ),
    { name: 'backgammon-game' }
  )
);
```

### 1.5 TypeScript Integration Best Practices

#### Strict Type Inference

```typescript
// Use the double function call pattern for proper typing
export const useGameStore = create<GameStore>()(
  // middlewares here
  (set, get) => ({
    // implementation
  })
);

// NOT this (loses type inference):
export const useGameStore = create((set, get) => ({
  // ...
}));
```

#### Extracting State Types

```typescript
// Extract state type from store
type GameState = ReturnType<typeof useGameStore.getState>;

// For component props that depend on store state
interface BoardProps {
  points: GameState['points'];
  currentPlayer: GameState['currentPlayer'];
}
```

#### Type-Safe Actions

```typescript
type GameAction =
  | { type: 'ROLL_DICE' }
  | { type: 'MOVE'; from: number; to: number }
  | { type: 'END_TURN' }
  | { type: 'RESET' };

interface GameStore {
  state: GameState;
  dispatch: (action: GameAction) => void;
}

// Redux-like dispatch pattern if preferred
export const useGameStore = create<GameStore>()((set, get) => ({
  state: initialState,
  dispatch: (action) => {
    switch (action.type) {
      case 'ROLL_DICE':
        // ...
        break;
      case 'MOVE':
        // ...
        break;
      // etc.
    }
  },
}));
```

---

## 2. Selector Patterns

### 2.1 Fine-Grained Subscriptions

Zustand only re-renders components when the selected state changes. This is crucial for performance:

```typescript
// BAD: Component re-renders on ANY state change
function BadComponent() {
  const store = useGameStore();  // Subscribes to entire store
  return <div>{store.currentPlayer}</div>;
}

// GOOD: Component only re-renders when currentPlayer changes
function GoodComponent() {
  const currentPlayer = useGameStore((state) => state.currentPlayer);
  return <div>{currentPlayer}</div>;
}
```

### 2.2 Multiple Selectors

```typescript
function GameBoard() {
  // Multiple fine-grained subscriptions
  const points = useGameStore((state) => state.points);
  const currentPlayer = useGameStore((state) => state.currentPlayer);
  const dice = useGameStore((state) => state.dice);

  return (
    // ...
  );
}

// Or combine into one selector returning an object
// (requires shallow comparison - see below)
function GameBoard() {
  const { points, currentPlayer, dice } = useGameStore(
    (state) => ({
      points: state.points,
      currentPlayer: state.currentPlayer,
      dice: state.dice,
    }),
    shallow  // Important!
  );

  return (
    // ...
  );
}
```

### 2.3 Shallow Equality Comparisons

When selecting objects or arrays, use `shallow` to prevent unnecessary re-renders:

```typescript
import { shallow } from 'zustand/shallow';

function DiceDisplay() {
  // Without shallow: new object reference each render = always re-renders
  // With shallow: compares object properties = only re-renders when values change
  const { dice, usedDice } = useGameStore(
    (state) => ({
      dice: state.dice,
      usedDice: state.usedDice,
    }),
    shallow
  );

  return <DiceComponent dice={dice} used={usedDice} />;
}

// For array selections
function MovesHistory() {
  const history = useGameStore(
    (state) => state.moveHistory,
    shallow
  );

  return <HistoryList moves={history} />;
}
```

### 2.4 Derived State and Computed Values

#### In-Selector Computation

```typescript
// Computed in selector - recalculates each time it's called
function MoveOptions() {
  const validMoves = useGameStore((state) => {
    const { points, currentPlayer, dice, usedDice } = state;
    return calculateValidMoves(points, currentPlayer, dice, usedDice);
  });

  return <MoveList moves={validMoves} />;
}
```

#### Memoized Selectors

For expensive computations, use external memoization:

```typescript
// selectors/gameSelectors.ts
import { createSelector } from 'reselect';  // or use memoize-one

const selectPoints = (state: GameState) => state.points;
const selectCurrentPlayer = (state: GameState) => state.currentPlayer;
const selectDice = (state: GameState) => state.dice;
const selectUsedDice = (state: GameState) => state.usedDice;

// Memoized selector - only recalculates when inputs change
export const selectValidMoves = createSelector(
  [selectPoints, selectCurrentPlayer, selectDice, selectUsedDice],
  (points, currentPlayer, dice, usedDice) => {
    return calculateValidMoves(points, currentPlayer, dice, usedDice);
  }
);

// Usage in component
function MoveOptions() {
  const validMoves = useGameStore(selectValidMoves);
  return <MoveList moves={validMoves} />;
}
```

#### Store-Level Derived State

```typescript
// Compute in the store itself (runs on every state change)
export const useGameStore = create<GameStore>()((set, get) => ({
  points: [],
  currentPlayer: 'white',

  // Computed getter (not reactive, called on demand)
  getValidMoves: () => {
    const { points, currentPlayer, dice, usedDice } = get();
    return calculateValidMoves(points, currentPlayer, dice, usedDice);
  },

  // Alternative: subscribe pattern for reactive derived state
  _validMovesCache: [] as Move[],
}));

// External subscription to update derived state
useGameStore.subscribe(
  (state) => ({ points: state.points, dice: state.dice }),
  ({ points, dice }) => {
    const validMoves = calculateValidMoves(points, dice);
    useGameStore.setState({ _validMovesCache: validMoves });
  },
  { equalityFn: shallow }
);
```

### 2.5 Selector Hooks Pattern

Create reusable selector hooks:

```typescript
// hooks/useGameSelectors.ts
import { useGameStore } from '../stores/gameStore';
import { shallow } from 'zustand/shallow';

// Simple selectors
export const useCurrentPlayer = () =>
  useGameStore((state) => state.currentPlayer);

export const useDice = () =>
  useGameStore((state) => state.dice);

export const useGamePhase = () =>
  useGameStore((state) => state.gamePhase);

// Compound selectors
export const useBoardState = () =>
  useGameStore(
    (state) => ({
      points: state.points,
      bar: state.bar,
      bearOff: state.bearOff,
    }),
    shallow
  );

// Parameterized selectors
export const usePointState = (pointIndex: number) =>
  useGameStore((state) => state.points[pointIndex], shallow);

// Computed selectors
export const useIsMyTurn = (player: 'white' | 'black') =>
  useGameStore((state) => state.currentPlayer === player);

export const useCanRoll = () =>
  useGameStore((state) =>
    state.gamePhase === 'rolling' && state.dice === null
  );

export const useRemainingMoves = () =>
  useGameStore((state) => {
    if (!state.dice) return 0;
    const totalMoves = state.isDoubles ? 4 : 2;
    return totalMoves - state.usedDice.length;
  });
```

### 2.6 Avoiding Unnecessary Re-renders

#### Anti-Pattern: Inline Selectors with Object References

```typescript
// BAD: Creates new object reference each render
function Component() {
  const state = useGameStore((s) => ({ a: s.a, b: s.b }));  // Always re-renders!
}

// GOOD: Use shallow comparison
function Component() {
  const state = useGameStore((s) => ({ a: s.a, b: s.b }), shallow);
}

// BETTER: Define selector outside component
const selectAB = (state: GameState) => ({ a: state.a, b: state.b });

function Component() {
  const state = useGameStore(selectAB, shallow);
}
```

#### Anti-Pattern: Selecting Arrays

```typescript
// BAD: Always creates new array reference
function Component() {
  const items = useGameStore((s) => s.items.filter(i => i.active));
}

// GOOD: Memoize the filter
const selectActiveItems = createSelector(
  (state: GameState) => state.items,
  (items) => items.filter(i => i.active)
);

function Component() {
  const items = useGameStore(selectActiveItems);
}
```

---

## 3. Zustand vs Alternatives

### 3.1 Comparison with Redux Toolkit

| Aspect | Zustand | Redux Toolkit |
|--------|---------|---------------|
| **Bundle Size** | ~1KB | ~15KB (RTK + Redux) |
| **Boilerplate** | Minimal | Moderate (reduced from vanilla Redux) |
| **Learning Curve** | Low | Medium |
| **DevTools** | Via middleware | Built-in |
| **Middleware** | Optional | Required for async |
| **TypeScript** | Excellent | Excellent |
| **React Dependency** | Optional | Required (via react-redux) |
| **Server Components** | Compatible | Requires providers |
| **State Location** | External to React | External to React |

**When to Choose Redux Toolkit:**
- Very large applications with complex state interactions
- Need for advanced middleware ecosystem
- Team already familiar with Redux patterns
- Complex normalized data structures

**When to Choose Zustand (Recommended for Backgammon):**
- Smaller to medium applications
- Want minimal boilerplate
- Need to use state outside React
- Prefer simpler mental model
- Building games where performance matters

### 3.2 Comparison with Jotai/Recoil

| Aspect | Zustand | Jotai | Recoil |
|--------|---------|-------|--------|
| **Model** | Store-based | Atomic | Atomic |
| **Bundle Size** | ~1KB | ~2KB | ~20KB |
| **Mental Model** | Single store | Bottom-up atoms | Bottom-up atoms |
| **Derived State** | Via selectors | Derived atoms | Selectors |
| **Async** | In actions | Async atoms | Async selectors |
| **React Suspense** | Manual | Built-in | Built-in |
| **Code Splitting** | Per-store | Per-atom | Per-atom |
| **DevTools** | Via middleware | Via devtools | Built-in |

**Atomic State (Jotai/Recoil) is Better When:**
- Many independent pieces of state
- Heavy code splitting needs
- Deep component trees with scattered state access
- Suspense-first architecture

**Store-Based (Zustand) is Better When:**
- State is logically grouped (like a game)
- Actions need to access/modify multiple state pieces
- Using state outside React components
- Simpler debugging needs

### 3.3 When to Use React Query Instead

React Query (TanStack Query) is for **server state**, Zustand is for **client state**:

```typescript
// Server state: Game data from backend
const { data: game, isLoading } = useQuery({
  queryKey: ['game', gameId],
  queryFn: () => api.fetchGame(gameId),
});

// Client state: UI state, local game state during play
const selectedPiece = useGameStore((s) => s.selectedPiece);
const animationState = useUIStore((s) => s.animationState);
```

**Use React Query for:**
- Fetching game history from server
- Loading user profiles
- Fetching leaderboards
- Syncing game state with backend
- Caching API responses

**Use Zustand for:**
- Current game board state
- UI state (selected piece, valid moves display)
- Animation state
- Local preferences
- State that changes frequently during gameplay

### 3.4 Hybrid Approach (Recommended for Backgammon)

```typescript
// stores/gameStore.ts - Client-side game logic
export const useGameStore = create<GameState>()((set, get) => ({
  board: null,
  localMoves: [],  // Moves made locally, pending server confirmation

  applyLocalMove: (move: Move) => {
    set((state) => ({
      board: applyMoveToBoard(state.board, move),
      localMoves: [...state.localMoves, move],
    }));
  },

  syncWithServer: (serverState: ServerGameState) => {
    set({
      board: serverState.board,
      localMoves: [],  // Clear pending moves
    });
  },
}));

// hooks/useGame.ts - Combines both
export function useGame(gameId: string) {
  // Server state via React Query
  const { data: serverGame, refetch } = useQuery({
    queryKey: ['game', gameId],
    queryFn: () => api.fetchGame(gameId),
  });

  // Sync server state to Zustand
  useEffect(() => {
    if (serverGame) {
      useGameStore.getState().syncWithServer(serverGame);
    }
  }, [serverGame]);

  // Mutation for moves
  const moveMutation = useMutation({
    mutationFn: (move: Move) => api.makeMove(gameId, move),
    onMutate: (move) => {
      // Optimistic update via Zustand
      useGameStore.getState().applyLocalMove(move);
    },
    onError: () => {
      // Rollback by re-fetching
      refetch();
    },
  });

  // Return combined state
  const board = useGameStore((s) => s.board);

  return {
    board,
    makeMove: moveMutation.mutate,
    isLoading: moveMutation.isPending,
  };
}
```

---

## 4. Game State Modeling

### 4.1 Modeling Complex Game State

#### Complete Backgammon State Model

```typescript
// types/backgammon.ts

// Core game types
export type Player = 'white' | 'black';
export type Direction = 1 | -1;  // white moves positive, black moves negative

export interface Point {
  checkers: number;  // Positive = white, Negative = black (or use separate count + player)
}

// Alternative representation (more explicit)
export interface ExplicitPoint {
  count: number;
  owner: Player | null;
}

export interface BoardState {
  // 24 points, indexed 0-23
  // Point 0 = white's 1-point, Point 23 = white's 24-point (black's 1-point)
  points: Point[];

  // Checkers on the bar (hit checkers)
  bar: Record<Player, number>;

  // Checkers that have been borne off
  borneOff: Record<Player, number>;
}

export interface DiceState {
  values: [number, number] | null;
  remaining: number[];  // Available dice values to use
  isDoubles: boolean;
}

export interface Move {
  id: string;
  from: number | 'bar';  // Point index or 'bar'
  to: number | 'off';    // Point index or 'off' (bear off)
  dieUsed: number;
  wasHit: boolean;       // Did this move hit an opponent?
}

export interface Turn {
  id: string;
  player: Player;
  dice: DiceState;
  moves: Move[];
  timestamp: Date;
}

export interface DoublingCube {
  value: 1 | 2 | 4 | 8 | 16 | 32 | 64;
  owner: Player | 'center';
  lastDoubler: Player | null;
}

export type GamePhase =
  | 'not_started'
  | 'rolling_for_first'
  | 'playing'
  | 'doubling_offered'
  | 'game_over'
  | 'match_over';

export interface GameState {
  id: string;
  board: BoardState;
  currentPlayer: Player;
  phase: GamePhase;
  dice: DiceState;
  doublingCube: DoublingCube;
  winner: Player | null;
  winType: 'single' | 'gammon' | 'backgammon' | null;

  // Match play
  matchScore: Record<Player, number>;
  matchLength: number;
  crawfordGame: boolean;  // Crawford rule active?

  // History
  turnHistory: Turn[];
  currentTurnMoves: Move[];
}

// For multiplayer
export interface OnlineGameState extends GameState {
  players: {
    white: { id: string; name: string; connected: boolean };
    black: { id: string; name: string; connected: boolean };
  };
  spectators: string[];
  timeControl: {
    white: number;  // milliseconds remaining
    black: number;
    lastMoveAt: Date;
  };
  chat: ChatMessage[];
}
```

### 4.2 State Machine Patterns in Zustand

```typescript
// Using explicit state machine pattern
type GameMachineState =
  | { phase: 'idle' }
  | { phase: 'rolling_for_first'; rolls: Record<Player, number | null> }
  | { phase: 'rolling'; player: Player }
  | { phase: 'moving'; player: Player; dice: DiceState; movesLeft: number }
  | { phase: 'doubling_offered'; offeredBy: Player }
  | { phase: 'game_over'; winner: Player; winType: WinType };

interface GameMachine {
  machineState: GameMachineState;

  // Transitions
  startGame: () => void;
  rollForFirst: (player: Player) => void;
  roll: () => void;
  move: (move: Move) => void;
  endTurn: () => void;
  offerDouble: () => void;
  acceptDouble: () => void;
  declineDouble: () => void;
  resign: () => void;
}

export const useGameMachine = create<GameMachine & BoardState>()((set, get) => ({
  machineState: { phase: 'idle' },
  // ... board state

  startGame: () => {
    const currentPhase = get().machineState.phase;
    if (currentPhase !== 'idle') {
      console.warn(`Cannot start game from phase: ${currentPhase}`);
      return;
    }

    set({
      machineState: { phase: 'rolling_for_first', rolls: { white: null, black: null } },
      // Initialize board
    });
  },

  rollForFirst: (player) => {
    const state = get().machineState;
    if (state.phase !== 'rolling_for_first') return;

    const roll = Math.floor(Math.random() * 6) + 1;
    const newRolls = { ...state.rolls, [player]: roll };

    // Check if both have rolled
    if (newRolls.white !== null && newRolls.black !== null) {
      if (newRolls.white === newRolls.black) {
        // Tie, roll again
        set({ machineState: { phase: 'rolling_for_first', rolls: { white: null, black: null } } });
      } else {
        const firstPlayer = newRolls.white > newRolls.black ? 'white' : 'black';
        set({
          machineState: { phase: 'rolling', player: firstPlayer }
        });
      }
    } else {
      set({ machineState: { ...state, rolls: newRolls } });
    }
  },

  roll: () => {
    const state = get().machineState;
    if (state.phase !== 'rolling') return;

    const die1 = Math.floor(Math.random() * 6) + 1;
    const die2 = Math.floor(Math.random() * 6) + 1;
    const isDoubles = die1 === die2;

    set({
      machineState: {
        phase: 'moving',
        player: state.player,
        dice: {
          values: [die1, die2],
          remaining: isDoubles ? [die1, die1, die1, die1] : [die1, die2],
          isDoubles,
        },
        movesLeft: isDoubles ? 4 : 2,
      },
    });
  },

  move: (move) => {
    const state = get().machineState;
    if (state.phase !== 'moving') return;

    // Validate move...
    // Apply move to board...

    const newRemaining = state.dice.remaining.filter((d, i) =>
      i !== state.dice.remaining.indexOf(move.dieUsed)
    );

    if (newRemaining.length === 0 || !hasValidMoves(get(), newRemaining)) {
      // Turn is over, switch to rolling phase for other player
      set({
        machineState: {
          phase: 'rolling',
          player: state.player === 'white' ? 'black' : 'white'
        },
      });
    } else {
      set({
        machineState: {
          ...state,
          dice: { ...state.dice, remaining: newRemaining },
          movesLeft: state.movesLeft - 1,
        },
      });
    }
  },

  endTurn: () => {
    const state = get().machineState;
    if (state.phase !== 'moving') return;

    const nextPlayer = state.player === 'white' ? 'black' : 'white';
    set({
      machineState: { phase: 'rolling', player: nextPlayer },
    });
  },

  offerDouble: () => {
    const state = get().machineState;
    if (state.phase !== 'rolling') return;

    set({
      machineState: { phase: 'doubling_offered', offeredBy: state.player },
    });
  },

  acceptDouble: () => {
    const state = get().machineState;
    if (state.phase !== 'doubling_offered') return;

    // Double the cube value, give ownership to accepter
    // Continue with rolling phase for the offerer
    set({
      machineState: { phase: 'rolling', player: state.offeredBy },
      // Update doubling cube...
    });
  },

  declineDouble: () => {
    const state = get().machineState;
    if (state.phase !== 'doubling_offered') return;

    set({
      machineState: {
        phase: 'game_over',
        winner: state.offeredBy,
        winType: 'single',
      },
    });
  },

  resign: () => {
    const state = get().machineState;
    if (!['rolling', 'moving'].includes(state.phase)) return;

    const currentPlayer = 'player' in state ? state.player : null;
    if (!currentPlayer) return;

    set({
      machineState: {
        phase: 'game_over',
        winner: currentPlayer === 'white' ? 'black' : 'white',
        winType: 'single',
      },
    });
  },
}));
```

### 4.3 Undo/Redo Implementation

```typescript
interface UndoableState {
  board: BoardState;
  dice: DiceState;
  // Other game state...
}

interface UndoState {
  past: UndoableState[];
  present: UndoableState;
  future: UndoableState[];

  // Actions
  saveSnapshot: () => void;
  undo: () => void;
  redo: () => void;
  clearHistory: () => void;
}

// Separate undo store
export const useUndoStore = create<UndoState>()((set, get) => ({
  past: [],
  present: getInitialState(),
  future: [],

  saveSnapshot: () => {
    const gameState = useGameStore.getState();
    const snapshot: UndoableState = {
      board: gameState.board,
      dice: gameState.dice,
    };

    set((state) => ({
      past: [...state.past, state.present],
      present: snapshot,
      future: [],  // Clear redo stack on new action
    }));
  },

  undo: () => {
    const { past, present, future } = get();
    if (past.length === 0) return;

    const previous = past[past.length - 1];
    const newPast = past.slice(0, -1);

    // Apply previous state to game store
    useGameStore.setState({
      board: previous.board,
      dice: previous.dice,
    });

    set({
      past: newPast,
      present: previous,
      future: [present, ...future],
    });
  },

  redo: () => {
    const { past, present, future } = get();
    if (future.length === 0) return;

    const next = future[0];
    const newFuture = future.slice(1);

    // Apply next state to game store
    useGameStore.setState({
      board: next.board,
      dice: next.dice,
    });

    set({
      past: [...past, present],
      present: next,
      future: newFuture,
    });
  },

  clearHistory: () => set({
    past: [],
    future: [],
  }),
}));

// Integration with game actions
export const useGameStore = create<GameState>()((set, get) => ({
  // ... state

  makeMove: (from, to) => {
    // Save snapshot before move
    useUndoStore.getState().saveSnapshot();

    // Apply move
    set((state) => ({
      board: applyMove(state.board, from, to),
      // ...
    }));
  },
}));

// Component usage
function UndoRedoButtons() {
  const { past, future, undo, redo } = useUndoStore();

  return (
    <div>
      <button onClick={undo} disabled={past.length === 0}>
        Undo
      </button>
      <button onClick={redo} disabled={future.length === 0}>
        Redo
      </button>
    </div>
  );
}
```

### 4.4 History Tracking

```typescript
interface HistoryEntry {
  id: string;
  timestamp: Date;
  player: Player;
  action: GameAction;
  stateBefore: GameState;
  stateAfter: GameState;
}

interface HistoryStore {
  entries: HistoryEntry[];

  recordAction: (action: GameAction, stateBefore: GameState, stateAfter: GameState) => void;
  getEntriesForPlayer: (player: Player) => HistoryEntry[];
  getEntryById: (id: string) => HistoryEntry | undefined;
  exportHistory: () => string;  // JSON export
  clearHistory: () => void;
}

export const useHistoryStore = create<HistoryStore>()(
  persist(
    (set, get) => ({
      entries: [],

      recordAction: (action, stateBefore, stateAfter) => {
        const entry: HistoryEntry = {
          id: crypto.randomUUID(),
          timestamp: new Date(),
          player: stateBefore.currentPlayer,
          action,
          stateBefore,
          stateAfter,
        };

        set((state) => ({
          entries: [...state.entries, entry],
        }));
      },

      getEntriesForPlayer: (player) => {
        return get().entries.filter((e) => e.player === player);
      },

      getEntryById: (id) => {
        return get().entries.find((e) => e.id === id);
      },

      exportHistory: () => {
        return JSON.stringify(get().entries, null, 2);
      },

      clearHistory: () => set({ entries: [] }),
    }),
    {
      name: 'game-history',
      partialize: (state) => ({ entries: state.entries }),
    }
  )
);

// Middleware to auto-record history
const withHistory = (config: StateCreator<GameState>): StateCreator<GameState> =>
  (set, get, api) => {
    const wrappedSet: typeof set = (partial, replace) => {
      const stateBefore = get();
      set(partial, replace);
      const stateAfter = get();

      // Record significant state changes
      if (hasSignificantChange(stateBefore, stateAfter)) {
        const action = inferAction(stateBefore, stateAfter);
        useHistoryStore.getState().recordAction(action, stateBefore, stateAfter);
      }
    };

    return config(wrappedSet, get, api);
  };
```

---

## 5. Server Synchronization

### 5.1 Syncing with Backend State

```typescript
// types/api.ts
interface ServerGameState {
  id: string;
  version: number;  // For conflict detection
  board: BoardState;
  currentPlayer: Player;
  dice: DiceState | null;
  status: GameStatus;
  lastMoveAt: string;
}

interface SyncStore {
  serverVersion: number;
  lastSyncAt: Date | null;
  syncStatus: 'idle' | 'syncing' | 'error';
  pendingChanges: GameAction[];

  syncFromServer: (serverState: ServerGameState) => void;
  queueChange: (action: GameAction) => void;
  flushChanges: () => Promise<void>;
}

export const useSyncStore = create<SyncStore>()((set, get) => ({
  serverVersion: 0,
  lastSyncAt: null,
  syncStatus: 'idle',
  pendingChanges: [],

  syncFromServer: (serverState) => {
    const { serverVersion, pendingChanges } = get();

    // Only apply if server is ahead and no pending changes
    if (serverState.version > serverVersion && pendingChanges.length === 0) {
      useGameStore.setState({
        board: serverState.board,
        currentPlayer: serverState.currentPlayer,
        dice: serverState.dice,
      });

      set({
        serverVersion: serverState.version,
        lastSyncAt: new Date(),
      });
    }
  },

  queueChange: (action) => {
    set((state) => ({
      pendingChanges: [...state.pendingChanges, action],
    }));
  },

  flushChanges: async () => {
    const { pendingChanges, serverVersion } = get();
    if (pendingChanges.length === 0) return;

    set({ syncStatus: 'syncing' });

    try {
      const response = await api.submitActions({
        gameId: useGameStore.getState().id,
        baseVersion: serverVersion,
        actions: pendingChanges,
      });

      if (response.success) {
        set({
          serverVersion: response.newVersion,
          pendingChanges: [],
          syncStatus: 'idle',
          lastSyncAt: new Date(),
        });
      } else {
        // Conflict - need to handle
        set({ syncStatus: 'error' });
      }
    } catch (error) {
      set({ syncStatus: 'error' });
    }
  },
}));
```

### 5.2 Optimistic Updates

```typescript
interface OptimisticUpdate<T> {
  id: string;
  action: GameAction;
  previousState: T;
  timestamp: Date;
}

interface OptimisticStore {
  pendingUpdates: OptimisticUpdate<GameState>[];

  applyOptimistic: (action: GameAction) => string;  // Returns update ID
  confirmUpdate: (updateId: string) => void;
  revertUpdate: (updateId: string) => void;
  revertAll: () => void;
}

export const useOptimisticStore = create<OptimisticStore>()((set, get) => ({
  pendingUpdates: [],

  applyOptimistic: (action) => {
    const updateId = crypto.randomUUID();
    const previousState = useGameStore.getState();

    // Apply the action locally
    applyActionToStore(useGameStore, action);

    // Track the pending update
    set((state) => ({
      pendingUpdates: [
        ...state.pendingUpdates,
        {
          id: updateId,
          action,
          previousState,
          timestamp: new Date(),
        },
      ],
    }));

    return updateId;
  },

  confirmUpdate: (updateId) => {
    set((state) => ({
      pendingUpdates: state.pendingUpdates.filter((u) => u.id !== updateId),
    }));
  },

  revertUpdate: (updateId) => {
    const update = get().pendingUpdates.find((u) => u.id === updateId);
    if (!update) return;

    // Revert to previous state
    useGameStore.setState(update.previousState);

    // Remove from pending and reapply subsequent updates
    const { pendingUpdates } = get();
    const updateIndex = pendingUpdates.findIndex((u) => u.id === updateId);
    const remainingUpdates = pendingUpdates.slice(updateIndex + 1);

    // Reapply remaining updates
    for (const remaining of remainingUpdates) {
      applyActionToStore(useGameStore, remaining.action);
    }

    set({
      pendingUpdates: pendingUpdates.filter((u) => u.id !== updateId),
    });
  },

  revertAll: () => {
    const { pendingUpdates } = get();
    if (pendingUpdates.length === 0) return;

    // Revert to oldest pending update's previous state
    const oldest = pendingUpdates[0];
    useGameStore.setState(oldest.previousState);

    set({ pendingUpdates: [] });
  },
}));

// Usage with API calls
async function makeMove(from: number, to: number) {
  const action: GameAction = { type: 'MOVE', from, to };

  // Apply optimistically
  const updateId = useOptimisticStore.getState().applyOptimistic(action);

  try {
    await api.makeMove(from, to);
    useOptimisticStore.getState().confirmUpdate(updateId);
  } catch (error) {
    useOptimisticStore.getState().revertUpdate(updateId);
    throw error;
  }
}
```

### 5.3 Conflict Resolution Strategies

```typescript
type ConflictResolution = 'server_wins' | 'client_wins' | 'merge' | 'manual';

interface ConflictState {
  hasConflict: boolean;
  serverState: GameState | null;
  localState: GameState | null;
  conflictType: 'version_mismatch' | 'concurrent_move' | null;

  resolveConflict: (resolution: ConflictResolution) => void;
}

const resolvers: Record<ConflictResolution, (server: GameState, local: GameState) => GameState> = {
  server_wins: (server, _local) => server,
  client_wins: (_server, local) => local,
  merge: (server, local) => {
    // Intelligent merge - for backgammon, server state is generally authoritative
    // but we might want to preserve local UI state
    return {
      ...server,
      // Preserve local-only state
      selectedPiece: local.selectedPiece,
      highlightedMoves: local.highlightedMoves,
    };
  },
  manual: (server, local) => {
    // Don't auto-resolve, let UI handle it
    throw new Error('Manual resolution required');
  },
};

export const useConflictStore = create<ConflictState>()((set, get) => ({
  hasConflict: false,
  serverState: null,
  localState: null,
  conflictType: null,

  resolveConflict: (resolution) => {
    const { serverState, localState } = get();
    if (!serverState || !localState) return;

    const resolved = resolvers[resolution](serverState, localState);
    useGameStore.setState(resolved);

    set({
      hasConflict: false,
      serverState: null,
      localState: null,
      conflictType: null,
    });
  },
}));

// Detection during sync
function detectConflict(serverState: ServerGameState, localState: GameState): boolean {
  // Version mismatch
  if (serverState.version !== useSyncStore.getState().serverVersion + 1) {
    return true;
  }

  // Board state divergence
  if (!boardsEqual(serverState.board, localState.board)) {
    return true;
  }

  return false;
}
```

### 5.4 WebSocket Integration

```typescript
// stores/websocketStore.ts
interface WebSocketStore {
  socket: WebSocket | null;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  reconnectAttempts: number;

  connect: (gameId: string) => void;
  disconnect: () => void;
  send: (message: WSMessage) => void;
}

type WSMessage =
  | { type: 'MOVE'; from: number; to: number }
  | { type: 'ROLL_DICE' }
  | { type: 'END_TURN' }
  | { type: 'CHAT'; message: string }
  | { type: 'OFFER_DOUBLE' }
  | { type: 'ACCEPT_DOUBLE' }
  | { type: 'DECLINE_DOUBLE' };

type WSServerMessage =
  | { type: 'GAME_STATE'; state: ServerGameState }
  | { type: 'MOVE_MADE'; move: Move; player: Player }
  | { type: 'DICE_ROLLED'; dice: [number, number]; player: Player }
  | { type: 'TURN_ENDED'; nextPlayer: Player }
  | { type: 'GAME_OVER'; winner: Player; winType: WinType }
  | { type: 'PLAYER_CONNECTED'; player: Player }
  | { type: 'PLAYER_DISCONNECTED'; player: Player }
  | { type: 'ERROR'; message: string };

export const useWebSocketStore = create<WebSocketStore>()((set, get) => ({
  socket: null,
  connectionStatus: 'disconnected',
  reconnectAttempts: 0,

  connect: (gameId: string) => {
    const wsUrl = `${import.meta.env.VITE_WS_URL}/game/${gameId}/`;

    set({ connectionStatus: 'connecting' });

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      set({
        socket,
        connectionStatus: 'connected',
        reconnectAttempts: 0,
      });
    };

    socket.onmessage = (event) => {
      const message: WSServerMessage = JSON.parse(event.data);
      handleServerMessage(message);
    };

    socket.onerror = () => {
      set({ connectionStatus: 'error' });
    };

    socket.onclose = () => {
      set({ socket: null, connectionStatus: 'disconnected' });

      // Auto-reconnect logic
      const attempts = get().reconnectAttempts;
      if (attempts < 5) {
        setTimeout(() => {
          set({ reconnectAttempts: attempts + 1 });
          get().connect(gameId);
        }, Math.min(1000 * Math.pow(2, attempts), 30000));  // Exponential backoff
      }
    };

    set({ socket });
  },

  disconnect: () => {
    const { socket } = get();
    if (socket) {
      socket.close();
    }
    set({ socket: null, connectionStatus: 'disconnected' });
  },

  send: (message) => {
    const { socket, connectionStatus } = get();
    if (socket && connectionStatus === 'connected') {
      socket.send(JSON.stringify(message));
    } else {
      console.error('WebSocket not connected');
    }
  },
}));

// Message handler
function handleServerMessage(message: WSServerMessage) {
  switch (message.type) {
    case 'GAME_STATE':
      useGameStore.setState({
        board: message.state.board,
        currentPlayer: message.state.currentPlayer,
        dice: message.state.dice,
      });
      useSyncStore.getState().syncFromServer(message.state);
      break;

    case 'MOVE_MADE':
      // Apply opponent's move
      useGameStore.getState().applyServerMove(message.move);
      break;

    case 'DICE_ROLLED':
      if (message.player !== useGameStore.getState().localPlayer) {
        useGameStore.setState({
          dice: {
            values: message.dice,
            remaining: message.dice[0] === message.dice[1]
              ? [message.dice[0], message.dice[0], message.dice[0], message.dice[0]]
              : [...message.dice],
            isDoubles: message.dice[0] === message.dice[1],
          },
        });
      }
      break;

    case 'GAME_OVER':
      useGameStore.setState({
        phase: 'game_over',
        winner: message.winner,
        winType: message.winType,
      });
      break;

    case 'ERROR':
      console.error('Server error:', message.message);
      // Maybe revert optimistic updates
      useOptimisticStore.getState().revertAll();
      break;
  }
}

// Hook for components
export function useWebSocket(gameId: string) {
  const connect = useWebSocketStore((s) => s.connect);
  const disconnect = useWebSocketStore((s) => s.disconnect);
  const status = useWebSocketStore((s) => s.connectionStatus);

  useEffect(() => {
    connect(gameId);
    return () => disconnect();
  }, [gameId, connect, disconnect]);

  return { status };
}
```

---

## 6. Persistence and Hydration

### 6.1 Persist Middleware Usage

```typescript
import { create } from 'zustand';
import { persist, createJSONStorage, PersistOptions } from 'zustand/middleware';

interface GameState {
  board: BoardState;
  currentPlayer: Player;
  settings: GameSettings;
  // ... other state
}

// Basic persistence
export const useGameStore = create<GameState>()(
  persist(
    (set, get) => ({
      // Store implementation
    }),
    {
      name: 'backgammon-game',  // localStorage key
    }
  )
);
```

### 6.2 Selective Persistence

```typescript
interface FullState {
  // Persisted
  board: BoardState;
  currentPlayer: Player;
  turnHistory: Turn[];
  settings: GameSettings;

  // Not persisted (transient)
  selectedPiece: number | null;
  highlightedMoves: Move[];
  animationState: AnimationState;
  connectionStatus: 'connected' | 'disconnected';
}

export const useGameStore = create<FullState>()(
  persist(
    (set, get) => ({
      // Implementation
    }),
    {
      name: 'backgammon-game',
      partialize: (state) => ({
        // Only these fields are persisted
        board: state.board,
        currentPlayer: state.currentPlayer,
        turnHistory: state.turnHistory,
        settings: state.settings,
      }),
    }
  )
);

// Alternative: Exclude specific fields
const persistConfig: PersistOptions<FullState> = {
  name: 'backgammon-game',
  partialize: (state) =>
    Object.fromEntries(
      Object.entries(state).filter(
        ([key]) => !['selectedPiece', 'highlightedMoves', 'animationState', 'connectionStatus'].includes(key)
      )
    ) as Partial<FullState>,
};
```

### 6.3 Custom Storage

```typescript
// IndexedDB storage for larger state
import { get as idbGet, set as idbSet, del as idbDel } from 'idb-keyval';

const indexedDBStorage = {
  getItem: async (name: string) => {
    const value = await idbGet(name);
    return value ?? null;
  },
  setItem: async (name: string, value: string) => {
    await idbSet(name, value);
  },
  removeItem: async (name: string) => {
    await idbDel(name);
  },
};

export const useGameStore = create<GameState>()(
  persist(
    (set, get) => ({
      // Implementation
    }),
    {
      name: 'backgammon-game',
      storage: createJSONStorage(() => indexedDBStorage),
    }
  )
);

// Session storage for temporary state
export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      // Implementation
    }),
    {
      name: 'backgammon-session',
      storage: createJSONStorage(() => sessionStorage),
    }
  )
);
```

### 6.4 Migration Strategies

```typescript
interface GameStateV1 {
  board: number[][];  // Old format
  player: 'white' | 'black';
}

interface GameStateV2 {
  board: BoardState;  // New structured format
  currentPlayer: Player;
  version: 2;
}

interface GameStateV3 {
  board: BoardState;
  currentPlayer: Player;
  doublingCube: DoublingCube;  // New field
  version: 3;
}

type CurrentState = GameStateV3;

export const useGameStore = create<CurrentState>()(
  persist(
    (set, get) => ({
      // Current implementation
    }),
    {
      name: 'backgammon-game',
      version: 3,
      migrate: (persistedState: unknown, version: number): CurrentState => {
        let state = persistedState as any;

        // V1 -> V2 migration
        if (version < 2) {
          state = {
            board: convertV1BoardToV2(state.board),
            currentPlayer: state.player,
            version: 2,
          };
        }

        // V2 -> V3 migration
        if (version < 3) {
          state = {
            ...state,
            doublingCube: { value: 1, owner: 'center', lastDoubler: null },
            version: 3,
          };
        }

        return state as CurrentState;
      },
    }
  )
);

// Migration helper for complex transformations
function convertV1BoardToV2(oldBoard: number[][]): BoardState {
  return {
    points: oldBoard.map((point) => ({
      checkers: point[0] || 0,
      // Additional conversion logic
    })),
    bar: { white: 0, black: 0 },
    borneOff: { white: 0, black: 0 },
  };
}
```

### 6.5 SSR Considerations

```typescript
// For Next.js or other SSR frameworks

// Option 1: Skip hydration mismatch by using dynamic import
import dynamic from 'next/dynamic';

const GameBoard = dynamic(() => import('./GameBoard'), {
  ssr: false,  // Only render on client
});

// Option 2: Handle hydration explicitly
export const useGameStore = create<GameState>()(
  persist(
    (set, get) => ({
      // Implementation
    }),
    {
      name: 'backgammon-game',
      skipHydration: true,  // Don't auto-hydrate
    }
  )
);

// Then hydrate manually in a client component
'use client';

import { useEffect } from 'react';
import { useGameStore } from './stores/gameStore';

function HydrationHandler() {
  useEffect(() => {
    useGameStore.persist.rehydrate();
  }, []);

  return null;
}

// Option 3: Hydration-safe selectors
function useHydratedStore<T>(selector: (state: GameState) => T, fallback: T): T {
  const [isHydrated, setIsHydrated] = useState(false);
  const value = useGameStore(selector);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  return isHydrated ? value : fallback;
}

// Usage
function GameStatus() {
  const currentPlayer = useHydratedStore(
    (s) => s.currentPlayer,
    'white'  // SSR fallback
  );

  return <div>Current: {currentPlayer}</div>;
}
```

### 6.6 Hydration Events

```typescript
export const useGameStore = create<GameState>()(
  persist(
    (set, get) => ({
      // Implementation
    }),
    {
      name: 'backgammon-game',
      onRehydrateStorage: (state) => {
        console.log('Starting hydration...');

        return (state, error) => {
          if (error) {
            console.error('Hydration failed:', error);
            // Maybe reset to defaults
          } else {
            console.log('Hydration complete');
            // Post-hydration logic
            validateAndFixState(state);
          }
        };
      },
    }
  )
);

// Hook to wait for hydration
export function useHydrated() {
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const unsubscribe = useGameStore.persist.onFinishHydration(() => {
      setHydrated(true);
    });

    // Check if already hydrated
    if (useGameStore.persist.hasHydrated()) {
      setHydrated(true);
    }

    return unsubscribe;
  }, []);

  return hydrated;
}
```

---

## 7. Testing Zustand Stores

### 7.1 Unit Testing Stores

```typescript
// __tests__/gameStore.test.ts
import { act } from '@testing-library/react';
import { useGameStore } from '../stores/gameStore';

// Reset store before each test
beforeEach(() => {
  useGameStore.setState({
    board: createInitialBoard(),
    currentPlayer: 'white',
    dice: null,
    usedDice: [],
    gamePhase: 'rolling',
  });
});

describe('gameStore', () => {
  describe('rollDice', () => {
    it('should set dice values between 1 and 6', () => {
      const { rollDice } = useGameStore.getState();

      act(() => {
        rollDice();
      });

      const { dice } = useGameStore.getState();

      expect(dice).not.toBeNull();
      expect(dice![0]).toBeGreaterThanOrEqual(1);
      expect(dice![0]).toBeLessThanOrEqual(6);
      expect(dice![1]).toBeGreaterThanOrEqual(1);
      expect(dice![1]).toBeLessThanOrEqual(6);
    });

    it('should set gamePhase to moving after roll', () => {
      const { rollDice } = useGameStore.getState();

      act(() => {
        rollDice();
      });

      expect(useGameStore.getState().gamePhase).toBe('moving');
    });
  });

  describe('makeMove', () => {
    beforeEach(() => {
      // Set up a valid game state for moving
      useGameStore.setState({
        dice: [3, 5],
        usedDice: [],
        gamePhase: 'moving',
      });
    });

    it('should move a checker from one point to another', () => {
      const initialBoard = useGameStore.getState().board;
      const fromPoint = 5;  // White has 5 checkers here at start
      const toPoint = 2;    // 3 points away

      act(() => {
        useGameStore.getState().makeMove(fromPoint, toPoint);
      });

      const { board } = useGameStore.getState();

      expect(board.points[fromPoint].count).toBe(initialBoard.points[fromPoint].count - 1);
      expect(board.points[toPoint].count).toBe(initialBoard.points[toPoint].count + 1);
    });

    it('should mark die as used after move', () => {
      act(() => {
        useGameStore.getState().makeMove(5, 2);  // Uses the 3
      });

      expect(useGameStore.getState().usedDice).toContain(3);
    });

    it('should reject invalid moves', () => {
      const result = act(() => {
        return useGameStore.getState().makeMove(0, 10);  // Invalid move
      });

      expect(result).toBe(false);
    });
  });

  describe('endTurn', () => {
    it('should switch to other player', () => {
      useGameStore.setState({ currentPlayer: 'white' });

      act(() => {
        useGameStore.getState().endTurn();
      });

      expect(useGameStore.getState().currentPlayer).toBe('black');
    });

    it('should reset dice', () => {
      useGameStore.setState({ dice: [3, 5], usedDice: [3, 5] });

      act(() => {
        useGameStore.getState().endTurn();
      });

      expect(useGameStore.getState().dice).toBeNull();
      expect(useGameStore.getState().usedDice).toEqual([]);
    });
  });
});

// Testing computed values
describe('computed selectors', () => {
  it('should correctly calculate valid moves', () => {
    useGameStore.setState({
      board: createBoardWithPosition([
        { point: 5, count: 5, player: 'white' },
        { point: 7, count: 3, player: 'white' },
      ]),
      currentPlayer: 'white',
      dice: [3, 5],
      usedDice: [],
    });

    const validMoves = selectValidMoves(useGameStore.getState());

    expect(validMoves).toContainEqual({ from: 5, to: 2 });
    expect(validMoves).toContainEqual({ from: 5, to: 0 });
    expect(validMoves).toContainEqual({ from: 7, to: 4 });
  });

  it('should detect when player can bear off', () => {
    useGameStore.setState({
      board: createBoardWithPosition([
        // All white checkers in home board
        { point: 0, count: 5, player: 'white' },
        { point: 1, count: 5, player: 'white' },
        { point: 2, count: 5, player: 'white' },
      ]),
      currentPlayer: 'white',
    });

    expect(selectCanBearOff(useGameStore.getState())).toBe(true);
  });
});
```

### 7.2 Integration Testing with React

```typescript
// __tests__/GameBoard.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GameBoard } from '../components/GameBoard';
import { useGameStore } from '../stores/gameStore';

// Reset store before each test
beforeEach(() => {
  useGameStore.setState({
    board: createInitialBoard(),
    currentPlayer: 'white',
    dice: null,
    usedDice: [],
    gamePhase: 'rolling',
    selectedPiece: null,
  });
});

describe('GameBoard', () => {
  it('should render the board with initial position', () => {
    render(<GameBoard />);

    // Check that pieces are rendered
    expect(screen.getByTestId('point-5')).toHaveTextContent('5');  // 5 white pieces
    expect(screen.getByTestId('point-23')).toHaveTextContent('2'); // 2 black pieces
  });

  it('should show roll button when in rolling phase', () => {
    render(<GameBoard />);

    expect(screen.getByRole('button', { name: /roll/i })).toBeInTheDocument();
  });

  it('should show dice after rolling', async () => {
    render(<GameBoard />);

    await userEvent.click(screen.getByRole('button', { name: /roll/i }));

    await waitFor(() => {
      expect(screen.getByTestId('dice-display')).toBeInTheDocument();
    });
  });

  it('should highlight valid moves when piece is selected', async () => {
    // Set up game state with dice rolled
    useGameStore.setState({
      dice: [3, 5],
      usedDice: [],
      gamePhase: 'moving',
    });

    render(<GameBoard />);

    // Click on a piece to select it
    await userEvent.click(screen.getByTestId('point-5'));

    // Valid destination points should be highlighted
    expect(screen.getByTestId('point-2')).toHaveClass('highlighted');
    expect(screen.getByTestId('point-0')).toHaveClass('highlighted');
  });

  it('should move piece when clicking on valid destination', async () => {
    useGameStore.setState({
      dice: [3, 5],
      usedDice: [],
      gamePhase: 'moving',
    });

    render(<GameBoard />);

    // Select piece
    await userEvent.click(screen.getByTestId('point-5'));

    // Click destination
    await userEvent.click(screen.getByTestId('point-2'));

    // Verify move was made
    await waitFor(() => {
      expect(screen.getByTestId('point-5')).toHaveTextContent('4');
      expect(screen.getByTestId('point-2')).toHaveTextContent('1');
    });
  });
});

// Testing with custom render that includes providers
const customRender = (ui: React.ReactElement, options = {}) => {
  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>,
    options
  );
};

describe('GameBoard with React Query', () => {
  it('should sync with server state', async () => {
    // Mock API response
    server.use(
      rest.get('/api/game/:id', (req, res, ctx) => {
        return res(ctx.json({
          board: createTestBoard(),
          currentPlayer: 'black',
        }));
      })
    );

    customRender(<GameBoard gameId="123" />);

    await waitFor(() => {
      expect(useGameStore.getState().currentPlayer).toBe('black');
    });
  });
});
```

### 7.3 Mocking for Component Tests

```typescript
// __mocks__/stores/gameStore.ts
import { create } from 'zustand';

export const mockGameState = {
  board: createInitialBoard(),
  currentPlayer: 'white' as const,
  dice: null,
  usedDice: [],
  gamePhase: 'rolling' as const,
  selectedPiece: null,
};

export const mockGameActions = {
  rollDice: jest.fn(),
  makeMove: jest.fn(() => true),
  endTurn: jest.fn(),
  selectPiece: jest.fn(),
  resetGame: jest.fn(),
};

export const useGameStore = create(() => ({
  ...mockGameState,
  ...mockGameActions,
}));

// Helper to reset mock state
export const resetMockStore = () => {
  useGameStore.setState({
    ...mockGameState,
  });
  Object.values(mockGameActions).forEach(fn => fn.mockClear());
};
```

```typescript
// __tests__/DiceButton.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DiceButton } from '../components/DiceButton';
import { useGameStore, mockGameActions, resetMockStore } from '../stores/__mocks__/gameStore';

jest.mock('../stores/gameStore');

beforeEach(() => {
  resetMockStore();
});

describe('DiceButton', () => {
  it('should be disabled when dice are already rolled', () => {
    useGameStore.setState({ dice: [3, 5] });

    render(<DiceButton />);

    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('should call rollDice when clicked', async () => {
    render(<DiceButton />);

    await userEvent.click(screen.getByRole('button'));

    expect(mockGameActions.rollDice).toHaveBeenCalledTimes(1);
  });

  it('should be disabled when not current player turn', () => {
    useGameStore.setState({
      currentPlayer: 'black',
      localPlayer: 'white',
    });

    render(<DiceButton />);

    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### 7.4 Testing Async Actions

```typescript
// stores/gameStore.ts
export const useGameStore = create<GameStore>()((set, get) => ({
  // ...

  makeServerMove: async (from: number, to: number) => {
    set({ isLoading: true });

    try {
      const result = await api.makeMove(get().gameId, from, to);

      set({
        board: result.board,
        currentPlayer: result.currentPlayer,
        isLoading: false,
      });

      return true;
    } catch (error) {
      set({ isLoading: false, error: error.message });
      return false;
    }
  },
}));

// __tests__/gameStore.async.test.ts
import { waitFor } from '@testing-library/react';
import { useGameStore } from '../stores/gameStore';

// Mock the API
jest.mock('../api', () => ({
  makeMove: jest.fn(),
}));

import * as api from '../api';

describe('async actions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useGameStore.setState({
      gameId: 'test-game',
      board: createInitialBoard(),
      isLoading: false,
      error: null,
    });
  });

  it('should update state after successful server move', async () => {
    const newBoard = createBoardWithMove(5, 2);
    (api.makeMove as jest.Mock).mockResolvedValue({
      board: newBoard,
      currentPlayer: 'black',
    });

    const result = await useGameStore.getState().makeServerMove(5, 2);

    expect(result).toBe(true);
    expect(useGameStore.getState().board).toEqual(newBoard);
    expect(useGameStore.getState().currentPlayer).toBe('black');
    expect(useGameStore.getState().isLoading).toBe(false);
  });

  it('should set error state on failed move', async () => {
    (api.makeMove as jest.Mock).mockRejectedValue(new Error('Network error'));

    const result = await useGameStore.getState().makeServerMove(5, 2);

    expect(result).toBe(false);
    expect(useGameStore.getState().error).toBe('Network error');
    expect(useGameStore.getState().isLoading).toBe(false);
  });

  it('should set loading state during request', async () => {
    (api.makeMove as jest.Mock).mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 100))
    );

    const movePromise = useGameStore.getState().makeServerMove(5, 2);

    expect(useGameStore.getState().isLoading).toBe(true);

    await movePromise;

    expect(useGameStore.getState().isLoading).toBe(false);
  });
});
```

### 7.5 Testing Middleware

```typescript
// Testing persist middleware
describe('persistence', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('should persist state to localStorage', async () => {
    useGameStore.setState({
      board: createBoardWithMove(5, 2),
      currentPlayer: 'black',
    });

    // Wait for debounced persist
    await new Promise(resolve => setTimeout(resolve, 100));

    const stored = JSON.parse(localStorage.getItem('backgammon-game') || '{}');

    expect(stored.state.currentPlayer).toBe('black');
  });

  it('should restore state from localStorage on mount', () => {
    const savedState = {
      state: {
        board: createBoardWithMove(5, 2),
        currentPlayer: 'black',
      },
      version: 1,
    };

    localStorage.setItem('backgammon-game', JSON.stringify(savedState));

    // Create a new store instance (simulating page reload)
    const { useGameStore: freshStore } = jest.requireActual('../stores/gameStore');

    expect(freshStore.getState().currentPlayer).toBe('black');
  });
});

// Testing subscriptions
describe('subscriptions', () => {
  it('should call listener when state changes', () => {
    const listener = jest.fn();

    const unsubscribe = useGameStore.subscribe(
      (state) => state.currentPlayer,
      listener
    );

    useGameStore.setState({ currentPlayer: 'black' });

    expect(listener).toHaveBeenCalledWith('black', 'white');

    unsubscribe();
  });
});
```

---

## 8. Recommendations for Backgammon Project

### 8.1 Recommended Store Architecture

Based on the CLAUDE.md file specifying "one store per domain (game, user, ui)", here's the recommended structure:

```
frontend/
├── src/
│   ├── stores/
│   │   ├── index.ts           # Re-exports all stores
│   │   ├── gameStore.ts       # Core game state
│   │   ├── userStore.ts       # User/auth state
│   │   ├── uiStore.ts         # UI state
│   │   └── slices/            # For gameStore breakdown
│   │       ├── boardSlice.ts
│   │       ├── diceSlice.ts
│   │       └── turnSlice.ts
│   ├── selectors/
│   │   ├── gameSelectors.ts   # Memoized selectors
│   │   └── uiSelectors.ts
│   └── hooks/
│       ├── useGameSelectors.ts # Selector hooks
│       └── useGameActions.ts   # Action hooks
```

### 8.2 Recommended Tech Choices

1. **Use Zustand with these middleware:**
   - `devtools` for development debugging
   - `persist` for game state recovery
   - `immer` for cleaner board mutations

2. **Use React Query for:**
   - Fetching game history
   - Loading user profiles
   - Leaderboard data
   - Initial game state from server

3. **Use selectors for:**
   - Valid moves calculation (memoized)
   - Can bear off check
   - Game over detection
   - Score calculation

### 8.3 Key Implementation Priorities

1. **Phase 1: Core Game Store**
   - Board state management
   - Dice rolling
   - Move validation and application
   - Turn management

2. **Phase 2: Persistence**
   - Save game state locally
   - Resume interrupted games
   - Migration strategy for updates

3. **Phase 3: Server Sync**
   - WebSocket integration
   - Optimistic updates
   - Conflict resolution

4. **Phase 4: History & Undo**
   - Turn history tracking
   - Undo for local games
   - Export/import games

### 8.4 Sample Starting Implementation

```typescript
// stores/gameStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type { GameStore } from '../types/game';
import { createInitialBoard, applyMove, validateMove, checkGameOver } from '../utils/gameLogic';

export const useGameStore = create<GameStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // State
        board: createInitialBoard(),
        currentPlayer: 'white',
        dice: null,
        usedDice: [],
        gamePhase: 'not_started',
        winner: null,
        doublingCube: { value: 1, owner: 'center', lastDoubler: null },
        turnHistory: [],

        // Actions
        startGame: () => set((state) => {
          state.board = createInitialBoard();
          state.currentPlayer = 'white';
          state.gamePhase = 'rolling';
          state.winner = null;
          state.turnHistory = [];
        }),

        rollDice: () => set((state) => {
          const die1 = Math.floor(Math.random() * 6) + 1;
          const die2 = Math.floor(Math.random() * 6) + 1;

          state.dice = [die1, die2];
          state.usedDice = [];
          state.gamePhase = 'moving';
        }),

        makeMove: (from, to) => {
          const state = get();
          if (!validateMove(state, from, to)) return false;

          set((draft) => {
            draft.board = applyMove(draft.board, from, to, draft.currentPlayer);

            const dieUsed = Math.abs(to - from);
            draft.usedDice.push(dieUsed);

            if (checkGameOver(draft.board, draft.currentPlayer)) {
              draft.gamePhase = 'game_over';
              draft.winner = draft.currentPlayer;
            }
          });

          return true;
        },

        endTurn: () => set((state) => {
          state.currentPlayer = state.currentPlayer === 'white' ? 'black' : 'white';
          state.dice = null;
          state.usedDice = [];
          state.gamePhase = 'rolling';
        }),

        resetGame: () => set((state) => {
          state.board = createInitialBoard();
          state.currentPlayer = 'white';
          state.dice = null;
          state.usedDice = [];
          state.gamePhase = 'not_started';
          state.winner = null;
          state.turnHistory = [];
        }),
      })),
      {
        name: 'backgammon-game',
        partialize: (state) => ({
          board: state.board,
          currentPlayer: state.currentPlayer,
          gamePhase: state.gamePhase,
          doublingCube: state.doublingCube,
        }),
      }
    ),
    { name: 'backgammon' }
  )
);
```

---

## References

- Zustand GitHub: https://github.com/pmndrs/zustand
- Zustand Documentation: https://docs.pmnd.rs/zustand
- TanStack Query: https://tanstack.com/query
- Immer: https://immerjs.github.io/immer/

---

*This research document provides comprehensive guidance for implementing Zustand state management in the Backgammon AI Sandbox project. The patterns and examples are specifically tailored for game state management with TypeScript.*
