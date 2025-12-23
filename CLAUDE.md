# Backgammon AI Sandbox

A learning-focused project for exploring ML and AI techniques through backgammon.

## Project Purpose

This is a **sandbox for learning** - prioritizing experimentation and understanding over production polish. The goal is to explore various AI/ML approaches in a well-structured environment.

## Tech Stack

### Backend
- **Framework**: Django 5.x with Django REST Framework
- **Database**: PostgreSQL
- **Authentication**: Django's built-in auth + django-rest-framework-simplejwt for API tokens
- **Python**: 3.11+

### Frontend
- **Build Tool**: Vite
- **Framework**: React 18+ with TypeScript
- **State Management**: Zustand
- **Styling**: Tailwind CSS (recommended for rapid prototyping)
- **API Client**: TanStack Query (React Query) for server state

### AI/ML (future)
- PyTorch for neural networks
- NumPy for numerical operations
- Potential: stable-baselines3 for RL algorithms

## Project Structure

```
Backgammon/
├── backend/                 # Django project
│   ├── config/             # Django settings, urls, wsgi
│   ├── apps/
│   │   ├── accounts/       # User authentication & profiles
│   │   ├── game/           # Core game logic, models, API
│   │   └── ai/             # AI players, training, evaluation
│   ├── requirements/
│   │   ├── base.txt
│   │   ├── dev.txt
│   │   └── prod.txt
│   └── manage.py
├── frontend/               # Vite React app
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── stores/         # Zustand stores
│   │   ├── hooks/          # Custom React hooks
│   │   ├── api/            # API client functions
│   │   ├── types/          # TypeScript types
│   │   └── utils/          # Utility functions
│   └── ...
├── docs/                   # Documentation & learning notes
└── scripts/                # Utility scripts
```

## Coding Conventions

### Python/Django
- Use type hints throughout
- Follow PEP 8, enforced by ruff
- Use Django's class-based views for complex logic, function-based for simple endpoints
- Models should be thin; business logic in service modules
- Write docstrings for public APIs

### TypeScript/React
- Functional components only
- Use TypeScript strict mode
- Prefer named exports
- Zustand stores: one store per domain (game, user, ui)
- Use React Query for all server state

### General
- Commits should be atomic and descriptive
- Branch naming: `feature/`, `fix/`, `experiment/`
- This is a learning project - comments explaining "why" are encouraged

## Game Domain

### Backgammon Rules Reference
- 24 points (triangles) on the board
- Each player has 15 checkers
- Movement determined by two dice
- Goal: bear off all checkers first
- Key concepts: hitting, re-entering from bar, bearing off, doubling cube

### Key Entities
- `Game`: A single game instance with state
- `Board`: 24 points + bar + bear-off for each player
- `Move`: A single checker movement
- `Turn`: All moves in a player's turn (up to 4 with doubles)
- `Player`: Human or AI player

## Commands

```bash
# Backend
cd backend
python manage.py runserver
python manage.py test

# Frontend
cd frontend
npm run dev
npm run build
npm run test

# Database
python manage.py migrate
python manage.py createsuperuser
```

## Environment Variables

Backend expects:
- `DATABASE_URL` or individual `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`
- `SECRET_KEY`
- `DEBUG` (default: False)
- `ALLOWED_HOSTS`

Frontend expects:
- `VITE_API_URL` (backend API base URL)

## AI Exploration Roadmap

1. **Random Player** - Baseline opponent
2. **Heuristic Player** - Hand-crafted position evaluation
3. **Minimax/Expectimax** - Classical game tree search
4. **TD-Learning** - Temporal difference learning (TD-Gammon approach)
5. **Neural Network Evaluation** - Deep learning for position assessment
6. **Reinforcement Learning** - Policy gradient methods
7. **MCTS** - Monte Carlo Tree Search adaptation for stochastic games

## Notes for Claude

- This is a learning sandbox - explain concepts when implementing new AI techniques
- Prefer clarity over cleverness in code
- When adding ML features, include comments explaining the approach
- Feel free to suggest experiments or alternative approaches
- Database migrations should be reviewed before applying
