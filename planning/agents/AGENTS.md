# Agent Manifest

This document describes the specialized agents available for the Backgammon project. Each agent is defined in a separate markdown file and provides domain-specific expertise.

## How to Use Agents

When working with Claude Code, reference these agent definitions to invoke specialized thinking:

```
"Think like the [agent-name] expert defined in planning/agents/[agent-name].md"
```

Or ask Claude to:
```
"Apply the Django API Expert thinking framework to design this endpoint"
```

## Available Agents

### Backend Agents

| Agent | File | Use When |
|-------|------|----------|
| **Django API Expert** | `django-api-expert.md` | Designing REST APIs, authentication, database queries, real-time features |
| **Data Structures Expert** | `data-structures-expert.md` | Board representation, move generation, state serialization, performance |

### Frontend Agents

| Agent | File | Use When |
|-------|------|----------|
| **React Frontend Expert** | `react-frontend-expert.md` | Component design, hooks, performance optimization, animations |
| **State Management Expert** | `state-management-expert.md` | Zustand stores, React Query, synchronization, state machines |

### Game & AI Agents

| Agent | File | Use When |
|-------|------|----------|
| **Game Theory Expert** | `game-theory-expert.md` | AI strategy, position evaluation, probability calculations, search algorithms |
| **ML/AI Expert** | `ml-ai-expert.md` | Neural networks, TD-learning, training loops, PyTorch implementation |

### Infrastructure Agents

| Agent | File | Use When |
|-------|------|----------|
| **Testing Expert** | `testing-expert.md` | Test strategies, pytest, Vitest, E2E tests, mocking |
| **DevOps Expert** | `devops-expert.md` | Docker, CI/CD, database management, deployment |

## Agent Thinking Frameworks

Each agent includes a "Thinking Framework" section that outlines:

1. **Key considerations** - What to always think about in that domain
2. **Common patterns** - Proven approaches and code examples
3. **Trade-off analysis** - How to evaluate alternatives
4. **Questions to ask** - Prompts to ensure thorough analysis

## Combining Agents

For complex tasks, combine multiple agent perspectives:

**Example: Implementing a "make move" feature**

1. **Data Structures Expert**: Design the Move data structure
2. **Django API Expert**: Design the API endpoint
3. **State Management Expert**: Design the Zustand action
4. **React Frontend Expert**: Design the UI interaction
5. **Testing Expert**: Design the test strategy

## Creating New Agents

When adding a new agent:

1. Create a new markdown file in `planning/agents/`
2. Follow the established format:
   - Role description
   - Expertise areas
   - Thinking framework
   - Code examples
   - Questions to ask
3. Update this manifest

## Agent Evolution

These agents should evolve as the project grows. After implementing features, update agent files with:

- New patterns discovered
- Lessons learned
- Project-specific conventions
- Links to relevant code examples
