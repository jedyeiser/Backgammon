"""
RuleSet pattern for multi-game support.

This module provides an abstract base class for game rule implementations,
allowing the platform to support multiple board games with different rules.

Usage:
    from apps.game.rulesets import RuleSetRegistry, BackgammonRuleSet

    # Get a ruleset by game type
    ruleset_class = RuleSetRegistry.get('backgammon')
    ruleset = ruleset_class(game_state)

    # List available games
    available = RuleSetRegistry.get_available_games()
"""
from .base import BaseRuleSet
from .backgammon import BackgammonRuleSet
from .catan import CatanRuleSet
from .registry import RuleSetRegistry

__all__ = [
    'BaseRuleSet',
    'BackgammonRuleSet',
    'CatanRuleSet',
    'RuleSetRegistry',
]
