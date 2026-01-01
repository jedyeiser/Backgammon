"""
Settlers of Catan rule implementation.

This module provides a framework for Catan rules. Note that this is a
placeholder implementation that demonstrates the RuleSet pattern and
provides a foundation for future development.

Game Overview:
    - 3-4 players compete to build settlements, cities, and roads
    - Resources are gathered based on dice rolls and settlement placement
    - First player to 10 victory points wins
"""
import random
from typing import List, Dict, Any, Optional

from .base import BaseRuleSet


class CatanRuleSet(BaseRuleSet):
    """
    Settlers of Catan game rules implementation.

    This is a placeholder that demonstrates how the RuleSet pattern
    enables adding new games to the platform.

    Catan-specific features:
    - Hex-based board with resources
    - 3-4 player support
    - Settlement, city, and road building
    - Resource trading
    - Victory point system
    """

    game_type = 'catan'
    display_name = 'Settlers of Catan'
    min_players = 3
    max_players = 4
    requires_dice = True

    # Resource types
    RESOURCES = ['brick', 'lumber', 'ore', 'grain', 'wool']

    # Building costs
    COSTS = {
        'road': {'brick': 1, 'lumber': 1},
        'settlement': {'brick': 1, 'lumber': 1, 'grain': 1, 'wool': 1},
        'city': {'ore': 3, 'grain': 2},
        'development_card': {'ore': 1, 'grain': 1, 'wool': 1},
    }

    # Victory points to win
    VICTORY_POINTS_TO_WIN = 10

    def get_initial_state(self) -> Dict[str, Any]:
        """Return initial Catan game state."""
        return {
            'board': self._generate_board(),
            'players': {},
            'player_order': [],
            'current_player_index': 0,
            'phase': 'setup',  # setup, main, robber
            'setup_round': 1,  # 1 or 2 for initial placement
            'dice_rolled': False,
            'robber_hex': None,
            'development_cards_deck': self._create_dev_card_deck(),
            'longest_road_holder': None,
            'largest_army_holder': None,
            'turn_number': 0,
        }

    def _generate_board(self) -> Dict[str, Any]:
        """
        Generate a standard Catan board.

        Note: This is a simplified representation. A full implementation
        would include proper hex coordinates and adjacency calculations.
        """
        # Standard Catan has 19 hexes
        hex_resources = (
            ['desert'] +
            ['brick'] * 3 +
            ['lumber'] * 4 +
            ['ore'] * 3 +
            ['grain'] * 4 +
            ['wool'] * 4
        )
        random.shuffle(hex_resources)

        # Number tokens (excluding 7, which triggers robber)
        number_tokens = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        random.shuffle(number_tokens)

        hexes = []
        token_idx = 0
        for i, resource in enumerate(hex_resources):
            hex_data = {
                'id': i,
                'resource': resource,
                'number': None if resource == 'desert' else number_tokens[token_idx],
                'has_robber': resource == 'desert',
            }
            if resource != 'desert':
                token_idx += 1
            hexes.append(hex_data)

        return {
            'hexes': hexes,
            'vertices': [None] * 54,  # Settlement/city positions
            'edges': [None] * 72,  # Road positions
        }

    def _create_dev_card_deck(self) -> List[str]:
        """Create the development card deck."""
        deck = (
            ['knight'] * 14 +
            ['victory_point'] * 5 +
            ['road_building'] * 2 +
            ['year_of_plenty'] * 2 +
            ['monopoly'] * 2
        )
        random.shuffle(deck)
        return deck

    def add_player(self, player_id: str, color: str) -> None:
        """Add a player to the game."""
        self.game_state['players'][player_id] = {
            'color': color,
            'resources': {r: 0 for r in self.RESOURCES},
            'settlements': [],
            'cities': [],
            'roads': [],
            'development_cards': [],
            'knights_played': 0,
            'victory_points': 0,
        }
        self.game_state['player_order'].append(player_id)

    def get_legal_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get all legal actions for the player."""
        actions = []
        phase = self.game_state.get('phase', 'setup')

        if phase == 'setup':
            actions.extend(self._get_setup_actions(player_id))
        elif phase == 'main':
            if not self.game_state.get('dice_rolled', False):
                actions.append({'type': 'roll_dice'})
            else:
                actions.extend(self._get_build_actions(player_id))
                actions.extend(self._get_trade_actions(player_id))
                actions.append({'type': 'end_turn'})
        elif phase == 'robber':
            actions.extend(self._get_robber_actions(player_id))

        return actions

    def _get_setup_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get legal setup phase actions."""
        player = self.game_state['players'].get(player_id, {})
        settlements = player.get('settlements', [])
        roads = player.get('roads', [])

        actions = []

        # During setup, place settlement then road
        setup_round = self.game_state.get('setup_round', 1)
        expected_settlements = setup_round
        expected_roads = len(settlements)

        if len(settlements) < expected_settlements:
            # Can place settlement at any empty vertex
            for i, vertex in enumerate(self.game_state['board']['vertices']):
                if vertex is None:
                    actions.append({
                        'type': 'place_settlement',
                        'position': i,
                    })
        elif len(roads) < len(settlements):
            # Can place road adjacent to last settlement
            last_settlement = settlements[-1] if settlements else 0
            for i, edge in enumerate(self.game_state['board']['edges']):
                if edge is None:
                    actions.append({
                        'type': 'build_road',
                        'edge': i,
                    })

        return actions

    def _get_build_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get legal building actions based on resources."""
        actions = []
        player = self.game_state['players'].get(player_id, {})
        resources = player.get('resources', {})

        # Check if can afford each building type
        for building, cost in self.COSTS.items():
            can_afford = all(resources.get(r, 0) >= amt for r, amt in cost.items())
            if can_afford:
                if building == 'road':
                    for i, edge in enumerate(self.game_state['board']['edges']):
                        if edge is None:
                            actions.append({'type': 'build_road', 'edge': i})
                elif building == 'settlement':
                    for i, vertex in enumerate(self.game_state['board']['vertices']):
                        if vertex is None:
                            actions.append({'type': 'build_settlement', 'position': i})
                elif building == 'city':
                    for pos in player.get('settlements', []):
                        actions.append({'type': 'build_city', 'position': pos})
                elif building == 'development_card':
                    if self.game_state.get('development_cards_deck'):
                        actions.append({'type': 'buy_development_card'})

        return actions

    def _get_trade_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get legal trading actions."""
        actions = []
        player = self.game_state['players'].get(player_id, {})
        resources = player.get('resources', {})

        # Bank trade (4:1 by default)
        for give_resource in self.RESOURCES:
            if resources.get(give_resource, 0) >= 4:
                for get_resource in self.RESOURCES:
                    if give_resource != get_resource:
                        actions.append({
                            'type': 'bank_trade',
                            'give': {give_resource: 4},
                            'get': {get_resource: 1},
                        })

        return actions

    def _get_robber_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get robber placement actions."""
        actions = []
        current_robber = self.game_state.get('robber_hex')

        for hex_data in self.game_state['board']['hexes']:
            if hex_data['id'] != current_robber:
                actions.append({
                    'type': 'move_robber',
                    'hex': hex_data['id'],
                })

        return actions

    def apply_action(self, player_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action and return result."""
        action_type = action.get('type')

        if action_type == 'roll_dice':
            return self._roll_dice(player_id)
        elif action_type == 'place_settlement':
            return self._place_settlement(player_id, action['position'], is_setup=True)
        elif action_type == 'build_settlement':
            return self._place_settlement(player_id, action['position'], is_setup=False)
        elif action_type == 'build_road':
            return self._build_road(player_id, action['edge'])
        elif action_type == 'build_city':
            return self._build_city(player_id, action['position'])
        elif action_type == 'buy_development_card':
            return self._buy_development_card(player_id)
        elif action_type == 'bank_trade':
            return self._bank_trade(player_id, action['give'], action['get'])
        elif action_type == 'move_robber':
            return self._move_robber(player_id, action['hex'])
        elif action_type == 'end_turn':
            return self._end_turn()

        raise ValueError(f"Unknown action type: {action_type}")

    def _roll_dice(self, player_id: str) -> Dict[str, Any]:
        """Roll dice and distribute resources."""
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        total = die1 + die2

        self.game_state['dice_rolled'] = True

        if total == 7:
            self.game_state['phase'] = 'robber'
            return {
                'dice': [die1, die2],
                'total': total,
                'robber_activated': True
            }

        # Distribute resources based on dice roll
        resources_distributed = {}
        for hex_data in self.game_state['board']['hexes']:
            if hex_data.get('number') == total and not hex_data.get('has_robber'):
                resource = hex_data['resource']
                if resource != 'desert':
                    # Would distribute to adjacent settlements/cities
                    # Simplified: skip detailed adjacency check
                    pass

        return {
            'dice': [die1, die2],
            'total': total,
            'resources_distributed': resources_distributed
        }

    def _place_settlement(self, player_id: str, position: int, is_setup: bool) -> Dict[str, Any]:
        """Place a settlement."""
        player = self.game_state['players'].get(player_id, {})

        if not is_setup:
            # Deduct resources
            for resource, amount in self.COSTS['settlement'].items():
                player['resources'][resource] -= amount

        self.game_state['board']['vertices'][position] = {
            'player': player_id,
            'type': 'settlement'
        }
        player.setdefault('settlements', []).append(position)
        player['victory_points'] = player.get('victory_points', 0) + 1

        return {'success': True, 'type': 'settlement', 'position': position}

    def _build_road(self, player_id: str, edge: int) -> Dict[str, Any]:
        """Build a road."""
        player = self.game_state['players'].get(player_id, {})

        if self.game_state.get('phase') != 'setup':
            for resource, amount in self.COSTS['road'].items():
                player['resources'][resource] -= amount

        self.game_state['board']['edges'][edge] = {'player': player_id}
        player.setdefault('roads', []).append(edge)

        return {'success': True, 'type': 'road', 'edge': edge}

    def _build_city(self, player_id: str, position: int) -> Dict[str, Any]:
        """Upgrade settlement to city."""
        player = self.game_state['players'].get(player_id, {})

        for resource, amount in self.COSTS['city'].items():
            player['resources'][resource] -= amount

        self.game_state['board']['vertices'][position]['type'] = 'city'
        player['settlements'].remove(position)
        player.setdefault('cities', []).append(position)
        player['victory_points'] += 1  # Cities worth 2, already had 1 for settlement

        return {'success': True, 'type': 'city', 'position': position}

    def _buy_development_card(self, player_id: str) -> Dict[str, Any]:
        """Buy a development card."""
        player = self.game_state['players'].get(player_id, {})
        deck = self.game_state.get('development_cards_deck', [])

        if not deck:
            raise ValueError("No development cards remaining")

        for resource, amount in self.COSTS['development_card'].items():
            player['resources'][resource] -= amount

        card = deck.pop()
        player.setdefault('development_cards', []).append(card)

        if card == 'victory_point':
            player['victory_points'] += 1

        return {'success': True, 'card': card}

    def _bank_trade(self, player_id: str, give: Dict, get: Dict) -> Dict[str, Any]:
        """Trade with the bank."""
        player = self.game_state['players'].get(player_id, {})

        for resource, amount in give.items():
            player['resources'][resource] -= amount

        for resource, amount in get.items():
            player['resources'][resource] = player['resources'].get(resource, 0) + amount

        return {'success': True, 'gave': give, 'received': get}

    def _move_robber(self, player_id: str, hex_id: int) -> Dict[str, Any]:
        """Move the robber to a new hex."""
        # Remove robber from old location
        for hex_data in self.game_state['board']['hexes']:
            hex_data['has_robber'] = False

        # Place robber on new hex
        self.game_state['board']['hexes'][hex_id]['has_robber'] = True
        self.game_state['robber_hex'] = hex_id
        self.game_state['phase'] = 'main'

        return {'success': True, 'robber_hex': hex_id}

    def _end_turn(self) -> Dict[str, Any]:
        """End current player's turn."""
        self.game_state['dice_rolled'] = False
        player_order = self.game_state.get('player_order', [])
        current_idx = self.game_state.get('current_player_index', 0)

        self.game_state['current_player_index'] = (current_idx + 1) % len(player_order)
        self.game_state['turn_number'] = self.game_state.get('turn_number', 0) + 1

        return {'success': True, 'next_player': player_order[self.game_state['current_player_index']]}

    def check_winner(self) -> Optional[str]:
        """Check if any player has won."""
        for player_id, player_data in self.game_state.get('players', {}).items():
            if player_data.get('victory_points', 0) >= self.VICTORY_POINTS_TO_WIN:
                return player_id
        return None

    def get_current_player(self) -> str:
        """Get ID of current player."""
        player_order = self.game_state.get('player_order', [])
        if not player_order:
            return ''
        current_idx = self.game_state.get('current_player_index', 0)
        return player_order[current_idx]

    def validate_state(self) -> bool:
        """Validate game state."""
        players = self.game_state.get('players', {})
        return len(players) >= self.min_players
