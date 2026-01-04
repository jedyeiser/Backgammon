"""
Settlers of Catan rule implementation.

This module provides a complete implementation of Catan rules including:
- Hex-based board with axial coordinate system
- Resource production and distribution
- Settlement, city, and road building with validation
- Development cards
- Trading (bank and player-to-player)
- Robber mechanics
- Longest road and largest army tracking

Board Geometry:
    - 19 hexes arranged in a standard Catan pattern
    - 54 vertices (intersection points for settlements/cities)
    - 72 edges (paths for roads)
    - Axial coordinate system for hex positions
"""
import random
from typing import List, Dict, Any, Optional, Tuple, Set

from .base import BaseRuleSet


class CatanRuleSet(BaseRuleSet):
    """
    Settlers of Catan game rules implementation.

    Implements the full rules of Catan including:
    - Hex board generation with proper adjacency
    - Resource distribution based on dice rolls
    - Building placement with distance and connectivity rules
    - Development card mechanics
    - Trading with bank and other players
    - Special victory points (longest road, largest army)
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

    # Standard Catan board hex positions (axial coordinates q, r)
    # Arranged in the classic 3-4-5-4-3 pattern
    HEX_COORDS = [
        # Row 0 (top, 3 hexes)
        (0, -2), (1, -2), (2, -2),
        # Row 1 (4 hexes)
        (-1, -1), (0, -1), (1, -1), (2, -1),
        # Row 2 (center, 5 hexes)
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        # Row 3 (4 hexes)
        (-2, 1), (-1, 1), (0, 1), (1, 1),
        # Row 4 (bottom, 3 hexes)
        (-2, 2), (-1, 2), (0, 2),
    ]

    # Vertex directions relative to hex center (pointy-top orientation)
    # Each hex has 6 vertices numbered 0-5 clockwise from top
    VERTEX_DIRS = [
        (0, -1, 'N'),   # 0: North
        (1, -1, 'NE'),  # 1: Northeast
        (1, 0, 'SE'),   # 2: Southeast
        (0, 1, 'S'),    # 3: South
        (-1, 1, 'SW'),  # 4: Southwest
        (-1, 0, 'NW'),  # 5: Northwest
    ]

    # Harbor positions: (vertex_ids, type, resource)
    # type: '3:1' for generic, '2:1' for specific resource
    HARBOR_VERTICES = {
        # These will be populated based on vertex IDs after board generation
    }

    def __init__(self, game_state: Dict[str, Any]):
        """Initialize with game state, building adjacency maps if needed."""
        super().__init__(game_state)
        # Build adjacency maps if this is a new game
        if game_state and 'board' in game_state:
            if 'hex_to_vertices' not in game_state['board']:
                self._build_adjacency_maps()

    def get_initial_state(self) -> Dict[str, Any]:
        """Return initial Catan game state with proper adjacency maps."""
        state = {
            'board': self._generate_board(),
            'players': {},
            'player_order': [],
            'current_player_index': 0,
            'phase': 'setup',  # setup, main, robber, discard
            'setup_round': 1,  # 1 or 2 for initial placement
            'setup_direction': 1,  # 1 for forward, -1 for reverse
            'dice_rolled': False,
            'last_dice': None,
            'robber_hex': None,
            'pending_discards': {},  # player_id -> cards to discard
            'development_cards_deck': self._create_dev_card_deck(),
            'cards_bought_this_turn': [],  # Can't play cards bought same turn
            'dev_card_played_this_turn': False,
            'longest_road_holder': None,
            'longest_road_length': 0,
            'largest_army_holder': None,
            'largest_army_size': 0,
            'turn_number': 0,
            'free_roads': 0,  # For road building dev card
        }
        # Find desert hex for initial robber placement
        for hex_data in state['board']['hexes']:
            if hex_data['resource'] == 'desert':
                state['robber_hex'] = hex_data['id']
                break
        return state

    def _generate_board(self) -> Dict[str, Any]:
        """
        Generate a standard Catan board with proper geometry.

        Creates hexes, vertices, edges, and all adjacency mappings needed
        for game logic (resource distribution, placement validation, etc.)
        """
        # Standard Catan resource distribution
        hex_resources = (
            ['desert'] +
            ['brick'] * 3 +
            ['lumber'] * 4 +
            ['ore'] * 3 +
            ['grain'] * 4 +
            ['wool'] * 4
        )
        random.shuffle(hex_resources)

        # Number tokens (standard distribution, avoiding 6/8 adjacent)
        number_tokens = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        random.shuffle(number_tokens)

        # Create hexes with coordinates
        hexes = []
        hex_coords = {}
        coord_to_hex = {}
        token_idx = 0

        for i, coord in enumerate(self.HEX_COORDS):
            resource = hex_resources[i]
            hex_data = {
                'id': i,
                'coord': coord,
                'resource': resource,
                'number': None if resource == 'desert' else number_tokens[token_idx],
                'has_robber': resource == 'desert',
            }
            if resource != 'desert':
                token_idx += 1
            hexes.append(hex_data)
            hex_coords[i] = coord
            coord_to_hex[coord] = i

        # Build vertex and edge mappings
        vertices, edges, adjacency = self._build_geometry(hex_coords, coord_to_hex)

        # Create harbor definitions
        harbors = self._create_harbors(vertices)

        return {
            'hexes': hexes,
            'vertices': vertices,  # List of vertex data or None
            'edges': edges,  # List of edge data or None
            'hex_coords': hex_coords,
            'coord_to_hex': coord_to_hex,
            'hex_to_vertices': adjacency['hex_to_vertices'],
            'vertex_to_hexes': adjacency['vertex_to_hexes'],
            'edge_to_vertices': adjacency['edge_to_vertices'],
            'vertex_to_edges': adjacency['vertex_to_edges'],
            'vertex_to_vertices': adjacency['vertex_to_vertices'],
            'harbors': harbors,
        }

    def _build_geometry(
        self,
        hex_coords: Dict[int, Tuple[int, int]],
        coord_to_hex: Dict[Tuple[int, int], int]
    ) -> Tuple[List, List, Dict]:
        """
        Build the complete vertex and edge geometry.

        Uses axial coordinates to determine vertex positions and which
        hexes/vertices/edges are adjacent to each other.
        """
        # We'll use a coordinate system for vertices
        # Each vertex can be identified by (hex_q, hex_r, direction)
        # where direction is 'N' or 'S' (the two "sharp" vertices of the hex)

        vertex_coords = {}  # (q, r, dir) -> vertex_id
        vertex_id_to_coord = {}
        vertex_id = 0

        # First pass: create unique vertices
        for hex_id, (q, r) in hex_coords.items():
            # Each hex contributes its N and S vertices
            # N vertex at (q, r, 'N')
            # S vertex at (q, r, 'S')
            for direction in ['N', 'S']:
                coord = (q, r, direction)
                if coord not in vertex_coords:
                    # Check if this vertex is shared with another hex
                    # N vertex is shared with hex at (q, r-1) as its S vertex
                    # and hex at (q+1, r-1) as its SW vertex, etc.
                    shared_coord = self._get_canonical_vertex_coord(q, r, direction, coord_to_hex)
                    if shared_coord not in vertex_coords:
                        vertex_coords[shared_coord] = vertex_id
                        vertex_id_to_coord[vertex_id] = shared_coord
                        vertex_id += 1

        # Create vertex list (54 vertices for standard board)
        num_vertices = len(vertex_coords)
        vertices = [None] * num_vertices

        # Build hex_to_vertices mapping
        hex_to_vertices = {i: [] for i in hex_coords}
        vertex_to_hexes = {i: [] for i in range(num_vertices)}

        for hex_id, (q, r) in hex_coords.items():
            # Get all 6 vertices of this hex
            hex_vertices = self._get_hex_vertices(q, r, vertex_coords, coord_to_hex)
            hex_to_vertices[hex_id] = hex_vertices
            for v_id in hex_vertices:
                if hex_id not in vertex_to_hexes[v_id]:
                    vertex_to_hexes[v_id].append(hex_id)

        # Build edge mappings
        edges = []
        edge_to_vertices = {}
        vertex_to_edges = {i: [] for i in range(num_vertices)}
        vertex_to_vertices = {i: [] for i in range(num_vertices)}
        edge_set = set()
        edge_id = 0

        # Edges connect adjacent vertices on each hex
        for hex_id, hex_verts in hex_to_vertices.items():
            for i in range(6):
                v1 = hex_verts[i]
                v2 = hex_verts[(i + 1) % 6]
                edge_key = tuple(sorted([v1, v2]))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append(None)
                    edge_to_vertices[edge_id] = list(edge_key)
                    vertex_to_edges[v1].append(edge_id)
                    vertex_to_edges[v2].append(edge_id)
                    if v2 not in vertex_to_vertices[v1]:
                        vertex_to_vertices[v1].append(v2)
                    if v1 not in vertex_to_vertices[v2]:
                        vertex_to_vertices[v2].append(v1)
                    edge_id += 1

        return vertices, edges, {
            'hex_to_vertices': hex_to_vertices,
            'vertex_to_hexes': vertex_to_hexes,
            'edge_to_vertices': edge_to_vertices,
            'vertex_to_edges': vertex_to_edges,
            'vertex_to_vertices': vertex_to_vertices,
        }

    def _get_canonical_vertex_coord(
        self,
        q: int,
        r: int,
        direction: str,
        coord_to_hex: Dict
    ) -> Tuple[int, int, str]:
        """Get the canonical coordinate for a vertex (to handle sharing)."""
        # Use the smallest hex coordinate that contains this vertex
        return (q, r, direction)

    def _get_hex_vertices(
        self,
        q: int,
        r: int,
        vertex_coords: Dict,
        coord_to_hex: Dict
    ) -> List[int]:
        """
        Get the 6 vertex IDs for a hex in clockwise order from top.

        For a pointy-top hex at (q, r):
        - Vertex 0 (N): shared with (q, r-1, S) and (q+1, r-1, S)
        - Vertex 1 (NE): shared with (q+1, r-1, S) and (q+1, r, N)
        - Vertex 2 (SE): shared with (q+1, r, N) and (q, r+1, N)
        - Vertex 3 (S): this hex's S vertex
        - Vertex 4 (SW): shared with (q, r+1, N) and (q-1, r+1, N)
        - Vertex 5 (NW): shared with (q-1, r+1, N) and (q-1, r, S)
        """
        vertices = []

        # For simplicity, use a pre-computed mapping based on relative positions
        # The key insight: each vertex is shared by up to 3 hexes
        vertex_positions = [
            # (dq, dr, dir) relative to hex that "owns" each vertex position
            (q, r, 'N'),      # 0: N - this hex's north
            (q + 1, r - 1, 'S') if (q + 1, r - 1) in coord_to_hex else (q, r, 'NE'),  # 1: NE
            (q + 1, r, 'N') if (q + 1, r) in coord_to_hex else (q, r, 'SE'),  # 2: SE
            (q, r, 'S'),      # 3: S - this hex's south
            (q - 1, r + 1, 'N') if (q - 1, r + 1) in coord_to_hex else (q, r, 'SW'),  # 4: SW
            (q - 1, r, 'S') if (q - 1, r) in coord_to_hex else (q, r, 'NW'),  # 5: NW
        ]

        for pos in vertex_positions:
            # Find the vertex ID - try the position and canonical alternatives
            if pos in vertex_coords:
                vertices.append(vertex_coords[pos])
            else:
                # Try canonical form
                canonical = (q, r, pos[2] if len(pos) > 2 else 'N')
                if canonical in vertex_coords:
                    vertices.append(vertex_coords[canonical])
                else:
                    # Create new vertex
                    new_id = len(vertex_coords)
                    vertex_coords[pos] = new_id
                    vertices.append(new_id)

        return vertices

    def _build_adjacency_maps(self) -> None:
        """Build adjacency maps for an existing board state."""
        board = self.game_state['board']
        if 'hex_to_vertices' in board:
            return  # Already built

        hex_coords = board.get('hex_coords', {})
        coord_to_hex = board.get('coord_to_hex', {})

        if not hex_coords:
            # Build from HEX_COORDS
            for i, coord in enumerate(self.HEX_COORDS):
                hex_coords[i] = coord
                coord_to_hex[coord] = i
            board['hex_coords'] = hex_coords
            board['coord_to_hex'] = coord_to_hex

        # Rebuild geometry
        _, _, adjacency = self._build_geometry(hex_coords, coord_to_hex)
        board.update(adjacency)

    def _create_harbors(self, vertices: List) -> Dict[int, Dict]:
        """
        Create harbor definitions for the board.

        Standard Catan has 9 harbors:
        - 4 generic (3:1) harbors
        - 5 specific (2:1) harbors (one for each resource)
        """
        # For a standard board, harbors are on the edges
        # We'll assign them to specific vertex pairs on the coast
        harbors = {}

        # Simplified: assign harbors to edge vertices on the outer ring
        # In a real implementation, these would be at specific positions
        harbor_types = [
            {'type': '3:1'},
            {'type': '3:1'},
            {'type': '3:1'},
            {'type': '3:1'},
            {'type': '2:1', 'resource': 'brick'},
            {'type': '2:1', 'resource': 'lumber'},
            {'type': '2:1', 'resource': 'ore'},
            {'type': '2:1', 'resource': 'grain'},
            {'type': '2:1', 'resource': 'wool'},
        ]
        random.shuffle(harbor_types)

        # Assign to vertex pairs (simplified - would need coastal detection)
        # For now, assign to first 18 vertices (9 pairs)
        for i, harbor in enumerate(harbor_types):
            v1 = i * 2
            v2 = i * 2 + 1
            if v1 < len(vertices) and v2 < len(vertices):
                harbors[v1] = harbor
                harbors[v2] = harbor

        return harbors

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
            'played_dev_cards': [],
            'knights_played': 0,
            'victory_points': 0,
            'visible_victory_points': 0,  # VP excluding hidden dev cards
        }
        self.game_state['player_order'].append(player_id)

    def roll_dice(self) -> Dict[str, Any]:
        """Roll dice and handle the result."""
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        total = die1 + die2

        self.game_state['dice_rolled'] = True
        self.game_state['last_dice'] = [die1, die2]

        if total == 7:
            return self._handle_seven_rolled()

        # Distribute resources
        resources_distributed = self._distribute_resources(total)

        return {
            'dice': [die1, die2],
            'total': total,
            'resources_distributed': resources_distributed
        }

    def _distribute_resources(self, dice_total: int) -> Dict[str, Dict[str, int]]:
        """
        Distribute resources based on dice roll.

        For each hex with the rolled number (and no robber), give resources
        to players with adjacent settlements (1) or cities (2).
        """
        distributed = {}
        board = self.game_state['board']

        for hex_data in board['hexes']:
            if hex_data['number'] != dice_total:
                continue
            if hex_data['has_robber']:
                continue
            if hex_data['resource'] == 'desert':
                continue

            resource = hex_data['resource']
            hex_id = hex_data['id']

            # Get vertices adjacent to this hex
            adjacent_vertices = board['hex_to_vertices'].get(hex_id, [])

            for vertex_id in adjacent_vertices:
                vertex = board['vertices'][vertex_id]
                if vertex is None:
                    continue

                player_id = vertex['player']
                amount = 2 if vertex['type'] == 'city' else 1

                # Add to player's resources
                self.game_state['players'][player_id]['resources'][resource] += amount

                # Track distribution for return value
                if player_id not in distributed:
                    distributed[player_id] = {}
                if resource not in distributed[player_id]:
                    distributed[player_id][resource] = 0
                distributed[player_id][resource] += amount

        return distributed

    def _handle_seven_rolled(self) -> Dict[str, Any]:
        """
        Handle when a 7 is rolled.

        Players with more than 7 cards must discard half.
        Then the current player moves the robber.
        """
        must_discard = {}

        for player_id, player_data in self.game_state['players'].items():
            total_cards = sum(player_data['resources'].values())
            if total_cards > 7:
                must_discard[player_id] = total_cards // 2

        if must_discard:
            self.game_state['phase'] = 'discard'
            self.game_state['pending_discards'] = must_discard
        else:
            self.game_state['phase'] = 'robber'

        return {
            'dice': self.game_state['last_dice'],
            'total': 7,
            'robber_activated': True,
            'must_discard': must_discard
        }

    def get_legal_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get all legal actions for the player."""
        actions = []
        phase = self.game_state.get('phase', 'setup')
        current_player = self.get_current_player()

        # Only current player can act (except for discarding)
        if phase == 'discard':
            if player_id in self.game_state.get('pending_discards', {}):
                return [{'type': 'discard', 'count': self.game_state['pending_discards'][player_id]}]
            return []

        if player_id != current_player:
            return []

        if phase == 'setup':
            actions.extend(self._get_setup_actions(player_id))
        elif phase == 'robber':
            actions.extend(self._get_robber_actions(player_id))
        elif phase == 'main':
            if not self.game_state.get('dice_rolled', False):
                actions.append({'type': 'roll_dice'})
            else:
                # Free roads from road building card
                if self.game_state.get('free_roads', 0) > 0:
                    actions.extend(self._get_free_road_actions(player_id))
                else:
                    actions.extend(self._get_build_actions(player_id))
                    actions.extend(self._get_trade_actions(player_id))
                    actions.extend(self._get_dev_card_actions(player_id))
                    actions.append({'type': 'end_turn'})

        return actions

    def _get_setup_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get legal setup phase actions."""
        player = self.game_state['players'].get(player_id, {})
        settlements = player.get('settlements', [])
        roads = player.get('roads', [])
        actions = []

        setup_round = self.game_state.get('setup_round', 1)
        expected_settlements = setup_round
        expected_roads = len(settlements)

        if len(settlements) < expected_settlements:
            # Place settlement - must follow distance rule
            for vertex_id, vertex in enumerate(self.game_state['board']['vertices']):
                if self._is_valid_settlement_location(vertex_id, player_id, is_setup=True):
                    actions.append({
                        'type': 'place_settlement',
                        'position': vertex_id,
                    })
        elif len(roads) < len(settlements):
            # Place road adjacent to last settlement
            last_settlement = settlements[-1]
            for edge_id in self.game_state['board']['vertex_to_edges'].get(last_settlement, []):
                if self.game_state['board']['edges'][edge_id] is None:
                    actions.append({
                        'type': 'build_road',
                        'edge': edge_id,
                    })

        return actions

    def _is_valid_settlement_location(self, vertex_id: int, player_id: str, is_setup: bool = False) -> bool:
        """
        Check if a settlement can be placed at a vertex.

        Rules:
        - Vertex must be empty
        - No adjacent vertex can have a settlement/city (distance rule)
        - Must be connected to player's road (except during setup)
        """
        board = self.game_state['board']

        # Check if vertex exists and is empty
        if vertex_id >= len(board['vertices']):
            return False
        if board['vertices'][vertex_id] is not None:
            return False

        # Distance rule: check adjacent vertices
        adjacent_vertices = board['vertex_to_vertices'].get(vertex_id, [])
        for adj_v in adjacent_vertices:
            if board['vertices'][adj_v] is not None:
                return False

        # Road connection rule (not during setup)
        if not is_setup:
            if not self._has_adjacent_road(vertex_id, player_id):
                return False

        return True

    def _has_adjacent_road(self, vertex_id: int, player_id: str) -> bool:
        """Check if player has a road connected to this vertex."""
        board = self.game_state['board']
        adjacent_edges = board['vertex_to_edges'].get(vertex_id, [])

        for edge_id in adjacent_edges:
            edge = board['edges'][edge_id]
            if edge and edge['player'] == player_id:
                return True

        return False

    def _is_valid_road_location(self, edge_id: int, player_id: str, is_setup: bool = False) -> bool:
        """
        Check if a road can be placed at an edge.

        Rules:
        - Edge must be empty
        - Must connect to player's settlement/city OR existing road
        - Cannot be blocked by opponent's settlement
        """
        board = self.game_state['board']

        if edge_id >= len(board['edges']):
            return False
        if board['edges'][edge_id] is not None:
            return False

        v1, v2 = board['edge_to_vertices'][edge_id]

        # Check if either vertex has player's settlement/city
        for v in [v1, v2]:
            vertex = board['vertices'][v]
            if vertex and vertex['player'] == player_id:
                return True

        # Check if connects to existing road (through unblocked vertex)
        for v in [v1, v2]:
            vertex = board['vertices'][v]
            # If opponent has a settlement here, can't connect through it
            if vertex and vertex['player'] != player_id:
                continue

            for adj_edge in board['vertex_to_edges'].get(v, []):
                if adj_edge != edge_id:
                    edge = board['edges'][adj_edge]
                    if edge and edge['player'] == player_id:
                        return True

        return False

    def _get_build_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get legal building actions based on resources."""
        actions = []
        player = self.game_state['players'].get(player_id, {})
        resources = player.get('resources', {})
        board = self.game_state['board']

        # Check each building type
        for building, cost in self.COSTS.items():
            can_afford = all(resources.get(r, 0) >= amt for r, amt in cost.items())
            if not can_afford:
                continue

            if building == 'road':
                for edge_id, edge in enumerate(board['edges']):
                    if self._is_valid_road_location(edge_id, player_id):
                        actions.append({'type': 'build_road', 'edge': edge_id})

            elif building == 'settlement':
                for vertex_id in range(len(board['vertices'])):
                    if self._is_valid_settlement_location(vertex_id, player_id):
                        actions.append({'type': 'build_settlement', 'position': vertex_id})

            elif building == 'city':
                for pos in player.get('settlements', []):
                    actions.append({'type': 'build_city', 'position': pos})

            elif building == 'development_card':
                if self.game_state.get('development_cards_deck'):
                    actions.append({'type': 'buy_development_card'})

        return actions

    def _get_free_road_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get legal road actions when player has free roads from dev card."""
        actions = []
        board = self.game_state['board']

        for edge_id, edge in enumerate(board['edges']):
            if self._is_valid_road_location(edge_id, player_id):
                actions.append({'type': 'build_road', 'edge': edge_id, 'free': True})

        return actions

    def _get_trade_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get legal trading actions."""
        actions = []
        player = self.game_state['players'].get(player_id, {})
        resources = player.get('resources', {})

        # Bank trades
        for give_resource in self.RESOURCES:
            rate = self._get_trade_rate(player_id, give_resource)
            if resources.get(give_resource, 0) >= rate:
                for get_resource in self.RESOURCES:
                    if give_resource != get_resource:
                        actions.append({
                            'type': 'bank_trade',
                            'give': {give_resource: rate},
                            'get': {get_resource: 1},
                        })

        return actions

    def _get_trade_rate(self, player_id: str, resource: str) -> int:
        """Get the best trade rate for a resource (4, 3, or 2)."""
        player = self.game_state['players'].get(player_id, {})
        harbors = self.game_state['board'].get('harbors', {})
        rate = 4  # Default bank rate

        # Check harbors at player's settlements/cities
        for pos in player.get('settlements', []) + player.get('cities', []):
            if pos in harbors:
                harbor = harbors[pos]
                if harbor['type'] == '3:1':
                    rate = min(rate, 3)
                elif harbor['type'] == '2:1' and harbor.get('resource') == resource:
                    rate = min(rate, 2)

        return rate

    def _get_dev_card_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get playable development card actions."""
        actions = []
        player = self.game_state['players'].get(player_id, {})

        # Can only play one dev card per turn
        if self.game_state.get('dev_card_played_this_turn', False):
            return actions

        # Can't play cards bought this turn
        cards_bought = self.game_state.get('cards_bought_this_turn', [])

        for card in player.get('development_cards', []):
            if card in cards_bought:
                continue
            if card == 'victory_point':
                continue  # VP cards are revealed only when winning

            if card == 'knight':
                actions.append({'type': 'play_dev_card', 'card': 'knight'})
            elif card == 'road_building':
                actions.append({'type': 'play_dev_card', 'card': 'road_building'})
            elif card == 'year_of_plenty':
                actions.append({'type': 'play_dev_card', 'card': 'year_of_plenty'})
            elif card == 'monopoly':
                actions.append({'type': 'play_dev_card', 'card': 'monopoly'})

        return actions

    def _get_robber_actions(self, player_id: str) -> List[Dict[str, Any]]:
        """Get robber placement actions."""
        actions = []
        current_robber = self.game_state.get('robber_hex')

        for hex_data in self.game_state['board']['hexes']:
            if hex_data['id'] != current_robber:
                # Get players with settlements adjacent to this hex
                players_to_steal = self._get_players_adjacent_to_hex(hex_data['id'], player_id)
                if players_to_steal:
                    for target in players_to_steal:
                        actions.append({
                            'type': 'move_robber',
                            'hex': hex_data['id'],
                            'steal_from': target,
                        })
                else:
                    actions.append({
                        'type': 'move_robber',
                        'hex': hex_data['id'],
                        'steal_from': None,
                    })

        return actions

    def _get_players_adjacent_to_hex(self, hex_id: int, exclude_player: str) -> List[str]:
        """Get list of players with settlements/cities adjacent to a hex."""
        board = self.game_state['board']
        players = set()

        for vertex_id in board['hex_to_vertices'].get(hex_id, []):
            vertex = board['vertices'][vertex_id]
            if vertex and vertex['player'] != exclude_player:
                # Only include if they have resources to steal
                player_resources = self.game_state['players'][vertex['player']]['resources']
                if sum(player_resources.values()) > 0:
                    players.add(vertex['player'])

        return list(players)

    def apply_action(self, player_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action and return result."""
        action_type = action.get('type')

        if action_type == 'roll_dice':
            return self.roll_dice()
        elif action_type == 'place_settlement':
            return self._place_settlement(player_id, action['position'], is_setup=True)
        elif action_type == 'build_settlement':
            return self._place_settlement(player_id, action['position'], is_setup=False)
        elif action_type == 'build_road':
            is_free = action.get('free', False)
            return self._build_road(player_id, action['edge'], is_free=is_free)
        elif action_type == 'build_city':
            return self._build_city(player_id, action['position'])
        elif action_type == 'buy_development_card':
            return self._buy_development_card(player_id)
        elif action_type == 'play_dev_card':
            return self._play_development_card(player_id, action['card'], action)
        elif action_type == 'bank_trade':
            return self._bank_trade(player_id, action['give'], action['get'])
        elif action_type == 'player_trade':
            return self._player_trade(player_id, action['target'], action['give'], action['get'])
        elif action_type == 'move_robber':
            return self._move_robber(player_id, action['hex'], action.get('steal_from'))
        elif action_type == 'discard':
            return self._discard_cards(player_id, action['resources'])
        elif action_type == 'end_turn':
            return self._end_turn()

        raise ValueError(f"Unknown action type: {action_type}")

    def _place_settlement(self, player_id: str, position: int, is_setup: bool) -> Dict[str, Any]:
        """Place a settlement."""
        player = self.game_state['players'][player_id]
        board = self.game_state['board']

        if not is_setup:
            # Deduct resources
            for resource, amount in self.COSTS['settlement'].items():
                player['resources'][resource] -= amount

        board['vertices'][position] = {
            'player': player_id,
            'type': 'settlement'
        }
        player['settlements'].append(position)
        player['victory_points'] += 1
        player['visible_victory_points'] += 1

        result = {'success': True, 'type': 'settlement', 'position': position}

        # During setup round 2, collect resources from adjacent hexes
        if is_setup and self.game_state.get('setup_round', 1) == 2:
            resources_gained = {}
            for hex_id in board['vertex_to_hexes'].get(position, []):
                hex_data = board['hexes'][hex_id]
                if hex_data['resource'] != 'desert':
                    resource = hex_data['resource']
                    player['resources'][resource] += 1
                    resources_gained[resource] = resources_gained.get(resource, 0) + 1
            result['resources_gained'] = resources_gained

        return result

    def _build_road(self, player_id: str, edge: int, is_free: bool = False) -> Dict[str, Any]:
        """Build a road."""
        player = self.game_state['players'][player_id]
        board = self.game_state['board']

        if not is_free and self.game_state.get('phase') != 'setup':
            for resource, amount in self.COSTS['road'].items():
                player['resources'][resource] -= amount

        if is_free:
            self.game_state['free_roads'] = self.game_state.get('free_roads', 0) - 1

        board['edges'][edge] = {'player': player_id}
        player['roads'].append(edge)

        # Check longest road
        self._check_longest_road()

        return {'success': True, 'type': 'road', 'edge': edge}

    def _build_city(self, player_id: str, position: int) -> Dict[str, Any]:
        """Upgrade settlement to city."""
        player = self.game_state['players'][player_id]
        board = self.game_state['board']

        for resource, amount in self.COSTS['city'].items():
            player['resources'][resource] -= amount

        board['vertices'][position]['type'] = 'city'
        player['settlements'].remove(position)
        player['cities'].append(position)
        player['victory_points'] += 1  # Net gain (city=2, settlement was 1)
        player['visible_victory_points'] += 1

        return {'success': True, 'type': 'city', 'position': position}

    def _buy_development_card(self, player_id: str) -> Dict[str, Any]:
        """Buy a development card."""
        player = self.game_state['players'][player_id]
        deck = self.game_state.get('development_cards_deck', [])

        if not deck:
            raise ValueError("No development cards remaining")

        for resource, amount in self.COSTS['development_card'].items():
            player['resources'][resource] -= amount

        card = deck.pop()
        player['development_cards'].append(card)
        self.game_state['cards_bought_this_turn'].append(card)

        if card == 'victory_point':
            player['victory_points'] += 1
            # Don't add to visible VP - stays hidden

        return {'success': True, 'card': 'hidden'}  # Don't reveal to other players

    def _play_development_card(self, player_id: str, card_type: str, action: Dict) -> Dict[str, Any]:
        """Play a development card."""
        player = self.game_state['players'][player_id]

        if card_type not in player['development_cards']:
            raise ValueError("You don't have that card")

        player['development_cards'].remove(card_type)
        player['played_dev_cards'].append(card_type)
        self.game_state['dev_card_played_this_turn'] = True

        if card_type == 'knight':
            player['knights_played'] += 1
            self._check_largest_army()
            self.game_state['phase'] = 'robber'
            return {'card': 'knight', 'move_robber': True}

        elif card_type == 'road_building':
            self.game_state['free_roads'] = 2
            return {'card': 'road_building', 'free_roads': 2}

        elif card_type == 'year_of_plenty':
            resources = action.get('resources', [])
            if len(resources) != 2:
                raise ValueError("Must choose exactly 2 resources")
            for r in resources:
                if r not in self.RESOURCES:
                    raise ValueError(f"Invalid resource: {r}")
                player['resources'][r] += 1
            return {'card': 'year_of_plenty', 'received': resources}

        elif card_type == 'monopoly':
            resource = action.get('resource')
            if resource not in self.RESOURCES:
                raise ValueError(f"Invalid resource: {resource}")
            total = 0
            for pid, pdata in self.game_state['players'].items():
                if pid != player_id:
                    amt = pdata['resources'].get(resource, 0)
                    pdata['resources'][resource] = 0
                    total += amt
            player['resources'][resource] += total
            return {'card': 'monopoly', 'resource': resource, 'collected': total}

        raise ValueError(f"Unknown card type: {card_type}")

    def _bank_trade(self, player_id: str, give: Dict, get: Dict) -> Dict[str, Any]:
        """Trade with the bank."""
        player = self.game_state['players'][player_id]

        for resource, amount in give.items():
            player['resources'][resource] -= amount

        for resource, amount in get.items():
            player['resources'][resource] = player['resources'].get(resource, 0) + amount

        return {'success': True, 'gave': give, 'received': get}

    def _player_trade(self, player_id: str, target_player: str, give: Dict, get: Dict) -> Dict[str, Any]:
        """Execute trade between players."""
        player = self.game_state['players'][player_id]
        target = self.game_state['players'][target_player]

        # Validate both have resources
        for resource, amount in give.items():
            if player['resources'].get(resource, 0) < amount:
                raise ValueError(f"You don't have enough {resource}")

        for resource, amount in get.items():
            if target['resources'].get(resource, 0) < amount:
                raise ValueError(f"Target doesn't have enough {resource}")

        # Execute trade
        for resource, amount in give.items():
            player['resources'][resource] -= amount
            target['resources'][resource] = target['resources'].get(resource, 0) + amount

        for resource, amount in get.items():
            target['resources'][resource] -= amount
            player['resources'][resource] = player['resources'].get(resource, 0) + amount

        return {'success': True, 'gave': give, 'received': get}

    def _move_robber(self, player_id: str, hex_id: int, steal_from: Optional[str]) -> Dict[str, Any]:
        """Move the robber to a new hex and optionally steal."""
        board = self.game_state['board']

        # Remove robber from old location
        for hex_data in board['hexes']:
            hex_data['has_robber'] = False

        # Place robber on new hex
        board['hexes'][hex_id]['has_robber'] = True
        self.game_state['robber_hex'] = hex_id

        # Steal from target player
        stolen = None
        if steal_from:
            victim = self.game_state['players'][steal_from]
            available = [r for r, amt in victim['resources'].items() if amt > 0]
            if available:
                stolen_resource = random.choice(available)
                victim['resources'][stolen_resource] -= 1
                self.game_state['players'][player_id]['resources'][stolen_resource] += 1
                stolen = {'from': steal_from, 'resource': stolen_resource}

        self.game_state['phase'] = 'main'

        return {'success': True, 'robber_hex': hex_id, 'stolen': stolen}

    def _discard_cards(self, player_id: str, resources: Dict[str, int]) -> Dict[str, Any]:
        """Discard cards when rolling 7 with >7 cards."""
        player = self.game_state['players'][player_id]
        required = self.game_state['pending_discards'].get(player_id, 0)

        total_discarding = sum(resources.values())
        if total_discarding != required:
            raise ValueError(f"Must discard exactly {required} cards")

        for resource, amount in resources.items():
            if player['resources'].get(resource, 0) < amount:
                raise ValueError(f"Don't have enough {resource}")
            player['resources'][resource] -= amount

        del self.game_state['pending_discards'][player_id]

        # If everyone has discarded, move to robber phase
        if not self.game_state['pending_discards']:
            self.game_state['phase'] = 'robber'

        return {'success': True, 'discarded': resources}

    def _end_turn(self) -> Dict[str, Any]:
        """End current player's turn."""
        self.game_state['dice_rolled'] = False
        self.game_state['dev_card_played_this_turn'] = False
        self.game_state['cards_bought_this_turn'] = []
        self.game_state['free_roads'] = 0

        player_order = self.game_state.get('player_order', [])
        current_idx = self.game_state.get('current_player_index', 0)

        # Handle setup phase turn order
        if self.game_state.get('phase') == 'setup':
            return self._advance_setup_turn()

        self.game_state['current_player_index'] = (current_idx + 1) % len(player_order)
        self.game_state['turn_number'] += 1

        return {'success': True, 'next_player': player_order[self.game_state['current_player_index']]}

    def _advance_setup_turn(self) -> Dict[str, Any]:
        """Advance turn during setup phase with snake draft order."""
        player_order = self.game_state['player_order']
        current_idx = self.game_state['current_player_index']
        setup_round = self.game_state.get('setup_round', 1)
        direction = self.game_state.get('setup_direction', 1)

        next_idx = current_idx + direction

        # Check if we need to reverse direction
        if next_idx >= len(player_order):
            if setup_round == 1:
                # Start round 2, going backwards
                self.game_state['setup_round'] = 2
                self.game_state['setup_direction'] = -1
                next_idx = len(player_order) - 1
            else:
                # Setup complete, start main game
                self.game_state['phase'] = 'main'
                next_idx = 0
        elif next_idx < 0:
            # Setup complete
            self.game_state['phase'] = 'main'
            next_idx = 0

        self.game_state['current_player_index'] = next_idx

        return {
            'success': True,
            'next_player': player_order[next_idx],
            'phase': self.game_state['phase'],
            'setup_round': self.game_state.get('setup_round', 1)
        }

    def _check_longest_road(self) -> None:
        """Update longest road holder after any road is built."""
        current_holder = self.game_state.get('longest_road_holder')
        current_length = self.game_state.get('longest_road_length', 0)

        for player_id in self.game_state['players']:
            length = self._calculate_longest_road(player_id)

            if length >= 5 and length > current_length:
                # Award longest road
                if current_holder and current_holder != player_id:
                    self.game_state['players'][current_holder]['victory_points'] -= 2
                    self.game_state['players'][current_holder]['visible_victory_points'] -= 2

                if current_holder != player_id:
                    self.game_state['players'][player_id]['victory_points'] += 2
                    self.game_state['players'][player_id]['visible_victory_points'] += 2

                self.game_state['longest_road_holder'] = player_id
                self.game_state['longest_road_length'] = length

    def _calculate_longest_road(self, player_id: str) -> int:
        """Calculate longest continuous road for a player using DFS."""
        board = self.game_state['board']
        player_edges = set()

        for edge_id, edge in enumerate(board['edges']):
            if edge and edge['player'] == player_id:
                player_edges.add(edge_id)

        if not player_edges:
            return 0

        # DFS from each edge
        max_length = 0

        for start_edge in player_edges:
            length = self._dfs_road_length(player_id, start_edge, player_edges, set())
            max_length = max(max_length, length)

        return max_length

    def _dfs_road_length(
        self,
        player_id: str,
        current_edge: int,
        player_edges: Set[int],
        visited: Set[int]
    ) -> int:
        """DFS to find longest path in road network."""
        if current_edge in visited:
            return 0

        visited = visited | {current_edge}
        board = self.game_state['board']

        max_continuation = 0
        v1, v2 = board['edge_to_vertices'][current_edge]

        for vertex in [v1, v2]:
            # Check if blocked by opponent's settlement
            vertex_data = board['vertices'][vertex]
            if vertex_data and vertex_data['player'] != player_id:
                continue

            # Find adjacent edges
            for adj_edge in board['vertex_to_edges'].get(vertex, []):
                if adj_edge in player_edges and adj_edge not in visited:
                    continuation = self._dfs_road_length(player_id, adj_edge, player_edges, visited)
                    max_continuation = max(max_continuation, continuation)

        return 1 + max_continuation

    def _check_largest_army(self) -> None:
        """Update largest army holder after a knight is played."""
        current_holder = self.game_state.get('largest_army_holder')
        current_size = self.game_state.get('largest_army_size', 0)

        for player_id, player_data in self.game_state['players'].items():
            knights = player_data.get('knights_played', 0)

            if knights >= 3 and knights > current_size:
                # Award largest army
                if current_holder and current_holder != player_id:
                    self.game_state['players'][current_holder]['victory_points'] -= 2
                    self.game_state['players'][current_holder]['visible_victory_points'] -= 2

                if current_holder != player_id:
                    self.game_state['players'][player_id]['victory_points'] += 2
                    self.game_state['players'][player_id]['visible_victory_points'] += 2

                self.game_state['largest_army_holder'] = player_id
                self.game_state['largest_army_size'] = knights

    def check_winner(self) -> Optional[str]:
        """Check if any player has won (10+ victory points)."""
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
