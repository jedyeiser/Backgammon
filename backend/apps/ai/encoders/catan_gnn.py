"""
Graph Neural Network encoder for Catan.

Represents the Catan board as a heterogeneous graph with:
- Hex nodes: Resource production points
- Vertex nodes: Settlement/city locations
- Edge nodes: Road locations

This graph structure naturally handles variable board sizes
and captures the spatial relationships that matter for strategy.

The encoding provides:
- Fixed-size output regardless of board size
- Preservation of spatial relationships
- Support for different board configurations
"""
from typing import Any, Dict, List, Optional, Tuple
import copy

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseEncoder


class CatanGraphEncoder(BaseEncoder):
    """
    Encode Catan game state as graph data for GNN processing.

    Creates a heterogeneous graph representation with three node types:
    - hex: Resource tiles (19 for standard board)
    - vertex: Settlement locations (54 for standard)
    - edge: Road locations (72 for standard)

    Features include:
    - Hex: Resource type, number token, production probability, robber
    - Vertex: Owner, building type, harbor access
    - Edge: Owner, road presence

    The encoder also creates edge indices for message passing:
    - hex_to_vertex: Which hexes connect to which vertices
    - vertex_to_vertex: Which vertices are adjacent
    - vertex_to_edge: Which vertices connect to which edges

    Example:
        encoder = CatanGraphEncoder()
        graph_data = encoder.encode_graph(game_state, player_id='player_1')

        # graph_data contains:
        # - hex_features: [num_hexes, hex_feature_dim]
        # - vertex_features: [num_vertices, vertex_feature_dim]
        # - edge_features: [num_edges, edge_feature_dim]
        # - adjacency: dict of edge index tensors
    """

    # Feature dimensions
    HEX_FEATURE_DIM = 13  # Resource one-hot (6) + number (1) + prob (1) + robber (1) + production (4)
    VERTEX_FEATURE_DIM = 12  # Owner one-hot (5) + building (3) + harbor (3) + position (1)
    EDGE_FEATURE_DIM = 6  # Owner one-hot (5) + has_road (1)

    # Resource types
    RESOURCES = ['brick', 'lumber', 'ore', 'grain', 'wool', 'desert']

    # Dice roll probabilities
    DICE_PROBS = {
        2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
        8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
    }

    def __init__(
        self,
        max_players: int = 4,
        include_opponent_view: bool = True,
    ):
        """
        Initialize the Catan graph encoder.

        Args:
            max_players: Maximum number of players to encode.
            include_opponent_view: If True, encode from player's perspective.
        """
        self.max_players = max_players
        self.include_opponent_view = include_opponent_view

    @property
    def input_size(self) -> int:
        """Return approximate feature size (actual size is variable)."""
        # This is for compatibility - GNNs handle variable sizes
        return self.HEX_FEATURE_DIM + self.VERTEX_FEATURE_DIM + self.EDGE_FEATURE_DIM

    def encode(
        self,
        game_state: Dict[str, Any],
        player: Optional[str] = None,
    ) -> 'np.ndarray':
        """
        Encode game state as a flattened feature vector.

        Note: For GNNs, use encode_graph() instead for full graph structure.
        This method provides a pooled summary for compatibility.

        Args:
            game_state: Catan game state.
            player: Player perspective.

        Returns:
            Flattened feature array (pooled graph features).
        """
        graph_data = self.encode_graph(game_state, player)

        # Pool graph features into fixed-size vector
        pooled = np.concatenate([
            np.mean(graph_data['hex_features'], axis=0),
            np.max(graph_data['hex_features'], axis=0),
            np.mean(graph_data['vertex_features'], axis=0),
            np.max(graph_data['vertex_features'], axis=0),
            np.mean(graph_data['edge_features'], axis=0),
            np.max(graph_data['edge_features'], axis=0),
            graph_data['global_features'],
        ])

        return pooled

    def encode_tensor(
        self,
        game_state: Dict[str, Any],
        player: Optional[str] = None,
    ) -> 'torch.Tensor':
        """
        Encode as PyTorch tensor.

        For GNNs, use encode_graph_tensors() for full graph structure.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for tensor encoding")

        features = self.encode(game_state, player)
        return torch.from_numpy(features).float()

    def encode_graph(
        self,
        game_state: Dict[str, Any],
        player: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Encode game state as graph data.

        Returns complete graph structure for GNN processing.

        Args:
            game_state: Catan game state dictionary.
            player: Player ID for perspective (first in order if None).

        Returns:
            Dictionary with:
            - hex_features: numpy array [num_hexes, hex_dim]
            - vertex_features: numpy array [num_vertices, vertex_dim]
            - edge_features: numpy array [num_edges, edge_dim]
            - adjacency: dict of edge index arrays
            - global_features: numpy array of global state
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy required for encoding")

        board = game_state.get('board', {})
        players = game_state.get('players', {})
        player_order = game_state.get('player_order', list(players.keys()))

        # Determine player perspective
        if player is None and player_order:
            player = player_order[0]

        # Get player index for relative encoding
        player_idx = player_order.index(player) if player in player_order else 0

        # Encode each node type
        hex_features = self._encode_hexes(board, game_state)
        vertex_features = self._encode_vertices(board, players, player_order, player_idx)
        edge_features = self._encode_edges(board, players, player_order, player_idx)

        # Build adjacency
        adjacency = self._build_adjacency(board)

        # Global features
        global_features = self._encode_global(game_state, player, player_order, player_idx)

        return {
            'hex_features': hex_features,
            'vertex_features': vertex_features,
            'edge_features': edge_features,
            'adjacency': adjacency,
            'global_features': global_features,
            'num_hexes': len(board.get('hexes', [])),
            'num_vertices': len(board.get('vertices', [])),
            'num_edges': len(board.get('edges', [])),
        }

    def encode_graph_tensors(
        self,
        game_state: Dict[str, Any],
        player: Optional[str] = None,
    ) -> Dict[str, 'torch.Tensor']:
        """
        Encode as PyTorch tensors for GNN processing.

        Args:
            game_state: Catan game state.
            player: Player perspective.

        Returns:
            Dictionary with tensor versions of graph data.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        graph_data = self.encode_graph(game_state, player)

        return {
            'hex_features': torch.from_numpy(graph_data['hex_features']).float(),
            'vertex_features': torch.from_numpy(graph_data['vertex_features']).float(),
            'edge_features': torch.from_numpy(graph_data['edge_features']).float(),
            'hex_to_vertex': torch.from_numpy(graph_data['adjacency']['hex_to_vertex']).long(),
            'vertex_to_vertex': torch.from_numpy(graph_data['adjacency']['vertex_to_vertex']).long(),
            'vertex_to_edge': torch.from_numpy(graph_data['adjacency']['vertex_to_edge']).long(),
            'global_features': torch.from_numpy(graph_data['global_features']).float(),
        }

    def _encode_hexes(
        self,
        board: Dict[str, Any],
        game_state: Dict[str, Any],
    ) -> 'np.ndarray':
        """Encode hex node features."""
        hexes = board.get('hexes', [])
        num_hexes = len(hexes)

        features = np.zeros((num_hexes, self.HEX_FEATURE_DIM), dtype=np.float32)

        for i, hex_data in enumerate(hexes):
            # Resource type one-hot (6)
            resource = hex_data.get('resource', 'desert')
            if resource in self.RESOURCES:
                features[i, self.RESOURCES.index(resource)] = 1.0

            # Number token (normalized to 0-1)
            number = hex_data.get('number')
            if number:
                features[i, 6] = number / 12.0

            # Production probability
            if number and number in self.DICE_PROBS:
                features[i, 7] = self.DICE_PROBS[number] * 36  # Scale to 0-6

            # Has robber
            features[i, 8] = 1.0 if hex_data.get('has_robber', False) else 0.0

            # Expected production by resource type (one-hot style)
            if resource != 'desert' and number in self.DICE_PROBS:
                prob = self.DICE_PROBS[number]
                res_idx = self.RESOURCES.index(resource)
                if res_idx < 4:  # Exclude desert
                    features[i, 9 + res_idx] = prob * 36

        return features

    def _encode_vertices(
        self,
        board: Dict[str, Any],
        players: Dict[str, Any],
        player_order: List[str],
        player_idx: int,
    ) -> 'np.ndarray':
        """Encode vertex node features."""
        vertices = board.get('vertices', [])
        num_vertices = len(vertices)
        harbors = board.get('harbors', {})

        features = np.zeros((num_vertices, self.VERTEX_FEATURE_DIM), dtype=np.float32)

        for i, vertex in enumerate(vertices):
            if vertex is None:
                # Empty vertex
                features[i, 0] = 1.0  # "No owner" one-hot position
                continue

            vertex_player = vertex.get('player')
            building_type = vertex.get('type', 'settlement')

            # Owner one-hot (relative to current player)
            # 0 = no owner, 1 = current player, 2-4 = other players
            if vertex_player:
                try:
                    owner_idx = player_order.index(vertex_player)
                    relative_idx = ((owner_idx - player_idx) % len(player_order)) + 1
                    if relative_idx < 5:
                        features[i, relative_idx] = 1.0
                except ValueError:
                    features[i, 0] = 1.0
            else:
                features[i, 0] = 1.0

            # Building type one-hot (empty, settlement, city)
            if building_type == 'settlement':
                features[i, 6] = 1.0
            elif building_type == 'city':
                features[i, 7] = 1.0
            else:
                features[i, 5] = 1.0  # Empty

            # Harbor access (generic 3:1, specific 2:1)
            if i in harbors:
                harbor = harbors[i]
                if harbor.get('type') == '3:1':
                    features[i, 8] = 1.0
                elif harbor.get('type') == '2:1':
                    features[i, 9] = 1.0
                    # Could encode specific resource, but keeping simple

            # Normalized position (approximate center distance)
            # This helps the network understand board geometry
            features[i, 11] = i / max(num_vertices - 1, 1)

        return features

    def _encode_edges(
        self,
        board: Dict[str, Any],
        players: Dict[str, Any],
        player_order: List[str],
        player_idx: int,
    ) -> 'np.ndarray':
        """Encode edge node features."""
        edges = board.get('edges', [])
        num_edges = len(edges)

        features = np.zeros((num_edges, self.EDGE_FEATURE_DIM), dtype=np.float32)

        for i, edge in enumerate(edges):
            if edge is None:
                features[i, 0] = 1.0  # No owner
                continue

            edge_player = edge.get('player')

            # Owner one-hot (relative)
            if edge_player:
                try:
                    owner_idx = player_order.index(edge_player)
                    relative_idx = ((owner_idx - player_idx) % len(player_order)) + 1
                    if relative_idx < 5:
                        features[i, relative_idx] = 1.0
                except ValueError:
                    features[i, 0] = 1.0
            else:
                features[i, 0] = 1.0

            # Has road
            features[i, 5] = 1.0 if edge_player else 0.0

        return features

    def _build_adjacency(
        self,
        board: Dict[str, Any],
    ) -> Dict[str, 'np.ndarray']:
        """Build edge index arrays for message passing."""
        # Hex to vertex edges
        hex_to_vertex_edges = []
        for hex_id, vertex_ids in board.get('hex_to_vertices', {}).items():
            for v_id in vertex_ids:
                # Bidirectional
                hex_to_vertex_edges.append([int(hex_id), v_id])
                hex_to_vertex_edges.append([v_id, int(hex_id)])

        # Vertex to vertex edges
        vertex_to_vertex_edges = []
        for v1, neighbors in board.get('vertex_to_vertices', {}).items():
            for v2 in neighbors:
                vertex_to_vertex_edges.append([int(v1), v2])

        # Vertex to edge (roads)
        vertex_to_edge_edges = []
        for v_id, edge_ids in board.get('vertex_to_edges', {}).items():
            for e_id in edge_ids:
                vertex_to_edge_edges.append([int(v_id), e_id])
                vertex_to_edge_edges.append([e_id, int(v_id)])

        return {
            'hex_to_vertex': np.array(hex_to_vertex_edges, dtype=np.int64).T if hex_to_vertex_edges else np.zeros((2, 0), dtype=np.int64),
            'vertex_to_vertex': np.array(vertex_to_vertex_edges, dtype=np.int64).T if vertex_to_vertex_edges else np.zeros((2, 0), dtype=np.int64),
            'vertex_to_edge': np.array(vertex_to_edge_edges, dtype=np.int64).T if vertex_to_edge_edges else np.zeros((2, 0), dtype=np.int64),
        }

    def _encode_global(
        self,
        game_state: Dict[str, Any],
        player: str,
        player_order: List[str],
        player_idx: int,
    ) -> 'np.ndarray':
        """Encode global game state features."""
        players = game_state.get('players', {})

        # Global features:
        # - Current player's resources (5)
        # - Current player's VP (1)
        # - Current player's dev cards count (1)
        # - Current player's settlements/cities/roads count (3)
        # - Opponents' visible VP (up to 3)
        # - Game phase one-hot (4)
        # - Turn number normalized (1)
        # - Dice last rolled (2)

        features = np.zeros(20, dtype=np.float32)

        if player and player in players:
            player_data = players[player]

            # Resources (normalized)
            for i, resource in enumerate(self.RESOURCES[:5]):
                features[i] = min(player_data.get('resources', {}).get(resource, 0), 10) / 10.0

            # Victory points
            features[5] = player_data.get('visible_victory_points', 0) / 10.0

            # Dev cards
            features[6] = len(player_data.get('development_cards', [])) / 10.0

            # Buildings
            features[7] = len(player_data.get('settlements', [])) / 5.0
            features[8] = len(player_data.get('cities', [])) / 4.0
            features[9] = len(player_data.get('roads', [])) / 15.0

        # Opponents' VP
        opp_idx = 0
        for pid in player_order:
            if pid != player and opp_idx < 3:
                opp_data = players.get(pid, {})
                features[10 + opp_idx] = opp_data.get('visible_victory_points', 0) / 10.0
                opp_idx += 1

        # Game phase
        phase = game_state.get('phase', 'main')
        phase_map = {'setup': 0, 'main': 1, 'robber': 2, 'discard': 3}
        if phase in phase_map:
            features[13 + phase_map[phase]] = 1.0

        # Turn number
        features[17] = min(game_state.get('turn_number', 0), 100) / 100.0

        # Last dice
        dice = game_state.get('last_dice', [0, 0])
        if dice:
            features[18] = dice[0] / 6.0 if len(dice) > 0 else 0
            features[19] = dice[1] / 6.0 if len(dice) > 1 else 0

        return features

    def decode(
        self,
        features: 'np.ndarray',
        template_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Decode features back to game state (approximate).

        Note: GNN features are not fully reversible due to pooling.
        This provides a rough reconstruction for debugging.
        """
        # Return a copy of template state - full decoding is complex
        return copy.deepcopy(template_state)
