"""A* pathfinding implementation for hex grid."""
from typing import List, Optional, Dict, Set, Tuple
import heapq
import logging

from .hex import Hex
from .hex_state import HexState

def a_star(start_hex: Hex,
           target_hex: Hex,
           valid_states: frozenset[HexState] = frozenset({HexState.SOLID, HexState.CRACKED})) -> Optional[List[Hex]]:
    """Find the shortest path between two hexes using A* algorithm.

    Args:
        start_hex: The starting hex
        target_hex: The target hex to reach
        valid_states: List of hex states that are valid for movement

    Returns:
        A list of hexes representing the path from start to target (inclusive),
        or None if no path exists
    """
    from .game import Game

    # Get the game instance
    game = Game.instance
    if not game:
        logging.error("Game instance not available for pathfinding")
        return None

    # can't move into occupied or to-be-occupied hexes
    occupied = set()
    for enemy in game.enemies:
        occupied.add(enemy.current_hex)
        if enemy.target_hex is not None:
            occupied.add(enemy.target_hex)

    # Initialize the open and closed sets
    open_set: List[Tuple[float, int, Hex]] = []  # (f_score, counter, hex)
    counter = 0  # Used as a tiebreaker for equal f_scores

    # Add the start hex to the open set
    heapq.heappush(open_set, (0, counter, start_hex))
    counter += 1

    # Track where each hex came from for path reconstruction
    came_from: Dict[Hex, Optional[Hex]] = {start_hex: None}

    # g_score: cost from start to current hex
    g_score: Dict[Hex, float] = {start_hex: 0}

    # f_score: estimated total cost from start to goal through current hex
    f_score: Dict[Hex, float] = {start_hex: game.hex_distance(start_hex, target_hex)}

    # Set to track hexes we've already processed
    closed_set: Set[Hex] = set()

    while open_set:
        # Get the hex with the lowest f_score
        _, _, current = heapq.heappop(open_set)

        # If we've reached the target, reconstruct and return the path
        if current == target_hex:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))

        # Add current to closed set
        closed_set.add(current)

        # Check all neighbors
        for neighbor in game.get_hex_neighbors(current):
            # Skip if this neighbor is not in a valid state
            if neighbor.state not in valid_states:
                continue

            # Skip if we've already processed this neighbor
            if neighbor in closed_set:
                continue

            # Don't consider occupied hexes
            if neighbor in occupied:
                continue

            # Calculate tentative g_score
            tentative_g_score = g_score[current] + 1  # Cost is 1 for each step

            # If this neighbor is not in open set or we found a better path
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # Update path and scores
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + game.hex_distance(neighbor, target_hex)

                # Add to open set if not already there
                if neighbor not in [item[2] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1

    # If we get here, no path was found
    return None
