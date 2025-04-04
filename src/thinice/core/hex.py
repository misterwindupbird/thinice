"""Hex tile component for the ice breaking game."""
import logging
import random
import math
import pygame
from typing import List, Tuple, Optional, Dict, Set, Union
import numpy
from pathlib import Path

from .animation_manager import AnimationManager
from .hex_state import HexState
from .ice_fragment import IceFragment
from .crack import Crack
from ..config.settings import hex_grid, crack as crack_config, water, animation, display
from ..utils.geometry import (
    Point, 
    calculate_hex_vertices,
    calculate_edge_points,
    point_in_hex
)

TILE_CACHE = {}

class Hex:
    """Represents a hexagonal tile in the game grid."""
    
    def __init__(self, x: float, y: float,
                 grid_x: int, grid_y: int,
                 animation_manager: AnimationManager,
                 color: Tuple[int, int, int] = None,
                 state: HexState = HexState.SOLID,
                 height: float = 0):
        """Initialize a new hex tile.
        
        Args:
            x: Center x coordinate
            y: Center y coordinate
            grid_x: Grid column index
            grid_y: Grid row index
            color: Initial color of the hex
            state: Initial state of the hex
        """
        self.center = (x, y)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.vertices = calculate_hex_vertices(self.center, hex_grid.RADIUS)
        self.edge_points = calculate_edge_points(self.vertices)
        self.state = state
        self.cracks: List[Crack] = []
        self.color = color if color else self._init_color()
        self.height = height
        
        # For broken ice fragments
        self.ice_fragments = []
        self.fragment_colors = []
        self.broken_surface = None
        self.fragment_sprites = pygame.sprite.Group()
        self.break_time = 0
        
        # For transition animation
        self.transition_start_time = 0
        self.transition_duration = 0.4  # Faster animation (was 1.0 second)
        self.transition_progress = 0.0  # 0.0 to 1.0
        self.original_fragment_positions = []  # To store initial positions for animation
        self.animation_manager = animation_manager

    def __eq__(self, other):
        return isinstance(other, Hex) and self.grid_x == other.grid_x and self.grid_y == other.grid_y

    def __hash__(self):
        return hash((self.grid_x, self.grid_y))

    def __repr__(self):
        return f"HEX({self.grid_x}, {self.grid_y})"
        
    def _init_color(self) -> Tuple[int, int, int]:
        """Initialize the color of the hex tile with slight variations."""
        # Base color is white with slight variations
        base_r, base_g, base_b = 255, 255, 255
        
        # Add slight variations to create a more natural look
        grey_var = random.randint(-3, -0)  # Slight grey variation
        blue_var = random.randint(-1, -0)   # Slight blue tint
        
        return (
            min(255, max(0, base_r + grey_var)),
            min(255, max(0, base_g + grey_var)),
            min(255, max(0, base_b + grey_var + blue_var))
        )
    
    def get_shared_edge_index(self, neighbor: 'Hex') -> int:
        """Get the index of the edge shared with a neighbor.
        
        Args:
            neighbor: Adjacent hex tile
            
        Returns:
            Index of the shared edge or -1 if not adjacent
        """
        delta = (self.grid_x - neighbor.grid_x, self.grid_y - neighbor.grid_y)
        
        if self.grid_x % 2 == 0:  # Even row
            edge_map = {
                (0, 1): 0,    # Top
                (-1, 1): 1,   # Top-left
                (-1, 0): 2,   # Bottom-left
                (0, -1): 3,   # Bottom
                (1, 0): 4,    # Bottom-right
                (1, 1): 5     # Top-right
            }
        else:  # Odd row
            edge_map = {
                (0, 1): 0,    # Top
                (-1, 0): 1,   # Top-left
                (-1, -1): 2,  # Bottom-left
                (0, -1): 3,   # Bottom
                (1, -1): 4,   # Bottom-right
                (1, 0): 5     # Top-right
            }
        
        return edge_map.get(delta, -1)
    
    def has_crack_to_point(self, point: Point, threshold: float = 5.0) -> bool:
        """Check if any crack ends near a point.
        
        Args:
            point: Target point to check
            threshold: Maximum distance to consider as "near"
            
        Returns:
            True if a crack ends near the point
        """
        return any(
            ((crack.points[-1][0] - point[0])**2 + 
             (crack.points[-1][1] - point[1])**2) ** 0.5 < threshold
            for crack in self.cracks
        )
    
    def add_straight_crack(self, end_point: Point) -> Optional[Crack]:
        """Add a new crack from center to end point.
        
        Args:
            end_point: Target end point for the crack
            
        Returns:
            The created crack or None if invalid
        """
        # Remove the boundary check since edge points are calculated from hex vertices
        # and should always be valid endpoints for cracks
        
        crack = Crack(self.center)
        num_segments = max(
            crack_config.MIN_SEGMENTS,
            int(((end_point[0] - self.center[0])**2 + 
                 (end_point[1] - self.center[1])**2) ** 0.5 / 
                crack_config.SEGMENT_LENGTH)
        )
        
        crack.extend_to(end_point, num_segments)
        self.cracks.append(crack)
        return crack
    
    def add_secondary_cracks(self) -> None:
        """Add smaller cracks between adjacent primary cracks."""
        if len(self.cracks) < 2:
            return
            
        current_secondary = sum(1 for c in self.cracks if c.is_secondary)
        if current_secondary >= crack_config.MAX_SECONDARY_CRACKS:
            return
            
        for i, crack1 in enumerate(self.cracks[:-1]):
            if current_secondary >= crack_config.MAX_SECONDARY_CRACKS:
                break
                
            for crack2 in self.cracks[i+1:]:
                if current_secondary >= crack_config.MAX_SECONDARY_CRACKS:
                    break
                    
                # Only consider primary cracks
                if crack1.is_secondary or crack2.is_secondary:
                    continue
                    
                # Check if these cracks are not too close
                p1 = crack1.points[-1]  # End point of crack1
                p2 = crack2.points[-1]  # End point of crack2
                distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
                
                # Only add a secondary crack if the endpoints are somewhat close but not too close
                if distance > hex_grid.RADIUS * 0.3 and distance < hex_grid.RADIUS * 0.8:
                    # Create a new crack between these two points
                    new_crack = Crack(p1, is_secondary=True)
                    new_crack.extend_to(p2, 3)  # More segments for better visual
                    self.cracks.append(new_crack)
                    current_secondary += 1
    
    def _point_in_hex(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the hexagon."""
        x, y = point
        dx = abs(x - self.center[0])
        dy = abs(y - self.center[1])
        
        # Quick check using bounding box
        if dx > hex_grid.RADIUS or dy > hex_grid.RADIUS:
            return False
            
        # More precise check
        # For a regular hexagon, we can check if the point is within the radius
        distance = math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        return distance <= hex_grid.RADIUS * 0.95  # Slightly smaller to keep inside visible boundary
    
    
    def crack(self, neighbors: List['Hex']) -> None:
        """Initiate cracking of the hex tile.
        
        Args:
            neighbors: List of adjacent hex tiles
        """
        if self.state != HexState.SOLID:
            return
            
        # Change to CRACKING state instead of directly to CRACKED
        self.state = HexState.CRACKING
        self.transition_start_time = pygame.time.get_ticks() / 1000.0  # Current time in seconds
        self.transition_duration = animation.CRACKING_DURATION
        self.transition_progress = 0.0
        self.animation_manager.blocking_animations += 1
        logging.debug(f'{self.animation_manager.blocking_animations=}')
        
        # Store neighbors for connecting cracks during animation
        self.cracking_neighbors = neighbors
        
        # Store cracks to be added during animation
        self.pending_cracks = []
        self.pending_secondary_cracks = []
        
        # Connect to cracked neighbors - store these connections for animation
        for neighbor in neighbors:
            if neighbor.state not in {HexState.CRACKED, HexState.CRACKING, HexState.BREAKING, HexState.BROKEN}:
                continue
                
            edge_index = self.get_shared_edge_index(neighbor)
            if edge_index == -1:
                continue
                
            shared_point = self.edge_points[edge_index]
            # Ensure neighbor has connecting crack
            if not neighbor.has_crack_to_point(shared_point):
                # Store this crack to be added during animation
                self.pending_cracks.append(("neighbor", neighbor, shared_point))
            
            # Store our connecting crack for animation
            self.pending_cracks.append(("self", self, shared_point))

        # Add minimum number of cracks - store for animation
        num_cracks = random.randint(crack_config.MIN_CRACKS, crack_config.MAX_CRACKS)
        
        # Pick unused edges for new cracks
        used_edges = set()
        for _, _, point in self.pending_cracks:
            if point in self.edge_points:
                used_edges.add(point)
        
        available_edges = [
            i for i, point in enumerate(self.edge_points)
            if point not in used_edges
        ]
        
        # Shuffle available edges
        random.shuffle(available_edges)
        
        # Add cracks up to the minimum number
        while len(self.pending_cracks) < num_cracks and available_edges:
            edge_index = available_edges.pop()
            self.pending_cracks.append(("self", self, self.edge_points[edge_index]))
        
        # Pre-calculate secondary cracks for animation
        # We'll store them but only show them in the second half of the animation
        self._calculate_secondary_cracks()
    
    def _calculate_secondary_cracks(self) -> None:
        """Pre-calculate secondary cracks for animation."""
        # This is similar to add_secondary_cracks but stores the cracks instead of adding them
        if len(self.pending_cracks) < 2:
            return
            
        # Count how many secondary cracks we can add
        max_secondary = crack_config.MAX_SECONDARY_CRACKS
        
        # Create a list of primary crack endpoints and their edge indices
        primary_cracks = []
        for crack_type, target_hex, end_point in self.pending_cracks:
            if crack_type == "self":  # Only consider cracks in this hex
                # Find which edge this crack connects to
                for i, edge_point in enumerate(self.edge_points):
                    if ((end_point[0] - edge_point[0])**2 + 
                        (end_point[1] - edge_point[1])**2) < 1.0:  # Close enough to be the same point
                        primary_cracks.append((end_point, i))
                        break
        
        # Generate secondary cracks between primary cracks that connect to adjacent edges
        for i, (p1, edge1) in enumerate(primary_cracks[:-1]):
            if len(self.pending_secondary_cracks) >= max_secondary:
                break
                
            for j, (p2, edge2) in enumerate(primary_cracks[i+1:], i+1):
                if len(self.pending_secondary_cracks) >= max_secondary:
                    break
                    
                # Check if these cracks connect to adjacent edges
                edge_diff = abs(edge1 - edge2)
                if edge_diff != 1 and edge_diff != 5:  # Edges 0 and 5 are also adjacent
                    continue
                    
                # Calculate random points along each primary crack
                # For primary crack 1: random point between center and endpoint
                t1 = random.uniform(0.3, 0.7)  # Random point 30-70% along the crack
                point1_x = self.center[0] + (p1[0] - self.center[0]) * t1
                point1_y = self.center[1] + (p1[1] - self.center[1]) * t1
                
                # For primary crack 2: random point between center and endpoint
                t2 = random.uniform(0.3, 0.7)  # Random point 30-70% along the crack
                point2_x = self.center[0] + (p2[0] - self.center[0]) * t2
                point2_y = self.center[1] + (p2[1] - self.center[1]) * t2
                
                # Calculate a midpoint with some randomness
                mid_x = (point1_x + point2_x) / 2
                mid_y = (point1_y + point2_y) / 2
                
                # Add some randomness to the midpoint
                deviation = hex_grid.RADIUS * crack_config.MAX_DEVIATION
                mid_x += random.uniform(-deviation, deviation)
                mid_y += random.uniform(-deviation, deviation)
                
                # Ensure the midpoint is inside the hex
                if not self._point_in_hex((mid_x, mid_y)):
                    continue
                
                # Store this secondary crack with its connection points
                self.pending_secondary_cracks.append(((point1_x, point1_y), (point2_x, point2_y), mid_x, mid_y))
    
    def break_ice(self) -> None:
        """Break the ice if the hex is in a cracked state."""
        if self.state != HexState.CRACKED:
            return
        
        # Add edge cracks to ensure ice can detach from neighboring hexes
        self._add_edge_cracks()
        
        # Transition to BREAKING state
        self.state = HexState.BREAKING
        self.transition_start_time = pygame.time.get_ticks() / 1000.0
        
        # Increase transition duration for a slower, more visible animation
        self.transition_duration = animation.BREAKING_DURATION
        self.transition_progress = 0.0
        self.animation_manager.blocking_animations += 1
        logging.debug(f'{self.animation_manager.blocking_animations=}')

        # Clear cached surfaces
        self.broken_surface = None
        
        # Clear any existing fragment sprites
        self.fragment_sprites.empty()
        
        # Find ice fragments but don't create sprites yet
        self._find_ice_fragments()
        
        # Store original positions for animation
        self.original_fragment_positions = []
        for fragment in self.ice_fragments:
            # Calculate center of fragment
            center_x = sum(p[0] for p in fragment) / len(fragment)
            center_y = sum(p[1] for p in fragment) / len(fragment)
            self.original_fragment_positions.append((center_x, center_y))

    def _add_edge_cracks(self) -> None:
        """Add cracks along the perimeter of the hex to detach fragments."""
        # Add cracks along the edges (between vertices)
        for i in range(6):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % 6]
            
            # Create a crack along this edge
            edge_crack = Crack(v1, is_secondary=True)
            edge_crack.extend_to(v2, 3)  # More segments for better visual effect
            self.cracks.append(edge_crack)
            
            # Connect edge cracks to the nearest existing crack endpoint
            # Create 1-2 intermediate points along each edge
            num_points = random.randint(1, 2)
            for j in range(num_points):
                # Calculate position along edge
                t = (j + 1) / (num_points + 1)
                edge_point = (
                    v1[0] * (1 - t) + v2[0] * t,
                    v1[1] * (1 - t) + v2[1] * t
                )
                
                # Find nearest interior crack endpoint
                nearest_point = None
                min_distance = float('inf')
                
                for crack in self.cracks:
                    # Skip edge cracks we just added
                    if crack == edge_crack:
                        continue
                        
                    if len(crack.points) >= 2:
                        # Check start and end points of cracks
                        for point in [crack.points[0], crack.points[-1]]:
                            # Calculate distance from point to edge point
                            dist = ((point[0] - edge_point[0]) ** 2 + 
                                    (point[1] - edge_point[1]) ** 2) ** 0.5
                            
                            # If this is close enough but not too close
                            if dist < min_distance and dist > 10:
                                min_distance = dist
                                nearest_point = point
                
                # If found a good connection point, create a connecting crack
                if nearest_point and min_distance < hex_grid.RADIUS * 0.75:
                    new_crack = Crack(nearest_point, is_secondary=True)
                    new_crack.extend_to(edge_point, 2)
                    self.cracks.append(new_crack)
    
    def _find_ice_fragments(self) -> None:
        """Find ice fragments based on existing crack patterns and assign random colors."""
        # Use a flood fill approach to identify contiguous regions
        # First, create a grid representation of the hex
        grid_size = 50  # Resolution of the grid
        grid = [[False for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Calculate bounds for mapping to grid
        min_x = min(v[0] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        
        # Function to convert world coordinates to grid coordinates
        def world_to_grid(x, y):
            grid_x = int((x - min_x) / (max_x - min_x) * (grid_size - 1))
            grid_y = int((y - min_y) / (max_y - min_y) * (grid_size - 1))
            return max(0, min(grid_size - 1, grid_x)), max(0, min(grid_size - 1, grid_y))
        
        # Function to convert grid coordinates to world coordinates
        def grid_to_world(grid_x, grid_y):
            x = min_x + (grid_x / (grid_size - 1)) * (max_x - min_x)
            y = min_y + (grid_y / (grid_size - 1)) * (max_y - min_y)
            return x, y
        
        # Fill the grid with the hex shape
        total_hex_cells = 0
        for y in range(grid_size):
            for x in range(grid_size):
                world_x, world_y = grid_to_world(x, y)
                if self._point_in_hex((world_x, world_y)):
                    grid[y][x] = True
                    total_hex_cells += 1
        
        # Draw cracks on the grid (set cells to False)
        for crack in self.cracks:
            for i in range(len(crack.points) - 1):
                p1 = crack.points[i]
                p2 = crack.points[i + 1]
                
                # Draw a thick line for the crack
                for t in range(101):  # 101 points for smooth line
                    t_val = t / 100.0
                    x = p1[0] * (1 - t_val) + p2[0] * t_val
                    y = p1[1] * (1 - t_val) + p2[1] * t_val
                    
                    # Make the crack thicker
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if dx*dx + dy*dy <= 4:  # Circular thickness
                                gx, gy = world_to_grid(x + dx, y + dy)
                                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                                    grid[gy][gx] = False
        
        # Use flood fill to identify contiguous regions
        visited = [[False for _ in range(grid_size)] for _ in range(grid_size)]
        fragments = []
        
        for y in range(grid_size):
            for x in range(grid_size):
                if grid[y][x] and not visited[y][x]:
                    # Start a new fragment
                    fragment_cells = []
                    queue = [(x, y)]
                    visited[y][x] = True
                    
                    while queue:
                        cx, cy = queue.pop(0)
                        fragment_cells.append((cx, cy))
                        
                        # Check neighbors
                        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                                grid[ny][nx] and not visited[ny][nx]):
                                queue.append((nx, ny))
                                visited[ny][nx] = True
                    
                    # Convert cells to world coordinates and create a polygon
                    if len(fragment_cells) > 5:  # Ignore very small fragments
                        # Use convex hull to create a clean polygon
                        world_points = [grid_to_world(cx, cy) for cx, cy in fragment_cells]
                        hull = self._convex_hull(world_points)
                        
                        if len(hull) >= 3:  # Need at least 3 points for a polygon
                            fragments.append({
                                'cells': fragment_cells,
                                'hull': hull,
                                'size': len(fragment_cells)
                            })
        
        # Calculate the maximum size threshold using the configurable percentage
        max_size_threshold = total_hex_cells * hex_grid.MAX_FRAGMENT_SIZE_PERCENT
        
        # Process large fragments and break them into smaller ones
        final_fragments = []
        fragments_to_process = fragments.copy()
        
        while fragments_to_process:
            fragment = fragments_to_process.pop(0)
            
            if fragment['size'] > max_size_threshold:
                # This fragment is too large, break it into smaller pieces
                
                # Create a grid representation of just this fragment
                fragment_grid = [[False for _ in range(grid_size)] for _ in range(grid_size)]
                for cx, cy in fragment['cells']:
                    fragment_grid[cy][cx] = True
                
                # Choose a random starting point for the new crack
                cells = fragment['cells']
                start_idx = random.randint(0, len(cells) - 1)
                start_cell = cells[start_idx]
                
                # Choose a random direction and length for the crack
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                direction = random.choice(directions)
                
                # Create a crack of random length
                crack_length = random.randint(grid_size // 4, grid_size // 2)
                
                # Draw the crack on the fragment grid
                cx, cy = start_cell
                for _ in range(crack_length):
                    cx += direction[0]
                    cy += direction[1]
                    
                    # Check if we're still in bounds and in the fragment
                    if (0 <= cx < grid_size and 0 <= cy < grid_size and fragment_grid[cy][cx]):
                        # Add some thickness to the crack
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = cx + dx, cy + dy
                                if (0 <= nx < grid_size and 0 <= ny < grid_size and fragment_grid[ny][nx]):
                                    fragment_grid[ny][nx] = False
                
                # Find the new sub-fragments using flood fill
                sub_visited = [[False for _ in range(grid_size)] for _ in range(grid_size)]
                sub_fragments = []
                
                for y in range(grid_size):
                    for x in range(grid_size):
                        if fragment_grid[y][x] and not sub_visited[y][x]:
                            # Start a new sub-fragment
                            sub_fragment_cells = []
                            queue = [(x, y)]
                            sub_visited[y][x] = True
                            
                            while queue:
                                scx, scy = queue.pop(0)
                                sub_fragment_cells.append((scx, scy))
                                
                                # Check neighbors
                                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                    nx, ny = scx + dx, scy + dy
                                    if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                                        fragment_grid[ny][nx] and not sub_visited[ny][nx]):
                                        queue.append((nx, ny))
                                        sub_visited[ny][nx] = True
                            
                            # Convert cells to world coordinates and create a polygon
                            if len(sub_fragment_cells) > 5:  # Ignore very small fragments
                                # Use convex hull to create a clean polygon
                                sub_world_points = [grid_to_world(cx, cy) for cx, cy in sub_fragment_cells]
                                sub_hull = self._convex_hull(sub_world_points)
                                
                                if len(sub_hull) >= 3:  # Need at least 3 points for a polygon
                                    sub_fragments.append({
                                        'cells': sub_fragment_cells,
                                        'hull': sub_hull,
                                        'size': len(sub_fragment_cells)
                                    })
                
                # Add the sub-fragments back to the processing queue
                fragments_to_process.extend(sub_fragments)
            else:
                # This fragment is small enough, keep it
                final_fragments.append(fragment['hull'])
        
        # Store fragments and use the original hex color for all fragments
        self.ice_fragments = final_fragments
        self.fragment_colors = []
        
        for _ in range(len(final_fragments)):
            # Use the original hex color for all fragments
            self.fragment_colors.append(self.color)
    
    def _convex_hull(self, points):
        """Compute the convex hull of a set of points using Graham scan."""
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        if len(points) <= 3:
            return points
            
        # Find the lowest point
        lowest = min(points, key=lambda p: (p[1], p[0]))
        
        # Sort points by polar angle with respect to the lowest point
        sorted_points = sorted(points, key=lambda p: (
            math.atan2(p[1] - lowest[1], p[0] - lowest[0]),
            (p[0] - lowest[0])**2 + (p[1] - lowest[1])**2
        ))
        
        # Build the convex hull
        hull = [lowest]
        for p in sorted_points:
            while len(hull) >= 2 and cross_product(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
            
        return hull
    
    def _color_fragments(self, surface, size):
        """Identify and color ice fragments using flood fill on the rendered surface.
        
        Args:
            surface: The rendered surface with ice and cracks
            size: Size of the surface
            
        Returns:
            A new surface with colored fragments
        """
        # Create a new surface for the result
        result = pygame.Surface((size, size), pygame.SRCALPHA)
        result.blit(surface, (0, 0))  # Copy the original surface
        
        # Create a mask to track visited pixels
        visited = [[False for _ in range(size)] for _ in range(size)]
        
        # Water color for comparison (to avoid coloring water)
        water_colors = [
            water.BASE_COLOR,
            water.CRACK_COLOR
        ]
        
        # Function to check if a color is similar to water
        def is_water_color(color):
            if color[3] < 200:  # Check alpha (transparency)
                return True
                
            for wc in water_colors:
                # Check if color is similar to water color
                r_diff = abs(color[0] - wc[0])
                g_diff = abs(color[1] - wc[1])
                b_diff = abs(color[2] - wc[2])
                if r_diff + g_diff + b_diff < 60:  # Threshold for similarity
                    return True
            return False
        
        # Find and color ice fragments
        fragments = []  # Store fragment data for creating sprites
        
        for y in range(size):
            for x in range(size):
                if visited[y][x]:
                    continue
                    
                # Get pixel color
                color = surface.get_at((x, y))
                
                # Skip water pixels
                if is_water_color(color):
                    visited[y][x] = True
                    continue
                
                # Found an ice fragment, flood fill and color it
                # Use the original hex color instead of fragment colors
                fragment_color = self.color
                
                # Simple flood fill
                queue = [(x, y)]
                fragment_pixels = []
                
                while queue:
                    cx, cy = queue.pop(0)
                    if cx < 0 or cy < 0 or cx >= size or cy >= size or visited[cy][cx]:
                        continue
                        
                    pixel_color = surface.get_at((cx, cy))
                    if is_water_color(pixel_color):
                        visited[cy][cx] = True
                        continue
                    
                    # Add to fragment
                    visited[cy][cx] = True
                    fragment_pixels.append((cx, cy))
                    
                    # Add neighbors to queue
                    queue.append((cx+1, cy))
                    queue.append((cx-1, cy))
                    queue.append((cx, cy+1))
                    queue.append((cx, cy-1))
                
                # Color the fragment if it's large enough
                if len(fragment_pixels) > 10:
                    for px, py in fragment_pixels:
                        result.set_at((px, py), fragment_color)
                    
                    # Store fragment data for creating sprites
                    # Convert pixel coordinates to world coordinates
                    offset_x = size // 2
                    offset_y = size // 2
                    world_pixels = [(self.center[0] + (px - offset_x), 
                                    self.center[1] + (py - offset_y)) 
                                   for px, py in fragment_pixels]
                    
                    # Calculate fragment center
                    if world_pixels:
                        center_x = sum(p[0] for p in world_pixels) / len(world_pixels)
                        center_y = sum(p[1] for p in world_pixels) / len(world_pixels)
                        
                        # Create a convex hull for the fragment
                        if len(world_pixels) >= 3:
                            hull = self._convex_hull(world_pixels)
                            fragments.append({
                                'points': hull,
                                'color': fragment_color,
                                'center': (center_x, center_y)
                            })
        
        # Create sprite objects for each fragment
        for fragment in fragments:
            sprite = IceFragment(
                fragment['points'], 
                fragment['color'], 
                fragment['center'],
                self.center
            )
            self.fragment_sprites.add(sprite)
        
        return result
    
    def _draw_broken(self, screen: pygame.Surface, current_time: float = 0, non_broken_hexes: List['Hex'] = None) -> None:
        """Draw the broken state with widened cracks showing water beneath and colored fragments.
        
        Args:
            screen: Surface to draw on
            current_time: Current game time in seconds (optional)
            non_broken_hexes: List of hexes that are not in BROKEN state (for collision detection)
        """
        # First time setup - create the broken surface and fragment sprites
        if self.broken_surface is None:
            # Create a temporary surface for initial rendering
            surface_size = int(hex_grid.RADIUS * 3)
            temp_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            
            # Offset for drawing on the surface
            offset_x = surface_size // 2
            offset_y = surface_size // 2
            
            # Draw water underneath (dark blue)
            adjusted_vertices = [(x - self.center[0] + offset_x, y - self.center[1] + offset_y) 
                                for x, y in self.vertices]
            pygame.draw.polygon(temp_surface, water.BASE_COLOR, adjusted_vertices)
            
            # Draw the base ice (white)
            # pygame.draw.polygon(temp_surface, self.color, adjusted_vertices)
            
            # Draw widened cracks with water color
            for crack in self.cracks:
                # Skip invalid cracks
                if len(crack.points) < 2:
                    continue
                    
                # Draw widened cracks showing water underneath
                adjusted_points = [(p[0] - self.center[0] + offset_x, p[1] - self.center[1] + offset_y) 
                                for p in crack.points]
                                
                # Draw water through cracks - using water color
                pygame.draw.lines(temp_surface, water.CRACK_COLOR, False, adjusted_points, 10)
            
            # Now identify and color the fragments using a simple flood fill
            self.broken_surface = self._color_fragments(temp_surface, surface_size)
        
        # Draw the water underneath
        pygame.draw.polygon(screen, water.BASE_COLOR, self.vertices)
        
        # Update and draw the fragment sprites
        self.fragment_sprites.update(current_time, non_broken_hexes)
        self.fragment_sprites.draw(screen)
    
    def _draw_cracking(self, screen: pygame.Surface, current_time: float) -> None:
        """Draw the cracking animation with cracks growing from center outward.
        
        Args:
            screen: Surface to draw on
            font: Font for text rendering
            current_time: Current game time in seconds
        """
        # Draw base hex
        pygame.draw.polygon(screen, self.color, self.vertices)
        
        # Calculate progress for the animation
        progress = self.transition_progress
        
        # Create a surface for the cracks with masking
        hex_bounds = self._get_hex_bounds()
        width = int(hex_bounds[2] - hex_bounds[0] + 20)  # Add padding
        height = int(hex_bounds[3] - hex_bounds[1] + 20)
        offset_x = hex_bounds[0] - 10
        offset_y = hex_bounds[1] - 10
        
        # Create a hex mask for clipping
        hex_mask_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        hex_mask_surface.fill((0, 0, 0, 0))  # Start with fully transparent
        
        # Draw the hex shape in the mask
        local_vertices = [(x - offset_x, y - offset_y) for x, y in self.vertices]
        pygame.draw.polygon(hex_mask_surface, (255, 255, 255, 255), local_vertices)
        
        # Create a mask from the hex shape
        hex_mask = pygame.mask.from_surface(hex_mask_surface)
        
        # Create a surface for the cracks
        crack_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Track which primary cracks are visible and how far they've grown
        primary_crack_progress = {}
        
        # Process pending primary cracks based on animation progress
        for crack_type, target_hex, end_point in self.pending_cracks:
            # Only process cracks for this hex (neighbor cracks are handled by the neighbor)
            if crack_type == "self":
                # Calculate the visible portion of the crack based on progress
                # Start from center and grow outward
                start_point = self.center
                
                # Calculate total distance
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                total_distance = (dx*dx + dy*dy) ** 0.5
                
                # Calculate visible distance based on progress
                visible_distance = total_distance * progress
                
                if visible_distance > 0:
                    # Calculate the visible end point
                    t = visible_distance / total_distance
                    visible_end_x = start_point[0] + dx * t
                    visible_end_y = start_point[1] + dy * t
                    
                    # Store the progress for this primary crack
                    primary_crack_progress[end_point] = t
                    
                    # Draw the visible portion of the crack
                    local_start = (start_point[0] - offset_x, start_point[1] - offset_y)
                    local_end = (visible_end_x - offset_x, visible_end_y - offset_y)
                    
                    # Draw the crack
                    pygame.draw.line(crack_surface, water.CRACK_COLOR, local_start, local_end, 1)
        
        # Process secondary cracks - only show in the second half of the animation
        if progress > 0.5 and len(primary_crack_progress) >= 2:
            # Scale the progress for secondary cracks from 0 to 1 in the second half
            secondary_progress = (progress - 0.5) * 2
            
            for point1, point2, mid_x, mid_y in self.pending_secondary_cracks:
                # Calculate how far along each primary crack we need to be to see these points
                # We need to find which primary cracks these points belong to
                
                # For each primary crack endpoint
                for end_point, t_progress in primary_crack_progress.items():
                    # Calculate the current visible point on this primary crack
                    dx = end_point[0] - self.center[0]
                    dy = end_point[1] - self.center[1]
                    
                    # Check if point1 is on this primary crack
                    t1_on_crack = ((point1[0] - self.center[0]) / dx if dx != 0 else 
                                  (point1[1] - self.center[1]) / dy if dy != 0 else 0)
                    
                    # Check if point2 is on this primary crack
                    t2_on_crack = ((point2[0] - self.center[0]) / dx if dx != 0 else 
                                  (point2[1] - self.center[1]) / dy if dy != 0 else 0)
                    
                    # If point1 is on this crack and the crack has grown past it
                    if 0 <= t1_on_crack <= 1 and t_progress >= t1_on_crack:
                        # Draw from point1 towards midpoint based on secondary progress
                        dx1 = mid_x - point1[0]
                        dy1 = mid_y - point1[1]
                        t1 = secondary_progress
                        p1_to_mid_x = point1[0] + dx1 * t1
                        p1_to_mid_y = point1[1] + dy1 * t1
                        
                        # Draw the visible portion of the secondary crack
                        local_p1 = (point1[0] - offset_x, point1[1] - offset_y)
                        local_p1_to_mid = (p1_to_mid_x - offset_x, p1_to_mid_y - offset_y)
                        
                        # Draw the secondary crack
                        pygame.draw.line(crack_surface, water.CRACK_COLOR, local_p1, local_p1_to_mid, 1)
                    
                    # If point2 is on this crack and the crack has grown past it
                    if 0 <= t2_on_crack <= 1 and t_progress >= t2_on_crack:
                        # Draw from point2 towards midpoint based on secondary progress
                        dx2 = mid_x - point2[0]
                        dy2 = mid_y - point2[1]
                        t2 = secondary_progress
                        p2_to_mid_x = point2[0] + dx2 * t2
                        p2_to_mid_y = point2[1] + dy2 * t2
                        
                        # Draw the visible portion of the secondary crack
                        local_p2 = (point2[0] - offset_x, point2[1] - offset_y)
                        local_p2_to_mid = (p2_to_mid_x - offset_x, p2_to_mid_y - offset_y)
                        
                        # Draw the secondary crack
                        pygame.draw.line(crack_surface, water.CRACK_COLOR, local_p2, local_p2_to_mid, 1)
            
        # If we're fully cracked, add the actual cracks to the hex
        if progress >= 1.0:
            # Add the actual cracks to this hex and neighbors
            for crack_type, target_hex, end_point in self.pending_cracks:
                if crack_type == "self":
                    # Add the actual crack to this hex
                    self.add_straight_crack(end_point)
                elif crack_type == "neighbor":
                    # Add the actual crack to the neighbor
                    target_hex.add_straight_crack(end_point)
            
            # Add secondary cracks
            for point1, point2, mid_x, mid_y in self.pending_secondary_cracks:
                # Create a secondary crack from point1 to midpoint
                crack1 = Crack(point1)
                crack1.extend_to((mid_x, mid_y), crack_config.MIN_SEGMENTS)
                crack1.is_secondary = True
                self.cracks.append(crack1)
                
                # Create a secondary crack from point2 to midpoint
                crack2 = Crack(point2)
                crack2.extend_to((mid_x, mid_y), crack_config.MIN_SEGMENTS)
                crack2.is_secondary = True
                self.cracks.append(crack2)
        
        # Apply the hex mask to the crack surface
        crack_array = pygame.surfarray.pixels_alpha(crack_surface)
        hex_mask_array = pygame.surfarray.pixels_alpha(hex_mask_surface)
        
        # Only keep cracks where the hex mask is set
        crack_array[:] = numpy.minimum(crack_array, hex_mask_array)
        
        # Clean up to release surface lock
        del crack_array
        del hex_mask_array
        
        # Blit the crack surface onto the screen
        screen.blit(crack_surface, (offset_x, offset_y))
        
        # Draw coordinates
        if display.DRAW_OVERLAY:
            text = display.font.render(f"({self.grid_x},{self.grid_y})", True, hex_grid.TEXT_COLOR)
            text_rect = text.get_rect(center=self.center)
            screen.blit(text, text_rect)
    
    def _draw_cracked(self, screen: pygame.Surface) -> None:
        """Draw the cracked state with thin water-colored hairlines."""
        # Draw base hex
        pygame.draw.polygon(screen, self.color, self.vertices)
        
        # Draw cracks as thin hairlines with water color
        for crack in self.cracks:
            # Draw thinner lines with water color instead of using crack.draw()
            if len(crack.points) < 2:
                continue
                
            # Use water color for hairline cracks
            pygame.draw.lines(screen, water.CRACK_COLOR, False, crack.points, 1)
        
        # Draw coordinates only (no outline)
        if display.DRAW_OVERLAY:
            text = display.font.render(f"({self.grid_x},{self.grid_y})", True, hex_grid.TEXT_COLOR)
            text_rect = text.get_rect(center=self.center)
            screen.blit(text, text_rect)
        
    
    def _draw_solid(self, screen: pygame.Surface) -> None:
        """Draw the solid state."""

        pygame.draw.polygon(screen, self.color, self.vertices)

        if display.DRAW_OVERLAY:
            text = display.font.render(f"({self.grid_x},{self.grid_y})", True, hex_grid.TEXT_COLOR)
            text_rect = text.get_rect(center=self.center)
            screen.blit(text, text_rect)
    
    def _line_intersection(self, line1: Tuple[Tuple[float, float], Tuple[float, float]], 
                          line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Find the intersection point of two line segments if it exists."""
        # Extract points
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        
        # Calculate denominators
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:  # Lines are parallel
            return None
            
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        # Check if intersection is within both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
        
        return None
    
    def _draw_breaking(self, screen: pygame.Surface, current_time: float, non_broken_hexes: List['Hex'] = None) -> None:
        """Draw the breaking animation state with widening cracks and separating fragments.
        
        Args:
            screen: Surface to draw on
            current_time: Current game time in seconds
            non_broken_hexes: List of hexes that are not in BROKEN state (for collision detection)
        """
        # Use a cubic ease-out function for even smoother animation
        # This makes the start of the animation faster and the end slower
        t = 1 - (1 - self.transition_progress) ** 3  # Cubic ease-out for smoother feel
        
        if self.broken_surface is None or len(self.fragment_sprites) == 0:
            # First time setup - create the broken surface and fragment sprites
            surface_size = int(hex_grid.RADIUS * 3)
            temp_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            
            # Offset for drawing on the surface
            offset_x = surface_size // 2
            offset_y = surface_size // 2
            
            # Draw water underneath (dark blue)
            adjusted_vertices = [(x - self.center[0] + offset_x, y - self.center[1] + offset_y) 
                                for x, y in self.vertices]
            pygame.draw.polygon(temp_surface, water.BASE_COLOR, adjusted_vertices)
            
            # Draw the base ice (white)
            pygame.draw.polygon(temp_surface, self.color, adjusted_vertices)
            
            # Draw cracks with width based on transition progress
            for crack in self.cracks:
                if len(crack.points) < 2:
                    continue
                    
                adjusted_points = [(p[0] - self.center[0] + offset_x, p[1] - self.center[1] + offset_y) 
                                for p in crack.points]
                
                # Calculate crack width based on transition progress
                # Start with hairline (1px) and widen to 10px
                crack_width = 1 + 9 * t
                pygame.draw.lines(temp_surface, water.BASE_COLOR, False, adjusted_points, int(crack_width))
            
            # Create fragment sprites if they don't exist yet
            if len(self.fragment_sprites) == 0:
                for i, fragment in enumerate(self.ice_fragments):
                    # Calculate center of fragment
                    center_x = sum(p[0] for p in fragment) / len(fragment)
                    center_y = sum(p[1] for p in fragment) / len(fragment)
                    
                    # Create sprite
                    sprite = IceFragment(
                        fragment, 
                        self.color,
                        (center_x, center_y),
                        self.center
                    )
                    
                    # Disable bobbing and rotation until fully broken
                    sprite.bob_amount = 0
                    sprite.rotation = 0
                    
                    self.fragment_sprites.add(sprite)
            
            # Store the surface for reuse
            self.broken_surface = temp_surface
        
        # Create a layered rendering approach using separate surfaces
        # Get hex bounds for creating appropriately sized surfaces
        hex_bounds = self._get_hex_bounds()
        width = int(hex_bounds[2] - hex_bounds[0] + 40)  # Add more padding
        height = int(hex_bounds[3] - hex_bounds[1] + 40)
        offset_x = hex_bounds[0] - 20
        offset_y = hex_bounds[1] - 20
        
        # Create a hex mask for clipping (used for both particles and cracks)
        hex_mask_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        hex_mask_surface.fill((0, 0, 0, 0))  # Start with fully transparent
        
        # Draw the hex shape in the mask
        local_vertices = [(x - offset_x, y - offset_y) for x, y in self.vertices]
        pygame.draw.polygon(hex_mask_surface, (255, 255, 255, 255), local_vertices)
        
        # Create a mask from the hex shape
        hex_mask = pygame.mask.from_surface(hex_mask_surface)
        
        # Create a surface for the water (bottom layer)
        water_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Draw water with alpha based on transition progress
        if t > 0.1:  # Start showing water earlier
            water_alpha = int(220 * ((t - 0.1) / 0.9))  # Adjust formula for earlier start
            water_color_with_alpha = (*water.BASE_COLOR, water_alpha)
            pygame.draw.polygon(water_surface, water_color_with_alpha, local_vertices)
        
        # Create a surface for the ice (with reduced opacity as it breaks)
        ice_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Draw the base hex with solid color (no alpha)
        pygame.draw.polygon(ice_surface, self.color, local_vertices)
        
        # Apply a fade effect by adjusting the surface's alpha
        ice_alpha = int(100 * (1.0 - t * 0.7))
        ice_surface.set_alpha(ice_alpha)
        
        # 4. Create a surface for the cracks (top layer)
        crack_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Draw cracks with increasing width - ALWAYS VISIBLE
        crack_width = 1 + 9 * t
        for crack in self.cracks:
            if len(crack.points) < 2:
                continue
            
            # Adjust crack points to the local surface coordinates
            local_points = [(p[0] - offset_x, p[1] - offset_y) for p in crack.points]
            
            # Draw the crack with full opacity to ensure visibility
            pygame.draw.lines(crack_surface, water.BASE_COLOR, False, local_points, int(crack_width))
        
        # Apply the hex mask to the crack surface to keep cracks within the hex
        crack_array = pygame.surfarray.pixels_alpha(crack_surface)
        hex_mask_array = pygame.surfarray.pixels_alpha(hex_mask_surface)
        
        # Only keep cracks where the hex mask is set
        crack_array[:] = numpy.minimum(crack_array, hex_mask_array)
        
        # Clean up to release surface lock
        del crack_array
        del hex_mask_array
        
        # Now composite the layers in the correct order onto the screen
        # 1. First the water layer
        screen.blit(water_surface, (offset_x, offset_y))
        
        # 2. Then the ice layer
        screen.blit(ice_surface, (offset_x, offset_y))
        
        # 3. Finally, draw the cracks on top of everything to ensure visibility
        screen.blit(crack_surface, (offset_x, offset_y))
        
        # Update fragment positions based on transition progress
        for i, sprite in enumerate(self.fragment_sprites):
            if i < len(self.original_fragment_positions):
                original_pos = self.original_fragment_positions[i]
                
                # Remove the "flying away" effect - keep fragments in place
                # Just gradually increase bobbing and rotation as transition progresses
                
                # Update sprite position to original position (no offset)
                sprite.offset_x = 0
                sprite.offset_y = 0
                
                # Update sprite rect
                sprite.rect.x = int(original_pos[0] - sprite.rect.width / 2)
                sprite.rect.y = int(original_pos[1] - sprite.rect.height / 2)
                
                # Gradually increase bobbing as transition progresses
                if t > 0.5:  # Start bobbing halfway through the transition
                    bob_factor = (t - 0.5) / 0.5  # Scale from 0 to 1 in the last 50% of the transition
                    sprite.bob_amount = sprite.bob_amount * (1 - bob_factor) + (random.uniform(0.3, 0.8) * bob_factor)
                    sprite.rotation = sprite.rotation * (1 - bob_factor) + (random.uniform(-0.05, 0.05) * bob_factor)
        
        # Draw the fragments on top of everything
        self.fragment_sprites.draw(screen)
    
    def _get_hex_bounds(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of the hex.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        return (min_x, min_y, max_x, max_y)
    
    def _draw_land(self, screen: pygame.Surface) -> None:
        """Draw the land state."""
        pygame.draw.polygon(screen, self.color, self.vertices)

        # Load and resize tile image once, cache it
        # logging.debug(f"{self.height=}")
        if len(TILE_CACHE) == 0:
            for height, file in hex_grid.TILES.items():
                tile_path = Path(__file__).parents[1] / file
                original_tile = pygame.image.load(str(tile_path)).convert_alpha()
                TILE_CACHE[height] = pygame.transform.smoothscale(original_tile, (hex_grid.RADIUS * 1.2, hex_grid.RADIUS * 1.2))

        # Blit the cached tile image centered on the hex
        if self.height > 0.25:
            tile_image = TILE_CACHE["peak"]
        elif self.height > 0.2:
            tile_image = TILE_CACHE["high"]
        elif self.height > 0.1:
            tile_image = TILE_CACHE["mid"]
        else:
            tile_image = TILE_CACHE["low"]

        tile_rect = tile_image.get_rect(center=self.center)
        screen.blit(tile_image, tile_rect)

        if display.DRAW_OVERLAY:
            text = display.font.render(f"({self.grid_x},{self.grid_y})", True, hex_grid.TEXT_COLOR)
            text_rect = text.get_rect(center=self.center)
            screen.blit(text, text_rect)
    
    def draw(self, screen: pygame.Surface, current_time: float, non_broken_hexes: List['Hex'] = None) -> None:
        """Draw the hex tile based on its current state.
        
        Args:
            screen: Surface to draw on
            current_time: Current game time in seconds
            non_broken_hexes: List of hexes that are not in BROKEN state (for collision detection)
        """
        if self.state == HexState.SOLID:
            self._draw_solid(screen)
        elif self.state == HexState.CRACKING:
            # Update transition progress
            elapsed = current_time - self.transition_start_time
            self.transition_progress = min(1.0, elapsed / self.transition_duration)
            
            # Draw the cracking animation
            self._draw_cracking(screen, current_time)
            
            # Check if transition is complete
            if self.transition_progress >= 1.0:
                self.state = HexState.CRACKED
                logging.debug(f'{self}: finished CRACKING')
                self.animation_manager.blocking_animations -= 1

        elif self.state == HexState.CRACKED:
            self._draw_cracked(screen)
        elif self.state == HexState.BREAKING:
            # Update transition progress
            elapsed = current_time - self.transition_start_time
            self.transition_progress = min(1.0, elapsed / self.transition_duration)
            
            # Draw the breaking animation
            self._draw_breaking(screen, current_time, non_broken_hexes)
            
            # Check if transition is complete
            if self.transition_progress >= 1.0:
                self.state = HexState.BROKEN
                logging.debug(f'{self}: finished BREAKING')
                self.animation_manager.blocking_animations -= 1

        elif self.state == HexState.BROKEN:
            self._draw_broken(screen, current_time, non_broken_hexes) 
        elif self.state == HexState.LAND:
            self._draw_land(screen)
    
    def is_broken(self) -> bool:
        """Check if the hex is in a broken state.
        
        Returns:
            True if the hex is broken or breaking, False otherwise
        """
        return self.state in [HexState.BROKEN, HexState.BREAKING]

    def get_neighbor(self, edge_index: int, hex_grid: List[List['Hex']]) -> Optional['Hex']:
        """Get the neighboring hex based on the given edge index.
        
        Args:
            edge_index: The index of the edge to find the neighbor for.
            hex_grid: The grid of hexes to search within.
            
        Returns:
            The neighboring hex if it exists, otherwise None.
        """
        # Directions for even and odd rows
        directions = [
            [(0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1)],  # even row
            [(0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]     # odd row
        ][self.grid_x % 2]
        
        if 0 <= edge_index < len(directions):
            dx, dy = directions[edge_index]
            nx, ny = self.grid_x + dx, self.grid_y + dy
            if 0 <= nx < len(hex_grid) and 0 <= ny < len(hex_grid[0]):
                return hex_grid[nx][ny]
        return None