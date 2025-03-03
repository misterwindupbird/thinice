"""Hex tile component for the ice breaking game."""
import random
import math
import pygame
from typing import List, Tuple, Optional, Dict, Set

from .hex_state import HexState
from .crack import Crack
from ..config.settings import hex_grid, crack as crack_config, water
from ..utils.geometry import (
    Point, 
    calculate_hex_vertices,
    calculate_edge_points,
    point_in_hex
)


class IceFragment(pygame.sprite.Sprite):
    """Represents a floating ice fragment that can be animated."""
    
    def __init__(self, points: List[Tuple[float, float]], color: Tuple[int, int, int], 
                 center: Tuple[float, float], hex_center: Tuple[float, float]):
        """Initialize a new ice fragment.
        
        Args:
            points: List of points defining the fragment polygon
            color: RGB color of the fragment
            center: Center point of the fragment
            hex_center: Center point of the parent hex
        """
        super().__init__()
        
        # Store original points and properties
        self.original_points = points.copy()
        self.color = color
        self.center = center
        self.hex_center = hex_center
        
        # Calculate bounding box for the sprite
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        
        # Create a surface for the fragment
        width = max(1, int(max_x - min_x + 4))  # Add padding
        height = max(1, int(max_y - min_y + 4))
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Adjust points to the surface coordinates
        self.surface_points = [(p[0] - min_x + 2, p[1] - min_y + 2) for p in points]
        
        # Draw the fragment on the surface
        if len(self.surface_points) >= 3:
            pygame.draw.polygon(self.image, self.color, self.surface_points)
            pygame.draw.polygon(self.image, (200, 220, 230), self.surface_points, 1)  # Outline
        
        # Set the sprite's rectangle
        self.rect = self.image.get_rect()
        self.rect.x = int(min_x)
        self.rect.y = int(min_y)
        
        # Animation properties
        self.dx = random.uniform(-0.3, 0.3)
        self.dy = random.uniform(-0.3, 0.3)
        self.rotation = random.uniform(-0.05, 0.05)
        self.bob_phase = random.random() * 2 * math.pi
        self.bob_speed = random.uniform(0.5, 1.5)
        self.bob_amount = random.uniform(0.3, 0.8)
        
        # Current state
        self.angle = 0
        self.offset_x = 0
        self.offset_y = 0
        self.creation_time = pygame.time.get_ticks() / 1000.0
        
        # Collision properties
        self.radius = max(width, height) / 2  # Simple circular collision
    
    def update(self, current_time: float, non_broken_hexes: List['Hex'] = None) -> None:
        """Update the fragment's position and rotation.
        
        Args:
            current_time: Current game time in seconds
            non_broken_hexes: List of hexes that are not in BROKEN state
        """
        # Calculate time since creation
        time_since_creation = current_time - self.creation_time
        
        # Calculate potential new position
        max_drift = 15  # Maximum drift distance
        new_offset_x = min(max_drift, self.dx * time_since_creation)
        new_offset_y = min(max_drift, self.dy * time_since_creation)
        
        # Add bobbing motion
        bob_y = math.sin(current_time * self.bob_speed + self.bob_phase) * self.bob_amount
        new_offset_y += bob_y
        
        # Check for collisions with non-broken hexes
        new_center_x = self.center[0] + new_offset_x
        new_center_y = self.center[1] + new_offset_y
        
        # Only apply movement if it doesn't cause a collision
        if non_broken_hexes:
            collision = False
            for hex in non_broken_hexes:
                # Simple distance-based collision detection
                distance = math.sqrt((new_center_x - hex.center[0])**2 + (new_center_y - hex.center[1])**2)
                if distance < (self.radius + hex_grid.RADIUS - 5):  # Subtract a small buffer
                    # Calculate bounce direction (away from the hex)
                    dx = new_center_x - hex.center[0]
                    dy = new_center_y - hex.center[1]
                    
                    # Normalize and reverse direction
                    length = math.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx /= length
                        dy /= length
                        
                        # Reverse direction with a small bounce effect
                        self.dx = -self.dx + dx * 0.05
                        self.dy = -self.dy + dy * 0.05
                    
                    collision = True
                    break
            
            if not collision:
                self.offset_x = new_offset_x
                self.offset_y = new_offset_y
        else:
            # No collision checking, just apply the movement
            self.offset_x = new_offset_x
            self.offset_y = new_offset_y
        
        # Update rotation
        max_rotation = math.pi / 6  # 30 degrees
        self.angle = min(max_rotation, self.rotation * time_since_creation)
        
        # Update the sprite's position
        self.rect.x = int(self.center[0] - self.rect.width / 2 + self.offset_x)
        self.rect.y = int(self.center[1] - self.rect.height / 2 + self.offset_y)
        
        # If we need to update the image due to rotation
        if abs(self.angle) > 0.01:
            self._update_rotated_image()
    
    def _update_rotated_image(self) -> None:
        """Update the sprite's image with the current rotation."""
        # Create a new surface for the rotated fragment
        width = self.image.get_width()
        height = self.image.get_height()
        
        # Create a new surface
        new_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Rotate the points around the center of the surface
        center_x = width / 2
        center_y = height / 2
        
        rotated_points = []
        for point in self.surface_points:
            # Calculate point relative to center
            rel_x = point[0] - center_x
            rel_y = point[1] - center_y
            
            # Apply rotation
            rot_x = rel_x * math.cos(self.angle) - rel_y * math.sin(self.angle)
            rot_y = rel_x * math.sin(self.angle) + rel_y * math.cos(self.angle)
            
            # Translate back
            rotated_points.append((rot_x + center_x, rot_y + center_y))
        
        # Draw the rotated fragment
        if len(rotated_points) >= 3:
            pygame.draw.polygon(new_surface, self.color, rotated_points)
            pygame.draw.polygon(new_surface, (200, 220, 230), rotated_points, 1)  # Outline
        
        # Update the image
        self.image = new_surface

class Hex:
    """Represents a hexagonal tile in the game grid."""
    
    def __init__(self, x: float, y: float, grid_x: int, grid_y: int):
        """Initialize a new hex tile.
        
        Args:
            x: Center x coordinate
            y: Center y coordinate
            grid_x: Grid column index
            grid_y: Grid row index
        """
        self.center = (x, y)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.vertices = calculate_hex_vertices(self.center, hex_grid.RADIUS)
        self.edge_points = calculate_edge_points(self.vertices)
        self.state = HexState.SOLID
        self.cracks: List[Crack] = []
        self._init_color()
        
        # For broken ice fragments
        self.ice_fragments = []
        self.fragment_colors = []
        self.broken_surface = None
        self.fragment_sprites = pygame.sprite.Group()
        self.break_time = 0
        
    def _init_color(self) -> None:
        """Initialize the hex color with slight random variations."""
        grey_var = random.randint(-5, 5)  # Base variation in grey
        blue_var = random.randint(0, 3)   # Slight additional blue variation
        
        base_r, base_g, base_b = hex_grid.ICE_BASE_COLOR
        self.color = (
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
        if not point_in_hex(end_point, self.center, hex_grid.RADIUS):
            return None
            
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
            
        self.state = HexState.CRACKED  # Directly transition to CRACKED
        
        # Connect to cracked neighbors
        for neighbor in neighbors:
            if neighbor.state != HexState.CRACKED:
                continue
                
            edge_index = self.get_shared_edge_index(neighbor)
            if edge_index == -1:
                continue
                
            shared_point = self.edge_points[edge_index]
            
            # Ensure neighbor has connecting crack
            if not neighbor.has_crack_to_point(shared_point):
                neighbor.add_straight_crack(shared_point)
            
            # Add our connecting crack
            self.add_straight_crack(shared_point)
        
        # Add minimum number of cracks
        while len(self.cracks) < random.randint(
            crack_config.MIN_CRACKS,
            crack_config.MAX_CRACKS
        ):
            # Pick unused edge
            used_edges = {crack.points[-1] for crack in self.cracks}
            available = [
                i for i, point in enumerate(self.edge_points)
                if point not in used_edges
            ]
            
            if not available:
                break
                
            self.add_straight_crack(self.edge_points[random.choice(available)])
        
        # Add secondary cracks
        self.add_secondary_cracks()
    
    def break_ice(self) -> None:
        """Break the ice by adding the minimum necessary cracks.
        Transitions to the BROKEN state.
        """
        if self.state != HexState.CRACKED:
            return
            
        print(f"Breaking ice at ({self.grid_x}, {self.grid_y})")
        
        # Add edge cracks to ensure ice can detach from neighboring hexes
        self._add_edge_cracks()
        print(f"Total cracks after adding edge cracks: {len(self.cracks)}")
        
        # Transition to BROKEN state
        self.state = HexState.BROKEN
        self.break_time = pygame.time.get_ticks() / 1000.0
        
        # Clear cached surfaces
        self.broken_surface = None
        
        # Clear any existing fragment sprites
        self.fragment_sprites.empty()
    
    def _add_edge_cracks(self) -> None:
        """Add cracks along the perimeter of the hex to detach fragments."""
        print("Adding edge cracks")
        
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
        
        print(f"Total cracks after adding edge cracks: {len(self.cracks)}")
    
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
        for y in range(grid_size):
            for x in range(grid_size):
                world_x, world_y = grid_to_world(x, y)
                if self._point_in_hex((world_x, world_y)):
                    grid[y][x] = True
        
        # Draw cracks on the grid (set cells to False)
        for crack in self.cracks:
            if len(crack.points) < 2:
                continue
                
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
                            fragments.append(hull)
        
        # Store fragments and use the original hex color for all fragments
        self.ice_fragments = fragments
        self.fragment_colors = []
        
        for _ in range(len(fragments)):
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
            pygame.draw.polygon(temp_surface, self.color, adjusted_vertices)
            
            # Draw widened cracks with water color
            water_crack_color = water.CRACK_COLOR  # Use the color from settings
            for crack in self.cracks:
                # Skip invalid cracks
                if len(crack.points) < 2:
                    continue
                    
                # Draw widened cracks showing water underneath
                adjusted_points = [(p[0] - self.center[0] + offset_x, p[1] - self.center[1] + offset_y) 
                                for p in crack.points]
                                
                # Draw water through cracks - using water color
                pygame.draw.lines(temp_surface, water_crack_color, False, adjusted_points, 10)
            
            # Now identify and color the fragments using a simple flood fill
            self.broken_surface = self._color_fragments(temp_surface, surface_size)
        
        # Draw the water underneath
        pygame.draw.polygon(screen, water.BASE_COLOR, self.vertices)
        
        # Update and draw the fragment sprites
        self.fragment_sprites.update(current_time, non_broken_hexes)
        self.fragment_sprites.draw(screen)
    
    def _draw_cracked(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
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
        text = font.render(f"({self.grid_x},{self.grid_y})", True, hex_grid.TEXT_COLOR)
        text_rect = text.get_rect(center=self.center)
        screen.blit(text, text_rect)
        
    
    def _draw_solid(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the solid state."""
        pygame.draw.polygon(screen, self.color, self.vertices)
        # Border removed
        text = font.render(f"({self.grid_x},{self.grid_y})", True, hex_grid.TEXT_COLOR)
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
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font, current_time: float, non_broken_hexes: List['Hex'] = None) -> None:
        """Draw the hex tile based on its current state.
        
        Args:
            screen: Surface to draw on
            font: Font for text rendering
            current_time: Current game time in seconds
            non_broken_hexes: List of hexes that are not in BROKEN state (for collision detection)
        """
        if self.state == HexState.SOLID:
            self._draw_solid(screen, font)
        elif self.state == HexState.CRACKED:
            self._draw_cracked(screen, font)
        elif self.state == HexState.BROKEN:
            self._draw_broken(screen, current_time, non_broken_hexes) 