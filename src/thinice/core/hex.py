"""Hex tile component for the ice breaking game."""
import random
import math
import pygame
from typing import List, Tuple, Optional

from .hex_state import HexState
from .crack import Crack
from ..config.settings import hex_grid, crack as crack_config, water
from ..utils.geometry import (
    Point, 
    calculate_hex_vertices,
    calculate_edge_points,
    point_in_hex
)


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
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.center: Point = (x, y)
        
        # Calculate vertices and edge points
        self.vertices = calculate_hex_vertices(self.center, hex_grid.RADIUS)
        self.edge_points = calculate_edge_points(self.vertices)
        
        # Initialize state and visual properties
        self.state = HexState.SOLID
        self._init_color()
        
        # Initialize components
        self.cracks: List[Crack] = []
        
        # Cached surfaces
        self.broken_surface: Optional[pygame.Surface] = None
        
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
        
        # Clear cached surfaces
        self.broken_surface = None
    
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
    
    def _draw_broken(self, screen: pygame.Surface) -> None:
        """Draw the broken state with widened cracks showing water beneath."""
        # If we have a cached surface, use it
        if self.broken_surface is not None:
            screen.blit(
                self.broken_surface,
                (self.center[0] - hex_grid.RADIUS * 1.5,
                 self.center[1] - hex_grid.RADIUS * 1.5)
            )
            return
            
        # Create a cached surface for better performance
        surface_size = int(hex_grid.RADIUS * 3)
        self.broken_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        
        # Offset for drawing on the surface
        offset_x = surface_size // 2
        offset_y = surface_size // 2
        
        # Draw the base ice 
        adjusted_vertices = [(x - self.center[0] + offset_x, y - self.center[1] + offset_y) 
                            for x, y in self.vertices]
        pygame.draw.polygon(self.broken_surface, self.color, adjusted_vertices)
        
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
            pygame.draw.lines(self.broken_surface, water_crack_color, False, adjusted_points, 10)
        
        # Draw cached surface
        screen.blit(
            self.broken_surface,
            (self.center[0] - hex_grid.RADIUS * 1.5,
             self.center[1] - hex_grid.RADIUS * 1.5)
        )
    
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
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font, current_time: float) -> None:
        """Draw the hex tile.
        
        Args:
            screen: Surface to draw on
            font: Font for coordinate display
            current_time: Current game time in seconds
        """
        if self.state == HexState.BROKEN:
            self._draw_broken(screen)
        elif self.state == HexState.CRACKED:
            self._draw_cracked(screen, font)
        else:  # SOLID state
            self._draw_solid(screen, font) 