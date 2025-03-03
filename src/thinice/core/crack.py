"""Crack component for ice breaking visualization."""
from typing import List, Tuple
import pygame
from ..config.settings import crack as config
from ..utils.geometry import Point, generate_jagged_line

class Crack:
    """Represents a crack in the ice."""
    
    def __init__(self, start_point: Point, is_secondary: bool = False):
        """Initialize a new crack.
        
        Args:
            start_point: Starting point of the crack
            is_secondary: Whether this is a secondary (thinner) crack
        """
        self.points: List[Point] = [start_point]
        self.is_secondary = is_secondary
        self.thickness = 2 if is_secondary else 3
        self.total_length = 0.0
    
    def add_point(self, point: Point) -> None:
        """Add a new point to the crack.
        
        Args:
            point: The point to add
        """
        if self.points:
            prev = self.points[-1]
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]
            self.total_length += (dx * dx + dy * dy) ** 0.5
        self.points.append(point)
    
    def extend_to(self, end_point: Point, num_segments: int) -> None:
        """Extend the crack to an endpoint with a jagged line.
        
        Args:
            end_point: The target end point
            num_segments: Number of segments in the jagged line
        """
        if not self.points:
            return
            
        new_points = generate_jagged_line(
            self.points[-1], 
            end_point,
            num_segments,
            config.MAX_DEVIATION
        )
        
        # Skip the first point as it's already in self.points
        for point in new_points[1:]:
            self.add_point(point)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the crack on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        if len(self.points) < 2:
            return
            
        # Draw shadow (wider for more visibility)
        for i in range(len(self.points) - 1):
            pygame.draw.line(
                screen,
                config.SHADOW_COLOR,
                (self.points[i][0] + 1, self.points[i][1] + 1),
                (self.points[i+1][0] + 1, self.points[i+1][1] + 1),
                self.thickness + 2
            )
        
        # Draw crack
        for i in range(len(self.points) - 1):
            pygame.draw.line(
                screen,
                config.COLOR,
                self.points[i],
                self.points[i+1],
                self.thickness
            ) 