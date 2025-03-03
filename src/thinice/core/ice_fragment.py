"""Ice fragment component for the ice breaking game."""
import math
import random
from typing import List, Tuple

import pygame

from thinice.config import animation, hex_grid


class IceFragment:
    """Represents a fragment of ice that breaks off during the breaking animation."""

    def __init__(
        self, 
        points: List[Tuple[float, float]], 
        center: Tuple[float, float],
        hex_radius: float
    ):
        """Initialize a new ice fragment.
        
        Args:
            points: List of points defining the fragment polygon
            center: Center point of the hex
            hex_radius: Radius of the hex
        """
        self.original_points = list(points)
        self.center = center
        self.hex_radius = hex_radius
        
        # Calculate fragment center
        self.fragment_center = self._calculate_center(points)
        
        # Calculate fragment size (approximate area)
        self.size = self._calculate_size(points)
        size_factor = min(1.0, self.size / (math.pi * hex_radius * hex_radius))
        
        # Movement parameters - smaller fragments move more
        # Reduce drift distance for more natural movement
        self.drift_distance = random.uniform(0.05, 0.15) * hex_radius * (1 - size_factor * 0.7)
        self.drift_angle = random.uniform(0, 2 * math.pi)
        self.drift_x = math.cos(self.drift_angle) * self.drift_distance
        self.drift_y = math.sin(self.drift_angle) * self.drift_distance
        
        # Rotation parameters - smaller fragments rotate more
        self.rotation = 0
        # Reduce rotation speed for more natural movement
        self.rotation_speed = random.uniform(-15, 15) * (1 - size_factor * 0.6)
        
        # Bobbing parameters - subtle vertical movement
        self.bob_phase = random.uniform(0, 2 * math.pi)
        # Reduce bob amplitude for more natural movement
        self.bob_amplitude = random.uniform(0.5, 1.5) * (1 - size_factor * 0.5)
        
        # Sinking parameters - improve sinking effect
        # Increase sink depth for more dramatic sinking
        self.sink_depth = random.uniform(0.7, 1.2) * hex_radius * 0.4
        # Start sinking earlier for more natural transition
        self.sink_start = random.uniform(0.1, 0.3)
        
        # Transformed points (updated during animation)
        self.transformed_points = list(points)

    def _calculate_center(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate the center of the fragment."""
        if not points:
            return self.center
            
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (x_sum / len(points), y_sum / len(points))

    def _calculate_size(self, points: List[Tuple[float, float]]) -> float:
        """Calculate approximate size (area) of the fragment."""
        if len(points) < 3:
            return 0
            
        # Simple approximation using shoelace formula
        area = 0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2

    def _point_in_hex(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the hexagon."""
        x, y = point
        dx = abs(x - self.center[0])
        dy = abs(y - self.center[1])
        
        # Quick check using bounding box
        if dx > self.hex_radius or dy > self.hex_radius:
            return False
            
        # More precise check
        # For a regular hexagon, we can check if the point is within the radius
        distance = math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        return distance <= self.hex_radius * 0.95  # Slightly smaller to keep inside visible boundary

    def _clip_to_hex(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Clip points to ensure they stay within the hex boundary."""
        clipped_points = []
        for point in points:
            if self._point_in_hex(point):
                clipped_points.append(point)
            else:
                # Find intersection with hex boundary
                # For simplicity, we'll just push the point toward the center
                dx = point[0] - self.center[0]
                dy = point[1] - self.center[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    # Scale to hex radius
                    scale = (self.hex_radius * 0.9) / distance
                    new_x = self.center[0] + dx * scale
                    new_y = self.center[1] + dy * scale
                    clipped_points.append((new_x, new_y))
                else:
                    clipped_points.append(self.center)
                    
        return clipped_points

    def update(self, progress: float) -> None:
        """Update the fragment position and rotation based on animation progress.
        
        Args:
            progress: Animation progress from 0 to 1
        """
        # Calculate drift based on progress - slow down drift over time
        drift_progress = min(1.0, progress * 1.5)  # Drift happens in first 2/3 of animation
        current_drift_x = self.drift_x * drift_progress
        current_drift_y = self.drift_y * drift_progress
        
        # Calculate bobbing effect - reduce bobbing as sinking increases
        bob_factor = max(0, 1 - (progress - self.sink_start) * 2) if progress > self.sink_start else 1.0
        bob_offset = math.sin(self.bob_phase + progress * 8) * self.bob_amplitude * bob_factor
        
        # Calculate sinking effect (starts earlier and accelerates)
        sink_amount = 0
        if progress > self.sink_start:
            # Use exponential curve for sinking to accelerate towards the end
            sink_progress = (progress - self.sink_start) / (1 - self.sink_start)
            sink_progress = sink_progress * sink_progress  # Square for acceleration
            sink_amount = self.sink_depth * sink_progress
        
        # Calculate rotation - slow down rotation as sinking increases
        rotation_factor = max(0, 1 - (progress - self.sink_start) * 1.5) if progress > self.sink_start else 1.0
        current_rotation = self.rotation + self.rotation_speed * progress * rotation_factor
        
        # Apply transformations to points
        self.transformed_points = []
        for x, y in self.original_points:
            # Calculate offset from fragment center
            dx = x - self.fragment_center[0]
            dy = y - self.fragment_center[1]
            
            # Apply rotation
            rad_angle = math.radians(current_rotation)
            rotated_x = dx * math.cos(rad_angle) - dy * math.sin(rad_angle)
            rotated_y = dx * math.sin(rad_angle) + dy * math.cos(rad_angle)
            
            # Apply drift, bob, and sink
            final_x = self.fragment_center[0] + rotated_x + current_drift_x
            final_y = self.fragment_center[1] + rotated_y + bob_offset + sink_amount
            
            self.transformed_points.append((final_x, final_y))
            
        # Ensure points stay within hex boundary
        self.transformed_points = self._clip_to_hex(self.transformed_points)

    def get_transformed_points(self) -> List[Tuple[float, float]]:
        """Get the current transformed points of the fragment."""
        return self.transformed_points

    def get_points(self) -> List[Tuple[float, float]]:
        """Get the original points of the fragment."""
        return self.original_points

    def draw(self, screen: pygame.Surface, color: Tuple[int, int, int]) -> None:
        """Draw the fragment on the screen.
        
        Args:
            screen: Surface to draw on
            color: Color of the fragment
        """
        if len(self.transformed_points) < 3:
            return  # Can't draw a polygon with less than 3 points
            
        # Draw shadow
        shadow_points = [(x + 2, y + 2) for x, y in self.transformed_points]
        pygame.draw.polygon(screen, (0, 0, 0), shadow_points)
        
        # Draw fragment
        pygame.draw.polygon(screen, color, self.transformed_points)
        pygame.draw.polygon(screen, (255, 255, 255), self.transformed_points, 1) 