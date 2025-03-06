import logging
import random
import math
import pygame
from typing import List, Tuple

from ..config.settings import hex_grid


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

        # Bump animation properties
        self.bump_force_x = 0
        self.bump_force_y = 0
        self.bump_decay = 0.9  # How quickly the bump effect fades
        self.bump_active = False
        self.bump_start_time = 0

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
            non_broken_hexes: List of hexes that are not in BROKEN state (for collision detection)
        """
        # Calculate time since creation
        time_since_creation = current_time - self.creation_time

        # Process bump effect if active
        if self.bump_active:
            # Apply decay to bump forces
            bump_time = current_time - self.bump_start_time
            if bump_time > 0.5 or (abs(self.bump_force_x) < 0.1 and abs(self.bump_force_y) < 0.1):
                # End bump effect if it's been active for half a second or forces are negligible
                self.bump_active = False
                self.bump_force_x = 0
                self.bump_force_y = 0
            else:
                # Apply decay
                self.bump_force_x *= self.bump_decay
                self.bump_force_y *= self.bump_decay

        # Skip movement if bob_amount or rotation is zero (during transition)
        if self.bob_amount == 0 and self.rotation == 0 and not self.bump_active:
            # Just update the sprite's position based on current offset
            self.rect.x = int(self.center[0] - self.rect.width / 2 + self.offset_x)
            self.rect.y = int(self.center[1] - self.rect.height / 2 + self.offset_y)
            return

        # Calculate potential new position
        max_drift = 15  # Maximum drift distance
        new_offset_x = min(max_drift, max(-max_drift, self.dx * time_since_creation + self.bump_force_x))
        new_offset_y = min(max_drift, max(-max_drift, self.dy * time_since_creation + self.bump_force_y))

        # Add bobbing motion
        bob_y = math.sin(current_time * self.bob_speed + self.bob_phase) * self.bob_amount
        new_offset_y += bob_y

        # Calculate new center position
        new_center_x = self.center[0] + new_offset_x
        new_center_y = self.center[1] + new_offset_y

        # Bounds checking - ensure fragment doesn't move too far from hex center
        max_distance_from_hex = hex_grid.RADIUS * 1.5
        dx_from_hex = new_center_x - self.hex_center[0]
        dy_from_hex = new_center_y - self.hex_center[1]
        distance_from_hex = math.sqrt(dx_from_hex ** 2 + dy_from_hex ** 2)

        if distance_from_hex > max_distance_from_hex:
            # Scale back to maximum allowed distance
            scale_factor = max_distance_from_hex / distance_from_hex
            new_offset_x = (self.hex_center[0] + dx_from_hex * scale_factor) - self.center[0]
            new_offset_y = (self.hex_center[1] + dy_from_hex * scale_factor) - self.center[1]
            new_center_x = self.center[0] + new_offset_x
            new_center_y = self.center[1] + new_offset_y

        # Only apply movement if it doesn't cause a collision
        if non_broken_hexes:
            collision = False
            for hex in non_broken_hexes:
                # Simple distance-based collision detection
                distance = math.sqrt((new_center_x - hex.center[0]) ** 2 + (new_center_y - hex.center[1]) ** 2)
                if distance < (self.radius + hex_grid.RADIUS - 5):  # Subtract a small buffer
                    # Calculate bounce direction (away from the hex)
                    dx = new_center_x - hex.center[0]
                    dy = new_center_y - hex.center[1]

                    # Normalize and reverse direction
                    length = math.sqrt(dx ** 2 + dy ** 2)
                    if length > 0:
                        dx /= length
                        dy /= length

                        # Reverse direction with a small bounce effect
                        self.dx = -self.dx + dx * 0.05
                        self.dy = -self.dy + dy * 0.05

                        # Push the fragment away from the collision point
                        # Calculate minimum distance needed to resolve collision
                        push_distance = (self.radius + hex_grid.RADIUS - 5) - distance
                        if push_distance > 0:
                            # Move fragment away from hex by push_distance
                            new_offset_x = self.offset_x + dx * push_distance
                            new_offset_y = self.offset_y + dy * push_distance

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

    def apply_bump(self, force: float = 5.0) -> None:
        """Apply a bump force to this fragment from the hex center.
        
        Args:
            force: Base force magnitude of the bump
        """
        # Calculate direction from hex center to fragment
        dx = self.center[0] + self.offset_x - self.hex_center[0]
        dy = self.center[1] + self.offset_y - self.hex_center[1]
        
        # Calculate distance
        distance = math.sqrt(dx**2 + dy**2)
        
        # Normalize direction
        if distance > 0:
            dx /= distance
            dy /= distance
            
            # Apply force in the direction away from hex center
            self.bump_force_x = dx * force
            self.bump_force_y = dy * force
            
            # Add a bit of random variation
            self.bump_force_x += random.uniform(-0.5, 0.5)
            self.bump_force_y += random.uniform(-0.5, 0.5)
            
            # Activate bump effect
            self.bump_active = True
            self.bump_start_time = pygame.time.get_ticks() / 1000.0
            
            # Also add a small rotation impulse
            self.rotation += random.uniform(-0.02, 0.02) * force

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
