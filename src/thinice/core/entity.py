"""Entity classes for game objects that can move and be drawn."""
from abc import ABC, abstractmethod
import math
import pygame
from typing import Tuple, Optional
import time

from .hex import Hex
from ..config.settings import display

class Entity(ABC):
    """Abstract base class for game entities like player and enemies."""
    
    def __init__(self, hex: Hex, color: Tuple[int, int, int] = (255, 255, 255)):
        """Initialize the entity.
        
        Args:
            hex: The hex tile the entity is on
            color: RGB color tuple for the entity
        """
        self.current_hex = hex
        self.target_hex = None
        self.color = color
        self.glyph = "?"  # Default glyph, should be overridden
        self.radius = 20  # Default radius
        
        # Animation properties
        self.animation_start_time = 0
        self.animation_duration = 0.3  # seconds
        self.is_moving = False
        self.move_start_pos = (0, 0)
        self.move_end_pos = (0, 0)
    
    @abstractmethod
    def update(self, current_time: float) -> None:
        """Update the entity state.
        
        Args:
            current_time: Current game time in seconds
        """
        pass
    
    def move(self, target_hex: Hex, current_time: float) -> bool:
        """Start moving to an adjacent hex with animation.
        
        Args:
            target_hex: The hex to move to
            current_time: Current game time in seconds
            
        Returns:
            True if movement started, False if invalid move
        """
        # Check if target is adjacent to current hex
        if target_hex not in self.get_adjacent_hexes():
            print(f"Cannot move to non-adjacent hex ({target_hex.grid_x}, {target_hex.grid_y})")
            return False
            
        # Check if target is broken
        if target_hex.is_broken():
            print(f"Cannot move to broken hex ({target_hex.grid_x}, {target_hex.grid_y})")
            return False
            
        # Start animation
        self.target_hex = target_hex
        self.is_moving = True
        self.animation_start_time = current_time
        self.move_start_pos = self.current_hex.center
        self.move_end_pos = target_hex.center
        
        print(f"Moving from ({self.current_hex.grid_x}, {self.current_hex.grid_y}) to ({target_hex.grid_x}, {target_hex.grid_y})")
        return True
    
    def get_adjacent_hexes(self) -> list:
        """Get list of adjacent hexes.
        
        Returns:
            List of adjacent hex tiles
        """
        from .game import Game  # Import here to avoid circular imports
        
        # Get the game instance (this is a bit of a hack, but works for now)
        game = Game.instance
        return game.get_hex_neighbors(self.current_hex)
    
    def draw(self, screen: pygame.Surface, current_time: float, scroll_x: float = 0, scroll_y: float = 0) -> None:
        """Draw the entity on the screen.
        
        Args:
            screen: Pygame surface to draw on
            current_time: Current game time in seconds
            scroll_x: Horizontal scroll offset (default 0)
            scroll_y: Vertical scroll offset (default 0)
        """
        # Calculate position based on animation
        if self.is_moving:
            progress = min(1.0, (current_time - self.animation_start_time) / self.animation_duration)
            
            # Ease in-out function: progress^2 * (3 - 2 * progress)
            eased_progress = progress * progress * (3 - 2 * progress)
            
            # Interpolate between start and end positions
            x = self.move_start_pos[0] * (1 - eased_progress) + self.move_end_pos[0] * eased_progress
            y = self.move_start_pos[1] * (1 - eased_progress) + self.move_end_pos[1] * eased_progress
            
            # Check if animation is complete
            if progress >= 1.0:
                self.is_moving = False
                self.current_hex = self.target_hex
                self.target_hex = None
                x, y = self.current_hex.center
        else:
            x, y = self.current_hex.center
        
        # Draw the entity as a circle with glyph
        # pygame.draw.circle(screen, self.color, (x, y), self.radius)
        
        # Draw the glyph
        font = pygame.font.SysFont(display.FONT_NAME, int(self.radius * 1.5))
        text = font.render(self.glyph, True, (0, 0, 0))
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
        

class Player(Entity):
    """Player entity controlled by the user."""
    
    def __init__(self, hex: Hex):
        """Initialize the player.
        
        Args:
            hex: The hex tile the player starts on
        """
        super().__init__(hex, color=(255, 0, 0))  # Red color
        self.glyph = "@"
        self.radius = 20
    
    def update(self, current_time: float) -> None:
        """Update the player state.
        
        Args:
            current_time: Current game time in seconds
        """
        # Currently just handles animation updates, which are done in the base class draw method
        pass 