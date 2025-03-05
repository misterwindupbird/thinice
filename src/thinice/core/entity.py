"""Entity classes for game objects that can move and be drawn."""
from abc import ABC, abstractmethod
import math
import pygame
from typing import Tuple, Optional, Any
import time
import logging

from .hex import Hex
from .hex_state import HexState
from ..config.settings import display

# We'll use a TYPE_CHECKING approach to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .game import Game

class Entity(ABC):
    """Abstract base class for game entities like player and enemies."""
    
    def __init__(self, hex: Hex, glyph: str = "?", color: Tuple[int, int, int] = (255, 255, 255)):
        """Initialize the entity.
        
        Args:
            hex: The hex tile the entity is on
            glyph: Character to represent the entity
            color: RGB color tuple for the entity
        """
        self.current_hex = hex
        self.target_hex = None
        self.color = color
        self.glyph = glyph
        self.radius = 20  # Default radius
        self.position = hex.center  # Current position (for animation)
        
        # Animation properties
        self.animation_start_time = 0
        self.animation_duration = 0.3  # seconds
        self.is_moving = False
        self.move_start_pos = (0, 0)
        self.move_end_pos = (0, 0)
        
        # Callback for when animation completes
        self.on_animation_complete = None
    
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
        # Start animation
        self.target_hex = target_hex
        self.is_moving = True
        self.animation_start_time = current_time
        self.animation_duration = 0.3  # Regular move is faster
        self.animation_type = "move"

        self.move_start_pos = self.current_hex.center
        self.move_end_pos = target_hex.center
        
        # Signal that movement has started (for floating text)
        self.on_move_start()
        
        return True

    def on_move_start(self) -> None:
        """Called when movement starts. Can be overridden by subclasses."""
        pass
    
    def get_adjacent_hexes(self) -> list:
        """Get list of adjacent hexes.
        
        Returns:
            List of adjacent hex tiles
        """
        # Get the game instance (this is a bit of a hack, but works for now)
        # We use a function to get the Game class to avoid circular imports
        game_instance = self._get_game_instance()
        if game_instance:
            return game_instance.get_hex_neighbors(self.current_hex)
        return []
    
    def _get_game_instance(self) -> Any:
        """Get the Game instance safely without circular imports.
        
        Returns:
            The Game instance or None
        """
        # This is a workaround to avoid circular imports
        # We import Game only when needed
        import sys
        if 'src.thinice.core.game' in sys.modules:
            return sys.modules['src.thinice.core.game'].Game.instance
        return None
    
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
            
            # Custom ease-in function: progress^3 for slow start and abrupt stop
            eased_progress = progress ** 3
            
            # Interpolate between start and end positions
            x = self.move_start_pos[0] * (1 - eased_progress) + self.move_end_pos[0] * eased_progress
            y = self.move_start_pos[1] * (1 - eased_progress) + self.move_end_pos[1] * eased_progress
            
            # Remove any jump arc
            # Check if animation is complete
            if progress >= 1.0:
                self.is_moving = False
                self.current_hex = self.target_hex
                self.target_hex = None
                x, y = self.current_hex.center
                
                # Call the animation complete callback if it exists
                if self.on_animation_complete:
                    self.on_animation_complete()
                    self.on_animation_complete = None
        else:
            x, y = self.position
        
        # Draw the entity as a circle with the glyph
        pygame.draw.circle(screen, self.color, (x, y), self.radius)
        
        # Draw the glyph
        font = pygame.font.SysFont(display.FONT_NAME, int(self.radius * 1.5))
        text = font.render(self.glyph, True, (255, 255, 255))  # White text for better visibility
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
        
class Player(Entity):
    """Player entity that can move between hex tiles."""
    
    def __init__(self, start_hex):
        """Initialize the player entity.
        
        Args:
            start_hex: The starting hex tile
        """
        super().__init__(start_hex, "@", (255, 100, 100))
        self.animation_type = "none"  # Track the type of animation: "none", "move", "jump", "sprint"
        self.sprint_path = []  # Store the path for sprint animation
        self.sprint_current_index = 0  # Current position in the sprint path
        self.sprint_next_time = 0  # Time to move to next hex in sprint
    

    def jump(self, target_hex, current_time):
        """Perform a jump to a target hex that is 2 steps away.
        
        Args:
            target_hex: The target hex to jump to
            current_time: Current game time in seconds
        """
        # Start animation with longer duration
        self.target_hex = target_hex
        self.is_moving = True
        self.animation_start_time = current_time
        self.animation_duration = 0.5  # Longer duration for jump
        self.move_start_pos = self.current_hex.center
        self.move_end_pos = target_hex.center
        self.animation_type = "jump"
    
    def sprint(self, path, current_time):
        """Move directly to the end hex of the path with a single animation.
        
        Args:
            path: List of hexes to sprint through (including start and end)
            current_time: Current game time in seconds
        """
        if len(path) < 2:
            return

        # Move directly to the last hex in the path
        end_hex = path[-1]
        self.target_hex = end_hex
        self.is_moving = True
        self.animation_type = "sprint"
        self.animation_start_time = current_time
        self.move_start_pos = self.current_hex.center
        self.move_end_pos = end_hex.center
        self.animation_duration = 0.5  # Longer duration for sprint

        # Add floating text for SPRINT
        game_instance = self._get_game_instance()
        if game_instance:
            game_instance.add_floating_text("SPRINT", self.current_hex.center, (255, 0, 0))

        # Start the move
        self.is_moving = True

    def update(self, current_time):
        """Update the player animation.
        
        Args:
            current_time: Current game time in seconds
        """
        if not self.is_moving:
            return
        
        if self.animation_type == "sprint":
            self._update_sprint(current_time)
        else:
            self._update_regular_animation(current_time)

    def _update_regular_animation(self, current_time):
        """Update regular move or jump animation.

        Args:
            current_time: Current game time in seconds
        """
        # Calculate progress (0.0 to 1.0)
        elapsed = current_time - self.animation_start_time
        progress = min(1.0, elapsed / self.animation_duration)

        # Interpolate position
        self.position = (
            self.move_start_pos[0] + (self.move_end_pos[0] - self.move_start_pos[0]) * progress,
            self.move_start_pos[1] + (self.move_end_pos[1] - self.move_start_pos[1]) * progress
        )

        # Check if animation is complete
        if progress >= 1.0:
            self.is_moving = False
            self.current_hex = self.target_hex
            self.position = self.current_hex.center

            # Call the on_animation_complete callback if it exists
            if hasattr(self, 'on_animation_complete') and self.on_animation_complete:
                self.on_animation_complete()
                self.on_animation_complete = None

            self.animation_type = "none"
    
    def _update_sprint(self, current_time):
        """This method is no longer needed and has been removed."""
        pass
    
    def on_move_start(self):
        """Called when the player starts moving."""
        # Create floating text for movement
        game_instance = self._get_game_instance()
        if game_instance:
            game_instance.add_floating_text("MOVE", self.current_hex.center, (255, 100, 100))

    def draw(self, screen: pygame.Surface, current_time: float, scroll_x: float = 0, scroll_y: float = 0) -> None:
        """Draw the player with special animations for jumping.

        Args:
            screen: Pygame surface to draw on
            current_time: Current game time in seconds
            scroll_x: Horizontal scroll offset (default 0)
            scroll_y: Vertical scroll offset (default 0)
        """
        # Calculate position based on animation
        if self.is_moving:
            progress = min(1.0, (current_time - self.animation_start_time) / self.animation_duration)

            # Custom ease-in function: progress^3 for slow start and abrupt stop
            eased_progress = progress ** 3

            # Interpolate between start and end positions
            x = self.move_start_pos[0] * (1 - eased_progress) + self.move_end_pos[0] * eased_progress
            y = self.move_start_pos[1] * (1 - eased_progress) + self.move_end_pos[1] * eased_progress

            # Remove any jump arc
            # Check if animation is complete
            if progress >= 1.0:
                self.is_moving = False
                self.current_hex = self.target_hex
                self.target_hex = None
                x, y = self.current_hex.center

                # Call the animation complete callback if it exists
                if self.on_animation_complete:
                    self.on_animation_complete()
                    self.on_animation_complete = None
        else:
            x, y = self.current_hex.center

        # Draw the player as a circle with the glyph
        pygame.draw.circle(screen, self.color, (x, y), self.radius)

        # Draw the glyph in white for better visibility
        font = pygame.font.SysFont(display.FONT_NAME, int(self.radius * 1.5))
        text = font.render(self.glyph, True, (255, 255, 255))  # White text
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)


class Wolf(Entity):
    """Player entity that can move between hex tiles."""

    def __init__(self, start_hex):
        """Initialize the player entity.

        Args:
            start_hex: The starting hex tile
        """
        super().__init__(start_hex, "W", (255, 100, 100))
        self.animation_type = "none"  # Track the type of animation: "none", "move", "jump", "sprint"
