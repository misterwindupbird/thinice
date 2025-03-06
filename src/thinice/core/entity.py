"""Entity classes for game objects that can move and be drawn."""
from abc import ABC, abstractmethod
import math
import pygame
from typing import Tuple, Optional, Any, Callable
from pathlib import Path
import logging
import random

from .animation_manager import AnimationManager
from .hex import Hex
from .hex_state import HexState
from ..config.settings import hex_grid, display

# We'll use a TYPE_CHECKING approach to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .game import Game

IMAGE_DIR = Path(__file__).parents[1] / 'images'

class Entity(pygame.sprite.Sprite, ABC):
    """Abstract base class for game entities like player and enemies."""

    _id_counter = 0

    def __init__(self, hex: Hex,
                 animation_manager: AnimationManager,
                 token: str):
        """Initialize the entity.
        
        Args:
            hex: The hex tile the entity is on
            animation_manager: The animation manager
            token: Image filename for the entity's token
        """
        # Initialize the Sprite base class
        pygame.sprite.Sprite.__init__(self)
        
        self.id = Entity._id_counter
        Entity._id_counter += 1

        self.current_hex = hex
        self.target_hex = None
        self.radius = 20  # Default radius
        self.position = hex.center  # Current position (for animation)
        
        # Animation properties
        self.animation_type = "none"
        self.animation_start_time = 0
        self.animation_duration = 0.3  # seconds
        self.is_moving = False
        self.move_start_pos = (0, 0)
        self.move_end_pos = (0, 0)
        self.animation_manager = animation_manager
        
        # Load the image and set up the rect for Sprite
        self.original_image = pygame.image.load(IMAGE_DIR / token)
        self.image = pygame.transform.smoothscale(self.original_image, (hex_grid.RADIUS, hex_grid.RADIUS))
        self.rect = self.image.get_rect(center=self.position)

        # Callback for when animation completes
        self.on_animation_complete = None

    def __repr__(self):
        return f"{self.__class__.__name__}(ID={self.id})"

    def __eq__(self, other):
        """Entities are equal if they have the same ID."""
        return isinstance(other, Entity) and self.id == other.id

    def __hash__(self):
        """Hash based on unique ID, making it usable in sets/dicts."""
        return hash(self.id)

    def update(self, current_time: float) -> None:
        """Update the entity state.
        
        Args:
            current_time: Current game time in seconds
        """
        # If the entity is dead, don't update it
        if self.animation_type == "dead":
            # Just log occasionally to avoid spamming the console
            if random.random() < 0.01:  # Only log about 1% of the time
                logging.debug(f"{self} is dead, not updating")
            return
            
        # Calculate progress (0.0 to 1.0)
        elapsed = current_time - self.animation_start_time
        progress = min(1.0, elapsed / self.animation_duration)

        if self.animation_type != "none":
            logging.debug(f"{self} Updated {self.animation_type}: {progress}")
            
        # Handle different animation types
        if self.animation_type == "drown":
            self._update_drown_animation(progress)
            return
        elif not self.is_moving and self.animation_type not in ["move", "jump", "sprint", "pushed"]:
            # No animation in progress
            return

        # Handle regular movement animations
        self._update_regular_animation(progress)
        
        # Update the rect position for Sprite
        self.rect.center = self.position
    
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
        logging.debug(f'{self}: start moving')
        self.animation_manager.blocking_animations += 1

    def drown(self, current_time: float) -> bool:
        """Start drowning animation.
        
        Args:
            current_time: Current game time in seconds
            
        Returns:
            True if drowning started
        """
        logging.debug(f'{self}: start drowning at time {current_time}')

        # Make sure we're not already drowning
        if self.animation_type == "drown":
            logging.debug(f'{self}: already drowning, ignoring duplicate call')
            return False

        self.animation_manager.blocking_animations += 1
        self.animation_start_time = current_time
        self.animation_duration = 0.5  # Slightly longer for drowning animation
        self.animation_type = "drown"
        
        # Reset any movement flags
        self.is_moving = False
        
        return True

    def _update_drown_animation(self, progress: float) -> None:
        """Update drowning animation.
        
        Args:
            progress: Animation progress from 0.0 to 1.0
        """

        if progress >= 1.0:
            logging.info(f'{self}: finished drowning, marking as dead')
            self.animation_manager.blocking_animations -= 1

            # Call the on_animation_complete callback if it exists
            if hasattr(self, 'on_animation_complete') and self.on_animation_complete:
                callback = self.on_animation_complete
                self.on_animation_complete = None
                callback()  # Call the callback after clearing it

            # Mark as dead - this will be picked up by the all_animations_completed_callback
            self.animation_type = "dead"
            logging.info(f'{self}: animation_type set to "dead"')
        else:
            # Scale the image down as the entity drowns
            self.radius = int(20 * (1-progress))
            new_size = int(hex_grid.RADIUS * (1-progress))
            if new_size > 0:  # Prevent scaling to zero which would cause errors
                self.image = pygame.transform.smoothscale(self.original_image, (new_size, new_size))
                self.rect = self.image.get_rect(center=self.position)

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

    def _update_regular_animation(self, progress):
        """Update regular move or jump animation.

        Args:
            progress: Animation progress from 0.0 to 1.0
        """
        # Interpolate position
        self.position = (
            self.move_start_pos[0] + (self.move_end_pos[0] - self.move_start_pos[0]) * progress,
            self.move_start_pos[1] + (self.move_end_pos[1] - self.move_start_pos[1]) * progress
        )
        
        # Update the rect position for Sprite
        self.rect.center = self.position

        # Check if animation is complete
        if progress >= 1.0:
            self.is_moving = False
            self.current_hex = self.target_hex
            self.position = self.current_hex.center
            self.rect.center = self.position  # Update rect one final time

            logging.debug(f'{self}: finished regular animation')
            self.animation_manager.blocking_animations -= 1

            # Call the on_animation_complete callback if it exists
            if hasattr(self, 'on_animation_complete') and self.on_animation_complete:
                callback = self.on_animation_complete
                self.on_animation_complete = None
                callback()  # Call the callback after clearing it to avoid recursion issues

            # Reset animation type only if it's not being changed by the callback
            # This allows drowning to start properly after a push
            if self.animation_type in ["move", "jump", "sprint", "pushed"]:
                self.animation_type = "none"

    def draw(self, screen: pygame.Surface, current_time: float) -> None:
        """Draw the entity.
        
        This method is kept for backward compatibility.
        In a full sprite-based system, you would use sprite group's draw method instead.

        Args:
            screen: Pygame surface to draw on
            current_time: Current game time in seconds
        """
        # The sprite's image and rect are already updated in the update method
        # Just blit the image at the rect position
        screen.blit(self.image, self.rect.topleft)

class Player(Entity):
    """Player entity that can move between hex tiles."""
    
    def __init__(self, start_hex, animation_manager: AnimationManager):
        """Initialize the player entity.
        
        Args:
            start_hex: The starting hex tile
            animation_manager: The animation manager
        """
        super().__init__(start_hex, animation_manager, 'player_token.png')

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

        logging.debug(f'{self}: started jump')
        self.animation_manager.blocking_animations += 1
    
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

        logging.debug(f'{self}: started sprint')
        self.animation_manager.blocking_animations += 1

    def on_move_start(self):
        """Called when the player starts moving."""
        super().on_move_start()

        # Create floating text for movement
        game_instance = self._get_game_instance()
        if game_instance:
            game_instance.add_floating_text("MOVE", self.current_hex.center, (255, 100, 100))

class Wolf(Entity):
    """Enemy wolf entity that can be pushed by the player."""

    def __init__(self, start_hex, animation_manager: AnimationManager):
        """Initialize the wolf entity.

        Args:
            start_hex: The starting hex tile
            animation_manager: The animation manager
        """
        super().__init__(start_hex,
                         animation_manager=animation_manager,
                         token='wolf_token.png')

        self.animation_type = "none"  # Track the type of animation: "none", "move", "jump", "sprint"

    def pushed(self, target_hex: Hex, current_time: float) -> None:
        """Start pushed animation when the wolf is pushed by the player.
        
        Args:
            target_hex: The hex to push to
            current_time: Current game time in seconds
        """
        self.target_hex = target_hex
        self.is_moving = True
        self.animation_start_time = current_time
        self.animation_duration = 0.2  # Regular move is faster
        self.animation_type = "pushed"
        self.move_start_pos = self.current_hex.center
        self.move_end_pos = target_hex.center

        logging.debug(f'{self}: pushed to {self.target_hex}')
        self.animation_manager.blocking_animations += 1

