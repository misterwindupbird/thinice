"""Entity classes for game objects that can move and be drawn."""
from abc import ABC, abstractmethod
import math
import pygame
from typing import Tuple, Optional, Any, Callable
from pathlib import Path
import logging

from .animation_manager import AnimationManager
from .hex import Hex
from .hex_state import HexState
from ..config.settings import hex_grid, display

# We'll use a TYPE_CHECKING approach to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .game import Game

IMAGE_DIR = Path(__file__).parents[1] / 'images'

class Entity(ABC):
    """Abstract base class for game entities like player and enemies."""

    _id_counter = 0

    def __init__(self, hex: Hex,
                 animation_manager: AnimationManager,
                 token: str):
        """Initialize the entity.
        
        Args:
            hex: The hex tile the entity is on
            glyph: Character to represent the entity
            color: RGB color tuple for the entity
        """
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
        self.image = pygame.transform.smoothscale(pygame.image.load(IMAGE_DIR / token), (hex_grid.RADIUS, hex_grid.RADIUS))

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
        # Calculate progress (0.0 to 1.0)
        elapsed = current_time - self.animation_start_time
        progress = min(1.0, elapsed / self.animation_duration)

        if self.animation_type == "drown":
            self._update_drown_animation(progress)
        if not self.is_moving:
            return

        self._update_regular_animation(progress)
    
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

        logging.debug(f'{self}: start drowning')

        self.animation_manager.blocking_animations += 1
        self.animation_start_time = current_time
        self.animation_duration = 0.3
        self.animation_type = "drown"

        return True

    def _update_drown_animation(self, progress: float) -> None:

        if progress >= 1.0:

            logging.debug(f'{self}: finished drowning')
            self.animation_manager.blocking_animations -= 1

            # Call the on_animation_complete callback if it exists
            if hasattr(self, 'on_animation_complete') and self.on_animation_complete:
                self.on_animation_complete()
                self.on_animation_complete = None

            self.animation_type = "dead"

        else:
            self.radius = int(20 * (1-progress))


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
            current_time: Current game time in seconds
        """

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

            logging.debug(f'{self}: finished regular animation')
            self.animation_manager.blocking_animations -= 1

            # Call the on_animation_complete callback if it exists
            if hasattr(self, 'on_animation_complete') and self.on_animation_complete:
                self.on_animation_complete()
                self.on_animation_complete = None

            self.animation_type = "none"

    def draw(self, screen: pygame.Surface, current_time: float) -> None:
        """Draw the player with special animations for jumping.

        Args:
            screen: Pygame surface to draw on
            current_time: Current game time in seconds
        """
        # Calculate position based on animation
        if self.is_moving:
            progress = min(1.0, (current_time - self.animation_start_time) / self.animation_duration)

            # Custom ease-in function: progress^3 for slow start and abrupt stop
            eased_progress = progress ** 3

            # Interpolate between start and end positions
            x = self.move_start_pos[0] * (1 - eased_progress) + self.move_end_pos[0] * eased_progress
            y = self.move_start_pos[1] * (1 - eased_progress) + self.move_end_pos[1] * eased_progress
        else:
            x, y = self.current_hex.center

        # # Load and resize the wolf token to match the previous circle size
        # token_size = self.radius * 2  # Match the diameter of the previous circle
        # scaled_token = pygame.transform.smoothscale(self.image, (token_size, token_size))
        #
        # Get rectangle for proper centering
        token_rect = self.image.get_rect(center=(x, y))

        # Draw the wolf token at the calculated position
        screen.blit(self.image, token_rect.topleft)

class Player(Entity):
    """Player entity that can move between hex tiles."""
    
    def __init__(self, start_hex, animation_manager: AnimationManager):
        """Initialize the player entity.
        
        Args:
            start_hex: The starting hex tile
        """
        super().__init__(start_hex, animation_manager,  'player_token.png')


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
    """Player entity that can move between hex tiles."""

    def __init__(self, start_hex, animation_manager: AnimationManager):
        """Initialize the player entity.

        Args:
            start_hex: The starting hex tile
        """
        super().__init__(start_hex,
                         animation_manager=animation_manager,
                         token='wolf_token.png')

        self.animation_type = "none"  # Track the type of animation: "none", "move", "jump", "sprint"
