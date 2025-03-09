"""Main game class for the ice breaking game."""
from typing import List, Optional, Tuple
import os
import tkinter as tk
import pygame
from scipy.stats import energy_distance
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import random
import logging
from enum import Enum, auto
from pathlib import Path
import textwrap

from .animation_manager import AnimationManager
from .hex import Hex
from .hex_state import HexState
from .entity import Player, Wolf, HealthRestore
from .floating_text import FloatingText
from ..config import settings
from ..config.settings import worldgen, game_settings, game_over_messages, health_restore_messages, hex_grid

# Add a test logging statement to verify logging is working
logging.info("Logging test: Game started")

HEART_CACHE = dict()
DEBUG_AREA_NAVIGATION = False

class Area:
    """Stores the state of a 21x15 area when it's not active."""
    
    def __init__(self, enemies):
        # Store hex states and colors
        self.hex_states = [[None for _ in range(settings.hex_grid.GRID_HEIGHT)] 
                          for _ in range(settings.hex_grid.GRID_WIDTH)]
        self.hex_colors = [[None for _ in range(settings.hex_grid.GRID_HEIGHT)] 
                          for _ in range(settings.hex_grid.GRID_WIDTH)]
        self.hex_heightmap = [[None for _ in range(settings.hex_grid.GRID_HEIGHT)]
                          for _ in range(settings.hex_grid.GRID_WIDTH)]

        # Store complete hex data for BROKEN and CRACKED hexes
        self.broken_hex_data = {}  # Key: (x, y), Value: dict of hex properties for BROKEN hexes
        self.cracked_hex_data = {}  # Key: (x, y), Value: dict of hex properties for CRACKED hexes
        
        # Just store enemy count, not their positions
        self.enemy_count = enemies
        
        # Track if this area has been generated yet
        self.generated = False
        self.has_health_restore = True

# Import WorldGenerator here, after Area is defined
from .world_generator import WorldGenerator


class GameRestartHandler(FileSystemEventHandler):
    """File system event handler for game auto-restart."""
    
    def on_modified(self, event):
        """Handle file modification event.
        
        Args:
            event: File system event
        """
        if event.src_path.endswith('.py'):
            # Get the current directory
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # Use -m to run as a module instead of a script
            python = sys.executable
            os.chdir(current_dir)  # Change to the project root directory
            os.execl(python, python, "-m", "thinice")

class GameState(Enum):
    PLAYER = auto() # waiting for player action
    ENEMY = auto()  # perform enemy actions

    def __str__(self) -> str:
        return self.name.lower().capitalize()

class Game:
    """Main game class managing the game loop and hex grid."""
    
    # Singleton instance for access from Entity classes
    instance = None


    def __init__(self, enable_watcher: bool = False):
        """Initialize the game.
        
        Args:
            enable_watcher: Whether to enable auto-restart on file changes
        """
        # Set singleton instance
        Game.instance = self
        
        # Set up file watcher
        self.observer = None
        if enable_watcher:
            self.observer = Observer()
            # Get the src directory path
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            self.observer.schedule(GameRestartHandler(), path=src_dir, recursive=True)
            self.observer.start()
        
        # Initialize display
        self._init_display()
        
        # Track keyboard state
        self.shift_pressed = False  # Track if shift is being pressed
        
        # Message display state
        self.is_showing_message = False
        
        # Initialize supergrid
        self.supergrid_size = worldgen.SUPERGRID_SIZE
        self.supergrid_position = [7, 7]  # Start in middle of supergrid

        # use Chebyshev distance for the base number of enemies
        self.areas = [[Area(max(abs(x-self.supergrid_position[0]), abs(y-self.supergrid_position[1])) + 1)
                       for x in range(self.supergrid_size)]
                       for y in range(self.supergrid_size)]
        
        # Generate the entire world
        world_generator = WorldGenerator(self)
        world_generator.generate_world()
        
        # Initialize world size with default values
        self.world_width = 0
        self.world_height = 0
        
        # Initialize game state
        self.animation_manager = AnimationManager(on_finished=self.all_animations_completed_callback)
        self.hexes: List[List[Hex]] = []
        
        # Initialize entity collections
        self.player = None
        self.enemies = []
        self.health_restore = None
        
        # Initialize sprite groups (moved up before _init_hex_grid)
        self.all_sprites = pygame.sprite.Group()
        self.player_sprite = pygame.sprite.GroupSingle()
        self.enemy_sprites = pygame.sprite.Group()
        self.health_restore_sprite = pygame.sprite.GroupSingle()

        self._init_hex_grid()  # This calculates world_width and world_height
        self.start_time = pygame.time.get_ticks() / 1000.0
        
        # Initialize player on a random SOLID hex
        self.player = self._init_player()
        
        # List to store active floating text animations
        self.floating_texts = []
        
        # Screen shake effect variables
        self.is_screen_shaking = False
        self.screen_shake_start = 0
        self.screen_shake_duration = 0
        self.screen_shake_intensity = 0
        
        # Delayed hex effects for STOMP
        self.has_pending_hex_effects = False
        self.pending_hex_effects = []
        self.hex_effect_start_time = 0
        self.hex_effect_delay = 0
        self.center_hex_for_effects = None  # Store the center hex for reference
        
        # Debug visualization
        self.show_jump_targets = False  # Set to False to hide valid jump targets

        self.game_state = GameState.PLAYER
        
        # Debug mode
        self.debug_mode = False
        
        # Initialize font for debug text
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 14)
        
        # Font for move overlay
        self.overlay_font = pygame.font.SysFont('Arial', 16, bold=True)
        
        # Move overlay state
        self.show_move_overlay = False
        
        # Initialize clock for frame rate control
        self.clock = pygame.time.Clock()

        # Initialize turn counter
        self.turn_counter = 0

    def _init_display(self) -> None:
        """Initialize the game display."""
        # Use tkinter to get screen info
        root = tk.Tk()
        left_monitor_x = root.winfo_screenwidth() * -1
        root.destroy()
        
        # Position window on left monitor
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{left_monitor_x + 1500},100"
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Set up display
        self.screen = pygame.display.set_mode((settings.display.WINDOW_WIDTH, settings.display.WINDOW_HEIGHT))
        pygame.display.set_caption("On Thin Ice")
        settings.display.font = pygame.font.SysFont(settings.display.FONT_NAME, settings.display.FONT_SIZE)
    
    def _init_hex_grid(self) -> None:
        """Initialize the hex grid.
        
        Creates the hex objects with proper world coordinates.
        Terrain generation is now handled by the WorldGenerator.
        """
        # Calculate hex dimensions
        hex_height = settings.hex_grid.RADIUS * 1.732  # sqrt(3)
        spacing_x = settings.hex_grid.RADIUS * 1.5
        spacing_y = hex_height
        
        # Create hex grid
        self.hexes = [[None for _ in range(settings.hex_grid.GRID_HEIGHT)] 
                     for _ in range(settings.hex_grid.GRID_WIDTH)]
        
        # Calculate world size
        self.world_width = settings.hex_grid.GRID_WIDTH * spacing_x
        self.world_height = settings.hex_grid.GRID_HEIGHT * spacing_y + (spacing_y / 2)
        
        logging.info(f"World size: {self.world_width}x{self.world_height}")
        logging.info(f"Window size: {settings.display.WINDOW_WIDTH}x{settings.display.WINDOW_HEIGHT}")
        logging.info(f"Hex radius: {settings.hex_grid.RADIUS}, spacing: {spacing_x}x{spacing_y}")
        
        # Create the hex grid with proper world coordinates
        for x in range(settings.hex_grid.GRID_WIDTH):
            for y in range(settings.hex_grid.GRID_HEIGHT):
                # Calculate center position in world coordinates
                center_x = x * spacing_x
                center_y = y * spacing_y
                
                # Apply offset for odd columns
                if x % 2 == 1:
                    center_y += spacing_y / 2
                
                # Create the hex with default state (will be updated by restore_area)
                self.hexes[x][y] = Hex(center_x, center_y, x, y, self.animation_manager, state=HexState.SOLID)
        
        # Load the current area's data
        self._restore_area()

    def _init_player(self, initial_hex: Hex | None=None) -> Player | None:
        """Initialize the player entity on a random solid hex.
        
        Returns:
            The player entity or None if no valid position found
        """
        if initial_hex is None:
            # start adjacent to the health restore if present, or else the center
            if self.health_restore:
                initial_hex = random.choice(self.get_hex_neighbors(self.health_restore.current_hex))
            else:
                initial_hex = self.hexes[hex_grid.GRID_HEIGHT // 2][hex_grid.GRID_HEIGHT // 2]

        player = Player(initial_hex, self.animation_manager, game_over_callback=self.game_over_screen)

        # Add player to sprite groups
        if hasattr(self, 'player_sprite'):
            self.player_sprite.add(player)
        if hasattr(self, 'all_sprites'):
            self.all_sprites.add(player)

        return player

    def get_hex_neighbors(self, hex: Hex) -> List[Hex]:
        """Get list of neighboring hex tiles.
        
        Args:
            hex: The hex tile to get neighbors for
            
        Returns:
            List of adjacent hex tiles
        """
        neighbors = []
        odd_row = hex.grid_x % 2
        
        # Directions for even and odd rows
        directions = [
            [(0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1)],  # even row
            [(0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]     # odd row
        ][odd_row]
        
        # Add all valid neighbors
        for dx, dy in directions:
            nx, ny = hex.grid_x + dx, hex.grid_y + dy
            if 0 <= nx < settings.hex_grid.GRID_WIDTH and 0 <= ny < settings.hex_grid.GRID_HEIGHT:
                neighbors.append(self.hexes[nx][ny])
        
        return neighbors
    
    def hex_distance(self, hex1: Hex, hex2: Hex) -> int:
        """Calculate the distance between two hexes in the grid.
        
        Args:
            hex1: First hex
            hex2: Second hex
            
        Returns:
            Integer distance (number of steps) between the hexes
        """
        # Convert to cube coordinates
        # For axial coordinates (x, y) to cube (x, z, y):
        # cube_x = x
        # cube_z = y
        # cube_y = -x-y
        
        x1, y1 = hex1.grid_x, hex1.grid_y
        x2, y2 = hex2.grid_x, hex2.grid_y
        
        # Adjust for odd-row offset in our coordinate system
        if x1 % 2 == 1:
            y1 = y1 + 0.5
        if x2 % 2 == 1:
            y2 = y2 + 0.5
            
        # Convert to cube coordinates
        x1_cube, z1_cube = x1, y1
        y1_cube = -x1_cube - z1_cube
        
        x2_cube, z2_cube = x2, y2
        y2_cube = -x2_cube - z2_cube
        
        # Calculate distance in cube coordinates
        distance = max(
            abs(x1_cube - x2_cube),
            abs(y1_cube - y2_cube),
            abs(z1_cube - z2_cube)
        )
        
        # Round to nearest integer (should already be an integer or very close)
        return round(distance)
    
    def get_hexes_at_distance(self, center_hex: Hex, distance: int) -> List[Hex]:
        """Get all hexes that are exactly the specified distance from the center hex.
        
        Args:
            center_hex: The center hex
            distance: The distance from the center
            
        Returns:
            List of hexes at the specified distance
        """
        result = []
        
        for x in range(settings.hex_grid.GRID_WIDTH):
            for y in range(settings.hex_grid.GRID_HEIGHT):
                hex = self.hexes[x][y]
                if self.hex_distance(center_hex, hex) == distance:
                    result.append(hex)
        
        return result
    
    def pixel_to_hex(self, px: float, py: float) -> Optional[Hex]:
        """Convert pixel coordinates to hex tile.
        
        Args:
            px: X coordinate in pixels
            py: Y coordinate in pixels
            
        Returns:
            The hex tile at the given coordinates, or None if outside grid
        """
        # Find the nearest hex center
        min_dist = float('inf')
        nearest_hex = None
        
        for row in self.hexes:
            for hex in row:
                dx = px - hex.center[0]
                dy = py - hex.center[1]
                dist = dx*dx + dy*dy
                
                if dist < min_dist and dist <= settings.hex_grid.RADIUS * settings.hex_grid.RADIUS:
                    min_dist = dist
                    nearest_hex = hex
        
        return nearest_hex
    
    def run(self) -> None:
        """Run the main game loop."""
        # Display introduction screen before starting the game
        self.show_text_screen()
        
        try:
            running = True
            
            while running:
                current_time = pygame.time.get_ticks() / 1000.0 - self.start_time
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            # Dismiss the overlay if it's showing
                            if self.show_move_overlay:
                                self.show_move_overlay = False
                            else:
                                self._handle_click(event.pos, current_time)
                        elif event.button == 3:  # Right click
                            # Toggle move overlay on right click if in player turn
                            if self.game_state == GameState.PLAYER and not self.animation_manager.blocking_animations > 0:
                                self.show_move_overlay = not self.show_move_overlay
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                            self.shift_pressed = True
                        elif DEBUG_AREA_NAVIGATION:
                            if event.key == pygame.K_LEFT:
                                self.navigate_area(-1, 0)
                            elif event.key == pygame.K_RIGHT:
                                self.navigate_area(1, 0)
                            elif event.key == pygame.K_UP:
                                self.navigate_area(0, -1)
                            elif event.key == pygame.K_DOWN:
                                self.navigate_area(0, 1)
                    elif event.type == pygame.KEYUP:
                        if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                            self.shift_pressed = False
                
                self._draw(current_time)
                pygame.display.flip()
                
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            pygame.quit()
    
    def add_floating_text(self, text: str, position: Tuple[float, float], color: Tuple[int, int, int] = (255, 50, 50)) -> None:
        """Add a floating text animation at the specified position.
        
        Args:
            text: The text to display
            position: Position (x, y) to display the text
            color: RGB color of the text
        """
        self.floating_texts.append(FloatingText(text, position, color))
    
    def get_valid_jump_targets(self, player_hex: Hex) -> List[Hex]:
        """Get all valid hexes that the player can jump to from their current position.
        
        A valid jump target is a hex that:
        1. Is exactly 2 steps away (a neighbor of a neighbor)
        2. Is not directly adjacent to the player
        3. Is in a SOLID state (not cracked, broken, or land)
        
        Args:
            player_hex: The player's current hex
            
        Returns:
            List of valid jump target hexes
        """
        if player_hex.state == HexState.LAND:
            return []

        # Get all neighbors of the player
        neighbors = self.get_hex_neighbors(player_hex)
        
        # Get all neighbors of neighbors (potential jump targets)
        potential_targets = set()
        for neighbor in neighbors:
            neighbor_neighbors = self.get_hex_neighbors(neighbor)
            for hex in neighbor_neighbors:
                # Only add if it's not the player's hex and not directly adjacent to player
                if hex != player_hex and hex not in neighbors:
                    potential_targets.add(hex)
        
        # Filter to only include SOLID hexes
        valid_targets = [hex for hex in potential_targets if hex.state == HexState.SOLID]
        
        return valid_targets
    
    def get_valid_sprint_targets(self, player_hex: Hex) -> List[Tuple[Hex, List[Hex]]]:
        """Get all valid hexes that the player can sprint to from their current position.
        
        A valid sprint target is a hex that:
        1. Is exactly 3 steps away in a straight line
        2. Is in a SOLID state (not cracked, broken, or land)
        3. All hexes in the path must be valid (not LAND or BROKEN)
        
        Args:
            player_hex: The player's current hex
            
        Returns:
            List of tuples containing (target_hex, path) where path is the list of hexes from start to end
        """
        valid_targets = []

        enemy_hexes = {enemy.current_hex for enemy in self.enemies}
        for edge_index in range(6):
            path = [player_hex]
            current_hex = player_hex
            valid_path = True
            
            for step in range(3):
                next_hex = current_hex.get_neighbor(edge_index, self.hexes)
                if not next_hex or next_hex.state != HexState.SOLID or next_hex in enemy_hexes:
                    valid_path = False
                    break
                path.append(next_hex)
                current_hex = next_hex
            
            if valid_path and path[-1].state == HexState.SOLID:
                valid_targets.append((path[-1], path))
        
        return valid_targets
    
    def _handle_click(self, pos: Tuple[int, int], current_time: float) -> None:
        """Handle mouse click event.
        
        Args:
            pos: Mouse position (x, y)
            current_time: Current game time in seconds
        """
        # Only handle clicks if no animations are running
        if self.animation_manager.blocking_animations > 0:
            return
            
        # Convert pixel coordinates to hex coordinates
        clicked_hex = self.pixel_to_hex(pos[0], pos[1])
        if not clicked_hex:
            return
            
        # Check if player was clicked
        if self.player and clicked_hex == self.player.current_hex and clicked_hex.state == HexState.SOLID:

            # Apply screen shake effect
            self.start_screen_shake(current_time)

            # Get all adjacent hexes for later processing
            neighbors = self.get_hex_neighbors(clicked_hex)

            # First, only crack the player's hex if it's SOLID
            clicked_hex.crack([])  # Empty list since we don't need to check neighbors here
            # Schedule the surrounding hexes to crack/break after a delay
            self.schedule_surrounding_hex_effects(clicked_hex, neighbors, current_time)
            return
            
        if self.game_state != GameState.PLAYER:
            logging.debug(f'Invalid! Game state is {self.game_state}.')
            return

        if not self.player:
            logging.warn("No PLAYER entity.")
            return

        # Find the clicked hex
        clicked_hex = self.pixel_to_hex(pos[0], pos[1])
        if not clicked_hex:
            return
            
        if self.shift_pressed:
            logging.info(f"SHIFT-CLICK at ({clicked_hex.grid_x}, {clicked_hex.grid_y}). "
                         f"Action={settings.game_settings.SHIFT_CLICK_ACTION}")
            match settings.game_settings.SHIFT_CLICK_ACTION:
                case "break":
                    # Don't break the hex if the player is standing on it
                    if self.player and clicked_hex == self.player.current_hex:
                        return

                    # Handle state transitions
                    if clicked_hex.state == HexState.SOLID:
                        clicked_hex.crack(neighbors=self.get_hex_neighbors(clicked_hex))
                    elif clicked_hex.state == HexState.CRACKED:
                        clicked_hex.break_ice()
                case "enemy":
                    # Create an enemy if there isn't one on the hex
                    if not self._hex_has_entity(clicked_hex):
                        self.add_enemy(clicked_hex)
            return

        # REGULAR CLICK
        logging.info(f"REGULAR CLICK at ({clicked_hex.grid_x}, {clicked_hex.grid_y})")
        # Check if player is already moving
        if self.player.is_moving:
            return

        # Check for SLIDE action (3 hexes away in a straight line)
        sprint_targets = self.get_valid_sprint_targets(self.player.current_hex)
        for target, path in sprint_targets:
            if clicked_hex == target:

                def sprint_completion():
                    direction = path[-2].get_shared_edge_index(path[-1])
                    logging.debug(f'{path=}, {direction=}')
                    for push_dir in [(direction - 1) % 6, direction, (direction + 1) % 6]:
                        for enemy in self.enemies:
                            if path[-1].get_neighbor(push_dir, self.hexes) == enemy.current_hex:
                                self.push_enemy(enemy, push_dir, pygame.time.get_ticks() / 1000.0, do_attack=False)

                self.player.sprint(path, current_time, sprint_completion)
                return
                
        # Check for JUMP action (2 hexes away)
        valid_jump_targets = self.get_valid_jump_targets(self.player.current_hex)
        if clicked_hex in valid_jump_targets:
            # Perform JUMP maneuver
            self.perform_jump(clicked_hex, current_time)
            return

        # Check for PUSH action (hex is adjacent and contains enemy)
        for enemy in self.enemies:
            if clicked_hex == enemy.current_hex:
                self.push_enemy(enemy, self.player.current_hex.get_shared_edge_index(clicked_hex), current_time)
                return

        # Regular move to adjacent hex?
        # Check if the clicked hex is adjacent to the player
        adjacent_hexes = self.get_hex_neighbors(self.player.current_hex)
        if clicked_hex in adjacent_hexes:
            # Try to move player to the clicked hex
            self.player.move(clicked_hex, current_time)


    def push_enemy(self, enemy: Wolf, direction: int, current_time: float, do_attack: bool = True) -> None:
        """Push an enemy in a direction.
        
        Args:
            enemy: The enemy to push
            direction: The direction to push in
            current_time: Current game time in seconds
            do_attack: Whether to do the attack animation first (default: True)
        """
        target_hex = enemy.current_hex.get_neighbor(direction, self.hexes)
        logging.info(f"push {enemy} in direction {direction} toward {target_hex}")
        
        def start_push():
            push_current_time = pygame.time.get_ticks() / 1000.0

            # If the target hex is broken, set up a callback to start drowning when push completes
            if target_hex and target_hex.state == HexState.BROKEN:
                def on_push_complete():
                    logging.info(f"{enemy} pushed into water, starting drowning animation")
                    # Use a slight delay to ensure the push animation is fully complete
                    enemy.drown(push_current_time + enemy.animation_duration)
                
                # Start the push animation with the drowning callback
                enemy.pushed(target_hex, push_current_time, on_push_complete)
            else:
                # Start the push animation without a callback
                enemy.pushed(target_hex, push_current_time)
            
            # Add floating text for PUSH
            self.start_screen_shake(push_current_time, intensity=3, duration=.1)

        if do_attack:
            # First, perform the attack animation
            self.player.attack(enemy, current_time, start_push)
        else:
            # Skip the attack animation and start the push directly
            start_push()


    def _draw(self, current_time: float) -> None:
        """Draw the game state.
        
        Args:
            current_time: Current game time in seconds
        """
        # Create a drawing surface
        draw_surface = pygame.Surface((self.world_width, self.world_height), pygame.SRCALPHA)
        draw_surface.fill((0, 0, 0, 0))  # Transparent background
        
        # Apply screen shake if active
        shake_offset_x, shake_offset_y = self.apply_screen_shake(current_time)
        
        # Draw hexes
        for row in self.hexes:
            for hex in row:
                if hex:
                    hex.draw(draw_surface, current_time)
        
        # Update all sprites
        self.all_sprites.update(current_time)
        
        # Draw all sprites except ice fragments (which are drawn by their parent hex)
        for sprite in self.all_sprites:
            # Skip IceFragment sprites - they're drawn by their parent hex
            if not hasattr(sprite, '__class__') or sprite.__class__.__name__ != 'IceFragment':
                sprite.draw(draw_surface, current_time)
        
        # Draw move overlay if enabled
        if self.show_move_overlay and self.game_state == GameState.PLAYER and self.player:
            self._draw_move_overlay(draw_surface)
        
        # Update and draw floating text animations
        active_texts = []
        for text in self.floating_texts:
            text.update(current_time)
            if text.is_active:
                text.draw(draw_surface)
                active_texts.append(text)
        self.floating_texts = active_texts
        
        # Apply the draw surface to the screen with shake offset
        self.screen.fill((0, 0, 0))  # Black background
        self.screen.blit(draw_surface, (shake_offset_x, shake_offset_y))

        self.draw_health()

        # Draw debug info if enabled
        if self.debug_mode:
            self._draw_debug_info()
        
        # Update the display
        pygame.display.flip()

    def draw_health(self, position=(10, 10), spacing=40):
        """Draws player's health using heart icons.

        Args:
            screen (pygame.Surface): The game screen.
            player (Player): The player object with `health` attribute.
            position (tuple): (x, y) coordinates for the first heart.
            spacing (int): Pixel spacing between hearts.
        """
        global HEART_CACHE
        if len(HEART_CACHE) == 0:
            # Global cache for heart images (preload for efficiency)
            HEART_CACHE = {
                "full": pygame.image.load(str(Path(__file__).parents[1] / "images/heart_full.png")).convert_alpha(),
                "empty": pygame.image.load(str(Path(__file__).parents[1] / "images/heart_empty.png")).convert_alpha(),
            }

        x, y = position

        for i in range(game_settings.MAX_HEALTH):
            heart_img = HEART_CACHE["full"] if i < Player.HEALTH else HEART_CACHE["empty"]
            self.screen.blit(heart_img, (x + i * spacing, y))  # Draw heart with spacing


    def start_screen_shake(self, current_time: float, duration: float = 0.3, intensity: int = 5) -> None:
        """Start a screen shake effect.
        
        Args:
            current_time: Current game time in seconds
        """
        self.screen_shake_start = current_time
        self.screen_shake_duration = duration  # seconds
        self.screen_shake_intensity = intensity  # pixels
        self.is_screen_shaking = True
    
    def schedule_surrounding_hex_effects(self, center_hex: Hex, neighbors: List[Hex], current_time: float) -> None:
        """Schedule effects on surrounding hexes after a delay.
        
        Args:
            center_hex: The center hex (player's hex)
            neighbors: List of neighboring hexes
            current_time: Current game time in seconds
        """
        self.hex_effect_start_time = current_time
        self.hex_effect_delay = 0.5  # Increased delay to ensure player hex is fully cracked
        self.pending_hex_effects = neighbors
        self.has_pending_hex_effects = True
        self.center_hex_for_effects = center_hex  # Store the center hex for reference
    
    def apply_screen_shake(self, current_time: float) -> Tuple[int, int]:
        """Calculate screen shake offset based on current time.
        
        Args:
            current_time: Current game time in seconds
            
        Returns:
            Tuple of (x_offset, y_offset) for screen shake
        """
        if not self.is_screen_shaking:
            return (0, 0)
            
        elapsed = current_time - self.screen_shake_start
        if elapsed >= self.screen_shake_duration:
            self.is_screen_shaking = False
            return (0, 0)
            
        # Calculate shake intensity based on time (decreases over time)
        remaining = 1.0 - (elapsed / self.screen_shake_duration)
        intensity = self.screen_shake_intensity * remaining
        
        # Generate random offset
        import random
        x_offset = random.uniform(-intensity, intensity)
        y_offset = random.uniform(-intensity, intensity)
        
        return (int(x_offset), int(y_offset))
    
    def process_pending_hex_effects(self, current_time: float) -> None:
        """Process any pending hex effects if their delay has elapsed.
        
        Args:
            current_time: Current game time in seconds
        """
        if not self.has_pending_hex_effects or not self.center_hex_for_effects:
            return
            
        # First, check if the center hex has completed its transition to CRACKED state
        # Only proceed if the center hex is fully CRACKED or BROKEN (not in transition)
        if self.center_hex_for_effects.state not in [HexState.CRACKED, HexState.BROKEN]:
            return
            
        # Now check if enough time has passed since scheduling
        elapsed = current_time - self.hex_effect_start_time
        if elapsed < self.hex_effect_delay:
            return
            
        # Time to apply effects to surrounding hexes
        for hex in self.pending_hex_effects:
            if hex.state == HexState.SOLID:
                # Pass the center hex and other neighbors to ensure cracks connect properly
                neighbors_for_crack = [self.center_hex_for_effects]
                # Also include other already cracked neighbors to ensure proper connections
                for neighbor in self.get_hex_neighbors(hex):
                    if neighbor.state == HexState.CRACKED and neighbor != self.center_hex_for_effects:
                        neighbors_for_crack.append(neighbor)
                
                hex.crack(neighbors_for_crack)  # Pass neighbors to connect cracks
            elif hex.state == HexState.CRACKED:
                hex.break_ice()

        # Clear pending effects
        self.has_pending_hex_effects = False
        self.pending_hex_effects = []
        self.center_hex_for_effects = None
    
    def perform_jump(self, target_hex: Hex, current_time: float) -> None:
        """Perform a jump maneuver to a hex that is exactly 2 steps away.
        
        Args:
            target_hex: The target hex to jump to
            current_time: Current game time in seconds
        """
        if not self.player or self.player.is_moving:
            return
            
        # Store the launch hex for later
        launch_hex = self.player.current_hex
        
        # Start cracking the launch hex immediately if it's SOLID
        if launch_hex.state == HexState.SOLID:
            launch_hex.crack([])
        elif launch_hex.state == HexState.CRACKED:
            launch_hex.break_ice()
        
        # Define callback for when jump animation completes
        def on_jump_complete():
            in_between = set(self.get_hex_neighbors(launch_hex)).intersection(set(self.get_hex_neighbors(target_hex)))
            in_between.add(target_hex)
            for ihex in in_between:
                if ihex.state == HexState.SOLID:
                    ihex.crack([])
                elif ihex.state == HexState.CRACKED:
                    ihex.break_ice()
        
        # Store the callback to be executed when animation completes
        self.player.on_animation_complete = on_jump_complete
        
        # Execute the jump using the new jump method
        self.player.jump(target_hex, current_time)
        
        logging.info(f"JUMP from ({launch_hex.grid_x}, {launch_hex.grid_y}) to ({target_hex.grid_x}, {target_hex.grid_y})")

    def all_animations_completed_callback(self) -> None:
        """Called when all animations have completed."""
        logging.info("All animations completed")
        
        # Check for dead enemies and remove them
        dead_enemies = [enemy for enemy in self.enemies if enemy.animation_type == "dead"]
        if dead_enemies:
            logging.info(f"Removing {len(dead_enemies)} dead enemies")
            for enemy in dead_enemies:
                logging.info(f"Removing dead enemy: {enemy}")
                self.enemies.remove(enemy)
                # Remove from sprite groups
                self.enemy_sprites.remove(enemy)
                self.all_sprites.remove(enemy)
            
        # Log the number of remaining enemies
        if self.enemies:
            logging.info(f"Remaining enemies: {len(self.enemies)}")
        else:
            logging.info("All enemies destroyed!")
        
        # Check if any enemies are in broken hexes and need to drown
        current_time = pygame.time.get_ticks() / 1000.0
        drowning_started = False
        for enemy in self.enemies:
            if enemy.current_hex.state == HexState.BROKEN and enemy.animation_type != "drown":
                logging.info(f"Starting drowning for {enemy} in broken hex")
                enemy.drown(current_time=current_time)
                drowning_started = True
                
        # Process any pending hex effects
        self.process_pending_hex_effects(current_time)

        # If we started new animations, don't switch state yet
        if self.animation_manager.blocking_animations > 0 or drowning_started:
            logging.info(f"Animations still in progress, not switching state. Blocking animations: {self.animation_manager.blocking_animations}")
            return
            

        if self.health_restore and self.player.current_hex == self.health_restore.current_hex:
            logging.info(f"Health restore for {self.player.current_hex}. Restoring...")
            if Player.HEALTH < game_settings.MAX_HEALTH:
               Player.HEALTH += 1

            self.remove_health_restore()
            self.display_message(random.choice(health_restore_messages))

        if self.player.current_hex.grid_x == 0:
            self.navigate_area(dx=-1)
        elif self.player.current_hex.grid_x == hex_grid.GRID_WIDTH - 1:
            self.navigate_area(dx=1)
        elif self.player.current_hex.grid_y == 0:
            self.navigate_area(dy=-1)
        elif self.player.current_hex.grid_y == hex_grid.GRID_HEIGHT - 2:
            self.navigate_area(dy=1)

        # Handle state transitions
        if self.game_state == GameState.PLAYER:
            # Player's turn is complete, switch to enemy turn
            self.game_state = GameState.ENEMY
            logging.info("Switching to ENEMY turn")
            self._enemy_ai(current_time)
        elif self.game_state == GameState.ENEMY:
            # Enemy's turn is complete, switch to player turn
            self.game_state = GameState.PLAYER
            logging.info("Switching to PLAYER turn")
            self.turn_counter += 1


    def _enemy_ai(self, current_time: float) -> None:
        """Run AI for all enemies."""
        
        # First, remove any dead enemies that might still be in the list
        dead_enemies = [enemy for enemy in self.enemies if enemy.animation_type == "dead"]
        if dead_enemies:
            logging.info(f"AI found {len(dead_enemies)} dead enemies to remove")
            for enemy in dead_enemies:
                logging.info(f"AI removing dead enemy: {enemy}")
                self.enemies.remove(enemy)
                # Remove from sprite groups
                self.enemy_sprites.remove(enemy)
                self.all_sprites.remove(enemy)
        
        # If there are no enemies, immediately switch back to player turn
        if not self.enemies:
            logging.info("No enemies left, switching back to player turn")
            self.game_state = GameState.PLAYER
            return
            
        # Track if any enemies moved
        any_enemy_moved = False
            
        for enemy in self.enemies:
            logging.debug(f'AI for {enemy}')

            # Skip dead or drowning enemies
            if enemy.animation_type in ["dead", "drown"]:
                logging.info(f"Skipping {enemy} with animation type {enemy.animation_type}")
                continue
                
            # If stunned, we lose our turn.
            if enemy.stunned:
                enemy.stunned = False
                continue

            # If player is adjacent, attack instead of moving
            if self.player and self.player.current_hex in self.get_hex_neighbors(enemy.current_hex):
                # Create a callback function to apply damage after animation
                def on_attack_complete():
                    current_time = pygame.time.get_ticks() / 1000.0
                    self.start_screen_shake(current_time=current_time)
                    self.player.take_damage()
                
                # Use the attack animation before applying damage
                enemy.attack(self.player, current_time, on_attack_complete)
                any_enemy_moved = True
                continue

            # Find path to player
            from .pathfinding import a_star  # avoid circular dependencies
            if self.player:
                path = a_star(enemy.current_hex, self.player.current_hex)
                logging.info(f'{path=}')
                if path is not None and len(path) > 1:
                    enemy.move(target_hex=path[1], current_time=current_time)
                    any_enemy_moved = True
        
        # If no enemies moved and no animations are running, we can immediately switch back to player turn
        if not any_enemy_moved and self.animation_manager.blocking_animations == 0:
            self.game_state = GameState.PLAYER

    def add_enemy(self, hex: Hex) -> None:
        """Add an enemy to the game.
        
        Args:
            hex: The hex to place the enemy on
        """
        enemy = Wolf(hex, self.animation_manager)
        self.enemies.append(enemy)
        
        # Add enemy to sprite groups if they exist
        if hasattr(self, 'enemy_sprites'):
            self.enemy_sprites.add(enemy)
        if hasattr(self, 'all_sprites'):
            self.all_sprites.add(enemy)

    def add_health_restore(self, hex: Hex) -> None:
        self.health_restore = HealthRestore(hex, self.animation_manager)
        # Add enemy to sprite groups if they exist
        if hasattr(self, 'health_restore_sprite'):
            self.health_restore_sprite.add(self.health_restore)
        if hasattr(self, 'all_sprites'):
            self.all_sprites.add(self.health_restore)


    def _hex_has_entity(self, hex: Hex) -> bool:
        """Check if a hex has an entity on it.
        
        Args:
            hex: The hex to check
            
        Returns:
            True if the hex has an entity, False otherwise
        """
        if hasattr(self, 'player') and self.player and self.player.current_hex == hex:
            return True
            
        if hasattr(self, 'enemies'):
            for enemy in self.enemies:
                if enemy.current_hex == hex:
                    return True
                
        return False

    def save_current_area(self):
        """Save the current screen state to the current area."""
        area = self.areas[self.supergrid_position[0]][self.supergrid_position[1]]
        area.generated = True
        
        # Save hex states and colors
        for x in range(len(self.hexes)):
            for y in range(len(self.hexes[x])):
                if self.hexes[x][y]:
                    hex_obj = self.hexes[x][y]
                    area.hex_states[x][y] = hex_obj.state
                    area.hex_colors[x][y] = hex_obj.color
                    
                    # For broken hexes, save complete data needed for visual appearance
                    if hex_obj.state == HexState.BROKEN:
                        key = (x, y)
                        area.broken_hex_data[key] = {
                            'cracks': hex_obj.cracks,
                            'broken_surface': hex_obj.broken_surface,
                            'fragment_sprites': hex_obj.fragment_sprites
                        }
                    
                    # For cracked hexes, save crack geometry
                    elif hex_obj.state == HexState.CRACKED:
                        key = (x, y)
                        # Only save if there are actual cracks
                        if hasattr(hex_obj, 'cracks') and hex_obj.cracks:
                            area.cracked_hex_data[key] = {
                                'cracks': hex_obj.cracks
                            }
        
        # Save enemy count
        area.enemy_count = len(self.enemies)

    def navigate_area(self, dx=0, dy=0):
        """Navigate to a different area in the supergrid.

        Args:
            dx: Change in x position (-1, 0, 1)
            dy: Change in y position (-1, 0, 1)
        """
        # Save current area state
        self.save_current_area()

        # Update supergrid position, ensuring we stay within bounds
        new_x = self.supergrid_position[0] + dx
        new_y = self.supergrid_position[1] + dy

        if new_x < 0 or new_x >= self.supergrid_size or new_y < 0 or new_y >= self.supergrid_size:
            self.show_text_screen(victory=True)
            pygame.quit()
            sys.exit()

        # If position hasn't changed, we're at the edge of the world
        if new_x == self.supergrid_position[0] and new_y == self.supergrid_position[1]:
            return False

        # Update position
        self.supergrid_position[0] = new_x
        self.supergrid_position[1] = new_y

        # Clear current entities
        # Clear enemies
        for enemy in self.enemies:
            enemy.kill()  # Remove from all sprite groups
        self.enemies = []
        if hasattr(self, 'enemy_sprites'):
            self.enemy_sprites.empty()

        # Clear ice fragment sprites
        if hasattr(self, 'all_sprites'):
            for sprite in list(self.all_sprites):
                if hasattr(sprite, '__class__') and sprite.__class__.__name__ == 'IceFragment':
                    sprite.kill()

        # Reset all hexes to clear any crack data
        # BUT don't clear fragment sprites or broken surfaces
        for row in self.hexes:
            for hex in row:
                if hex:
                    # Clear cracks only for now - we'll restore them properly in _restore_area
                    hex.cracks = []

        # Always restore the area (all areas should be generated)
        self._restore_area()

        # Reset player position
        if self.player:
            if hasattr(self, 'player_sprite'):
                self.player_sprite.empty()
            if hasattr(self, 'all_sprites'):
                self.all_sprites.remove(self.player)

        match dx:
            case -1:
                x = hex_grid.GRID_WIDTH - 2
            case 0:
                x = self.player.current_hex.grid_x
            case 1:
                x = 1
            case _:
                raise ValueError
        match dy:
            case -1:
                y = hex_grid.GRID_HEIGHT - (3 if self.player.current_hex.grid_x % 2 == 0 else 3)
            case 0:
                y = self.player.current_hex.grid_y
            case 1:
                y = 1
            case _:
                raise ValueError

        self.player = self._init_player(self.hexes[x][y])

        logging.info(f"Navigated to supergrid position: {self.supergrid_position}")
        logging.debug(f"Positioned player at {self.player.current_hex}")

        self.spawn_enemies()

        return True

    def _restore_area(self):
        """Restore the current area state."""

        area = self.areas[self.supergrid_position[0]][self.supergrid_position[1]]

        # Only restore if the area has been generated before
        if not area.generated:
            return
        
        # Restore hex states and colors
        for x in range(len(self.hexes)):
            for y in range(len(self.hexes[x])):
                if self.hexes[x][y] and area.hex_states[x][y] is not None:
                    # Get the current and saved states
                    current_hex = self.hexes[x][y]
                    saved_state = area.hex_states[x][y]
                    
                    current_hex.color = area.hex_colors[x][y]
                    current_hex.height = area.hex_heightmap[x][y]
                    
                    # Restore hex state with appropriate handling for special states
                    if saved_state == HexState.CRACKED:
                        # For cracked hexes, restore crack data if available
                        key = (x, y)
                        if key in area.cracked_hex_data:
                            hex_data = area.cracked_hex_data[key]
                            if 'cracks' in hex_data:
                                current_hex.cracks = hex_data['cracks']
                        
                        # Set state to CRACKED
                        current_hex.state = HexState.CRACKED
                    
                    elif saved_state == HexState.BROKEN:
                        # For broken hexes, restore the complete saved data
                        key = (x, y)
                        if key in area.broken_hex_data:
                            hex_data = area.broken_hex_data[key]
                            
                            # Restore all visual elements
                            current_hex.cracks = hex_data['cracks']
                            current_hex.broken_surface = hex_data['broken_surface']
                            current_hex.fragment_sprites = hex_data['fragment_sprites']
                        
                        # Set state to BROKEN after restoring all data
                        current_hex.state = HexState.BROKEN
                    else:
                        # For other states, just set the state directly
                        current_hex.state = saved_state
        
        # Clear current enemies
        self.enemies = []
        # Check if sprite groups exist before using them
        if hasattr(self, 'enemy_sprites'):
            self.enemy_sprites.empty()

        self.remove_health_restore()
        if area.has_health_restore:
            area.has_health_restore = False
            health_hex = self.find_health_restore_placement()
            self.add_health_restore(health_hex)
            logging.info(f'added health restore: {self.health_restore} -> {self.health_restore.current_hex}')
        else:
            logging.info(f'no health restore')
        
        # Don't spawn enemies automatically - just log the count
        if area.enemy_count > 0:
            logging.info(f"Area has {area.enemy_count} enemies (not spawning them automatically)")


    def spawn_enemies(self):
        # add some enemies
        enemy_positions = set()
        enemies_to_spawn = self.areas[self.supergrid_position[0]][self.supergrid_position[1]].enemy_count

        while enemies_to_spawn > 0:
            pack_size = min(random.randint(2, 6), enemies_to_spawn)
            pack_positions = []

            # find a valid starting position
            while True:
                x = random.randint(0, len(self.hexes) - 1)
                y = random.randint(0, len(self.hexes[0]) - 1)
                start_hex = self.hexes[x][y]
                if (
                        start_hex
                        and start_hex.state != HexState.BROKEN  # Can't spawn on broken ice
                        and self.hex_distance(self.player.current_hex, start_hex) >= game_settings.MIN_PACK_DISTANCE
                        and start_hex not in enemy_positions
                ):
                    pack_positions.append(start_hex)
                    break

                # Spread remaining pack members in 1-3 hex range
            for _ in range(pack_size - 1):
                valid_spots = []
                for dist in range(1, game_settings.MAX_PACK_SPREAD + 1):
                    valid_spots.extend(
                        [vhex for vhex in self.get_hexes_at_distance(pack_positions[0], dist)
                         if vhex.state != HexState.BROKEN and (vhex.grid_x, vhex.grid_y) not in enemy_positions]
                    )

                if valid_spots:
                    pack_positions.append(random.choice(valid_spots))

            # Create and place enemies
            for phex in pack_positions:
                self.add_enemy(phex)

            enemies_to_spawn -= pack_size
            enemy_positions.update(pack_positions)

        logging.info(f"Spawned enemies: {len(self.enemies)}")

    def remove_health_restore(self):
        if self.health_restore and hasattr(self, 'all_sprites'):
            self.all_sprites.remove(self.health_restore)
        self.health_restore = None
        if hasattr(self, 'health_restore_sprite'):
            self.health_restore_sprite.empty()


    def find_health_restore_placement(self) -> Hex:
        # Start BFS to find the closest LAND hex
        from collections import deque

        center_x = settings.hex_grid.GRID_WIDTH // 2
        center_y = settings.hex_grid.GRID_HEIGHT // 2
        queue = deque([(center_x, center_y)])
        visited = set(queue)

        while queue:
            x, y = queue.popleft()
            current_hex = self.hexes[x][y]

            if current_hex.state == HexState.LAND:
                logging.info(f"Closest LAND hex found at ({x}, {y})")
                return current_hex

            for neighbor in self.get_hex_neighbors(current_hex):
                nx, ny = neighbor.grid_x, neighbor.grid_y
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))




    def display_message(self, message: str) -> None:
        """Displays a message in a white box with drop shadow until user clicks.
        
        Args:
            message: The message to display
        """
        # Setup
        screen_width, screen_height = self.screen.get_size()
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 30)
        
        # Calculate box dimensions
        padding = 40
        wrapped_text = textwrap.wrap(message, width=40)  # Wrap text to fit nicely
        box_width = min(600, screen_width - 100)  # Max width or screen width minus margin
        line_height = 35
        text_height = len(wrapped_text) * line_height
        box_height = text_height + padding * 2
        
        # Default position in the center of the screen
        box_x = (screen_width - box_width) // 2
        box_y = (screen_height - box_height) // 2
        
        # Adjust position to avoid player
        if self.player:
            # Get player's screen position
            player_x, player_y = self.player.position
            
            # Define the player's area (add some margin)
            player_radius = 60  # Approximate space the player takes up
            player_area = pygame.Rect(
                player_x - player_radius,
                player_y - player_radius,
                player_radius * 2,
                player_radius * 2
            )
            
            # Define the message box area
            message_box = pygame.Rect(box_x, box_y, box_width, box_height)
            
            # Check if the message box overlaps with the player
            if message_box.colliderect(player_area):
                # Determine which quadrant the player is in and position the box in the opposite quadrant
                if player_x < screen_width // 2:
                    # Player is on the left side, move box to right
                    box_x = screen_width - box_width - 50
                else:
                    # Player is on the right side, move box to left
                    box_x = 50
                    
                if player_y < screen_height // 2:
                    # Player is in the top half, move box to bottom
                    box_y = screen_height - box_height - 50
                else:
                    # Player is in the bottom half, move box to top
                    box_y = 50
        
        # Animation parameters
        fade_duration = 500  # ms
        start_time = pygame.time.get_ticks()
        
        # Take a screenshot of the current game state to use as background
        # This prevents flickering by avoiding continuous redraws
        background = self.screen.copy()
        
        # Save the current game state
        self.is_showing_message = True
        
        # Message loop
        while self.is_showing_message:
            current_time = pygame.time.get_ticks()
            elapsed = current_time - start_time
            alpha = min(255, int(255 * (elapsed / fade_duration)))  # Fade in effect
            
            # Check for exit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and elapsed > fade_duration:
                    self.is_showing_message = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.is_showing_message = False
                    break
            
            # Draw the static background instead of redrawing the game
            self.screen.blit(background, (0, 0))
            
            # Draw drop shadow
            shadow_offset = 8
            shadow_surface = pygame.Surface((box_width, box_height))
            shadow_surface.fill((50, 50, 50))
            shadow_surface.set_alpha(min(128, alpha // 2))
            self.screen.blit(shadow_surface, (box_x + shadow_offset, box_y + shadow_offset))
            
            # Draw white box
            box_surface = pygame.Surface((box_width, box_height))
            box_surface.fill((255, 255, 255))
            box_surface.set_alpha(alpha)
            self.screen.blit(box_surface, (box_x, box_y))
            
            # Draw text
            if alpha > 128:  # Only start drawing text once the box is somewhat visible
                for i, line in enumerate(wrapped_text):
                    text_surface = font.render(line, True, (0, 0, 0))
                    text_alpha = min(255, int(510 * (elapsed / fade_duration)) - 255)
                    text_surface.set_alpha(text_alpha)
                    text_rect = text_surface.get_rect(
                        center=(box_x + box_width // 2, 
                                box_y + padding + i * line_height + line_height // 2)
                    )
                    self.screen.blit(text_surface, text_rect)
            
            pygame.display.flip()
            clock.tick(60)

    def show_text_screen(self, victory=False) -> None:
        """Show an introduction screen with game instructions."""
        # Setup
        screen_width, screen_height = self.screen.get_size()
        clock = pygame.time.Clock()
        font_title = pygame.font.SysFont(None, 36, italic=True)
        font_body = pygame.font.SysFont(None, 28)
        
        # Introduction text
        if victory:
            text = [
                "On Thin Ice",
                "",
                "The ice cracks one last time behind you and wind no longer carries the hunt. You step onto solid ground.",
                "No more shifting, no more breaking, no more cold water waiting below.",
                "",
                "You look down. A trail of footprints leads ahead, half-buried in the snow. Not yours. But ",
                "they match.",
                "",
                "The sky is clear now. The stars look different here.",
                "",
                "A structure looms in the distance, half-buried. Rusted metal, shattered glass. A door.",
                "",
                "Inside, the walls are covered in names. Some have been scratched out.",
                "",
                "One remains.",
                "",
                "Yours."
            ]
        else:
            text = [
                "On Thin Ice",
                "",
                "The ice is fragile. The water is deathly cold.",
                "",
                "MOVE one hex in any direction. This is safe.",
                "ATTACK an adjacent enemy. This will only push them back.",
                "JUMP two hexes in any direction. This will damage the ice you jump from and land on.",
                "STOMP on your own hex to damage it and everything around it. You cannot stomp cracked ice.",
                "SLIDE exactly three squares on uncracked ice and knock back enemies.",
                "",
                "When on land you can only MOVE and ATTACK. There is no difference in the land terrains.",
                "",
                "RIGHT-CLICK to show your available moves.",
                "",
                "Seven screens from the start is safety.",
                "",
                "You are being hunted."
            ]
        
        # Animation parameters
        fade_duration = 700  # ms for text fade in
        start_time = pygame.time.get_ticks()
        
        # White background that's already visible
        self.screen.fill((255, 255, 255))
        pygame.display.flip()
        
        # Wait for user click
        waiting = True
        done_fading = False
        
        while waiting:
            current_time = pygame.time.get_ticks()
            elapsed = current_time - start_time
            
            # Calculate fade-in for text
            fade_progress = min(1.0, elapsed / fade_duration)
            text_alpha = int(255 * fade_progress)
            
            # Draw white background
            self.screen.fill((255, 255, 255))
            
            # Draw text with fade-in effect
            y_offset = 100  # Starting position from top
            
            # Draw title (first line) with different formatting
            title_line = text[0]
            title_surface = font_title.render(title_line, True, (0, 0, 0))
            title_surface.set_alpha(text_alpha)
            title_rect = title_surface.get_rect(center=(screen_width // 2, y_offset))
            self.screen.blit(title_surface, title_rect)
            y_offset += 60  # More space after title
            
            # Draw remaining lines
            for i, line in enumerate(text[1:]):
                if line:  # Skip empty lines in rendering but add space
                    text_surface = font_body.render(line, True, (0, 0, 0))
                    text_surface.set_alpha(text_alpha)
                    text_rect = text_surface.get_rect(center=(screen_width // 2, y_offset))
                    self.screen.blit(text_surface, text_rect)
                y_offset += 30  # Standard line spacing
            
            # Once we've finished fading in, show a prompt to continue
            if fade_progress >= 1.0:
                done_fading = True
                continue_text = font_body.render("Click anywhere to continue...", True, (100, 100, 100))
                continue_rect = continue_text.get_rect(center=(screen_width // 2, screen_height - 100))
                self.screen.blit(continue_text, continue_rect)
            
            # Check for events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and done_fading:
                    waiting = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif done_fading:
                        waiting = False
                        break
            
            pygame.display.flip()
            clock.tick(60)

    def game_over_screen(self) -> None:
        """Smoothly fades to white over `fade_duration`, then fades in 'Game Over' over `text_fade_duration`."""
        fade_duration = 2000  # Total fade time in milliseconds (3 seconds)
        text_fade_duration = 500  # 'Game Over' fade-in time (0.5 seconds)

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 40, italic=True)
        screen_width, screen_height = self.screen.get_size()

        # **Randomly pick a Game Over message**
        message = random.choice(game_over_messages)

        # **Auto-wrap the message into multiple lines**
        wrapped_text = textwrap.wrap(message, width=60)  # Adjust width for best fit

        start_time = pygame.time.get_ticks()
        self.start_screen_shake(pygame.time.get_ticks() / 1000.0, 0.3, 10)

        # **Step 1: Fade to White with a Slower Start**
        while True:
            elapsed_time = pygame.time.get_ticks() - start_time - 1000
            if elapsed_time <= 0:
                clock.tick(60)
                continue

            fade_progress = min(1, elapsed_time / fade_duration)  # Scale between 0 and 1

            # **Ease-in curve for smoother fade (slow start, faster finish)**
            alpha = int(255 * (fade_progress ** 0.5))  # Square root curve

            if fade_progress >= 1:
                break  # Stop once fully white

            # **Apply the fading white overlay**
            fade_overlay = pygame.Surface((screen_width, screen_height))
            fade_overlay.fill((255, 255, 255))
            fade_overlay.set_alpha(alpha / 2)  # Set opacity
            self.screen.blit(fade_overlay, (0, 0))

            pygame.display.flip()
            clock.tick(60)  # Maintain smooth animation

        # **Step 2: Full White Screen Before Fading in 'Game Over'**
        self.screen.fill((255, 255, 255))
        pygame.display.flip()

        # **Step 3: Smoothly Fade in 'Game Over' Over `text_fade_duration`**
        start_time = pygame.time.get_ticks()

        while True:
            elapsed_time = pygame.time.get_ticks() - start_time
            fade_progress = min(1, elapsed_time / text_fade_duration)

            if fade_progress >= 1:
                break  # Stop when fully faded in

            self.screen.fill((255, 255, 255))

            # **Render multi-line text with auto-wrap**
            text_alpha = int(fade_progress * 255)
            y_offset = screen_height // 2 - (len(wrapped_text) * 30)  # Center vertically

            for i, line in enumerate(wrapped_text):
                text_surface = font.render(line, True, (0, 0, 0))
                text_surface.set_alpha(text_alpha)
                text_rect = text_surface.get_rect(center=(screen_width // 2, y_offset + i * 50))
                self.screen.blit(text_surface, text_rect)

            pygame.display.flip()
            clock.tick(60)

        # **Step 4: Wait for Mouse Click to Restart**
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False  # Exit loop and restart game
        pygame.quit()
        sys.exit()

    def _draw_move_overlay(self, surface: pygame.Surface) -> None:
        """Draw the move overlay showing available actions.
        
        Args:
            surface: Surface to draw on
        """
        if not self.player:
            return
            
        player_hex = self.player.current_hex
        neighbors = self.get_hex_neighbors(player_hex)
        
        # Draw "STOMP" on player's hex if it's not cracked
        if player_hex.state == HexState.SOLID:
            text = self.overlay_font.render("STOMP", True, (0, 100, 255))
            text_rect = text.get_rect(center=player_hex.center)
            surface.blit(text, text_rect)
        
        # Draw "MOVE" or "PUSH" on neighboring hexes
        for hex in neighbors:
            if hex.state not in [HexState.BROKEN, HexState.BREAKING]:
                # Check if there's an enemy
                has_enemy = False
                for enemy in self.enemies:
                    if enemy.current_hex == hex:
                        has_enemy = True
                        break
                        
                if has_enemy:
                    text = self.overlay_font.render("PUSH", True, (0, 100, 255))
                else:
                    text = self.overlay_font.render("MOVE", True, (0, 100, 255))
                    
                text_rect = text.get_rect(center=hex.center)
                surface.blit(text, text_rect)
        
        # Only show JUMP and SLIDE options if player is not on a LAND hex
        if player_hex.state != HexState.LAND:
            # Get hexes for jumping (2 hexes away)
            # We'll use a similar approach to get_valid_jump_targets but with different filtering
            potential_jump_targets = set()
            for neighbor in neighbors:
                neighbor_neighbors = self.get_hex_neighbors(neighbor)
                for hex in neighbor_neighbors:
                    # Only add if it's not the player's hex and not directly adjacent to player
                    if hex != player_hex and hex not in neighbors:
                        potential_jump_targets.add(hex)
            
            # Show JUMP for valid targets - ONLY SOLID hexes, not cracked or broken
            for hex in potential_jump_targets:
                # Can only jump to SOLID hexes
                if hex.state == HexState.SOLID:
                    # Check if there's an enemy
                    has_enemy = False
                    for enemy in self.enemies:
                        if enemy.current_hex == hex:
                            has_enemy = True
                            break
                    
                    if not has_enemy:
                        text = self.overlay_font.render("JUMP", True, (0, 100, 255))
                        text_rect = text.get_rect(center=hex.center)
                        surface.blit(text, text_rect)
            
            # Add SLIDE for hexes 3 steps away in each of the 6 directions
            # Check each of the 6 directions from the player
            for direction in range(6):
                # Start at player's hex
                current_hex = player_hex
                path = [current_hex]
                
                # Move 3 times in the same direction
                for step in range(3):
                    next_hex = current_hex.get_neighbor(direction, self.hexes)
                    if not next_hex:
                        # Hit edge of grid
                        break
                    path.append(next_hex)
                    current_hex = next_hex
                
                # Check if we have a valid path of 4 hexes (including start)
                if len(path) != 4:
                    continue
                
                target_hex = path[-1]
                
                # Check if all hexes in the path (except player's hex) are SOLID and have no enemies
                path_valid = True
                for i, hex in enumerate(path):
                    # Skip player's hex - it can be cracked
                    if i == 0:
                        continue
                    
                    # All other hexes must be SOLID
                    if hex.state != HexState.SOLID:
                        path_valid = False
                        break
                        
                    # Check if hex has an enemy
                    for enemy in self.enemies:
                        if enemy.current_hex == hex:
                            path_valid = False
                            break
                    
                    if not path_valid:
                        break
                
                # If path is valid, show SLIDE or SLAM
                if path_valid:
                    # Check if there's an enemy in the next hex after the slide (4th hex)
                    # or in one of its adjacent hexes
                    will_slam = False
                    
                    # Check for the next hex in the same direction (4th hex)
                    fourth_hex = target_hex.get_neighbor(direction, self.hexes)
                    if fourth_hex:
                        # Check if there's an enemy in the 4th hex
                        for enemy in self.enemies:
                            if enemy.current_hex == fourth_hex:
                                will_slam = True
                                break
                                
                        # If no enemy in the 4th hex, check adjacent hexes to the 4th hex
                        if not will_slam:
                            # Check the hexes adjacent to the direction we came from
                            # These are direction-1, direction, direction+1
                            for adj_dir in [(direction - 1) % 6, direction, (direction + 1) % 6]:
                                adj_hex = target_hex.get_neighbor(adj_dir, self.hexes)
                                if adj_hex:
                                    for enemy in self.enemies:
                                        if enemy.current_hex == adj_hex:
                                            will_slam = True
                                            break
                                    
                                    if will_slam:
                                        break
                    
                    # Show SLAM or SLIDE based on whether there's an enemy to push
                    if will_slam:
                        text = self.overlay_font.render("SLAM", True, (0, 100, 255))
                    else:
                        text = self.overlay_font.render("SLIDE", True, (0, 100, 255))
                        
                    text_rect = text.get_rect(center=target_hex.center)
                    surface.blit(text, text_rect)


