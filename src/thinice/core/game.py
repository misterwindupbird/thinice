"""Main game class for the ice breaking game."""
from typing import List, Optional, Tuple, Dict
import os
import tkinter as tk
import pygame
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import random
import logging
from enum import Enum, auto

from .animation_manager import AnimationManager
from .hex import Hex
from .hex_state import HexState
from .entity import Player, Wolf
from .floating_text import FloatingText
from ..config import settings

# Add a test logging statement to verify logging is working
logging.info("Logging test: Game started")


class Area:
    """Stores the state of a 21x15 area when it's not active."""
    
    def __init__(self):
        # Store hex states and colors
        self.hex_states = [[None for _ in range(settings.hex_grid.GRID_HEIGHT)] 
                          for _ in range(settings.hex_grid.GRID_WIDTH)]
        self.hex_colors = [[None for _ in range(settings.hex_grid.GRID_HEIGHT)] 
                          for _ in range(settings.hex_grid.GRID_WIDTH)]
        
        # Just store enemy count, not their positions
        self.enemy_count = 0
        
        # Track if this area has been generated yet
        self.generated = False

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
        
        # Initialize supergrid
        self.supergrid_size = 15
        self.supergrid_position = [7, 7]  # Start in middle of supergrid
        self.areas = [[Area() for _ in range(self.supergrid_size)] 
                     for _ in range(self.supergrid_size)]
        
        # Initialize world size with default values
        self.world_width = 0
        self.world_height = 0
        
        # Initialize game state
        self.animation_manager = AnimationManager(on_finished=self.all_animations_completed_callback)
        self.hexes: List[List[Hex]] = []
        self._init_hex_grid()  # This calculates world_width and world_height
        self.start_time = pygame.time.get_ticks() / 1000.0

        # Initialize sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.player_sprite = pygame.sprite.GroupSingle()
        self.enemy_sprites = pygame.sprite.Group()
        
        # Initialize player on a random SOLID hex
        self.player = self._init_player()
        self.enemies = []
        
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
        
        # Initialize clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # Initialize game over flag
        self.game_over = False
        
        # Initialize win flag
        self.win = False
        
        # Initialize turn counter
        self.turn_counter = 0
        
        # Initialize score
        self.score = 0
        
        # Initialize high score
        self.high_score = 0
        
        # Initialize game over text
        self.game_over_text = None
        
        # Initialize win text
        self.win_text = None
        
        # Initialize restart text
        self.restart_text = None
        
        # Initialize quit text
        self.quit_text = None
        
        # Initialize game over surface
        self.game_over_surface = None
        
        # Initialize win surface
        self.win_surface = None
        
        # Initialize restart surface
        self.restart_surface = None
        
        # Initialize quit surface
        self.quit_surface = None
        
        # Initialize game over rect
        self.game_over_rect = None
        
        # Initialize win rect
        self.win_rect = None
        
        # Initialize restart rect
        self.restart_rect = None
        
        # Initialize quit rect
        self.quit_rect = None

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
        pygame.display.set_caption("Hex Grid")
        settings.display.font = pygame.font.SysFont(settings.display.FONT_NAME, settings.display.FONT_SIZE)
    
    def _init_hex_grid(self) -> None:
        """Initialize the hex grid.
        
        - Map fills the entire screen
        - Hexes not entirely on screen are automatically LAND
        - 10 random hexes in the visible area are LAND
        - Hexes adjacent to LAND are pure white
        - The rest are SOLID with blue-grey gradient
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
        
        # Track all LAND hexes for adjacency check later
        visible_area = []
        edge_hexes = []
        land_hexes = []
        
        # Create the hex grid with proper world coordinates
        for x in range(settings.hex_grid.GRID_WIDTH):
            for y in range(settings.hex_grid.GRID_HEIGHT):
                # Calculate center position in world coordinates
                center_x = x * spacing_x
                center_y = y * spacing_y
                
                # Apply offset for odd columns
                if x % 2 == 1:
                    center_y += spacing_y / 2
                
                # Create the hex
                is_land = False
                
                # Check if the hex is at the edge of the grid
                if (x == 0 or x == settings.hex_grid.GRID_WIDTH - 1 or 
                    y == 0 or y == settings.hex_grid.GRID_HEIGHT - 1):
                    is_land = True
                    edge_hexes.append((x, y))
                # Check if the hex is not fully visible on screen
                elif (center_x - settings.hex_grid.RADIUS < 0 or 
                      center_x + settings.hex_grid.RADIUS > settings.display.WINDOW_WIDTH or
                      center_y - settings.hex_grid.RADIUS < 0 or 
                      center_y + settings.hex_grid.RADIUS > settings.display.WINDOW_HEIGHT):
                    is_land = True
                    edge_hexes.append((x, y))
                else:
                    visible_area.append((x, y))
                
                # Create the hex with the appropriate state
                if is_land:
                    # Generate a random green shade for land
                    base_r, base_g, base_b = settings.land.BASE_COLOR
                    color_variation = settings.land.COLOR_VARIATION
                    r = min(255, max(0, base_r + random.randint(-color_variation, color_variation)))
                    g = min(255, max(0, base_g + random.randint(-color_variation, color_variation)))
                    b = min(255, max(0, base_b + random.randint(-color_variation, color_variation)))
                    color = (r, g, b)
                    self.hexes[x][y] = Hex(center_x, center_y, x, y, self.animation_manager, color=color, state=HexState.LAND)
                    land_hexes.append((x, y))
                else:
                    # Create a regular ice hex with a pure white to blue-grey gradient
                    # No pink/purple tints - pure cool blue-grey only
                    
                    # Random value between 0.0 and 1.0 to determine position in the color range
                    color_position = random.random()
                    
                    # Create pure blue-grey by reducing red more dramatically
                    # White (255,255,255) to blue-grey (220,240,255)
                    ice_r = int(255 - (color_position * 35))  # Reduce red more (255 to 220)
                    ice_g = int(255 - (color_position * 15))  # Keep green higher (255 to 240)
                    ice_b = 255  # Keep blue at maximum
                    
                    # Very minimal variation to maintain color purity
                    tiny_variation = 2
                    ice_r = min(255, max(215, ice_r + random.randint(-tiny_variation, tiny_variation)))
                    ice_g = min(255, max(235, ice_g + random.randint(-tiny_variation, tiny_variation)))
                    ice_b = 255  # No variation in blue to ensure we stay in the blue spectrum
                    
                    ice_color = (ice_r, ice_g, ice_b)
                    self.hexes[x][y] = Hex(center_x, center_y, x, y, self.animation_manager, color=ice_color)
        
        # Select 10 random hexes from the visible area to be LAND
        if len(visible_area) > 10:
            random_land_positions = random.sample(visible_area, 10)
            
            for x, y in random_land_positions:
                # Generate a random green shade for land
                base_r, base_g, base_b = settings.land.BASE_COLOR
                color_variation = settings.land.COLOR_VARIATION
                r = min(255, max(0, base_r + random.randint(-color_variation, color_variation)))
                g = min(255, max(0, base_g + random.randint(-color_variation, color_variation)))
                b = min(255, max(0, base_b + random.randint(-color_variation, color_variation)))
                color = (r, g, b)
                
                # Convert existing hex to LAND
                self.hexes[x][y] = Hex(self.hexes[x][y].center[0], self.hexes[x][y].center[1], 
                                      x, y,
                                      animation_manager=self.animation_manager,
                                      color=color,
                                      state=HexState.LAND)
                land_hexes.append((x, y))

        # Second pass: Make hexes adjacent to LAND hexes pure white
        white_hexes_count = 0
        for land_x, land_y in land_hexes:
            # Get neighbors of this LAND hex
            land_hex = self.hexes[land_x][land_y]
            neighbors = self.get_hex_neighbors(land_hex)
            
            # Set each non-LAND neighbor to pure white
            for neighbor in neighbors:
                if neighbor.state != HexState.LAND:
                    # Create a new hex with the same properties but pure white color
                    pure_white = (255, 255, 255)
                    self.hexes[neighbor.grid_x][neighbor.grid_y] = Hex(
                        neighbor.center[0], neighbor.center[1], 
                        neighbor.grid_x, neighbor.grid_y,
                        animation_manager=self.animation_manager,
                        color=pure_white
                    )
                    white_hexes_count += 1
        
    def _init_player(self) -> Player:
        """Initialize the player entity on a random solid hex.
        
        Returns:
            The player entity
        """
        # Find all solid hexes
        solid_hexes = []
        for row in self.hexes:
            for hex in row:
                if hex.state == HexState.SOLID:
                    solid_hexes.append(hex)
        
        # Choose a random solid hex
        if solid_hexes:
            start_hex = random.choice(solid_hexes)
            player = Player(start_hex, self.animation_manager)
            
            # Add player to sprite groups
            self.player_sprite.add(player)
            self.all_sprites.add(player)
            
            return player
        else:
            logging.error("No solid hexes found for player start position!")
            return None
    
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
        try:
            running = True
            
            while running:
                current_time = pygame.time.get_ticks() / 1000.0 - self.start_time
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            self._handle_click(event.pos, current_time)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                            self.shift_pressed = True
                        elif event.key == pygame.K_LEFT:
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
    
    def find_straight_line_path(self, start_hex: Hex, end_hex: Hex) -> Optional[List[Hex]]:
        """Find a straight line path between two hexes if one exists.
        
        Args:
            start_hex: Starting hex
            end_hex: Ending hex
            
        Returns:
            List of hexes in the path including start and end, or None if no straight line exists
        """
        # Check if the hexes are in a straight line by checking if they share a direction
        # In a hex grid, there are 6 possible directions
        
        # Get the grid coordinates
        start_x, start_y = start_hex.grid_x, start_hex.grid_y
        end_x, end_y = end_hex.grid_x, end_hex.grid_y
        
        # Adjust for odd-row offset in our coordinate system
        if start_x % 2 == 1:
            start_y = start_y + 0.5
        if end_x % 2 == 1:
            end_y = end_y + 0.5
        
        # Calculate the vector between the hexes
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Check if the distance is exactly 3
        distance = self.hex_distance(start_hex, end_hex)
        if distance != 3:
            return None
        
        # Check if it's a straight line by seeing if the ratio of dx to dy matches one of the 6 directions
        # The 6 directions in a hex grid are:
        # (1,0), (0.5,0.75), (-0.5,0.75), (-1,0), (-0.5,-0.75), (0.5,-0.75)
        
        # Normalize the vector
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return None
            
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Define the 6 directions (normalized)
        directions = [
            (1, 0),                    # East
            (0.5, 0.866),              # Northeast
            (-0.5, 0.866),             # Northwest
            (-1, 0),                   # West
            (-0.5, -0.866),            # Southwest
            (0.5, -0.866)              # Southeast
        ]
        
        # Check if our normalized vector is close to any of these directions
        is_straight = False
        for dir_x, dir_y in directions:
            # Calculate dot product to check alignment
            dot_product = dx_norm * dir_x + dy_norm * dir_y
            if dot_product > 0.95:  # Allow some small error
                is_straight = True
                break
                
        if not is_straight:
            return None
        
        # Now find the intermediate hexes
        path = [start_hex]
        
        # Find the two intermediate hexes
        for step in range(1, 3):
            # Calculate the position at this step
            ratio = step / 3.0
            intermediate_x = start_x + dx * ratio
            intermediate_y = start_y + dy * ratio
            
            # Find the nearest hex to this position
            nearest_hex = None
            min_distance = float('inf')
            
            # Check all hexes at distance 'step' from start
            hexes_at_distance = self.get_hexes_at_distance(start_hex, step)
            for hex in hexes_at_distance:
                hex_x, hex_y = hex.grid_x, hex.grid_y
                if hex_x % 2 == 1:
                    hex_y = hex_y + 0.5
                
                # Calculate distance to the ideal position
                dist = ((hex_x - intermediate_x)**2 + (hex_y - intermediate_y)**2)**0.5
                if dist < min_distance:
                    min_distance = dist
                    nearest_hex = hex
            
            if nearest_hex:
                path.append(nearest_hex)
            else:
                return None
        
        # Add the end hex
        path.append(end_hex)
        
        return path
    
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
            
        # Debug: Add wolf on right click
        if pygame.mouse.get_pressed()[2]:  # Right mouse button
            if clicked_hex.state == HexState.SOLID and not self._hex_has_entity(clicked_hex):
                self.add_enemy(clicked_hex)
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
                        # Create floating text for CRACK action
                        self.add_floating_text("CRACK!", clicked_hex.center, (255, 70, 70))
                    elif clicked_hex.state == HexState.CRACKED:
                        clicked_hex.break_ice()
                        # Create floating text for BREAK action
                        self.add_floating_text("BREAK!", clicked_hex.center, (255, 30, 30))
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

        # Check if clicked on player's hex (STOMP action)
        if clicked_hex == self.player.current_hex and clicked_hex.state == HexState.SOLID:
            # Create floating text for STOMP action - no "CRACK" text for STOMP
            self.add_floating_text("STOMP!", self.player.current_hex.center, (255, 0, 0))

            # Apply screen shake effect
            self.start_screen_shake(current_time)

            # Get all adjacent hexes for later processing
            neighbors = self.get_hex_neighbors(clicked_hex)

            # First, only crack the player's hex if it's SOLID
            if clicked_hex.state == HexState.SOLID:
                clicked_hex.crack([])  # Empty list since we don't need to check neighbors here
                # Schedule the surrounding hexes to crack/break after a delay
                self.schedule_surrounding_hex_effects(clicked_hex, neighbors, current_time)
            elif clicked_hex.state == HexState.CRACKED:
                clicked_hex.break_ice()
                # Schedule the surrounding hexes to crack/break after a delay
                self.schedule_surrounding_hex_effects(clicked_hex, neighbors, current_time)
            return


        # Check for SPRINT action (3 hexes away in a straight line)
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
            self.add_floating_text("PUSH", enemy.current_hex.center, (255, 0, 0))
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
        
        # Draw all sprites
        # Note: We're not using the built-in draw method because our sprites have custom drawing logic
        # Instead, we call the draw method on each sprite individually
        for sprite in self.all_sprites:
            sprite.draw(draw_surface, current_time)
        
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
        
        # Draw supergrid position indicator
        pos_text = f"Area: {self.supergrid_position[0]},{self.supergrid_position[1]}"
        pos_surface = settings.display.font.render(pos_text, True, (255, 255, 255))
        self.screen.blit(pos_surface, (10, 10))
        
        # Draw debug info if enabled
        if self.debug_mode:
            self._draw_debug_info()
        
        # Update the display
        pygame.display.flip()
    
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
        
        # Add floating text for LEAP
        self.add_floating_text("JUMP!", self.player.current_hex.center, (255, 50, 50))
        
        # Define callback for when jump animation completes
        def on_jump_complete():
            # Crack or break the landing hex
            if target_hex.state == HexState.SOLID:
                target_hex.crack([])
            elif target_hex.state == HexState.CRACKED:
                target_hex.break_ice()
        
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
                
        # If we started new animations, don't switch state yet
        if self.animation_manager.blocking_animations > 0 or drowning_started:
            logging.info(f"Animations still in progress, not switching state. Blocking animations: {self.animation_manager.blocking_animations}")
            return
            
        # Process any pending hex effects
        self.process_pending_hex_effects(current_time)
        
        # Handle state transitions
        if self.game_state == GameState.PLAYER:
            # Player's turn is complete, switch to enemy turn
            self.game_state = GameState.ENEMY
            logging.info("Switching to ENEMY turn")
            self._enemy_ai()
        elif self.game_state == GameState.ENEMY:
            # Enemy's turn is complete, switch to player turn
            self.game_state = GameState.PLAYER
            logging.info("Switching to PLAYER turn")
            self.turn_counter += 1


    def _enemy_ai(self) -> None:
        """Run AI for all enemies."""
        current_time = pygame.time.get_ticks() / 1000.0
        
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
                self.add_floating_text("ATTACK", enemy.current_hex.center)
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
        
        # Add enemy to sprite groups
        self.enemy_sprites.add(enemy)
        self.all_sprites.add(enemy)

    def _hex_has_entity(self, hex: Hex) -> bool:
        """Check if a hex has an entity on it.
        
        Args:
            hex: The hex to check
            
        Returns:
            True if the hex has an entity, False otherwise
        """
        if self.player and self.player.current_hex == hex:
            return True
            
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
                    area.hex_states[x][y] = self.hexes[x][y].state
                    area.hex_colors[x][y] = self.hexes[x][y].color
        
        # Save enemy count
        area.enemy_count = len(self.enemies)

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
                    
                    # Update color first 
                    if area.hex_colors[x][y]:
                        current_hex.color = area.hex_colors[x][y]
                    
                    # Handle different hex states - just set the state without recreating visuals
                    # The visual state will be restored from the existing sprites when drawn
                    if saved_state in [HexState.CRACKED, HexState.BROKEN]:
                        # Just set the state directly without creating new cracks
                        # This avoids duplicate cracks
                        current_hex.state = saved_state
                        
                        # Only create cracks if none exist yet
                        if len(current_hex.cracks) == 0 and saved_state == HexState.CRACKED:
                            # Add a few simple cracks
                            edge_points = current_hex.edge_points
                            # Add 3-5 random cracks
                            num_cracks = random.randint(3, 5)
                            for _ in range(num_cracks):
                                # Pick a random edge point for the crack
                                end_point = random.choice(edge_points)
                                current_hex.add_straight_crack(end_point)
                            
                            # Add some secondary cracks
                            current_hex.add_secondary_cracks()
                        
                        # For broken hexes, create ice fragments if needed
                        if saved_state == HexState.BROKEN and not current_hex.ice_fragments:
                            if hasattr(current_hex, '_find_ice_fragments'):
                                current_hex._find_ice_fragments()
                    else:
                        # For other states, just set the state directly
                        current_hex.state = saved_state
        
        # Clear current enemies
        self.enemies = []
        self.enemy_sprites.empty()
        
        # For now, we don't restore enemy positions, just log the count
        logging.info(f"Area has {area.enemy_count} enemies (not spawning them for now)")

    def navigate_area(self, dx, dy):
        """Navigate to a different area in the supergrid.
        
        Args:
            dx: Change in x position (-1, 0, 1)
            dy: Change in y position (-1, 0, 1)
        """
        # Save current area state
        self.save_current_area()
        
        # Update supergrid position, ensuring we stay within bounds
        new_x = max(0, min(self.supergrid_size - 1, self.supergrid_position[0] + dx))
        new_y = max(0, min(self.supergrid_size - 1, self.supergrid_position[1] + dy))
        
        # If position hasn't changed, we're at the edge of the world
        if new_x == self.supergrid_position[0] and new_y == self.supergrid_position[1]:
            return False
        
        # Update position
        self.supergrid_position[0] = new_x
        self.supergrid_position[1] = new_y
        
        # Clear current enemies - explicitly remove from all sprite groups
        for enemy in self.enemies:
            enemy.kill()  # Remove from all sprite groups
        self.enemies = []
        self.enemy_sprites.empty()
        
        area = self.areas[self.supergrid_position[0]][self.supergrid_position[1]]
        if area.generated:
            # Restore existing area
            self._restore_area()
        else:
            # Generate new area with existing code
            self._init_hex_grid()
        
        # Reset player position
        if self.player:
            self.player_sprite.empty()
            self.all_sprites.remove(self.player)
        self.player = self._init_player()
        
        logging.info(f"Navigated to supergrid position: {self.supergrid_position}")
        return True


