"""Main game class for the ice breaking game."""
from typing import List, Optional, Tuple
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
        
        # Initialize world size with default values
        self.world_width = 0
        self.world_height = 0
        
        # Initialize game state
        self.animation_manager = AnimationManager(on_finished=self.all_animations_completed_callback)
        self.hexes: List[List[Hex]] = []
        self._init_hex_grid()  # This calculates world_width and world_height
        self.start_time = pygame.time.get_ticks() / 1000.0

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
        """Initialize the player on a random SOLID hex.
        
        Returns:
            The initialized Player entity
        """
        # Find all SOLID hexes
        solid_hexes = []
        for x in range(settings.hex_grid.GRID_WIDTH):
            for y in range(settings.hex_grid.GRID_HEIGHT):
                if self.hexes[x][y].state == HexState.SOLID:
                    solid_hexes.append(self.hexes[x][y])
        
        # Choose a random SOLID hex
        if solid_hexes:
            start_hex = random.choice(solid_hexes)
            return Player(start_hex, self.animation_manager)
        else:
            # Fallback to first hex if no SOLID hexes
            logging.warning("Warning: No SOLID hexes found for player start. Using first hex.")
            return Player(self.hexes[0][0])
    
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
        
        for edge_index in range(6):
            path = [player_hex]
            current_hex = player_hex
            valid_path = True
            
            for step in range(3):
                next_hex = current_hex.get_neighbor(edge_index, self.hexes)
                if not next_hex or next_hex.state in [HexState.LAND, HexState.BROKEN]:
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
        """Handle a mouse click at the given position.
        
        Args:
            pos: Mouse position (x, y) in screen coordinates
            current_time: Current game time in seconds
        """
        if self.game_state != GameState.PLAYER:
            logging.debug(f'Invalid! Game state is {self.game_state}.')
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
                    # Create an enemy is there isn't one on the hex, remove it if there is one.
                    self.enemies.append(Wolf(start_hex=clicked_hex, animation_manager=self.animation_manager))
        else:
            # REGULAR CLICK
            # Check if clicked on player's hex (STOMP action)
            logging.info(f"REGULAR CLICK at ({clicked_hex.grid_x}, {clicked_hex.grid_y})")
            if self.player and clicked_hex == self.player.current_hex:
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
            else:
                # Check if player is already moving
                if self.player and self.player.is_moving:
                    return
                
                # Check for SPRINT action (3 hexes away in a straight line)
                if self.player:
                    sprint_targets = self.get_valid_sprint_targets(self.player.current_hex)
                    for target, path in sprint_targets:
                        if clicked_hex == target:
                            self.player.sprint(path, current_time)
                            return
                
                # Check for JUMP action (2 hexes away)
                if self.player:
                    valid_jump_targets = self.get_valid_jump_targets(self.player.current_hex)
                    if clicked_hex in valid_jump_targets:
                        # Perform JUMP maneuver
                        self.perform_jump(clicked_hex, current_time)
                        return
                
                # Regular move to adjacent hex
                if self.player:
                    # Check if the clicked hex is adjacent to the player
                    adjacent_hexes = self.get_hex_neighbors(self.player.current_hex)
                    if clicked_hex in adjacent_hexes:
                        # Try to move player to the clicked hex
                        self.player.move(clicked_hex, current_time)
    
    def _draw(self, current_time: float) -> None:
        """Draw the game state.
        
        Args:
            current_time: Current game time in seconds
        """
        # Clear the screen
        self.screen.fill(settings.display.BACKGROUND_COLOR)
        
        # Process any pending hex effects
        self.process_pending_hex_effects(current_time)
        
        # Calculate screen shake offset
        shake_offset = self.apply_screen_shake(current_time)
        
        # Collect non-broken hexes for collision detection
        non_broken_hexes = []
        
        # Create a temporary surface for drawing with shake effect
        if self.is_screen_shaking:
            temp_surface = pygame.Surface((settings.display.WINDOW_WIDTH, settings.display.WINDOW_HEIGHT))
            temp_surface.fill(settings.display.BACKGROUND_COLOR)
            draw_surface = temp_surface
        else:
            draw_surface = self.screen
        
        # Draw all hexes
        for x in range(settings.hex_grid.GRID_WIDTH):
            for y in range(settings.hex_grid.GRID_HEIGHT):
                hex = self.hexes[x][y]
                
                # Only add to non-broken list if this hex is not broken
                if hex.state != HexState.BROKEN:
                    non_broken_hexes.append(hex)
                
                # Draw the hex
                if hex.state in [HexState.BROKEN, HexState.BREAKING]:
                    hex.draw(draw_surface, current_time, non_broken_hexes)
                else:
                    hex.draw(draw_surface, current_time)
        
        # Draw player using the Entity draw method
        if self.player:
            self.player.draw(draw_surface, current_time)
            
            # Update player animation
            self.player.update(current_time)
        else:
            logging.warning("Warning: Player is None!")

        for enemy in self.enemies:
            enemy.draw(draw_surface, current_time)
            enemy.update(current_time)
        
        # Update and draw floating text animations
        active_texts = []
        for text in self.floating_texts:
            text.update(current_time)
            if text.is_active:
                text.draw(draw_surface)
                active_texts.append(text)
        
        # Remove inactive texts
        self.floating_texts = active_texts
        
        # Debug info
        debug_text = f"Player: {self.player.current_hex.grid_x}, {self.player.current_hex.grid_y}"
        debug_surface = settings.display.font.render(debug_text, True, (255, 255, 255))
        draw_surface.blit(debug_surface, (10, 10))
        
        # Apply screen shake by blitting the temp surface with an offset
        if self.is_screen_shaking:
            self.screen.blit(temp_surface, shake_offset)
    
    def start_screen_shake(self, current_time: float) -> None:
        """Start a screen shake effect.
        
        Args:
            current_time: Current game time in seconds
        """
        self.screen_shake_start = current_time
        self.screen_shake_duration = 0.3  # seconds
        self.screen_shake_intensity = 5  # pixels
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
        """ All blocking animations have completed. """
        logging.info(f'Received animation completion callback. {self.game_state=}')

        # if any enemies are now in a BROKEN hex, we need to drown them
        for enemy in self.enemies:
            if enemy.current_hex.state == HexState.BROKEN:
                enemy.drown(current_time=pygame.time.get_ticks() / 1000.0 - self.start_time,
                            completion_callback=lambda: self.enemies.remove(enemy))

        # if we started any new animations, we won't switch state yet
        if self.animation_manager.blocking_animations > 0:
            return

        match self.game_state:
            case GameState.PLAYER:
                # all player-related animations have completed
                self.game_state = GameState.ENEMY
                self._enemy_ai()
            case GameState.ENEMY:
                # all enemy-related animations have completed
                self.game_state = GameState.PLAYER


    def _enemy_ai(self) -> None:
        current_time = pygame.time.get_ticks() / 1000.0 - self.start_time
        for enemy in self.enemies:
            if self.player.current_hex in self.get_hex_neighbors(enemy.current_hex):
                self.add_floating_text("ATTACK", enemy.current_hex.center)
                continue

            from .pathfinding import a_star # avoid circular dependencies
            path = a_star(enemy.current_hex, self.player.current_hex)
            logging.info(f'{path=}')
            if path is not None and len(path) > 1:
                enemy.move(target_hex=path[1], current_time=current_time)

        if self.animation_manager.blocking_animations == 0:
            self.game_state = GameState.PLAYER


