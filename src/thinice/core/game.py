"""Main game class for the ice breaking game."""
from typing import List, Optional, Tuple
import os
import tkinter as tk
import pygame
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import random

from .hex import Hex
from .hex_state import HexState
from ..config.settings import display, hex_grid, land

class GameRestartHandler(FileSystemEventHandler):
    """File system event handler for game auto-restart."""
    
    def on_modified(self, event):
        """Handle file modification event.
        
        Args:
            event: File system event
        """
        if event.src_path.endswith('game.py'):
            print("Game file changed, restarting...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

class Game:
    """Main game class managing the game loop and hex grid."""
    
    def __init__(self, enable_watcher: bool = False):
        """Initialize the game.
        
        Args:
            enable_watcher: Whether to enable auto-restart on file changes
        """
        # Set up file watcher
        self.observer = None
        if enable_watcher:
            self.observer = Observer()
            self.observer.schedule(GameRestartHandler(), path='.', recursive=False)
            self.observer.start()
        
        # Initialize display
        self._init_display()
        
        # Initialize viewport parameters with default values
        self.scroll_x = 0
        self.scroll_y = 0
        self.scroll_speed = 15  # Pixels per scroll unit
        
        # Initialize player
        self.player_hex = None  # Current hex the player is on
        self.player_color = (255, 0, 0)  # Red
        self.player_radius = hex_grid.RADIUS // 2  # Half the hex radius
        self.shift_pressed = False  # Track if shift is being pressed
        
        # Initialize world size with default values
        self.world_width = 0
        self.world_height = 0
        self.max_scroll_x = 0
        self.max_scroll_y = 0
        
        # Initialize game state
        self.hexes: List[List[Hex]] = []
        self._init_hex_grid()  # This calculates world_width and world_height
        self.start_time = pygame.time.get_ticks() / 1000.0
        
        # Now calculate maximum scroll limits based on the initialized world size
        self.max_scroll_x = max(0, self.world_width - display.WINDOW_WIDTH)
        self.max_scroll_y = max(0, self.world_height - display.WINDOW_HEIGHT)
        
        print(f"Scroll limits: ({self.max_scroll_x}, {self.max_scroll_y})")
    
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
        self.screen = pygame.display.set_mode((display.WINDOW_WIDTH, display.WINDOW_HEIGHT))
        pygame.display.set_caption("Hex Grid")
        display.font = pygame.font.SysFont(display.FONT_NAME, display.FONT_SIZE)
    
    def _init_hex_grid(self) -> None:
        """Initialize the hex grid.
        
        - Map fills the entire screen
        - Hexes not entirely on screen are automatically LAND
        - 10 random hexes in the visible area are LAND
        - Hexes adjacent to LAND are pure white
        - The rest are SOLID with blue-grey gradient
        """
        # Update grid dimensions to ensure the map fills the entire screen
        hex_grid.GRID_WIDTH = 100
        hex_grid.GRID_HEIGHT = 100
        
        # Calculate hex dimensions
        hex_height = hex_grid.RADIUS * 1.732  # sqrt(3)
        spacing_x = hex_grid.RADIUS * 1.5
        spacing_y = hex_height
        
        # Create hex grid
        self.hexes = [[None for _ in range(hex_grid.GRID_HEIGHT)] 
                     for _ in range(hex_grid.GRID_WIDTH)]
        
        # Calculate world size
        self.world_width = hex_grid.GRID_WIDTH * spacing_x
        self.world_height = hex_grid.GRID_HEIGHT * spacing_y + (spacing_y / 2)
        
        print(f"World size: {self.world_width}x{self.world_height}")
        print(f"Window size: {display.WINDOW_WIDTH}x{display.WINDOW_HEIGHT}")
        
        # Calculate how many hexes can fit entirely on the screen
        visible_area = []
        edge_hexes = []
        land_hexes = []  # Track all LAND hexes for adjacency check later
        
        # Create the hex grid with proper world coordinates
        for x in range(hex_grid.GRID_WIDTH):
            for y in range(hex_grid.GRID_HEIGHT):
                # Calculate center position in world coordinates
                center_x = x * spacing_x
                center_y = y * spacing_y
                
                # Apply offset for odd columns
                if x % 2 == 1:
                    center_y += spacing_y / 2
                
                # Create the hex
                is_land = False
                
                # Convert world to screen coordinates
                screen_x = center_x - self.scroll_x
                screen_y = center_y - self.scroll_y
                
                # Check if the hex is fully visible on screen
                is_fully_visible = (
                    screen_x - hex_grid.RADIUS >= 0 and
                    screen_x + hex_grid.RADIUS <= display.WINDOW_WIDTH and
                    screen_y - hex_grid.RADIUS >= 0 and
                    screen_y + hex_grid.RADIUS <= display.WINDOW_HEIGHT
                )
                
                # Check if the hex is at least partially visible
                is_partially_visible = (
                    screen_x + hex_grid.RADIUS >= 0 and
                    screen_x - hex_grid.RADIUS <= display.WINDOW_WIDTH and
                    screen_y + hex_grid.RADIUS >= 0 and
                    screen_y - hex_grid.RADIUS <= display.WINDOW_HEIGHT
                )
                
                # Hexes not entirely on screen are LAND
                if is_partially_visible and not is_fully_visible:
                    is_land = True
                    edge_hexes.append((x, y))
                elif is_fully_visible:
                    visible_area.append((x, y))
                
                # Create the hex with the appropriate state
                if is_land:
                    # Generate a random green shade for land
                    base_r, base_g, base_b = land.BASE_COLOR
                    color_variation = land.COLOR_VARIATION
                    r = min(255, max(0, base_r + random.randint(-color_variation, color_variation)))
                    g = min(255, max(0, base_g + random.randint(-color_variation, color_variation)))
                    b = min(255, max(0, base_b + random.randint(-color_variation, color_variation)))
                    color = (r, g, b)
                    self.hexes[x][y] = Hex(center_x, center_y, x, y, color=color, state=HexState.LAND)
                    land_hexes.append((x, y))
                    print(f"Creating edge LAND hex at ({x}, {y})")
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
                    self.hexes[x][y] = Hex(center_x, center_y, x, y, color=ice_color)
        
        # Select 10 random hexes from the visible area to be LAND
        if len(visible_area) > 10:
            random_land_positions = random.sample(visible_area, 10)
            
            for x, y in random_land_positions:
                # Generate a random green shade for land
                base_r, base_g, base_b = land.BASE_COLOR
                color_variation = land.COLOR_VARIATION
                r = min(255, max(0, base_r + random.randint(-color_variation, color_variation)))
                g = min(255, max(0, base_g + random.randint(-color_variation, color_variation)))
                b = min(255, max(0, base_b + random.randint(-color_variation, color_variation)))
                color = (r, g, b)
                
                # Convert existing hex to LAND
                self.hexes[x][y] = Hex(self.hexes[x][y].center[0], self.hexes[x][y].center[1], 
                                      x, y, color=color, state=HexState.LAND)
                land_hexes.append((x, y))
                print(f"Creating random LAND hex at ({x}, {y})")
        
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
                        color=pure_white
                    )
                    white_hexes_count += 1
        
    
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
            if 0 <= nx < hex_grid.GRID_WIDTH and 0 <= ny < hex_grid.GRID_HEIGHT:
                neighbors.append(self.hexes[nx][ny])
        
        return neighbors
    
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
                
                if dist < min_dist and dist <= hex_grid.RADIUS * hex_grid.RADIUS:
                    min_dist = dist
                    nearest_hex = hex
        
        return nearest_hex
    
    def run(self) -> None:
        """Run the main game loop."""
        try:
            running = True
            scrolling = False
            last_mouse_pos = (0, 0)
            
            while running:
                current_time = pygame.time.get_ticks() / 1000.0 - self.start_time
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            self._handle_click(event.pos, current_time)
                        elif event.button == 2:  # Middle mouse button
                            scrolling = True
                            last_mouse_pos = event.pos
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 2:  # Middle mouse button
                            scrolling = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                            self.shift_pressed = True
                    elif event.type == pygame.KEYUP:
                        if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                            self.shift_pressed = False
                    elif event.type == pygame.MOUSEMOTION and scrolling:
                        # Calculate the difference from the last position
                        dx = event.pos[0] - last_mouse_pos[0]
                        dy = event.pos[1] - last_mouse_pos[1]
                        self._handle_scroll(-dx, -dy)
                        last_mouse_pos = event.pos
                    elif event.type == pygame.MOUSEWHEEL:
                        # Handle mouse wheel scrolling
                        # Vertical scrolling (y) is primary, horizontal (x) is secondary
                        scroll_y = event.y * self.scroll_speed
                        scroll_x = event.x * self.scroll_speed
                        self._handle_scroll(-scroll_x, -scroll_y)
                
                self._draw(current_time)
                pygame.display.flip()
                
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            pygame.quit()
    
    def _handle_click(self, pos: Tuple[int, int], current_time: float) -> None:
        """Handle a mouse click at the given position.
        
        Args:
            pos: Mouse position (x, y) in screen coordinates
            current_time: Current game time in seconds
        """
        # Find the clicked hex
        clicked_hex = self.pixel_to_hex(*pos)
        if not clicked_hex:
            return
            
        # Log the clicked hex
        print(f"Clicked hex at ({clicked_hex.grid_x}, {clicked_hex.grid_y})")
        
        if self.shift_pressed:
            # Move player to the clicked hex only if it's not broken
            if clicked_hex.state != HexState.BROKEN:
                self.player_hex = clicked_hex
                print(f"Player moved to ({clicked_hex.grid_x}, {clicked_hex.grid_y})")
            else:
                print("Cannot move player to broken hex!")
        else:
            # Don't break the hex if the player is standing on it
            if self.player_hex and clicked_hex == self.player_hex:
                print("Cannot break hex where player is standing!")
                return
            
            # Handle normal ice-breaking behavior
            neighbors = self.get_hex_neighbors(clicked_hex)
            
            # Handle state transitions
            if clicked_hex.state == HexState.SOLID:
                clicked_hex.crack(neighbors)
            elif clicked_hex.state == HexState.CRACKED:
                clicked_hex.break_ice()
    
    def _draw(self, current_time: float) -> None:
        """Draw the game state.
        
        Args:
            current_time: Current game time in seconds
        """
        # Clear the screen
        self.screen.fill(display.BACKGROUND_COLOR)
        
        # Calculate hex dimensions
        hex_height = hex_grid.RADIUS * 1.732
        spacing_x = hex_grid.RADIUS * 1.5
        spacing_y = hex_height
        
        # Determine which hexes are visible in the viewport
        # Convert viewport boundaries to grid coordinates
        min_visible_x = max(0, int(self.scroll_x / spacing_x) - 1)
        max_visible_x = min(hex_grid.GRID_WIDTH - 1, int((self.scroll_x + display.WINDOW_WIDTH) / spacing_x) + 1)
        min_visible_y = max(0, int(self.scroll_y / spacing_y) - 1)
        max_visible_y = min(hex_grid.GRID_HEIGHT - 1, int((self.scroll_y + display.WINDOW_HEIGHT) / spacing_y) + 1)
        
        # Collect visible non-broken hexes for collision detection
        non_broken_hexes = []
        
        # Draw only the visible hexes
        for x in range(min_visible_x, max_visible_x + 1):
            for y in range(min_visible_y, max_visible_y + 1):
                hex = self.hexes[x][y]
                
                # Calculate the screen position for this hex
                world_x, world_y = hex.center
                screen_x = world_x - self.scroll_x
                screen_y = world_y - self.scroll_y
                
                # Only add to non-broken list if this hex is visible and not broken
                if hex.state != HexState.BROKEN:
                    non_broken_hexes.append(hex)
                
                # Save original center
                original_center = hex.center
                
                # Set temporary center for drawing
                hex.center = (screen_x, screen_y)
                
                # Draw the hex
                if hex.state in [HexState.BROKEN, HexState.BREAKING]:
                    hex.draw(self.screen, current_time, non_broken_hexes)
                else:
                    hex.draw(self.screen, current_time)
                
                # Restore original center
                hex.center = original_center
        
        # Draw player if they exist on a hex
        if self.player_hex:
            # Convert world coordinates to screen coordinates
            player_world_x, player_world_y = self.player_hex.center
            player_screen_x = player_world_x - self.scroll_x
            player_screen_y = player_world_y - self.scroll_y
            
            # Only draw if the player is on screen
            if (player_screen_x + self.player_radius >= 0 and 
                player_screen_x - self.player_radius <= display.WINDOW_WIDTH and
                player_screen_y + self.player_radius >= 0 and 
                player_screen_y - self.player_radius <= display.WINDOW_HEIGHT):
                
                # Draw the player as a red circle
                pygame.draw.circle(
                    self.screen, 
                    self.player_color, 
                    (player_screen_x, player_screen_y), 
                    self.player_radius
                )
        
        # Remove debug overlay with hex coordinates 