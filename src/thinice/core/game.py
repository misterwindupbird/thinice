"""Main game class for the ice breaking game."""
from typing import List, Optional, Tuple
import os
import tkinter as tk
import pygame
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

from .hex import Hex
from .hex_state import HexState
from ..config.settings import display, hex_grid

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
        
        # Initialize game state
        self.hexes: List[List[Hex]] = []
        self._init_hex_grid()
        self.start_time = pygame.time.get_ticks() / 1000.0
    
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
        self.font = pygame.font.SysFont(display.FONT_NAME, display.FONT_SIZE)
    
    def _init_hex_grid(self) -> None:
        """Initialize the hex grid."""
        # Calculate hex dimensions
        hex_height = hex_grid.RADIUS * 1.732  # sqrt(3)
        spacing_x = hex_grid.RADIUS * 1.5
        spacing_y = hex_height
        
        # Calculate grid dimensions to center it
        grid_pixel_width = (hex_grid.GRID_WIDTH + 0.5) * spacing_x
        grid_pixel_height = (hex_grid.GRID_HEIGHT + 0.5) * spacing_y
        grid_start_x = (display.WINDOW_WIDTH - grid_pixel_width) // 2 + hex_grid.RADIUS
        grid_start_y = (display.WINDOW_HEIGHT - grid_pixel_height) // 2 + hex_grid.RADIUS
        
        # Create hex grid
        self.hexes = [[None for _ in range(hex_grid.GRID_HEIGHT)] 
                     for _ in range(hex_grid.GRID_WIDTH)]
        
        for x in range(hex_grid.GRID_WIDTH):
            for y in range(hex_grid.GRID_HEIGHT):
                # Calculate center position
                center_x = grid_start_x + x * spacing_x
                center_y = grid_start_y + y * spacing_y
                if x % 2:  # Offset odd columns
                    center_y += spacing_y // 2
                
                self.hexes[x][y] = Hex(center_x, center_y, x, y)
    
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
            while running:
                current_time = pygame.time.get_ticks() / 1000.0 - self.start_time
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            self._handle_click(event.pos, current_time)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                # Draw
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
            pos: Mouse position (x, y)
            current_time: Current game time in seconds
        """
        # Find the clicked hex
        clicked_hex = self.pixel_to_hex(*pos)
        if not clicked_hex:
            return
            
        # Log the clicked hex and its neighbors
        print(f"\nClicked hex ({clicked_hex.grid_x},{clicked_hex.grid_y})")
        neighbors = self.get_hex_neighbors(clicked_hex)
        for n in neighbors:
            if n:
                print(f"  Neighbor ({n.grid_x},{n.grid_y})",
                  f"edge={clicked_hex.get_shared_edge_index(n)}")
        
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
        self.screen.fill(display.BACKGROUND_COLOR)
        
        # Collect all non-broken hexes for collision detection
        non_broken_hexes = []
        for row in self.hexes:
            for hex in row:
                if hex.state != HexState.BROKEN:
                    non_broken_hexes.append(hex)
        
        # Draw all hexes
        for row in self.hexes:
            for hex in row:
                if hex.state == HexState.BROKEN:
                    # Pass non-broken hexes for collision detection
                    hex.draw(self.screen, self.font, current_time, non_broken_hexes)
                else:
                    hex.draw(self.screen, self.font, current_time) 