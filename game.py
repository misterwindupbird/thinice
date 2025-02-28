from typing import List

import pygame
import math
import random
import os
import tkinter as tk
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
from enum import Enum, auto

# Use tkinter to get screen info
root = tk.Tk()
left_monitor_x = root.winfo_screenwidth() * -1  # Get width of monitor
root.destroy()

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
HEX_RADIUS = 40
GRID_WIDTH = 10
GRID_HEIGHT = 10

# Position window on left monitor 
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{left_monitor_x + 1500},100"

# Calculate hex dimensions
HEX_HEIGHT = HEX_RADIUS * math.sqrt(3)
SPACING_X = HEX_RADIUS * 3/2
SPACING_Y = HEX_HEIGHT

# Calculate grid dimensions to center it
GRID_PIXEL_WIDTH = (GRID_WIDTH + 0.5) * SPACING_X  # Account for the offset columns
GRID_PIXEL_HEIGHT = (GRID_HEIGHT + 0.5) * SPACING_Y
GRID_START_X = (WINDOW_WIDTH - GRID_PIXEL_WIDTH) // 2 + HEX_RADIUS  # Add radius to shift right
GRID_START_Y = (WINDOW_HEIGHT - GRID_PIXEL_HEIGHT) // 2 + HEX_RADIUS  # Add radius to shift down

# Colors
BACKGROUND = (20, 20, 30)
LINE_COLOR = (30, 30, 45)  # Just slightly lighter than background
TEXT_COLOR = (240, 40, 55)  # Very subtle text color, just barely visible
CRACKED_COLOR = (250, 250, 260)  # Grey for cracked ice
BROKEN_COLOR = (20, 20, 30)   # Same as background for broken ice
CRACK_COLOR = (255, 255, 255)  # White cracks

class HexState(Enum):
    SOLID = auto()    # Uncracked ice
    CRACKED = auto()  # Has cracks but not broken
    BROKEN = auto()   # Completely broken

class GameRestartHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('game.py'):
            print("Game file changed, restarting...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

class Crack:
    def __init__(self, start_point):
        self.points = [start_point]
    
    def add_point(self, point):
        self.points.append(point)
    
    def get_nearest_point(self, target_point):
        nearest = min(self.points, key=lambda p: math.dist(p, target_point))
        return nearest, math.dist(nearest, target_point)
    
    def draw(self, screen):
        for i in range(len(self.points) - 1):
            pygame.draw.line(screen, CRACK_COLOR, self.points[i], self.points[i+1], 2)

class Hex:
    def __init__(self, x, y, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.center = (x, y)

        # Ordered for a flat-top hex
        self.vertices = [(self.center[0] + HEX_RADIUS * np.cos(a), self.center[1] - HEX_RADIUS * np.sin(a))
                         for a in np.radians([120, 60, 0, 300, 240, 180])]
        # Compute edge centers by averaging adjacent vertices
        self.edge_points = [((self.vertices[i][0] + self.vertices[(i + 1) % 6][0]) / 2,
                             (self.vertices[i][1] + self.vertices[(i + 1) % 6][1]) / 2)
                            for i in range(6)]

        self.state = HexState.SOLID
        self.color = (0, 0, random.randint(100, 255))
        self.cracks = []

    def pixel_to_hex(self, px, py):
        """Convert pixel coordinates to hex grid coordinates"""
        # Offset coordinates from grid start position
        px -= GRID_START_X
        py -= GRID_START_Y
        
        # Convert to axial coordinates
        q = (2.0/3 * px) / HEX_RADIUS
        r = (-1.0/3 * px + math.sqrt(3)/3 * py) / HEX_RADIUS
        
        # Convert to cube coordinates for rounding
        x = q
        z = r
        y = -x - z
        
        # Round cube coordinates
        rx = round(x)
        ry = round(y)
        rz = round(z)
        
        # Fix rounding errors
        x_diff = abs(rx - x)
        y_diff = abs(ry - y)
        z_diff = abs(rz - z)
        
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
            
        # Convert back to offset coordinates
        col = rx
        row = rz + (rx - (rx & 1)) // 2
        
        if 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT:
            return col, row
        return None
    
    def get_neighbors(self, x, y):
        """Get coordinates of adjacent hexes"""
        neighbors = []
        odd_row = x % 2
        
        # All possible neighbor offsets for even and odd rows
        directions = [
            [(0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1)],  # even row
            [(0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]     # odd row
        ][odd_row]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                neighbors.append((nx, ny))
        return neighbors


    def draw(self, screen, font):
        # Draw hex
        if self.state == HexState.BROKEN:
            color = BACKGROUND
        else:
            color = self.color
        pygame.draw.polygon(screen, color, self.vertices)
        
        # Draw cracks
        if self.state == HexState.CRACKED:
            for crack in self.cracks:
                crack.draw(screen)
        
        # Draw outline and coordinates for unbroken hexes
        if self.state != HexState.BROKEN:
            pygame.draw.polygon(screen, LINE_COLOR, self.vertices, 1)
            text = font.render(f"({self.grid_x},{self.grid_y})", True, TEXT_COLOR)
            text_rect = text.get_rect(center=self.center)
            screen.blit(text, text_rect)

    def crack(self, neighbors):
        if self.state != HexState.SOLID:
            return
        
        self.state = HexState.CRACKED
        print(f"\nCracking hex ({self.grid_x}, {self.grid_y})")
        
        # First, connect to all cracked neighbors
        cracked_neighbors = [n for n in neighbors if n.state == HexState.CRACKED]
        for neighbor in cracked_neighbors:
            print(f"  Connecting to neighbor ({neighbor.grid_x}, {neighbor.grid_y})")
            
            # Get the shared edge point
            edge_index = self.get_shared_edge_index(neighbor)
            shared_point = self.edge_points[edge_index]
            
            # Get the corresponding edge index in the neighbor
            neighbor_edge_index = neighbor.get_shared_edge_index(self)
            
            # Ensure neighbor has a crack to our shared edge
            if not neighbor.has_crack_to_point(shared_point):
                print(f"    Adding crack to neighbor at edge {neighbor_edge_index}")
                neighbor.add_straight_crack(neighbor.edge_points[neighbor_edge_index])
            
            # Add our connecting crack to the same point
            print(f"    Adding our crack to shared  {edge_index}")
            self.add_straight_crack(shared_point)

        min_cracks = np.random.randint(1, 4)
        while len(self.cracks) < min_cracks:
            # Pick an edge we haven't used yet
            used_edges = {crack.points[-1] for crack in self.cracks}
            available_edges = [i for i, point in enumerate(self.edge_points) 
                             if point not in used_edges]
            
            if not available_edges:  # If all edges are used, break
                break
                
            edge_index = random.choice(available_edges)
            print(f"  Adding random crack to edge {edge_index}")
            self.add_straight_crack(self.edge_points[edge_index])

    def add_straight_crack(self, end_point):
        """Create a straight crack from center to end point"""
        crack = Crack(self.center)
        crack.add_point(end_point)
        self.cracks.append(crack)
        return crack

    def has_crack_to_point(self, point, threshold=5):
        """Check if any crack ends near this point"""
        for crack in self.cracks:
            if math.dist(crack.points[-1], point) < threshold:
                return True
        return False

    def break_ice(self):
        if self.state == HexState.CRACKED:
            self.state = HexState.BROKEN

    def get_shared_edge_index(self, neighbor):
        """Get the index of the edge shared with neighbor"""
        delta = (self.grid_x - neighbor.grid_x, self.grid_y - neighbor.grid_y)
        if self.grid_x % 2 == 0:
            match delta:
                case (0, 1):
                    return 0
                case (-1, 1):
                    return 1
                case (-1, 0):
                    return 2
                case (0, -1):
                    return 3
                case (1, 0):
                    return 4
                case (1, 1):
                    return 5

        match delta:
            case (0, 1):
                return 0
            case (-1, 0):
                return 1
            case (-1, -1):
                return 2
            case (0, -1):
                return 3
            case (1, -1):
                return 4
            case (1, 0):
                return 5


class Game:
    def __init__(self, enable_watcher=False):
        # Set up file watcher only if requested
        self.observer = None
        if enable_watcher:
            self.observer = Observer()
            self.observer.schedule(GameRestartHandler(), path='.', recursive=False)
            self.observer.start()

        # Set up display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hex Grid")
        self.font = pygame.font.SysFont('Arial', 10)
        
        # Initialize hex grid
        self.hexes = [[None for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
        self.init_hex_grid()
    
    def init_hex_grid(self):
        """Create all hex objects with proper positions"""
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                # Calculate center position
                center_x = GRID_START_X + x * SPACING_X
                center_y = GRID_START_Y + y * SPACING_Y
                if x % 2:
                    center_y += SPACING_Y // 2
                
                self.hexes[x][y] = Hex(center_x, center_y, x, y)
    
    def get_hex_neighbors(self, hex: Hex) -> List[Hex]:
        """Get list of neighboring Hex objects"""
        neighbors = []
        odd_row = hex.grid_x % 2
        
        directions = [
            [(0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1)],  # even row
            [(0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]     # odd row
        ][odd_row]
        
        for dx, dy in directions:
            nx, ny = hex.grid_x + dx, hex.grid_y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                neighbors.append(self.hexes[nx][ny])
        return neighbors
    
    def pixel_to_hex(self, px: float, py: float) -> Hex:
        """Convert pixel coordinates to hex object"""
        # Offset coordinates from grid start position
        px -= GRID_START_X
        py -= GRID_START_Y
        
        # Convert to axial coordinates
        q = (2.0/3 * px) / HEX_RADIUS
        r = (-1.0/3 * px + math.sqrt(3)/3 * py) / HEX_RADIUS
        
        # Convert to cube coordinates for rounding
        x = q
        z = r
        y = -x - z
        
        # Round cube coordinates
        rx = round(x)
        ry = round(y)
        rz = round(z)
        
        # Fix rounding errors
        x_diff = abs(rx - x)
        y_diff = abs(ry - y)
        z_diff = abs(rz - z)
        
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
            
        # Convert back to offset coordinates
        col = rx
        row = rz + (rx - (rx & 1)) // 2
        
        if 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT:
            return self.hexes[col][row]
        return None
    
    def run(self):
        try:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            mouse_pos = pygame.mouse.get_pos()
                            clicked_hex = self.pixel_to_hex(*mouse_pos)
                            if clicked_hex:
                                # Debug info
                                print(f"\nClicked hex ({clicked_hex.grid_x},{clicked_hex.grid_y})")
                                neighbors = self.get_hex_neighbors(clicked_hex)
                                for n in neighbors:
                                    marker = "*" if n.state == HexState.CRACKED else " "
                                    print(f"  Neighbor ({n.grid_x},{n.grid_y}){marker}\tedge={clicked_hex.get_shared_edge_index(n)}")
                                
                                if clicked_hex.state == HexState.SOLID:
                                    clicked_hex.crack(neighbors)
                                elif clicked_hex.state == HexState.CRACKED:
                                    clicked_hex.break_ice()
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                # Draw
                self.screen.fill(BACKGROUND)
                for row in self.hexes:
                    for hex in row:
                        hex.draw(self.screen, self.font)
                pygame.display.flip()
                
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            pygame.quit()

if __name__ == '__main__':
    # Create game instance with file watcher disabled by default
    game = Game(enable_watcher=False)
    game.run() 