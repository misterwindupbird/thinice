import pygame
import math
import random
import os
import tkinter as tk
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import subprocess

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
TEXT_COLOR = (255, 255, 255)

class GameRestartHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('game.py'):
            print("Game file changed, restarting...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

class Game:
    def __init__(self):
        # Set up file watcher
        self.observer = Observer()
        self.observer.schedule(GameRestartHandler(), path='.', recursive=False)
        self.observer.start()

        # Try to force the window to open on the left monitor
        pygame.display.set_mode((1,1))  # Create dummy window
        pygame.display.quit()
        pygame.display.init()
        
        flags = pygame.NOFRAME | pygame.SHOWN
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        pygame.display.set_caption("Hex Grid")
        
        # Try to move window after creation
        time.sleep(0.1)  # Give the window time to appear
        window = pygame.display.get_wm_info()['window']
        
        # Try to force position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "-2820,100"
        pygame.display.flip()
        
        self.font = pygame.font.SysFont('Arial', 12)
        
        # Initialize hex data
        self.hex_counts = [[0 for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
        self.hex_colors = [[self.random_blue() for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
        
    def random_blue(self):
        """Generate a random shade of blue"""
        return (0, 0, random.randint(100, 255))
    
    def get_hex_points(self, center_x, center_y):
        """Calculate the six points of a hexagon"""
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            x = center_x + HEX_RADIUS * math.cos(angle)
            y = center_y + HEX_RADIUS * math.sin(angle)
            points.append((int(x), int(y)))
        return points
    
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
    
    def draw_hex_grid(self):
        """Draw the entire hex grid"""
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                # Calculate center of hex
                center_x = GRID_START_X + x * SPACING_X
                center_y = GRID_START_Y + y * SPACING_Y
                if x % 2:
                    center_y += SPACING_Y // 2
                
                # Get hex points
                points = self.get_hex_points(center_x, center_y)
                
                # Draw filled hex
                pygame.draw.polygon(self.screen, self.hex_colors[x][y], points)
                # Draw outline
                pygame.draw.polygon(self.screen, LINE_COLOR, points, 1)
                
                # Draw coordinates and count
                count = self.hex_counts[x][y]
                text = self.font.render(f"({x},{y})\n{count}", True, TEXT_COLOR)
                text_rect = text.get_rect(center=(center_x, center_y))
                self.screen.blit(text, text_rect)
    
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
                            hex_pos = self.pixel_to_hex(*mouse_pos)
                            if hex_pos:
                                x, y = hex_pos
                                self.hex_counts[x][y] += 1
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                            # Restart on Ctrl+R
                            python = sys.executable
                            os.execl(python, python, *sys.argv)
                
                # Draw
                self.screen.fill(BACKGROUND)
                self.draw_hex_grid()
                pygame.display.flip()
        finally:
            self.observer.stop()
            self.observer.join()
            pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.run() 