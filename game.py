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
LINE_COLOR = (240, 240, 245)  # Very faint lines, just barely visible against the ice
TEXT_COLOR = (100, 120, 140)  # Subtle bluish text
ICE_BASE_COLOR = (245, 245, 250)  # Almost white with just a tiny hint of blue
CRACK_COLOR = (0, 150, 190, 255)  # Deep cyan-blue for cracks with full opacity
CRACK_SHADOW = (0, 100, 130, 255)  # Darker blue-green for shadows with full opacity
WATER_BASE = (0, 70, 100)  # Deep water color for broken ice
WATER_VARIATION = 15  # Amount of random variation in water color

class HexState(Enum):
    SOLID = auto()    # Uncracked ice
    CRACKING = auto() # Currently animating crack formation
    CRACKED = auto()  # Has cracks but not broken
    BREAKING = auto() # Currently animating break
    BROKEN = auto()   # Completely broken

class GameRestartHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('game.py'):
            print("Game file changed, restarting...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

class Crack:
    def __init__(self, start_point, is_secondary=False):
        self.points = [start_point]
        self.is_secondary = is_secondary
        self.thickness = 1 if is_secondary else random.randint(2, 3)
        self.total_length = 0
    
    def add_point(self, point):
        self.points.append(point)
        # Calculate segment length when adding points
        if len(self.points) > 1:
            prev = self.points[-2]
            length = math.sqrt((point[0] - prev[0])**2 + (point[1] - prev[1])**2)
            self.total_length += length
    
    def draw(self, screen, progress=1.0):
        """Draw the crack directly without animation"""
        if len(self.points) < 2:
            return
            
        # Draw shadow
        for i in range(len(self.points) - 1):
            pygame.draw.line(screen, CRACK_SHADOW,
                           (self.points[i][0] + 1, self.points[i][1] + 1),
                           (self.points[i+1][0] + 1, self.points[i+1][1] + 1),
                           self.thickness + 1)
        
        # Draw crack
        for i in range(len(self.points) - 1):
            pygame.draw.line(screen, CRACK_COLOR,
                           self.points[i], self.points[i+1],
                           self.thickness)

class IceFragment:
    def __init__(self, points, center_x, center_y):
        self.original_points = points
        self.center_x = center_x
        self.center_y = center_y
        self.x_offset = 0
        self.y_offset = 0
        self.rotation = 0
        
        # Calculate center of mass and area
        total_x = sum(x for x, y in points)
        total_y = sum(y for x, y in points)
        self.com_x = total_x / len(points)
        self.com_y = total_y / len(points)
        
        # Calculate area as percentage of hex
        self.area = len(points) / 20  # Rough approximation
        
        # Random movement direction based on position relative to hex center
        dx = self.com_x - center_x
        dy = self.com_y - center_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.1:  # Prevent division by zero for central pieces
            angle = random.uniform(0, 2 * math.pi)
            self.dx = math.cos(angle)
            self.dy = math.sin(angle)
        else:
            # Move outward from center but limit distance
            self.dx = dx / dist
            self.dy = dy / dist
        
        # Smaller pieces move slightly faster but stay within bounds
        max_drift = HEX_RADIUS * 0.15  # Maximum drift distance
        self.max_offset = max_drift * (1 - self.area/2)  # Larger pieces move less
        self.spin = random.uniform(-15, 15) * (1 - self.area/2)  # Less rotation for larger pieces
        
    def update(self, dt, progress):
        # First phase (0-0.3): widen cracks
        # Second phase (0.3-0.7): drift apart
        # Final phase (0.7-1.0): settle into position
        
        if progress < 0.3:
            # Initial crack widening phase
            ease = progress / 0.3
            drift = ease * self.max_offset * 0.3
        elif progress < 0.7:
            # Drifting phase
            ease = (progress - 0.3) / 0.4
            drift = self.max_offset * (0.3 + 0.7 * ease)
        else:
            # Settling phase
            drift = self.max_offset
        
        # Apply easing and constraints
        self.x_offset = self.dx * drift
        self.y_offset = self.dy * drift
        
        # Rotation follows similar pattern
        if progress < 0.7:
            self.rotation = self.spin * progress
        else:
            self.rotation = self.spin  # Keep final rotation
        
    def get_transformed_points(self):
        # Rotate points around center of mass and apply offset
        cos_r = math.cos(math.radians(self.rotation))
        sin_r = math.sin(math.radians(self.rotation))
        transformed = []
        for x, y in self.original_points:
            # Translate to center of mass, rotate, translate back
            rx = (x - self.com_x) * cos_r - (y - self.com_y) * sin_r + self.com_x
            ry = (x - self.com_x) * sin_r + (y - self.com_y) * cos_r + self.com_y
            # Apply movement offset
            rx += self.x_offset
            ry += self.y_offset
            transformed.append((rx, ry))
        return transformed

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
        # Random variation in ice color with clamping
        base_r, base_g, base_b = ICE_BASE_COLOR
        # Very subtle variations
        grey_var = random.randint(-5, 5)  # Base variation in grey
        blue_var = random.randint(0, 3)   # Slight additional blue variation
        self.color = (
            min(255, max(0, base_r + grey_var)),
            min(255, max(0, base_g + grey_var)),
            min(255, max(0, base_b + grey_var + blue_var))
        )
        self.cracks = []
        self.ice_fragments = []  # Store floating ice fragments
        self.breaking_start_time = 0
        self.breaking_duration = 0.6  # Shorter animation duration
        self.cracking_start_time = 0
        self.cracking_duration = 0.4  # Duration for crack animation
        self.cracked_surface = None  # Cache for final cracked state
        self.broken_surface = None   # Cache for final broken state

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


    def draw(self, screen, font, current_time=0):
        if self.state == HexState.BROKEN:
            # Use cached broken state if available
            if self.broken_surface is None:
                # Create the final surface once
                self.broken_surface = pygame.Surface((HEX_RADIUS * 3, HEX_RADIUS * 3), pygame.SRCALPHA)
                surface_center = (HEX_RADIUS * 1.5, HEX_RADIUS * 1.5)
                
                # Draw water with static color
                adjusted_vertices = [(x - self.center[0] + surface_center[0], 
                                   y - self.center[1] + surface_center[1]) for x, y in self.vertices]
                pygame.draw.polygon(self.broken_surface, WATER_BASE, adjusted_vertices)
                
                # Draw fragments in their final positions
                for fragment in self.ice_fragments:
                    points = fragment.get_transformed_points()
                    adjusted_points = [(x - self.center[0] + surface_center[0],
                                     y - self.center[1] + surface_center[1]) for x, y in points]
                    
                    # Draw shadow
                    shadow_points = [(x + 2, y + 2) for x, y in adjusted_points]
                    pygame.draw.polygon(self.broken_surface, CRACK_SHADOW, shadow_points)
                    
                    # Draw ice fragment
                    pygame.draw.polygon(self.broken_surface, self.color, adjusted_points)
                    pygame.draw.polygon(self.broken_surface, (255, 255, 255, 30), adjusted_points, 1)
            
            # Draw the cached surface
            screen.blit(self.broken_surface, 
                       (self.center[0] - HEX_RADIUS * 1.5,
                        self.center[1] - HEX_RADIUS * 1.5))
            
        elif self.state == HexState.BREAKING:
            # Calculate animation progress
            progress = min(1.0, (current_time - self.breaking_start_time) / self.breaking_duration)
            
            # Draw static water base
            pygame.draw.polygon(screen, WATER_BASE, self.vertices)
            
            # Update and draw fragments
            for fragment in self.ice_fragments:
                fragment.update(1/60, progress)
                points = fragment.get_transformed_points()
                
                # Draw shadow and fragment
                shadow_points = [(x + 2, y + 2) for x, y in points]
                pygame.draw.polygon(screen, CRACK_SHADOW, shadow_points)
                pygame.draw.polygon(screen, self.color, points)
                pygame.draw.polygon(screen, (255, 255, 255, 30), points, 1)
            
            # Check if animation is complete
            if progress >= 1.0:
                self.state = HexState.BROKEN
                self.broken_surface = None  # Force recreation of cached surface
                
        elif self.state == HexState.CRACKED or self.state == HexState.CRACKING:
            # Draw base hex
            pygame.draw.polygon(screen, self.color, self.vertices)
            
            # Draw all cracks
            for crack in self.cracks:
                crack.draw(screen)
            
            # Draw outline and coordinates
            pygame.draw.polygon(screen, LINE_COLOR, self.vertices, 1)
            text = font.render(f"({self.grid_x},{self.grid_y})", True, (245, 245, 250))
            text_rect = text.get_rect(center=self.center)
            screen.blit(text, text_rect)
            
            # If cracking animation is complete, transition to cracked state
            if self.state == HexState.CRACKING:
                progress = min(1.0, (current_time - self.cracking_start_time) / self.cracking_duration)
                if progress >= 1.0:
                    self.state = HexState.CRACKED
            
        else:  # SOLID state
            # Draw base hex
            pygame.draw.polygon(screen, self.color, self.vertices)
            
            # Draw outline and coordinates
            pygame.draw.polygon(screen, LINE_COLOR, self.vertices, 1)
            text = font.render(f"({self.grid_x},{self.grid_y})", True, (245, 245, 250))
            text_rect = text.get_rect(center=self.center)
            screen.blit(text, text_rect)

    def _generate_ice_fragments(self):
        """Generate ice fragments based on crack pattern"""
        self.ice_fragments = []
        
        # Create additional cracks to break the ice into more pieces
        self._add_breaking_cracks()
        
        # Create a grid of potential points for better coverage
        points = set()
        points.add(self.center)  # Add center point
        
        # Add vertices and edge midpoints
        for vertex in self.vertices:
            points.add(vertex)
        for edge in self.edge_points:
            points.add(edge)
        
        # Add evenly distributed internal points in a hex pattern
        hex_angles = [i * math.pi / 3 for i in range(6)]  # Six directions
        distances = [HEX_RADIUS * d for d in [0.33, 0.66]]  # Two rings of points
        
        for dist in distances:
            for angle in hex_angles:
                x = self.center[0] + math.cos(angle) * dist
                y = self.center[1] + math.sin(angle) * dist
                points.add((x, y))
                
                # Add some slight random offset points for more natural shapes
                if random.random() < 0.5:
                    offset = HEX_RADIUS * 0.1
                    x_off = x + random.uniform(-offset, offset)
                    y_off = y + random.uniform(-offset, offset)
                    points.add((x_off, y_off))
        
        # Add points from all cracks
        for crack in self.cracks:
            for point in crack.points:
                points.add(point)
        
        # Convert to list and sort by angle from center for triangulation
        points = list(points)
        points.sort(key=lambda p: math.atan2(p[1] - self.center[1], p[0] - self.center[0]))
        
        def is_path_clear(p1, p2):
            for crack in self.cracks:
                for i in range(len(crack.points) - 1):
                    if line_segments_intersect(p1, p2, crack.points[i], crack.points[i + 1]):
                        return False
            return True
        
        # Create initial triangles
        triangles = []
        center_idx = points.index(self.center)
        
        # Create triangles in a more balanced way
        for i in range(len(points)):
            if i != center_idx:
                next_i = (i + 1) % len(points)
                if next_i != center_idx:
                    if is_path_clear(points[i], points[next_i]) and is_path_clear(points[center_idx], points[i]):
                        triangles.append([points[center_idx], points[i], points[next_i]])
        
        # Merge triangles into larger, more regular fragments
        fragments = []
        used = set()
        
        # Helper function to calculate fragment aspect ratio
        def get_aspect_ratio(fragment_points):
            if len(fragment_points) < 3:
                return 1.0
            xs = [x for x, y in fragment_points]
            ys = [y for x, y in fragment_points]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            if height == 0:
                return float('inf')
            return width / height
        
        # Try to merge triangles into roughly equal-sized, regular shapes
        for i, tri1 in enumerate(triangles):
            if i in used:
                continue
            
            fragment = tri1.copy()
            used.add(i)
            
            # Try to merge with other triangles while maintaining good shape
            merged = True
            while merged and len(fragment) < 8:  # Limit size for more fragments
                merged = False
                best_ratio = float('inf')
                best_tri = None
                best_idx = None
                
                for j, tri2 in enumerate(triangles):
                    if j in used:
                        continue
                    
                    # Check if triangles share an edge and aren't separated by a crack
                    shared_points = set(fragment) & set(tri2)
                    if len(shared_points) == 2:
                        p1, p2 = shared_points
                        if is_path_clear(p1, p2):
                            # Calculate what the merged shape would look like
                            test_fragment = fragment.copy()
                            test_fragment.extend([p for p in tri2 if p not in test_fragment])
                            ratio = get_aspect_ratio(test_fragment)
                            
                            # Keep the merge that results in the most regular shape
                            if 0.5 <= ratio <= 2.0 and ratio < best_ratio:
                                best_ratio = ratio
                                best_tri = tri2
                                best_idx = j
                
                if best_tri is not None:
                    fragment.extend([p for p in best_tri if p not in fragment])
                    used.add(best_idx)
                    merged = True
            
            # Only keep fragments that are mostly inside the hex and have good shape
            if self._is_fragment_mostly_inside(fragment) and 0.3 <= get_aspect_ratio(fragment) <= 3.0:
                fragments.append(fragment)
        
        # Create fragment objects
        for points in fragments:
            fragment = IceFragment(points, self.center[0], self.center[1])
            self.ice_fragments.append(fragment)
            
    def _is_fragment_mostly_inside(self, points):
        """Check if at least 75% of points are inside the hex"""
        inside_count = 0
        total_points = len(points)
        
        # Helper function to check if a point is inside the hex
        def point_in_hex(x, y):
            dx = abs(x - self.center[0]) / HEX_RADIUS
            dy = abs(y - self.center[1]) / HEX_RADIUS
            return dx <= 1.0 and dy <= math.sqrt(3)/2 and dy <= math.sqrt(3) * (1 - dx/2)
        
        for point in points:
            if point_in_hex(*point):
                inside_count += 1
                
        return inside_count >= total_points * 0.75
        
    def _add_breaking_cracks(self):
        """Add additional cracks during the breaking process"""
        # Add more cracks for better fragmentation
        num_new_cracks = random.randint(3, 5)  # Increased number of cracks
        
        # First, add cracks between existing crack points
        all_points = []
        for crack in self.cracks:
            all_points.extend(crack.points[1:-1])  # Exclude endpoints
            
        if len(all_points) >= 2:
            for _ in range(num_new_cracks):
                start = random.choice(all_points)
                end = random.choice([p for p in all_points if p != start])
                
                # Create a new jagged crack
                new_crack = Crack(start, is_secondary=True)
                
                # Calculate direction and distance
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Add more intermediate points for jagged appearance
                num_points = random.randint(3, 5)
                for i in range(1, num_points):
                    t = i / num_points
                    # Add random deviation perpendicular to the line
                    perpendicular_x = -dy / dist
                    perpendicular_y = dx / dist
                    deviation = random.uniform(-0.3, 0.3) * dist
                    
                    x = start[0] + dx * t + perpendicular_x * deviation
                    y = start[1] + dy * t + perpendicular_y * deviation
                    new_crack.add_point((x, y))
                
                new_crack.add_point(end)
                self.cracks.append(new_crack)
        
        # Then add some radial cracks from center
        num_radial = random.randint(2, 3)
        for _ in range(num_radial):
            angle = random.uniform(0, 2 * math.pi)
            end_x = self.center[0] + math.cos(angle) * HEX_RADIUS * 0.8
            end_y = self.center[1] + math.sin(angle) * HEX_RADIUS * 0.8
            self.add_straight_crack((end_x, end_y))

    def add_secondary_cracks(self):
        """Add smaller cracks between adjacent primary cracks"""
        if len(self.cracks) < 2:
            return
            
        # Maximum number of secondary cracks allowed
        MAX_SECONDARY_CRACKS = 5
        current_secondary_cracks = sum(1 for crack in self.cracks if crack.is_secondary)
        
        # For each pair of cracks
        for i in range(len(self.cracks)):
            if current_secondary_cracks >= MAX_SECONDARY_CRACKS:
                break
                
            for j in range(i + 1, len(self.cracks)):
                if current_secondary_cracks >= MAX_SECONDARY_CRACKS:
                    break
                    
                crack1 = self.cracks[i]
                crack2 = self.cracks[j]
                
                # Skip if both are secondary cracks
                if crack1.is_secondary and crack2.is_secondary:
                    continue
                
                # Only connect cracks going to adjacent edges
                end1 = crack1.points[-1]
                end2 = crack2.points[-1]
                
                # Find the indices of these endpoints in edge_points
                edge1_idx = -1
                edge2_idx = -1
                for idx, point in enumerate(self.edge_points):
                    if math.dist(point, end1) < 5:
                        edge1_idx = idx
                    if math.dist(point, end2) < 5:
                        edge2_idx = idx
                
                # Check if edges are adjacent (difference of 1 or 5)
                edge_diff = abs(edge1_idx - edge2_idx)
                if edge_diff == 1 or edge_diff == 5:
                    # 50% chance to add a secondary crack (reduced from 95%)
                    if random.random() < 0.5:
                        self.add_connecting_crack(crack1, crack2)
                        current_secondary_cracks += 1

    def add_connecting_crack(self, crack1, crack2):
        """Create a thin, straight crack connecting two existing cracks"""
        # Choose random points along each crack (not at the ends)
        points1 = crack1.points[1:-1]
        points2 = crack2.points[1:-1]
        
        if not points1 or not points2:
            return
            
        start = random.choice(points1)
        end = random.choice(points2)
        
        # Create a thin, straight crack
        crack = Crack(start, is_secondary=True)
        crack.thickness = 1  # Always thin
        
        # Calculate direction and distance
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        total_dist = math.sqrt(dx*dx + dy*dy)
        
        # Fewer segments for straighter appearance
        num_segments = max(2, int(total_dist / 20))  # Increased segment length for straighter lines
        
        # Very small maximum deviation for near-straight appearance
        max_deviation = 0.05  # Significantly reduced from previous value
        
        # Generate intermediate points with minimal deviation
        for i in range(1, num_segments):
            t = i / num_segments
            base_x = start[0] + dx * t
            base_y = start[1] + dy * t
            
            # Minimal perpendicular deviation
            perp_dx = -dy / total_dist
            perp_dy = dx / total_dist
            
            # Small random deviation that reduces near endpoints
            deviation = random.uniform(-max_deviation, max_deviation) * total_dist
            deviation *= math.sin(t * math.pi)  # Reduce deviation at endpoints
            
            point_x = base_x + perp_dx * deviation
            point_y = base_y + perp_dy * deviation
            
            crack.add_point((point_x, point_y))
        
        crack.add_point(end)
        self.cracks.append(crack)
        return crack

    def crack(self, neighbors):
        if self.state != HexState.SOLID:
            return
        
        self.state = HexState.CRACKING
        self.cracking_start_time = pygame.time.get_ticks() / 1000.0
        print(f"\nCracking hex ({self.grid_x}, {self.grid_y})")
        
        # First, connect to all cracked neighbors
        cracked_neighbors = [n for n in neighbors if n.state == HexState.CRACKED]
        for neighbor in cracked_neighbors:
            print(f"  Connecting to neighbor ({neighbor.grid_x}, {neighbor.grid_y})")
            
            # Get the shared edge point
            edge_index = self.get_shared_edge_index(neighbor)
            if edge_index == -1:  # Skip if no shared edge
                continue
                
            shared_point = self.edge_points[edge_index]
            
            # Get the corresponding edge index in the neighbor
            neighbor_edge_index = neighbor.get_shared_edge_index(self)
            if neighbor_edge_index == -1:  # Skip if no shared edge
                continue
            
            # Ensure neighbor has a crack to our shared edge
            if not neighbor.has_crack_to_point(shared_point):
                print(f"    Adding crack to neighbor at edge {neighbor_edge_index}")
                neighbor.add_straight_crack(neighbor.edge_points[neighbor_edge_index])
            
            # Add our connecting crack to the same point
            print(f"    Adding our crack to shared edge {edge_index}")
            self.add_straight_crack(shared_point)

        # Random number of minimum cracks between 2 and 4
        min_cracks = random.randint(2, 4)
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
        
        # Add secondary cracks after all primary cracks are created
        self.add_secondary_cracks()

    def add_straight_crack(self, end_point):
        """Create a jagged crack from center to end point"""
        crack = Crack(self.center)
        
        # Calculate the total distance and direction
        dx = end_point[0] - self.center[0]
        dy = end_point[1] - self.center[1]
        total_dist = math.sqrt(dx*dx + dy*dy)
        
        # Fewer segments for straighter appearance
        num_segments = max(3, int(total_dist / 15))
        
        # Reduced maximum deviation for less jaggedness
        max_deviation = 0.15  # Reduced from 0.25
        
        # Initialize prev_point with center
        prev_point = self.center
        
        # Generate intermediate points with controlled deviation
        for i in range(1, num_segments):
            t = i / num_segments
            base_x = self.center[0] + dx * t
            base_y = self.center[1] + dy * t
            
            # Calculate perpendicular direction
            perp_dx = -dy
            perp_dy = dx
            perp_length = math.sqrt(perp_dx*perp_dx + perp_dy*perp_dy)
            perp_dx /= perp_length
            perp_dy /= perp_length
            
            # Smaller deviation that reduces near endpoints
            deviation = random.uniform(-max_deviation, max_deviation) * total_dist
            deviation *= math.sin(t * math.pi)  # Reduce deviation at endpoints
            
            # Add point with controlled deviation
            point_x = base_x + perp_dx * deviation
            point_y = base_y + perp_dy * deviation
            
            # Add extra points for sharp angles only after first point
            if i > 1 and random.random() < 0.2:  # Reduced probability
                mid_x = (prev_point[0] + point_x) / 2
                mid_y = (prev_point[1] + point_y) / 2
                angle_point = (
                    mid_x + random.uniform(-0.1, 0.1) * total_dist,  # Reduced variation
                    mid_y + random.uniform(-0.1, 0.1) * total_dist
                )
                crack.add_point(angle_point)
            
            crack.add_point((point_x, point_y))
            prev_point = (point_x, point_y)
        
        crack.add_point(end_point)
        self.cracks.append(crack)
        return crack

    def has_crack_to_point(self, point, threshold=5):
        """Check if any crack ends near this point"""
        for crack in self.cracks:
            if math.dist(crack.points[-1], point) < threshold:
                return True
        return False

    def break_ice(self, current_time):
        if self.state == HexState.CRACKED:
            self.state = HexState.BREAKING
            self.breaking_start_time = current_time
            self._generate_ice_fragments()

    def get_shared_edge_index(self, neighbor):
        """Get the index of the edge shared with neighbor"""
        delta = (self.grid_x - neighbor.grid_x, self.grid_y - neighbor.grid_y)
        if self.grid_x % 2 == 0:  # Even row
            match delta:
                case (0, 1): return 0   # Top
                case (-1, 1): return 1  # Top-left
                case (-1, 0): return 2  # Bottom-left
                case (0, -1): return 3  # Bottom
                case (1, 0): return 4   # Bottom-right
                case (1, 1): return 5   # Top-right
        else:  # Odd row
            match delta:
                case (0, 1): return 0   # Top
                case (-1, 0): return 1  # Top-left
                case (-1, -1): return 2 # Bottom-left
                case (0, -1): return 3  # Bottom
                case (1, -1): return 4  # Bottom-right
                case (1, 0): return 5   # Top-right
        return -1  # No shared edge

def line_segments_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

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
        self.start_time = pygame.time.get_ticks() / 1000.0  # Convert to seconds
    
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
        
        # Directions for even and odd rows
        directions = [
            [(0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1)],  # even row
            [(0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]     # odd row
        ][odd_row]
        
        # Add all valid neighbors
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
                current_time = pygame.time.get_ticks() / 1000.0 - self.start_time
                
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
                                    clicked_hex.break_ice(current_time)
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                # Draw
                self.screen.fill(BACKGROUND)
                for row in self.hexes:
                    for hex in row:
                        hex.draw(self.screen, self.font, current_time)
                pygame.display.flip()
                
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            pygame.quit()

if __name__ == '__main__':
    # Create game instance with file watcher disabled by default
    game = Game(enable_watcher=True)
    game.run() 