"""World generator for creating and populating the game world."""
import random
import logging
from typing import Any
import noise
import numpy as np

from .hex_state import HexState
from ..config.settings import hex_grid, worldgen, land


class WorldGenerator:
    """Generates the world with a border and area-specific land hexes."""
    
    def __init__(self, game: Any):
        """Initialize the world generator.
        
        Args:
            game: The Game instance to generate world for
        """
        self.game = game
        self.supergrid_size = game.supergrid_size
        self.areas = game.areas
        
        # Calculate the dimensions of the entire world in hexes
        self.world_hex_width = hex_grid.GRID_WIDTH * self.supergrid_size
        self.world_hex_height = hex_grid.GRID_HEIGHT * self.supergrid_size
        
        # These will store the state of the entire world
        self.world_hex_states = None
        self.world_hex_colors = None
        self.world_hex_heightmap = None
    
    def generate_world(self):
        """Generate the world with 2-hex border and area-specific random land hexes."""
        logging.info("Generating world...")
        
        # Initialize the world grid
        self._init_world_grid()

        match worldgen.ALGORITHM:
            case "box":
                # Generate 2-hex thick land border around the entire world
                self._generate_world_border()

            case "noise":
                self._generate_world_from_noise()

        # Process each area
        for x in range(self.supergrid_size):
            for y in range(self.supergrid_size):
                # Generate random land hexes based on area coordinates
                self._generate_area(x, y)

        # Fill remaining hexes with ice
        self._fill_world_with_ice()
        
        logging.info("World generation complete!")
    
    def _init_world_grid(self):
        """Initialize the entire world grid."""
        # Create the world grid arrays
        self.world_hex_states = [[None for _ in range(self.world_hex_height)] 
                                for _ in range(self.world_hex_width)]
        self.world_hex_colors = [[None for _ in range(self.world_hex_height)] 
                                for _ in range(self.world_hex_width)]

    def _generate_world_from_noise(self):

        # Generate a heightmap for the entire world
        self.world_hex_heightmap = np.zeros((self.world_hex_width, self.world_hex_height))

        for x in range(self.world_hex_width):
            for y in range(self.world_hex_height):
                nx, ny = x / worldgen.SCALE, y / worldgen.SCALE  # Normalize coordinates for noise
                self.world_hex_heightmap[x, y] = noise.pnoise2(
                    nx, ny,
                    octaves=worldgen.OCTAVES,
                    persistence=worldgen.PERSISTENCE,
                    lacunarity=worldgen.LACUNARITY,
                    repeatx=self.world_hex_width,  # Ensure seamless tiling
                    repeaty=self.world_hex_height,
                    base=worldgen.SEED
               )

        # Normalize height values to 0-1
        min_val, max_val = self.world_hex_heightmap.min(), self.world_hex_heightmap.max()
        self.world_hex_heightmap = (self.world_hex_heightmap - min_val) / (max_val - min_val)

        # Determine land vs. ice threshold (75% of hexes should be ice)
        threshold = np.percentile(self.world_hex_heightmap, worldgen.ICE_PERCENT)
        self.world_hex_heightmap -= threshold

        # add some rivers
        self._generate_world_river([(-1, 7), (0, 7), (1, 7), (2, 7)])
        self._generate_world_river([(7, -1), (7, 0), (7, 1), (7, 2), (7, 3)])

        # add craters for more interesting terrain with open areas in the middle
        for x in range(worldgen.SUPERGRID_SIZE):
            for y in range(worldgen.SUPERGRID_SIZE):
                self._add_crater_to_area(x * hex_grid.GRID_WIDTH, y * hex_grid.GRID_HEIGHT)

        for x in range(self.world_hex_width):
            for y in range(self.world_hex_height):
                if self.world_hex_heightmap[x][y] > 0:
                    self.world_hex_states[x][y] = HexState.LAND
                    self.world_hex_colors[x][y] = (0, 0, 0)

    def _generate_world_river(self, start_path):
        """Determine a continuous river path across multiple Areas."""

        path = start_path

        while True:  # Let the river traverse ~6 Areas
            next_area = self._get_next_river_area(path)
            path.append(next_area)
            if next_area[0] < 0 or next_area[0] >= worldgen.SUPERGRID_SIZE or next_area[1] < 0 or next_area[1] >= worldgen.SUPERGRID_SIZE:
                break

        logging.debug(f"Generated river path: {path}")

        for i in range(1, len(path) - 2):
            self._generate_river_for_area(path[i], path[i-1], path[i + 1])
        return path

    def _get_next_river_area(self, path):
        """Choose a neighboring Area for the river to flow into."""

        current_head = path[-1]
        possible_moves = [(current_head[0]-1, current_head[1]),
                          (current_head[0]+1, current_head[1]),
                          (current_head[0], current_head[1]+1),
                          (current_head[0], current_head[1]-1)]

        # don't double back
        possible_moves.remove(path[-2])
        move = random.choice(possible_moves)
        return move

    def _generate_river_for_area(self, area, prev_area, next_area):
        """Create a river segment in an Area that connects from prev_area to next_area."""

        if prev_area is None or next_area is None:
            return  # Skip if we don't have both entry & exit points

        area_x, area_y = area
        area_width = hex_grid.GRID_WIDTH
        area_height = hex_grid.GRID_HEIGHT

        # **Compute entry and exit points**
        entry_x, entry_y = self._get_area_center(prev_area)
        exit_x, exit_y = self._get_area_center(next_area)

        # **Raise surrounding terrain** in the entire area before carving the river
        for x in range(area_x * area_width, (area_x + 1) * area_width):
            for y in range(area_y * area_height, (area_y + 1) * area_height):
                self.world_hex_heightmap[x, y] += 0.2  # Elevate land in the river area

        # **Carve a defined river path from entry to exit**
        river_width = 4  # Fixed width at entry and exit

        # Use Bresenham’s line algorithm to generate the core river path
        river_path = self._bresenham_line(entry_x, entry_y, exit_x, exit_y)

        # **Carve out the river, adding some width variation**
        for x, y in river_path:
            width_variation = random.randint(-1, 1)  # Make it a bit wider or narrower
            for dx in range(-river_width // 2 + width_variation, river_width // 2 + width_variation):
                if 0 <= x + dx < self.world_hex_width:
                    self.world_hex_heightmap[x + dx, y] = -1  # Lower height for water

    def _get_area_center(self, area):
        """Returns the center hex coordinates for a given Area."""
        if area is None:
            return None
        area_x, area_y = area
        center_x = (area_x * hex_grid.GRID_WIDTH) + (hex_grid.GRID_WIDTH // 2)
        center_y = (area_y * hex_grid.GRID_HEIGHT) + (hex_grid.GRID_HEIGHT // 2)
        return center_x, center_y

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to generate a smooth path from (x0, y0) to (x1, y1)."""
        path = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            path.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        path.append((x1, y1))  # Ensure we include the final point
        return path

    def _generate_area(self, grid_x, grid_y):
        """Generate an area with random land hexes based on coordinates.
        
        Args:
            grid_x: X position in the supergrid
            grid_y: Y position in the supergrid
        """
        # Mark area as generated
        self.areas[grid_x][grid_y].generated = True

    def _add_crater_to_area(self, start_x, start_y):

        crater_center_x = start_x + hex_grid.GRID_WIDTH // 2
        crater_center_y = start_y + hex_grid.GRID_HEIGHT // 2 - 1
        crater_radius = random.randint(4, 5)  # 8-10 hexes in diameter

        for x in range(start_x, start_x + hex_grid.GRID_WIDTH):
            for y in range(start_y, start_y + hex_grid.GRID_HEIGHT):
                # Compute distance from the crater center
                dx, dy = x - crater_center_x, y - crater_center_y
                dist = (dx ** 2 + dy ** 2) ** 0.5  # Approximate circular distance

                if dist < crater_radius:
                    # **Lower height inside the crater**
                    self.world_hex_heightmap[x, y] -= 0.4 * (1 - (dist / crater_radius))  # Smooth depression

                elif dist < crater_radius + 2:
                    # **Raise the terrain to form a crater rim**
                    self.world_hex_heightmap[x, y] += 0.4 * ((dist - crater_radius) / 2)  # Raised outer ridge


    def _add_random_land_in_area(self, start_x, start_y, width, height, count):
        """Add random land hexes in a specific area region.
        
        Args:
            start_x: World X coordinate of area's top-left
            start_y: World Y coordinate of area's top-left
            width: Width of the area in hexes
            height: Height of the area in hexes
            count: Number of land hexes to place
        """
        # Get all valid positions in this area (not already assigned)
        valid_positions = []
        for x in range(start_x, start_x + width):
            for y in range(start_y, start_y + height):
                if self.world_hex_states[x][y] is None:
                    valid_positions.append((x, y))
        
        # Place land hexes at random positions
        for _ in range(min(count, len(valid_positions))):
            if not valid_positions:
                break
                
            # Choose a random position
            idx = random.randint(0, len(valid_positions) - 1)
            x, y = valid_positions.pop(idx)
            
            # Set to LAND
            self.world_hex_states[x][y] = HexState.LAND
            self.world_hex_colors[x][y] = (0, 255, 0)
    
    def _fill_world_with_ice(self):
        """Fill all remaining hexes with ice, then divide into areas."""
        # Track hexes adjacent to LAND
        hexes_adjacent_to_land = set()
        
        # First identify all hexes adjacent to LAND
        for x in range(self.world_hex_width):
            for y in range(self.world_hex_height):
                if self.world_hex_states[x][y] == HexState.LAND:
                    # Check all neighbors
                    neighbors = self._get_hex_neighbors(x, y)
                    for nx, ny in neighbors:
                        if (0 <= nx < self.world_hex_width and 
                            0 <= ny < self.world_hex_height and
                            self.world_hex_states[nx][ny] is None):
                            hexes_adjacent_to_land.add((nx, ny))
        
        # Now fill in all remaining hexes
        for x in range(self.world_hex_width):
            for y in range(self.world_hex_height):
                if self.world_hex_states[x][y] is None:
                    # Set state to SOLID
                    self.world_hex_states[x][y] = HexState.SOLID
                    
                    # Choose color: pure white if adjacent to LAND, blue-grey gradient otherwise
                    if (x, y) in hexes_adjacent_to_land:
                        self.world_hex_colors[x][y] = (255, 255, 255)  # Pure white
                    else:
                        self.world_hex_colors[x][y] = self._generate_ice_color()
        
        # Now divide the world into areas
        self._divide_into_areas()
    
    def _divide_into_areas(self):
        """Divide the world grid into individual areas."""
        area_width = hex_grid.GRID_WIDTH
        area_height = hex_grid.GRID_HEIGHT
        
        for grid_x in range(self.supergrid_size):
            for grid_y in range(self.supergrid_size):
                # Get the area object
                area = self.areas[grid_x][grid_y]
                
                # Calculate the world position of this area's top-left corner
                world_start_x = grid_x * area_width
                world_start_y = grid_y * area_height
                
                # Initialize area hex states and colors if needed
                if area.hex_states is None:
                    area.hex_states = [[None for _ in range(area_height)] 
                                       for _ in range(area_width)]
                    area.hex_colors = [[None for _ in range(area_height)] 
                                       for _ in range(area_width)]
                
                # Copy the appropriate section of the world grid to this area
                for local_x in range(area_width):
                    for local_y in range(area_height):
                        world_x = world_start_x + local_x
                        world_y = world_start_y + local_y
                        
                        # Copy state and color
                        area.hex_states[local_x][local_y] = self.world_hex_states[world_x][world_y]
                        area.hex_colors[local_x][local_y] = self.world_hex_colors[world_x][world_y]
                        area.hex_heightmap[local_x][local_y] = self.world_hex_heightmap[world_x][world_y]
    
    def _get_hex_neighbors(self, x, y):
        """Get the coordinates of neighboring hexes.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List of (x,y) coordinate tuples for neighboring hexes
        """
        # Account for hex grid offset (odd-q system)
        odd_row = x % 2
        
        # Directions for even and odd rows
        if odd_row == 0:  # even row
            directions = [(0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1)]
        else:  # odd row
            directions = [(0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]
        
        # Calculate neighbor coordinates
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.world_hex_width and 0 <= ny < self.world_hex_height:
                neighbors.append((nx, ny))
        
        return neighbors
    
    # def _generate_land_color(self):
    #     """Generate a random color for land hexes.
    #
    #     Returns:
    #         RGB color tuple
    #     """
    #     base_r, base_g, base_b = land.BASE_COLOR
    #     variation = land.COLOR_VARIATION
    #
    #     r = min(255, max(0, base_r + random.randint(-variation, variation)))
    #     g = min(255, max(0, base_g + random.randint(-variation, variation)))
    #     b = min(255, max(0, base_b + random.randint(-variation, variation)))
    #
    #     return (r, g, b)
    
    def _generate_ice_color(self):
        """Generate a color for ice hexes (blue-grey gradient).
        
        Returns:
            RGB color tuple
        """
        # Random value between 0.0 and 1.0 for color position in gradient
        color_position = random.random()
        
        # White (255,255,255) to blue-grey (220,240,255)
        ice_r = int(255 - (color_position * 35))  # Reduce red more (255 to 220)
        ice_g = int(255 - (color_position * 15))  # Keep green higher (255 to 240)
        ice_b = 255  # Keep blue at maximum
        
        # Small variation
        tiny_variation = 2
        ice_r = min(255, max(215, ice_r + random.randint(-tiny_variation, tiny_variation)))
        ice_g = min(255, max(235, ice_g + random.randint(-tiny_variation, tiny_variation)))
        ice_b = 255  # No variation in blue
        
        return (ice_r, ice_g, ice_b) 