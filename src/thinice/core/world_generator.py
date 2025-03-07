"""World generator for creating and populating the game world."""
import random
import logging
from typing import List, Tuple, Any

from .hex_state import HexState
from ..config import settings

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
        self.world_hex_width = settings.hex_grid.GRID_WIDTH * self.supergrid_size
        self.world_hex_height = settings.hex_grid.GRID_HEIGHT * self.supergrid_size
        
        # These will store the state of the entire world
        self.world_hex_states = None
        self.world_hex_colors = None
    
    def generate_world(self):
        """Generate the world with 2-hex border and area-specific random land hexes."""
        logging.info("Generating world...")
        
        # Initialize the world grid
        self._init_world_grid()
        
        # Generate 2-hex thick land border around the entire world
        self._generate_world_border()
        
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
    
    def _generate_world_border(self):
        """Generate a 2-hex thick land border around the entire world."""
        border_width = 2  # Width of the border in hexes
        
        for x in range(self.world_hex_width):
            for y in range(self.world_hex_height):
                # Check if this hex is within the border
                if (x < border_width or x >= self.world_hex_width - border_width or
                    y < border_width or y >= self.world_hex_height - border_width):
                    self.world_hex_states[x][y] = HexState.LAND
                    self.world_hex_colors[x][y] = self._generate_land_color()
    
    def _generate_area(self, grid_x, grid_y):
        """Generate an area with random land hexes based on coordinates.
        
        Args:
            grid_x: X position in the supergrid
            grid_y: Y position in the supergrid
        """
        # Mark area as generated
        self.areas[grid_x][grid_y].generated = True
        
        # Calculate number of random land hexes (grid_x * grid_y)
        num_land_hexes = grid_x * grid_y
        
        # Get the world coordinates for this area
        area_width = settings.hex_grid.GRID_WIDTH
        area_height = settings.hex_grid.GRID_HEIGHT
        world_start_x = grid_x * area_width
        world_start_y = grid_y * area_height
        
        # Add random land hexes in the interior of this area
        self._add_random_land_in_area(world_start_x, world_start_y, area_width, area_height, num_land_hexes)
        
        # Set a random number of enemies
        self.areas[grid_x][grid_y].enemy_count = random.randint(1, 3)
    
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
            self.world_hex_colors[x][y] = self._generate_land_color()
    
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
        area_width = settings.hex_grid.GRID_WIDTH
        area_height = settings.hex_grid.GRID_HEIGHT
        
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
    
    def _generate_land_color(self):
        """Generate a random color for land hexes.
        
        Returns:
            RGB color tuple
        """
        base_r, base_g, base_b = settings.land.BASE_COLOR
        variation = settings.land.COLOR_VARIATION
        
        r = min(255, max(0, base_r + random.randint(-variation, variation)))
        g = min(255, max(0, base_g + random.randint(-variation, variation)))
        b = min(255, max(0, base_b + random.randint(-variation, variation)))
        
        return (r, g, b)
    
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