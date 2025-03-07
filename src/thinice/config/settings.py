"""Game configuration settings."""
from dataclasses import dataclass
from typing import Tuple
import pygame
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(module)s:%(lineno)d %(levelname)s %(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('game.log', mode='a')
    ]
)

@dataclass
class GameSettings:
    SHIFT_CLICK_ACTION = "enemy"

@dataclass
class DisplayConfig:
    """Display-related configuration."""
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800
    BACKGROUND_COLOR: Tuple[int, int, int] = (20, 20, 30)
    FONT_NAME: str = 'Arial'
    FONT_SIZE: int = 10
    DRAW_OVERLAY: bool = False
    font: pygame.font.Font = None


@dataclass
class HexConfig:
    """Hex grid configuration."""
    RADIUS: int = 40
    GRID_WIDTH: int = 21
    GRID_HEIGHT: int = 15
    LINE_COLOR: Tuple[int, int, int] = (240, 240, 245)
    TEXT_COLOR: Tuple[int, int, int] = (100, 120, 140)
    # ICE_BASE_COLOR: Tuple[int, int, int] = (245, 245, 250)
    MAX_FRAGMENT_SIZE_PERCENT: float = 0.2  # Maximum fragment size as percentage of hex area (15%)
    TILES = {"low": "images/tiles/Desert.png",
             "mid": "images/tiles/SnowWaste.png",
             "high": "images/tiles/IcyConnifer.png",
             "peak": "images/tiles/Connifer.png"}


@dataclass
class CrackConfig:
    """Crack-related configuration."""
    COLOR: Tuple[int, int, int, int] = (0, 150, 190, 255)
    SHADOW_COLOR: Tuple[int, int, int, int] = (0, 100, 130, 255)
    MIN_CRACKS: int = 2
    MAX_CRACKS: int = 4
    MIN_SEGMENTS: int = 3
    MAX_DEVIATION: float = 0.15
    SEGMENT_LENGTH: int = 15
    SECONDARY_CRACK_CHANCE: float = 0.5
    MAX_SECONDARY_CRACKS: int = 5

@dataclass
class WaterConfig:
    """Water-related configuration."""
    BASE_COLOR: Tuple[int, int, int] = (0, 70, 100)
    CRACK_COLOR: Tuple[int, int, int] = (70, 150, 200)

@dataclass
class AnimationConfig:
    """Animation timing configuration."""
    BREAKING_DURATION: float = 0.3
    CRACKING_DURATION: float = 0.4

@dataclass
class LandConfig:
    """Land hex configuration."""
    BASE_COLOR: Tuple[int, int, int] = (34, 139, 34)  # ForestGreen
    COLOR_VARIATION: int = 20  # Variation in green shades

@dataclass
class WorldGenerationConfig:
    SUPERGRID_SIZE = 15
    ALGORITHM = "noise" # noise, box
    SCALE = 5.
    OCTAVES = 4
    PERSISTENCE = 0.5
    LACUNARITY = 2.0
    SEED = 32
    ICE_PERCENT = 75

# Create global instances
game_settings = GameSettings()
display = DisplayConfig()
hex_grid = HexConfig()
crack = CrackConfig()
water = WaterConfig()
animation = AnimationConfig()
land = LandConfig()
worldgen = WorldGenerationConfig()