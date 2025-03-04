"""Game configuration settings."""
from dataclasses import dataclass
from typing import Tuple
import pygame

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
    GRID_WIDTH: int = 10
    GRID_HEIGHT: int = 10
    LINE_COLOR: Tuple[int, int, int] = (240, 240, 245)
    TEXT_COLOR: Tuple[int, int, int] = (100, 120, 140)
    # ICE_BASE_COLOR: Tuple[int, int, int] = (245, 245, 250)
    MAX_FRAGMENT_SIZE_PERCENT: float = 0.20  # Maximum fragment size as percentage of hex area (15%)

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
    # Probability distribution for secondary cracks between a pair of primaries
    # 20% chance of 0, 60% chance of 1, 20% chance of 2
    SECONDARY_CRACK_DISTRIBUTION: Tuple[float, float, float] = (0.2, 0.6, 0.2)

@dataclass
class WaterConfig:
    """Water-related configuration."""
    BASE_COLOR: Tuple[int, int, int] = (0, 70, 100)
    COLOR_VARIATION: int = 15
    RIPPLE_COLOR: Tuple[int, int, int] = (100, 200, 255)
    CRACK_COLOR: Tuple[int, int, int] = (70, 150, 200)

@dataclass
class AnimationConfig:
    """Animation timing configuration."""
    BREAKING_DURATION: float = 0.3
    CRACKING_DURATION: float = 0.4
    ENABLE_PARTICLES: bool = False  # Disable particles by default

@dataclass
class LandConfig:
    """Land hex configuration."""
    BASE_COLOR: Tuple[int, int, int] = (34, 139, 34)  # ForestGreen
    COLOR_VARIATION: int = 20  # Variation in green shades

# Create global instances
display = DisplayConfig()
hex_grid = HexConfig()
crack = CrackConfig()
water = WaterConfig()
animation = AnimationConfig()
land = LandConfig() 