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
    MAX_HEALTH = 3

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
    GRID_HEIGHT: int = 13
    LINE_COLOR: Tuple[int, int, int] = (240, 240, 245)
    TEXT_COLOR: Tuple[int, int, int] = (100, 120, 140)
    # ICE_BASE_COLOR: Tuple[int, int, int] = (245, 245, 250)
    MAX_FRAGMENT_SIZE_PERCENT: float = 0.2  # Maximum fragment size as percentage of hex area (15%)
    TILES = {"low": "images/tiles/Desert.png",
             "mid": "images/tiles/SnowWaste.png",
             "high": "images/tiles/IcyConnifer.png",
             "peak": "images/tiles/LowMountains.png"}


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

game_over_messages = [
    "You collapse into the snow, breath steaming. The last thing you see is not teeth, but eyes—calculating, expectant. They were always waiting for you.",
    "The ice shatters beneath you, but you do not fall. A dozen hungry mouths close in, and you finally understand: these were never wolves.",
    "They circle in silence, watching. Not hunger, not instinct—recognition. You were never meant to leave this place.",
    "As your blood stains the ice, your last thought is not fear, but memory. You've done this before.",
    "The beasts hesitate, just for a moment. Not mercy—doubt. Then the snow swallows your screams.",
    "You fought. You ran. You broke the ice. It was never enough. It was never going to be enough.",
    "The storm howls. The ice groans. The shapes in the dark tilt their heads, listening. You had almost made it.",
    "You die in the snow, the cold stealing your last breath. But the ice does not keep the dead for long.",
    "They move in perfect unison, too perfect. You should have realized sooner. You should have remembered.",
    "Your vision fades as the pack descends. And in the final moment, you see it—the flicker of something familiar beneath their skins."
]

health_restore_messages = [
    "A handprint on a frozen tree. The fingers are too long. Or maybe the ice stretched them.",
    "A campfire, long dead. Around it, small depressions in the snow. Someone sat here, waiting.",
    "A name, carved into bark. The cuts are deep, frantic. The snow has almost hidden it.",
    "A child’s footprints, leading nowhere. The last step is just… gone.",
    "A rope, snapped and stiff with frost. One end disappears into the snow.",
    "A message, scratched into ice: ‘Hold on.’ Below it, claw marks.",
    "A road sign, half-buried. It points nowhere.",
    "You find a single shoe. There is still a foot in it.",
    "The sky shifts. Not clouds—something behind them.",
    "A metal dog tag. The name is yours. The date is wrong.",
    "A scarf, tangled in dead reeds. It smells of smoke and something you cannot place.",
    "A whistle, still looped around a frozen neck. You blow it. Nothing answers.",
    "Beneath the snow, the bones are arranged too carefully.",
    "A trail of prints. Too many joints. Too many toes.",
    "A row of deep scratches in the frozen earth. Not claw marks. Not quite.",
]

# Create global instances
game_settings = GameSettings()
display = DisplayConfig()
hex_grid = HexConfig()
crack = CrackConfig()
water = WaterConfig()
animation = AnimationConfig()
land = LandConfig()
worldgen = WorldGenerationConfig()