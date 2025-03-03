"""Event types for the game."""
from enum import Enum, auto

import pygame


class EventType(Enum):
    """Custom event types for the game."""
    PLAY_SOUND = pygame.USEREVENT + 1
    GAME_OVER = pygame.USEREVENT + 2
    LEVEL_COMPLETE = pygame.USEREVENT + 3 