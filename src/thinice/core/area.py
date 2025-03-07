"""Area class for managing individual areas in the supergrid."""
from typing import List, Dict, Any, Optional, Tuple
import pygame
import logging

from .hex import Hex
from .entity import Entity, Player, Wolf
from .floating_text import FloatingText
from .hex_state import HexState


class Area:
    """Represents a single 21x15 area within the supergrid world."""
    
    def __init__(self, supergrid_x: int, supergrid_y: int):
        """Initialize an area at the specified supergrid coordinates.
        
        Args:
            supergrid_x: X-coordinate in the supergrid
            supergrid_y: Y-coordinate in the supergrid
        """
        self.supergrid_x = supergrid_x
        self.supergrid_y = supergrid_y
        
        # Core area data
        self.hexes: List[List[Hex]] = []
        self.player: Optional[Player] = None
        self.enemies: List[Wolf] = []
        self.floating_texts: List[FloatingText] = []
        
        # Sprites
        self.all_sprites: Optional[pygame.sprite.Group] = None
        self.player_sprite: Optional[pygame.sprite.GroupSingle] = None
        self.enemy_sprites: Optional[pygame.sprite.Group] = None
        
        # Effect state
        self.has_pending_hex_effects = False
        self.pending_hex_effects = []
        self.hex_effect_start_time = 0
        self.hex_effect_delay = 0
        self.center_hex_for_effects = None

        # Track if this area has been initialized
        self.is_initialized = False
        
    def initialize_sprites(self, animation_manager):
        """Initialize the sprite groups for this area.
        
        Args:
            animation_manager: The animation manager to use
        """
        self.all_sprites = pygame.sprite.Group()
        self.player_sprite = pygame.sprite.GroupSingle()
        self.enemy_sprites = pygame.sprite.Group()
        
        # Add existing entities to sprite groups
        if self.player:
            self.player.animation_manager = animation_manager
            self.player_sprite.add(self.player)
            self.all_sprites.add(self.player)
        
        for enemy in self.enemies:
            enemy.animation_manager = animation_manager
            self.enemy_sprites.add(enemy)
            self.all_sprites.add(enemy)
            
        # Update animation manager references in hexes
        for row in self.hexes:
            for hex in row:
                if hex:
                    hex.animation_manager = animation_manager
                    
        # Update floating texts
        for text in self.floating_texts:
            text.animation_manager = animation_manager
            
        self.is_initialized = True
            
    def save_state(self):
        """Save the current state of this area when it becomes inactive."""
        # Clear sprite groups but keep the entities
        if self.all_sprites:
            self.all_sprites.empty()
        if self.player_sprite:
            self.player_sprite.empty()
        if self.enemy_sprites:
            self.enemy_sprites.empty()
            
        # Keep hex and entity data intact
        # They will be reinitialized when this area becomes active again
        
    def __str__(self) -> str:
        return f"Area({self.supergrid_x}, {self.supergrid_y})" 