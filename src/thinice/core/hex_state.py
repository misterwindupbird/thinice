"""Hex state enumeration."""
from enum import Enum, auto

class HexState(Enum):
    """Represents the possible states of a hex tile."""
    
    SOLID = auto()     # Uncracked ice
    CRACKING = auto()  # Currently animating crack formation
    CRACKED = auto()   # Has cracks but not broken
    BREAKING = auto()  # Currently animating break
    BROKEN = auto()    # Completely broken
    
    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return self.name.lower().capitalize() 