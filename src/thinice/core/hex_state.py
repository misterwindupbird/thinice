"""Hex state enumeration."""
from enum import Enum, auto

class HexState(Enum):
    """Represents the possible states of a hex tile."""
    
    SOLID = auto()     # Uncracked ice
    CRACKING = auto()  # In the process of cracking (transition animation)
    CRACKED = auto()   # Has cracks but not broken
    BREAKING = auto()  # In the process of breaking (transition animation)
    BROKEN = auto()    # Completely broken
    LAND = auto()      # Represents land hexes
    
    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return self.name.lower().capitalize() 