from typing import Tuple
import pygame

from ..config import settings


class FloatingText:
    """Animated floating text that moves up and fades out."""

    def __init__(self, text: str, position: Tuple[float, float], color: Tuple[int, int, int] = (255, 50, 50)):
        """Initialize floating text.

        Args:
            text: The text to display
            position: Starting position (x, y)
            color: RGB color of the text (default is red)
        """
        self.text = text
        self.position = list(position)  # Convert to list for easier modification
        self.color = list(color)  # Convert to list to modify alpha
        self.alpha = 255
        self.start_time = pygame.time.get_ticks() / 1000.0
        self.duration = 1.2  # Total animation duration in seconds
        self.is_active = True
        self.font = pygame.font.SysFont(settings.display.FONT_NAME, 24, bold=True)

        # Create a glow surface (slightly larger text in a different color)
        self.glow_color = (color[0], min(255, color[1] + 50), min(255, color[2] + 50), 150)
        self.glow_font = pygame.font.SysFont(settings.display.FONT_NAME, 26, bold=True)

        # Start position slightly above the entity
        self.position[1] -= 30  # Start 30 pixels above the entity

    def update(self, current_time: float) -> None:
        """Update the floating text animation.

        Args:
            current_time: Current game time in seconds
        """
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.is_active = False
            return

        # Calculate progress (0.0 to 1.0)
        progress = elapsed / self.duration

        # Move upward very slightly
        self.position[1] -= 0.3  # Minimal upward movement

        # Fade out (alpha from 255 to 0)
        self.alpha = max(0, int(255 * (1.0 - progress)))

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the floating text on the screen.

        Args:
            screen: Pygame surface to draw on
        """
        if not self.is_active:
            return

        # Draw glow effect first (underneath)
        glow_text = self.glow_font.render(self.text, True, self.glow_color)
        glow_text.set_alpha(self.alpha // 2)  # Glow is more transparent
        glow_rect = glow_text.get_rect(center=(self.position[0], self.position[1]))
        screen.blit(glow_text, glow_rect)

        # Draw main text
        text_surface = self.font.render(self.text, True, self.color)
        text_surface.set_alpha(self.alpha)
        text_rect = text_surface.get_rect(center=(self.position[0], self.position[1]))
        screen.blit(text_surface, text_rect)
