"""Main entry point for the ice breaking game."""
from .core.game import Game

def main():
    """Run the game."""
    game = Game(enable_watcher=True)
    game.run()

if __name__ == '__main__':
    main() 