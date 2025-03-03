# On Thin Ice

Created for the [2025 7-Day Roguelike Challenge](https://itch.io/jam/7drl-challenge-2025).

![Walking on thin ice](<images/thinice.jpg>)

## Installation

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the package and dependencies:
   ```
   pip install -e .
   ```

## Running the Game

Run the game using:
```
python -m thinice
```

## Development

The game is structured as a Python package in the `src/thinice` directory. The main components are:

- `core/`: Core game classes (Hex, Crack, IceFragment, Game)
- `config/`: Game settings and configuration
- `utils/`: Utility functions and helpers

To run the game with auto-reload during development:
```
python -m thinice
```

## Controls

- Click on a hex to crack it
- Click on a cracked hex to break it
- Press ESC to exit
