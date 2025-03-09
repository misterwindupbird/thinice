# On Thin Ice

A [7-Day Roguelike](https://itch.io/jam/7drl-challenge-2025). 

![Walking on thin ice](<images/thinice.jpg>)


## Installation

Download the game for Mac on [itch.io](https://haikufactory.itch.io/on-thin-ice). If you want to play with the repo, just:

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

4. Run the game using:
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
