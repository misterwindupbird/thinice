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

## Notes/Postmortem

I've wanted to participate in the 7DRL for years and suddenly found myself between jobs with no excuse not to do so. 
This is my first game since I was a kid learning to program BASIC on my Commodore 64.

I work mostly with Python, so I wanted to use it for the game. After playing with `libtcod` for a bit I ended up 
using `Pygame` for the game loop. `Pygame` is good for handling the render loop and sprites, but I had to write all the
pathfinding, animation handlers, and Enemy AI myself. I spent probably 40% of my entire time writing and debugging
the ice-cracking animations, and I'm still not thrilled with them.

The terrain generation is a mix of Perlin noise and then some procedural river and lake generation to make the 
individual screens seem more distinct. Playing with this was the most fun part of the project and next
time I'll think I'll budget more time for this and less for animation.

I wanted to make sure I had a fairly solid "slice" by the end of the jam, but that meant a lot of stuff got cut. Biggest regrets in order:
1. Sound would have added a lot, I think.
2. I sketched out two additional enemy types, but ran out of time. One would have broken the ice when it moved, and one would have travelled *under* the ice and popped up through broken hexes.
3. All the land is the same. It might have been interesting to have impassable mountains or forests that break line of sight or beaches you can throw rocks from.

## Credits

I recently went on a trekking trip in Nepal. Gameplay is inspired by the cold and stark landscapes I found there, 
and the many hours I spent playing [Hoplite](https://apps.apple.com/ca/app/hoplite/id782438457) during my downtime. 

* I came across Dice Goblin's [Dwarf-Fortress-inspired hex tiles](https://dicegoblin.blog/dwarf-fortress-inspired-hex-tiles/) late into the week and loved the aesthetic. If I'd found it earlier this might have been the whole game.
* I found the heart icons online and saw they were free to use, but neglected to save the URL. Thank you, whoever made them!
* The font is [Alagard](https://www.dafont.com/alagard.font).
* The other asssets are either AI- or procedurally-generated.
