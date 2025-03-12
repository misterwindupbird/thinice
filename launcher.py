import sys
import os

# Ensure the thinice package is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import and run the game
from thinice.__main__ import main

if __name__ == "__main__":
    main()