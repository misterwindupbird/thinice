print("Initializing pygame...")
pygame.init()

print("Setting display mode...")
pygame.display.set_mode((960, 640), pygame.SCALED)

print("Creating game instance...")
game = Game(enable_watcher=False)

print("Entering game loop...")