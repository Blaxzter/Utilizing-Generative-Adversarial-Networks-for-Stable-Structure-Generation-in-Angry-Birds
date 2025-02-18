import matplotlib.pyplot as plt

from game_management.GameManager import GameManager
from level import Constants
from level.LevelElement import LevelElement
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from util.Config import Config


def add_block_to_level(level_idx=0, block_x=0, block_y=0):
    # 1. Load and visualize original level
    test_environment = TestEnvironment()
    test_level = test_environment.get_level(level_idx)
    visualizer = LevelVisualizer(line_size=2)

    # Create figure with two subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot original level
    visualizer.visualize_level_img(test_level, ax=ax1)
    ax1.set_title("Original Level Rasterized")    
    
    # Plot original level
    visualizer.create_img_of_level(test_level, use_grid=True, add_dots=False, ax=ax3)
    ax3.set_title("Original Level")
    
    # Add a red rectangle to visualize where we'll add the block
    BLOCK_NAME = 'RectTiny'
    ROTATED = True
    block_id = list(Constants.block_names.values()).index(BLOCK_NAME) + 1 + (1 if ROTATED else 0)
    (block_width, block_height) = Constants.block_sizes[block_id]  # Size of SquareHole block
    rect2 = plt.Rectangle((block_x, block_y), block_width / Constants.resolution, block_height / Constants.resolution, fill=False, color='red', linewidth=2)
    ax1.add_patch(rect2)

    rect3 = plt.Rectangle((block_x * Constants.resolution, block_y * Constants.resolution), block_width, block_height, fill=False, color='red', linewidth=2)
    ax3.add_patch(rect3)

    # 2. Add new block to level
    # Convert visualization coordinates to game coordinates
    game_x = block_x * Constants.resolution + block_width / 2  # Using Constants.resolution
    game_y = block_y * Constants.resolution + block_height / 2 # Using Constants.resolution
    
    new_block = LevelElement(
        id=len(test_level.blocks),
        type=BLOCK_NAME,  # Using a square block as example
        material="wood",
        x=game_x,  # Using converted coordinates
        y=game_y,  # Using converted coordinates
        rotation=0 if not ROTATED else 90,
    )
    test_level.blocks.append(new_block)
    test_level.create_polygons()

    # Plot modified level
    visualizer.visualize_level_img(test_level, ax=ax2)
    ax2.set_title("Level with Added Block Rasterized")
    
    # Plot modified level
    visualizer.create_img_of_level(test_level, use_grid=True, add_dots=False, ax=ax4)
    ax4.set_title("Level with Added Block")

    # # 3. Create XML and send to game
    config = Config.get_instance()
    game_manager = GameManager(conf=config)
    game_manager.start_game()


    # Create and copy level file
    game_manager.switch_to_level(test_level, wait_for_stable=False)

    plt.tight_layout()
    plt.show()

    # 4. Visualize screenshot of game
    fig, ax = plt.subplots(1, 1)
    visualizer.visualize_screenshot(game_manager.get_img(structure=True), ax=ax)
    plt.show()

    game_manager.stop_game()


if __name__ == "__main__":
    # Example: Add a block at position (2, 0)
    add_block_to_level(level_idx=1, block_x=30, block_y=28)