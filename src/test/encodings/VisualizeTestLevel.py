import matplotlib.pyplot as plt

from data_scripts.CreateEncodingData import create_element_for_each_block
from game_management.GameManager import GameManager
from level.Level import Level
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from util.Config import Config


def visualize_level(level = 0):

    test_environment = TestEnvironment()
    test_level = test_environment.get_level(level)
    visualizer = LevelVisualizer(line_size = 2)

    fig, ax = plt.subplots(1, 1)
    visualizer.create_img_of_level(test_level, use_grid = False, add_dots = False, ax = ax, material_color = True)
    plt.show()


def decode_test_level(direction = 'vertical', stacked = 3, x_offset = 0, y_offset = 0):
    elements, sizes = create_element_for_each_block(direction, stacked, x_offset, y_offset, diff_materials = False)

    test_level = Level.create_level_from_structure(elements)
    visualizer = LevelVisualizer(line_size = 1)

    fig, ax = plt.subplots(1, 1)
    visualizer.create_img_of_level(test_level, use_grid = False, add_dots = False, ax = ax)
    plt.show()


def visualize_test_level_with_screenshot(direction = 'vertical', stacked = 3, x_offset = 0, y_offset = 0):
    config = Config.get_instance()

    elements, sizes = create_element_for_each_block(direction, stacked, x_offset, y_offset, diff_materials = False)

    test_level = Level.create_level_from_structure(elements)
    visualizer = LevelVisualizer(line_size = 1)

    fig, axs = plt.subplots(1, 2)
    visualizer.create_img_of_level(test_level, use_grid = False, add_dots = False, ax = axs[0])

    game_manager = GameManager(conf = config)
    game_manager.start_game()

    game_manager.create_level_xml_file(element_idx = 4, elements = test_level.get_used_elements())
    game_manager.copy_game_levels(
        level_path = config.get_data_train_path(folder = 'temp'),
        rescue_level = False
    )
    game_manager.select_level(i = 4)

    visualizer.visualize_screenshot(game_manager.get_img(structure = True), ax = axs[1])

    game_manager.stop_game()
    plt.show()


if __name__ == '__main__':
    # decode_test_level(stacked = 2, direction = 'horizontal')
    # visualize_level()
    visualize_test_level_with_screenshot(stacked = 3, direction = 'vertical')
