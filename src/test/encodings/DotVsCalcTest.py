from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from generator.baseline.Baseline import BaselineGenerator
from level import Constants
from level.Level import Level
from level.LevelElement import LevelElement
from level.LevelReader import LevelReader
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from util.Config import Config


def generate_structure():
    config = Config.get_instance()

    level_dest = config.get_data_train_path(folder = 'generated/single_structure/')
    generator = BaselineGenerator()
    generator.settings(number_levels = 10, ground_structure_range = (1, 1), air_structure_range=(0, 0))
    generator.generate_level_init(folder_path = level_dest)


def visualize_multiple_level():
    # generate_structure()

    config = Config.get_instance()
    game_connection = GameConnection(conf = config)
    game_manager = GameManager(conf = config, game_connection = game_connection)
    # game_manager.start_game(is_running = False)

    counter = 0

    # config.game_folder_path = os.path.normpath('../science_birds/{os}')
    for level_path in sorted(Path(config.get_data_train_path(folder = 'generated/single_structure')).glob('*.xml')):
        level_reader = LevelReader()
        parse_level = level_reader.parse_level(str(level_path), use_blocks = True, use_pigs = True, use_platform = True)
        visualize_dot_vs_calc(parse_level)

        counter += 1
        if counter > 5:
            break

    game_manager.stop_game()


def visualize_level():
    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(0)
    visualize_dot_vs_calc(level)

def visualize_dot_vs_calc(level):
    level_visualizer = LevelVisualizer()
    level.filter_slingshot_platform()
    level.normalize()
    level.create_polygons()
    # game_manager.change_level(path = str(level_path))
    fig, ax = plt.subplots(1, 3, dpi = 100, figsize = (15, 5))
    # level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = True), ax = ax[0])
    level_visualizer.visualize_level_img(level, dot_version = False, ax = ax[1])
    level_visualizer.visualize_level_img(level, dot_version = True, ax = ax[2])
    # fig.suptitle(f'Level: {str(level_path.name)}', fontsize = 16)

    ax[0].set_title("Dot Encoding")
    ax[2].set_title("Dot Version")
    ax[1].set_title("Calc Version")

    fig.tight_layout()
    plt.show()


def visualize_dot_vs_calc(level):
    level_visualizer = LevelVisualizer(dot_size = 1)

    level.filter_slingshot_platform()
    level.normalize()
    level.create_polygons()

    fig, ax = plt.subplots(1, 1, dpi = 100)

    level_visualizer.create_img_of_level(level, add_dots = True, ax = ax, element_ids = False)

    ax.set_title("Level visualization with dots")

    fig.tight_layout()
    plt.show()


def get_wrong_dot_encoding():
    index = 11
    size = Constants.block_sizes[index]

    level_visualizer = LevelVisualizer(dot_size = 1)

    for x_offset in np.linspace(0.1, 0.2, num = 10):
        elements = []
        start_x = 0

        for i in range(0, 2):
            start_x += size[0] / 2

            block_attribute = dict(
                type = Constants.block_names[index],
                material = 'wood',
                x = start_x,
                y = size[1] / 2,
                rotation = 90 if Constants.block_is_rotated[index] else 0
            )

            element = LevelElement(id = 0, **block_attribute)
            element.shape_polygon = element.create_geometry()
            elements.append(element)

            start_x += size[0] / 2 + Constants.resolution * 2 + x_offset

        fig, ax = plt.subplots(1, 2, dpi = 100)
        level = Level.create_level_from_structure(elements)

        level_visualizer.create_img_of_level(level, add_dots = True, ax = ax[0], element_ids = False)
        level_visualizer.visualize_level_img(level, dot_version = True, ax = ax[1])

        fig.suptitle("Same block wrongly encoded")

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    get_wrong_dot_encoding()
