from pathlib import Path

import matplotlib.pyplot as plt

from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from generator.baseline.Baseline import BaselineGenerator
from level.LevelVisualizer import LevelVisualizer
from util.Config import Config


def generate_structure():
    config = Config.get_instance()

    level_dest = config.get_data_train_path(folder = 'generated/single_structure/')
    generator = BaselineGenerator()
    generator.settings(number_levels = 10, ground_structure_range = (1, 1), air_structure_range=(0, 0))
    generator.generate_level_init(folder_path = level_dest)


def leve_visualisation(generate_new = False):
    if generate_new:
        generate_structure()

    config = Config.get_instance()
    game_connection = GameConnection(conf = config)
    game_manager = GameManager(conf = config, game_connection = game_connection)
    game_manager.start_game(is_running = False)

    # config.game_folder_path = os.path.normpath('../science_birds/{os}')
    for level_path in sorted(Path(config.get_data_train_path(folder = 'generated/single_structure')).glob('*.xml')):
        level_visualizer = LevelVisualizer()

        game_manager.change_level(path = str(level_path))

        fig, ax = plt.subplots(1, 1, dpi = 100, figsize=(5, 5))

        level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = True), ax = ax)
        fig.suptitle(f'Level: {str(level_path.name)}', fontsize = 16)

        fig.tight_layout()
        plt.show()

    game_manager.stop_game()


if __name__ == '__main__':
    leve_visualisation()
