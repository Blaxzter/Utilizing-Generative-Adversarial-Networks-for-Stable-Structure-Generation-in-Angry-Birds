from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt

from converter.to_text_converter.text2xml import txt2xml
from converter.to_text_converter.xml2text import xml2txt
from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from level.LevelReader import LevelReader
from level.LevelVisualizer import LevelVisualizer
from util.Config import Config


def test_level_decompilation():
    config = Config.get_instance()
    game_connection = GameConnection(conf = config)
    game_manager = GameManager(conf = config, game_connection = game_connection)

    level_reader = LevelReader()
    level_visualizer = LevelVisualizer()

    level_path = config.get_data_train_path(folder = 'generated/single_structure')

    converter = xml2txt(level_path)
    deconverted = txt2xml()

    game_manager.start_game(is_running = False)
    levels = Path(level_path).glob('*.xml')
    for level_idx, level_path in enumerate(sorted(levels)):
        vector = converter.xml2txt(str(level_path), True)

        deconverted_level_xml = deconverted.txt2xml(vector)

        parse_level = level_reader.parse_level(str(level_path), use_blocks = True, use_pigs = True, use_platform = True)
        parse_level.filter_slingshot_platform()

        parse_level.normalize()
        parse_level.create_polygons()

        game_manager.change_level(path = str(level_path))

        fig, ax = plt.subplots(1, 3, dpi = 300, figsize=(15, 5))

        level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = True), ax = ax[0])
        ax[0].set_title('Original')

        deconverted_level_file = config.get_deconverted_file()
        level_reader.write_xml_file_from_string(deconverted_level_xml, deconverted_level_file)
        game_manager.change_level(path = deconverted_level_file, stopTime = True)
        sleep(1)
        level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = True), ax = ax[1])
        ax[1].set_title('Decompiled Start')

        level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = True), ax = ax[2])
        ax[2].set_title('After load')

        fig.suptitle(f'Level: {str(level_path)}', fontsize = 16)

        fig.tight_layout()
        plt.show()

        if level_idx > 5:
            break

    game_manager.stop_game()


if __name__ == '__main__':
    test_level_decompilation()
