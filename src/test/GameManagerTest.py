import os
from pathlib import Path

from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from level.LevelReader import LevelReader
from level.LevelUtil import calc_structure_meta_data
from level.LevelVisualizer import LevelVisualizer
from util import ProgramArguments
from util.Config import Config


def screen_shot_test():
    parser = ProgramArguments.get_program_arguments()
    config = parser.parse_args()
    config.game_folder_path = os.path.normpath('../science_birds/{os}')

    config = Config(config)
    level_visualizer = LevelVisualizer()
    level_reader = LevelReader()
    game_connection = GameConnection(conf = config)
    game_manager = GameManager(conf = config, game_connection = game_connection)
    try:
        game_manager.start_game()
        for level_path in sorted(Path("../resources/data/source_files/structures/").glob('*.xml')):
            level = level_reader.parse_level(path = str(level_path), use_blocks = True, use_pigs = True, use_platform = True)
            game_manager.change_level(path = str(level_path))
            img = game_connection.create_level_img(structure = True)
            cropped_img = level_visualizer.visualize_screenshot(img)

        game_manager.stop_game()
    except KeyboardInterrupt:
        game_manager.stop_game()

def level_split_test():
    parser = ProgramArguments.get_program_arguments()
    config = parser.parse_args()
    config.game_folder_path = os.path.normpath('../science_birds/{os}')

    config = Config(config)
    game_connection = GameConnection(conf = config)
    game_manager = GameManager(conf = config, game_connection = game_connection)
    game_manager.start_game()

    data_dict = dict()

    for level in sorted(Path("source_files/generated/").glob('*.xml')):
        level_reader = LevelReader()
        parsed_level = level_reader.parse_level(str(level), use_blocks = True, use_pigs = True, use_platform = True)
        parsed_level.filter_slingshot_platform()

        level_idx = level.name.strip('level-').strip('.xml')

        parsed_level.normalize()
        parsed_level.create_polygons()
        level_structures = parsed_level.separate_structures()

        structure_data = dict()
        for idx, structure in enumerate(level_structures):
            current_structure_data = calc_structure_meta_data(structure)
            if current_structure_data.pig_amount == 0:
                continue
            xml_file = level_reader.create_level_from_structure(structure, move_to_ground = True)
            structure_data[idx] = dict(
                meta_data = current_structure_data,
                file = xml_file
            )

        game_connection.startAi(start_level = level_idx, end_level = level_idx)

        structure_imgs = parsed_level.create_img(per_structure = True, dot_version = True)
        game_connection.wait_till_all_level_played()
        data = game_connection.get_data()

if __name__ == '__main__':
    screen_shot_test()
