from pathlib import Path

import matplotlib.pyplot as plt

from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from generator.baseline.Baseline import BaselineGenerator
from level.LevelReader import LevelReader
from level.LevelUtil import calc_structure_meta_data
from level.LevelVisualizer import LevelVisualizer
from util.Config import Config


def generate_structure():
    config = Config.get_instance()

    level_dest = config.get_data_train_path(folder = 'generated/single_structure/')
    generator = BaselineGenerator()
    generator.settings(number_levels = 3, ground_structure_range = (1, 1), air_structure_range=(0, 0))
    generator.generate_level_init(folder_path = level_dest)


def leve_visualisation():
    # generate_structure()

    config = Config.get_instance()
    # config.game_folder_path = os.path.normpath('../science_birds/{os}')
    for level_path in sorted(Path(config.get_data_train_path(folder = 'generated/single_structure')).glob('*.xml')):

        game_connection = GameConnection(conf = config)
        game_manager = GameManager(conf = config, game_connection = game_connection)
        level_reader = LevelReader()
        level_visualizer = LevelVisualizer()

        parse_level = level_reader.parse_level(str(level_path), use_blocks = True, use_pigs = True, use_platform = True)
        parse_level.filter_slingshot_platform()

        parse_level.normalize()
        parse_level.create_polygons()
        structures = parse_level.separate_structures()
        structures = list(filter(lambda structure: calc_structure_meta_data(structure).total > 1, structures))

        game_manager.start_game(is_running = False)
        game_manager.change_level(path = str(level_path))

        fig, ax = plt.subplots(len(structures) + 1, 3, dpi = 500, figsize=(5, 10))

        level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = False), ax = ax[0, 0])
        level_visualizer.create_img_of_level(level = parse_level, element_ids = False, use_grid = True, add_dots = False, ax = ax[0, 1])
        level_visualizer.visualize_level_img(parse_level, dot_version = False, ax = ax[0, 2])

        level_counter = 1
        for idx, structure in enumerate(structures):
            meta_data = calc_structure_meta_data(structure)
            struct_doc = level_reader.create_level_from_structure(
                structure = structure,
                level = parse_level,
                move_to_ground = True
            )
            new_level_path = config.get_data_train_path(folder = 'structures')
            new_level_path = f'{new_level_path}/level-0{level_counter + 4}.xml'
            level_reader.write_xml_file(struct_doc, new_level_path)
            game_manager.change_level(path = new_level_path)

            new_level = level_reader.parse_level(new_level_path, use_platform = True)
            new_level.normalize()
            new_level.create_polygons()
            new_level.filter_slingshot_platform()
            new_level.separate_structures()

            level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = True), ax = ax[level_counter, 0])
            level_visualizer.create_img_of_structure(new_level.get_used_elements(), use_grid = True, element_ids = False, ax = ax[level_counter, 1])
            level_visualizer.visualize_level_img(new_level, per_structure = True, dot_version = True, ax = ax[level_counter, 2])
            level_counter += 1

        fig.tight_layout()
        plt.show()
        game_manager.stop_game()
        break


if __name__ == '__main__':
    leve_visualisation()
