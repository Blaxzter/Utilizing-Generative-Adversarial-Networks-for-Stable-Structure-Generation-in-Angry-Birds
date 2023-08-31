import pickle
from pathlib import Path

from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from data_scripts.CreateEncodingData import create_element_for_each_block
from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from level.LevelReader import LevelReader
from level.LevelVisualizer import LevelVisualizer
from util.Config import Config


class TestEnvironment:

    def __init__(self, level_folder = 'generated/single_structure'):
        self.config = Config.get_instance()

        self.game_connection = GameConnection(conf = self.config)
        self.game_manager = GameManager(conf = self.config, game_connection = self.game_connection)

        self.level_reader = LevelReader()
        self.level_visualizer = LevelVisualizer(line_size = 1)
        self.level_img_encoder = LevelImgEncoder()

        if level_folder is None:
            level_folder = 'generated/single_structure'
            self.level_path = self.config.get_data_train_path(folder = level_folder)
        else:
            self.level_path = level_folder

        self.levels = list(map(str, Path(self.level_path).glob('*.xml')))

    def __len__(self):
        return

    def iter_levels(self, start = 0, end = -1):
        if end == -1:
            end = len(self.levels)

        for level_counter, level_path in enumerate(self.levels[start:end]):
            level = self.level_reader.parse_level(level_path)
            level.normalize()
            level.create_polygons()
            yield level_counter, level

    def get_level(self, idx, normalize = True):
        level_path = self.levels[idx]
        level = self.level_reader.parse_level(level_path)
        if normalize:
            level.normalize()
        level.create_polygons()
        return level

    def start_game(self, is_running = False):
        self.game_manager.start_game(is_running = is_running)

    def get_levels(self):
        return sorted(self.levels)

    def load_test_outputs_of_model(self, model_name):
        loaded_model = model_name.replace(' ', '_').lower()
        store_imgs_pickle_file = self.config.get_gan_img_store(loaded_model)

        with open(store_imgs_pickle_file, 'rb') as f:
            loaded_outputs = pickle.load(f)

        self.loaded_outputs = loaded_outputs

        return loaded_outputs

    def return_loaded_gan_output_by_idx(self, idx = 0):
        if self.loaded_outputs is None:
            raise Exception('Pls load test output first')

        test_image = list(self.loaded_outputs.keys())[idx]
        return self.loaded_outputs[test_image]['output'], test_image

    def create_calibration_img(self, direction = 'vertical', stacked = 1, x_offset = 0, y_offset = 0, diff_materials = False):
        elements, sizes = create_element_for_each_block(direction = direction, stacked = stacked, x_offset = x_offset, y_offset = y_offset, diff_materials = diff_materials)

        return self.level_img_encoder.create_calculated_img(elements)

