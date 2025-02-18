import os
import shutil
import time
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt

from game_management.GameConnection import GameConnection
from level.LevelReader import LevelReader
from util.Config import Config


class GameManager:

    def __init__(self, conf: Config, game_connection: GameConnection = None):
        self.conf = conf
        self.rescue_level = conf.rescue_level
        self.rescue_level_path = conf.rescue_level_path

        self.game_is_running = False
        self.level_reader = LevelReader()

        if game_connection is not None:
            self.game_connection = game_connection
        else:
            self.game_connection = GameConnection(self.conf)

    def start_game(self, is_running = False):
        if self.game_is_running:
            if self.game_connection.client != None:
                logger.debug("Game is allready running")
                return
            else:
                logger.debug("Restart the game.")
                self.game_connection.stop_components()
                self.game_connection = GameConnection(self.conf)
        else:
            logger.debug("Start Game and Game Connection Server")

        self.game_connection.start()
        if not is_running:
            self.game_connection.start_game(self.conf.game_path)

        self.game_connection.wait_for_game_window()
        self.game_is_running = True

    def stop_game(self):
        logger.debug("Stop Game Components")
        self.game_connection.stop_components()

    def simulate_all_levels(self, start_idx = 0, end_idx = False, wait_for_stable = False, wait_for_response = False):
        self.game_connection.load_level_menu()
        return self.game_connection.simulate_all_levels(start_idx, end_idx, wait_for_stable, wait_for_response)

    def switch_to_level(self, level, element_idx = 4, stop_time = False, wait_for_stable = False, move_to_ground = True):
        return self.switch_to_level_elements(level.get_used_elements(), element_idx, stop_time, wait_for_stable, move_to_ground)

    def switch_to_level_elements(self, elements, element_idx = 4, stop_time = False, wait_for_stable = False, move_to_ground = True):
        level_path = self.create_level_xml_file(element_idx, elements, move_to_ground)

        return self.change_level(path = str(level_path), stopTime = stop_time, wait_for_stable = wait_for_stable)

    def create_levels_xml_file(self, level_list, delete_previous = True, store_level_name = None):
        level_paths = []

        if delete_previous:
            for level in Path(self.conf.get_data_train_path(folder = 'temp')).glob('*.*'):
                os.remove(level)

        for level_idx, level in enumerate(level_list):

            level_path = self.create_level_xml_file(
                element_idx = level_idx + 4,
                elements = level.get_used_elements()
            )

            if store_level_name is not None:
                shutil.copy(str(level_path), self.conf.good_generated_level(store_level_name))

            level_paths.append(level_path)

        return level_paths

    def create_level_xml_file(self, element_idx, elements, move_to_ground = True):
        level = self.level_reader.create_level_from_structure(elements, red_birds = True, move_to_ground = move_to_ground)
        level_folder = self.conf.get_data_train_path(folder = 'temp')
        level_number = f'0{element_idx}' if len(str(element_idx)) == 1 else str(element_idx)
        level_path = f'{level_folder}/level-{level_number}.xml'
        self.level_reader.write_xml_file(level, level_path)
        return level_path

    def copy_game_levels(self, level_path = None, rescue_level = None):
        if rescue_level is None:
            rescue_level = self.rescue_level

        if level_path is None:
            level_path = self.conf.generated_level_path

        if rescue_level:
            current_rescue_level_pat = self.rescue_level_path
            timestr = time.strftime("%Y%m%d-%H%M%S")
            current_rescue_level_path = current_rescue_level_pat.replace("{timestamp}", timestr)
            # check if folder exists
            if not os.path.exists(current_rescue_level_path):
                # recursivly create folder
                os.makedirs(current_rescue_level_path, exist_ok = True, mode = 0o777)

            for src_file in Path(self.conf.get_game_level_path()).glob('*.*'):
                shutil.move(str(src_file), current_rescue_level_path)
        else:
            for level in Path(self.conf.get_game_level_path()).glob('*.*'):
                os.remove(level)

        ret_copied_levels = []

        for src_file in Path(level_path).glob('*.*'):
            ret_copied_levels.append(src_file)
            shutil.move(str(src_file), self.conf.get_game_level_path())

        self.game_connection.load_level_menu()

        return ret_copied_levels

    def change_level(self, path, delete_level = True, wait_for_stable = False, stopTime = False):
        if delete_level:
            for level in Path(self.conf.get_game_level_path()).glob('*.*'):
                os.remove(level)

        shutil.copy(str(path), self.conf.get_game_level_path())
        self.game_connection.load_level_menu()
        return self.game_connection.change_level(index = 4, wait_for_stable = wait_for_stable, stopTime = stopTime)

    def create_img_of_level(self, index = 4):
        self.select_level(index)
        return self.get_img()

    def get_img(self, structure = True):
        img = self.game_connection.create_level_img(structure = structure)
        return img

    def create_img(self, structure = True):
        plt.imshow(self.game_connection.create_level_img(structure = structure))
        plt.show()

    def remove_game_levels(self):
        for level in Path(self.conf.get_game_level_path()).glob('*.*'):
            os.remove(level)

    def select_level(self, i, wait_for_stable = True, stopTime = False):
        self.game_connection.load_level_menu()
        self.game_connection.change_level(index = i, wait_for_stable = wait_for_stable, stopTime = stopTime)

    def go_to_menu(self):
        self.game_connection.go_to_menu()
