import json
import os
import pickle
import sys
import time
from copy import copy
from dataclasses import dataclass, asdict
from multiprocessing import Process, Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from loguru import logger
from tqdm.auto import tqdm
from generator.gan.BigGans import WGANGP128128_Multilayer
import tensorflow as tf

from converter.gan_processing.DecodingFunctions import DecodingFunctions
from converter.to_img_converter.MultiLayerStackDecoder import MultiLayerStackDecoder
from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from level.LevelVisualizer import LevelVisualizer
from test.visualization.DashVisualization import LevelVisualization

from util.Config import Config


# logger.add(sys.stdout, level="DEBUG")

@dataclass
class GeneratedDataset:
    """ Class that represents a generated data set """
    imgs: np.ndarray = None
    seed: np.ndarray = None
    predictions: np.ndarray = None
    name: str = None
    date: str = None


class QuantitativeSearch:

    def __init__(self, create_game_instances = 4, repeat_epochs = 20, level_per_epoch = 400):
        self.config = Config.get_instance()

        self.decoding_functions = DecodingFunctions(threshold_callback = lambda: 0.5)
        self.decoding_functions.set_rescaling(rescaling = tf.keras.layers.Rescaling)
        self.decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
        self.rescale_function = self.decoding_functions.rescale

        self.checkpoint_dir = self.config.get_new_model_path('Multilayer With Air (AIIDE)')
        self.gan = WGANGP128128_Multilayer(last_dim = 5)

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer = tf.keras.optimizers.Adam(1e-4),
            discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
            generator = self.gan.generator,
            discriminator = self.gan.discriminator
        )
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_prefix, max_to_keep = 2
        )
        self.checkpoint.restore(self.manager.latest_checkpoint)

        self.create_new_levels(1)
        self.time_data = []

        self.game_managers = []

        self.repeat_epochs = repeat_epochs
        self.create_game_instances = create_game_instances
        self.level_per_epoch = level_per_epoch

        self.multilayer_stack_decoder = MultiLayerStackDecoder()

        self.multilayer_stack_decoder.round_to_next_int = True
        self.multilayer_stack_decoder.custom_kernel_scale = True
        self.multilayer_stack_decoder.minus_one_border = True
        self.multilayer_stack_decoder.combine_layers = True
        self.multilayer_stack_decoder.negative_air_value = -1
        self.multilayer_stack_decoder.cutoff_point = 0.5
        self.multilayer_stack_decoder.display_decoding = False

        self.level_visualization = LevelVisualizer()
        self.decoding_functions = DecodingFunctions(threshold_callback = lambda: 0.5)

        self.done_epochs = 0

    def create_new_levels(self, n_amount = 400):
        seed = self.gan.create_random_vector_batch(batch = n_amount)
        generated_images, predictions = self.gan.create_img(seed)
        self.gan_outputs_reformatted = self.rescale_function(generated_images)
        return self.gan_outputs_reformatted

    def start(self):
        for i in range(self.create_game_instances):
            game_connection = GameConnection(conf = self.config, port = 9001 + i)
            game_manager: GameManager = GameManager(self.config, game_connection = game_connection)
            self.game_managers.append(game_manager)
            game_manager.start_game()

        time_data_output = self.config.get_grid_search_file(f"time_data_output_2")
        if Path(time_data_output).exists():
            with open(time_data_output, 'rb') as f:
                self.time_data = pickle.load(f)
        else:
            self.time_data = []

        self.done_epochs = len(self.time_data)

        quantitative_search_output = self.config.get_grid_search_file(f"quantitative_search_output_2")
        if Path(quantitative_search_output).exists():
            with open(quantitative_search_output, 'rb') as f:
                data_output_list = pickle.load(f)
        else:
            data_output_list = []

        for epoch in tqdm(range(self.done_epochs, self.repeat_epochs)):

            generate_levels_start = time.time()

            print("Start Generating")
            gan_output = self.create_new_levels(self.level_per_epoch)
            gan_output_list = []

            generate_levels_end = time.time()

            for output_idx in range(gan_output.shape[0]):
                current_output = gan_output[output_idx]
                gan_output_list.append(current_output)

            time_decode_start = time.time()
            print("Start Decoding")

            # Use 5 Processes to decode
            with Pool(5) as p:
                level_list = p.map(self.multilayer_stack_decoder.decode, gan_output_list)

            time_decode_end = time.time()

            time_simulate_start = time.time()

            print("Start Simulation")
            self.game_managers[0].create_levels_xml_file(level_list)
            self.game_managers[0].copy_game_levels(
                level_path = self.config.get_data_train_path(folder = 'temp'),
                rescue_level = False
            )

            for game_manager, (start_idx, end_idx) in \
                    zip(self.game_managers, list(zip(np.arange(0, 400, 100) + 4, np.arange(100, 500, 100) + 4))):
                game_manager.simulate_all_levels(start_idx = start_idx.item(), end_idx = end_idx.item(),
                                                 wait_for_stable = True,
                                                 wait_for_response = False)
            current_data_dict = dict()
            for level_idx, level in enumerate(level_list):
                level_metadata = level.get_level_metadata()
                current_data_dict[level_idx + 4] = dict()
                current_data_dict[level_idx + 4]['data'] = asdict(level_metadata)
                current_data_dict[level_idx + 4]['level'] = level
                current_data_dict[level_idx + 4]['gan_output'] = gan_output_list[level_idx]

            for game_manager in self.game_managers:
                response = game_manager.game_connection.wait_for_response()
                parsed = json.loads(response[1]['data'])
                for level_data in parsed['levelData']:
                    data_ = current_data_dict[level_data['level_index'] + 1]['data']
                    data_['damage'] = level_data['initial_damage']
                    data_['is_stable'] = level_data['is_stable']
                    data_['woodBlockDestroyed'] = level_data['woodBlockDestroyed']
                    data_['iceBlockDestroyed'] = level_data['iceBlockDestroyed']
                    data_['stoneBlockDestroyed'] = level_data['stoneBlockDestroyed']
                    data_['totalBlocksDestroyed'] = level_data['woodBlockDestroyed'] + level_data['iceBlockDestroyed'] + level_data['stoneBlockDestroyed']

            for game_manager in self.game_managers:
                game_manager.go_to_menu()

            time_simulate_end = time.time()

            time_data = dict(
                epoch = self.done_epochs,
                generate_levels = generate_levels_end - generate_levels_start,
                decode_levels = time_decode_end - time_decode_start,
                simulation = time_simulate_end - time_simulate_start
            )
            self.time_data.append(time_data)

            for key, value in current_data_dict.items():
                data_output_list.append(value)

            with open(quantitative_search_output, 'wb') as handle:
                pickle.dump(data_output_list, handle, protocol = pickle.HIGHEST_PROTOCOL)

            with open(time_data_output, 'wb') as handle:
                pickle.dump(self.time_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

            self.done_epochs = epoch
            print(f'Finished Epoch {self.done_epochs}')

        return True

    def stop(self):
        for game_manager in self.game_managers:
            game_manager.stop_game()

        self.game_managers = []



def load_data():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)

    print(len(data_output_list))

if __name__ == '__main__':
    # load_data()
    quantitativeSearch = QuantitativeSearch()
    continue_search = True
    while continue_search:
        try:
            if quantitativeSearch.start():
                continue_search = False
        except Exception as e:
            print(e)
            print("Error error")
            quantitativeSearch.stop()

