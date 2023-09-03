import json
import os
import pickle
import time
from copy import copy
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from converter.gan_processing.DecodingFunctions import DecodingFunctions
from converter.to_img_converter.MultiLayerStackDecoder import MultiLayerStackDecoder
from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from level.LevelVisualizer import LevelVisualizer
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


def create_data_set(n_amount = 200):
    """
    Function to create and store a new dataset
    """
    from generator.gan.BigGans import WGANGP128128_Multilayer
    import tensorflow as tf

    config = Config.get_instance()

    decoding_functions = DecodingFunctions(threshold_callback = lambda: 0.5)
    decoding_functions.set_rescaling(rescaling = tf.keras.layers.Rescaling)
    decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
    rescale_function = decoding_functions.rescale
    grid_search_dataset = config.get_grid_search_file("generated_data_set")

    checkpoint_dir = config.get_new_model_path('Multilayer With Air (AIIDE)')
    gan = WGANGP128128_Multilayer(last_dim = 5)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer = tf.keras.optimizers.Adam(1e-4),
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
        generator = gan.generator,
        discriminator = gan.discriminator
    )
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_prefix, max_to_keep = 2
    )
    checkpoint.restore(manager.latest_checkpoint)

    seed = gan.create_random_vector_batch(batch = n_amount)
    generated_images, predictions = gan.create_img(seed)
    gan_outputs_reformatted = rescale_function(generated_images)

    out_file_data = GeneratedDataset(
        imgs = gan_outputs_reformatted,
        seed = seed.numpy(),
        predictions = predictions,
        name = f'wgan_gp_128_128_multilayer_with_air',
        date = '20220816-202429'
    )

    with open(grid_search_dataset, 'wb') as handle:
        pickle.dump(out_file_data, handle, protocol = pickle.HIGHEST_PROTOCOL)


def do_grid_search(start_game = True):
    config = Config.get_instance()
    grid_search_dataset = config.get_grid_search_file("generated_data_set")
    grid_search_output = config.get_grid_search_file("grid_search_output_new")

    with open(grid_search_dataset, 'rb') as f:
        data: GeneratedDataset = pickle.load(f)

    level_visualization = LevelVisualizer()
    decoding_functions = DecodingFunctions(threshold_callback = lambda: 0.5)

    game_managers = []

    for i in range(4):
        game_connection = GameConnection(conf = config, port = 9001 + i)
        game_manager: GameManager = GameManager(config, game_connection = game_connection)
        game_managers.append(game_manager)
        if start_game:
            game_manager.start_game()

    multilayer_stack_decoder = MultiLayerStackDecoder()
    multilayer_stack_decoder.display_decoding = False
    gan_output = data.imgs

    gan_output_list = []

    for output_idx in range(gan_output.shape[0]):
        current_output = gan_output[output_idx]
        gan_output_list.append(current_output)

    if Path(grid_search_output).exists():
        with open(grid_search_output, 'rb') as f:
            data_output_list = pickle.load(f)
    else:
        data_output_list = []

    tested_parameter = data_output_list

    ret_value = True

    parameter_list = create_tests()
    for parameter_idx, parameter in tqdm(enumerate(parameter_list), total = len(parameter_list)):

        already_exists = False
        for comp_data in tested_parameter:
            comp_para = comp_data['parameter']
            shared_items = {k: comp_para[k] for k in comp_para.keys() if k in parameter and comp_para[k] == parameter[k]}
            if len(shared_items.items()) == len(parameter.items()):
                already_exists = True
                break

        if already_exists:
            continue

        try:
            run_single_evaluation(config, data_output_list, game_managers, gan_output_list, grid_search_output,
                                  multilayer_stack_decoder, parameter)
        except Exception as e:
            print(f"Inner error {e}: Restarting evaluation run")
            ret_value = False
            break

    for game_manager in game_managers:
        game_manager.stop_game()

    return ret_value


def run_single_evaluation(config, data_output_list, game_managers, gan_output_list, grid_search_output,
                          multilayer_stack_decoder, parameter):
    print('\n\n')
    logger.debug("Run parameters: " + str(parameter))

    data_output = dict()
    data_output['parameter'] = parameter

    # Set the parameters in the decoder
    for key, value in parameter.items():
        if key == 'negative_air_value' and value == 0:
            multilayer_stack_decoder.use_negative_air_value = False

        if hasattr(multilayer_stack_decoder, key):
            setattr(multilayer_stack_decoder, key, value)

    start = time.time()

    # Use 5 Processes toô decode
    with Pool(5) as p:
        level_list = p.map(multilayer_stack_decoder.decode, gan_output_list)

    intermediate = time.time()
    print(intermediate - start)

    game_managers[0].create_levels_xml_file(level_list)
    game_managers[0].copy_game_levels(
        level_path = config.get_data_train_path(folder = 'temp'),
        rescue_level = False
    )

    for game_manager, (start_idx, end_idx) in zip(game_managers,
                                                  list(zip(np.arange(0, 250, 50) + 4, np.arange(50, 250, 50) + 4))):
        game_manager.simulate_all_levels(start_idx = start_idx.item(), end_idx = end_idx.item(), wait_for_stable = True,
                                         wait_for_response = False)

    current_data_dict = dict()
    for level_idx, level in enumerate(level_list):
        level_metadata = level.get_level_metadata()
        current_data_dict[level_idx + 4] = dict(level_metadata = level_metadata)

    for game_manager in game_managers:
        response = game_manager.game_connection.wait_for_response()
        parsed = json.loads(response[1]['data'])
        for level_data in parsed['levelData']:
            sim_data = dict(
                damage = level_data['initial_damage'],
                is_stable = level_data['is_stable'],
                woodBlockDestroyed = level_data['woodBlockDestroyed'],
                iceBlockDestroyed = level_data['iceBlockDestroyed'],
                stoneBlockDestroyed = level_data['stoneBlockDestroyed']
            )
            current_data_dict[level_data['level_index'] + 1]['sim_data'] = sim_data

    for game_manager in game_managers:
        game_manager.go_to_menu()

    end = time.time()
    print(f'Run time: {end - start}')

    data_output['data'] = current_data_dict
    data_output['time'] = end - start

    data_output_list.append(data_output)
    with open(grid_search_output, 'wb') as handle:
        pickle.dump(data_output_list, handle, protocol = pickle.HIGHEST_PROTOCOL)
    # if parameter_idx == 3:
    #     break


def create_tests():
    parameter_dict = dict(
        round_to_next_int = dict(type = 'bool', default = False, values = [True, False]),
        custom_kernel_scale = dict(type = 'bool', default = True, values = [True, False]),
        minus_one_border = dict(type = 'bool', default = False, values = [True, False]),
        combine_layers = dict(type = 'bool', default = False, values = [True, False]),
        negative_air_value = dict(type = 'number', default = -2, values = [-10, -5, -2, -1, 0]),
        cutoff_point = dict(type = 'number', default = 0.85, values = [0.1, 0.5, 0.8, 0.95]),
    )

    test_parameter = []

    def get_parameter_set(parameter_dict, indexes = [0 for _ in range(len(parameter_dict.values()))], skip_to = 0):
        test_parameter.append({
            parameter_name: parameter_dict[parameter_name]['values'][indexes[c_index]] for c_index, parameter_name in
            enumerate(parameter_dict.keys())
        })

        for c_index, parameter_name in enumerate(parameter_dict.keys()):
            if c_index < skip_to:
                continue

            c_value = parameter_dict[parameter_name]['values']
            if indexes[c_index] < len(c_value) - 1:
                next_indexes = copy(indexes)
                next_indexes[c_index] += 1
                get_parameter_set(parameter_dict, next_indexes, skip_to = c_index)

    get_parameter_set(parameter_dict)
    return test_parameter


def eval_grid_search():
    config = Config.get_instance()
    grid_search_output = config.get_grid_search_file("grid_search_output")

    with open(grid_search_output, 'rb') as f:
        data = pickle.load(f)

    print(data)


if __name__ == '__main__':
    # create_data_set()
    continue_search = True
    while continue_search:
        try:
            if do_grid_search():
                continue_search = False
        except Exception as e:
            print(e)
            print("Error error")

    # eval_grid_search()

    # print(len(create_tests()))
