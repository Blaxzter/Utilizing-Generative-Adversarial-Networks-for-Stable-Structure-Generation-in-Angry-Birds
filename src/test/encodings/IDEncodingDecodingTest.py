import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from converter.to_img_converter.LevelIdImgDecoder import LevelIdImgDecoder
from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from game_management.GameManager import GameManager
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from util.Config import Config


def encode_decode_test(test_with_game, multi_layer = False, true_one_hot = False):
    level_id_img_decoder = LevelIdImgDecoder()
    test_environment = TestEnvironment('generated/single_structure')
    level_visualizer = LevelVisualizer()

    if test_with_game:
        game_manager = GameManager(Config.get_instance())
        game_manager.start_game()

    for level_idx, level in test_environment.iter_levels(start = 0, end = 5):

        fig, axs = plt.subplots(2 if test_with_game else 1, 2)
        axs = axs.flatten()

        if test_with_game:
            game_manager.switch_to_level_elements(level.get_used_elements(), 4)
            img = game_manager.get_img(structure = True)
            axs[0].imshow(img)

        img_rep = create_encoding(level, multi_layer, true_one_hot)
        if multi_layer and not true_one_hot:
            ret_img = level_id_img_decoder.create_single_layer_img(img_rep)
        else:
            ret_img = img_rep

        decoded_level = level_id_img_decoder.decode_level(ret_img)

        level_visualizer.create_img_of_level(level, ax = axs[2 if test_with_game else 0], use_grid = False)
        level_visualizer.create_img_of_level(decoded_level, ax = axs[3 if test_with_game else 1], use_grid = False)

        if level == decoded_level:
            print(f'Not equal {level_idx}')

        if test_with_game:
            game_manager.switch_to_level_elements(decoded_level.get_used_elements(), 5)
            img = game_manager.get_img(structure = True)
            axs[1].imshow(img)

        plt.show()

    if test_with_game:
        game_manager.stop_game()


def compare_multilayer_with_single_layer():
    level_id_img_decoder = LevelIdImgDecoder()
    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(0)

    single_layer = create_encoding(level, multilayer = False)
    multilayer = create_encoding(level, multilayer = True)

    reconstructed_single_layer = level_id_img_decoder.create_single_layer_img(multilayer)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(single_layer)
    axs[1].imshow(reconstructed_single_layer)

    if np.alltrue(np.equal(single_layer, reconstructed_single_layer)):
        logger.debug("All the same")
    else:
        logger.debug("False")

    plt.show()


def create_encoding(level, multilayer = False, true_one_hot = False):
    level_img_encoder = LevelImgEncoder()
    return level_img_encoder.create_one_element_img(
        element_list = level.get_used_elements(),
        air_layer = False,
        multilayer = multilayer,
        true_one_hot = true_one_hot
    )


if __name__ == '__main__':
    encode_decode_test(test_with_game = False, multi_layer = False, true_one_hot = False)
    #compare_multilayer_with_single_layer()
