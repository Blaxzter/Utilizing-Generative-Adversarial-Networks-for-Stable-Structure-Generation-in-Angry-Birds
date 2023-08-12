import pickle
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from matplotlib.patches import Rectangle

from converter import MathUtil
from converter.gan_processing.DecodingFunctions import DecodingFunctions
from converter.to_img_converter import DecoderUtils
from converter.to_img_converter.DecoderUtils import recalibrate_blocks
from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from converter.to_img_converter.MultiLayerStackDecoder import MultiLayerStackDecoder
from data_scripts.CreateEncodingData import create_element_for_each_block
from level import Constants
from level.Level import Level
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from test.TestUtils import plot_img, plot_matrix_complete
from util.Config import Config

config = Config.get_instance()

plot_stuff = True
plot_to_file = True

img_counter = 0

block_data = config.get_encoding_data(f"encoding_res_{Constants.resolution}")
if type(block_data) is not str:
    resolution = block_data['resolution']
    del block_data['resolution']


def get_cutoff_point(layer):
    frequency, bins = np.histogram(layer, bins = 100)
    if plot_stuff:
        plot_img(layer, 'Original')

    center = (bins[-1] - bins[0]) / 2
    highest_lowest_value = bins[0]
    for i in range(20):
        highest_count = frequency.argmax()
        highest_count_value = bins[highest_count]
        frequency[highest_count] = 0
        if highest_count_value < center:
            # print(highest_lowest_value, highest_count_value)
            if highest_lowest_value < highest_count_value:
                highest_lowest_value = highest_count_value

    print(highest_lowest_value, center, bins[-1])
    return highest_lowest_value


def _get_pig_position(bird_layer, bird_cutoff = 0.5):
    current_img = np.copy(bird_layer)

    current_img = np.flip(current_img, axis = 0)

    highest_lowest_value = get_cutoff_point(bird_layer)
    current_img[current_img <= highest_lowest_value] = -1

    kernel = MathUtil.get_circular_kernel(7)
    kernel = kernel / np.sum(kernel)
    padded_img = np.pad(current_img, 6, 'constant', constant_values = -1)
    bird_probabilities = cv2.filter2D(padded_img, -1, kernel)[6:-6, 6:-6]

    if plot_stuff:
        plot_img(bird_probabilities, 'Bird Filter')

    bird_probabilities[bird_probabilities < bird_cutoff] = 0
    trimmed_bird_img, trim_data = DecoderUtils.trim_img(bird_probabilities, ret_trims = True)

    if plot_stuff:
        plot_img(bird_probabilities, 'After top trimming')

    max_height, max_width = trimmed_bird_img.shape
    top_space, bottom_space, left_space, right_space = trim_data

    bird_positions = []
    trim_counter = 0
    while not np.all(trimmed_bird_img < 0.00001):
        bird_position = np.unravel_index(np.argmax(trimmed_bird_img), trimmed_bird_img.shape)
        y, x = bird_position
        bird_positions.append([
            top_space + y, left_space + x
        ])

        # Remove the location the bird was picked
        x_cords = np.arange(x - 6, x + 6 + 1, 1)
        y_cords = np.arange(y - 6, y + 6 + 1, 1)
        x_cords[x_cords < 0] = 0
        x_cords[x_cords >= max_width] = max_width - 1
        y_cords[y_cords < 0] = 0
        y_cords[y_cords >= max_height] = max_height - 1
        x_pos, y_pos = np.meshgrid(y_cords, x_cords)

        trimmed_bird_img[x_pos, y_pos] = 0
        if plot_stuff:
            plt.imshow(trimmed_bird_img)
            plt.title(f'Bird Removed at: x_pos: {x} ,y_pos: {y}')
            if plot_to_file:
                global img_counter
                plt.savefig(config.get_conv_debug_img_file(f'{img_counter}_{trim_counter}_bird_after_trim'))
                img_counter += 1
                plt.close()
            else:
                plt.show()

        trim_counter += 1

    return bird_positions


def decode_gan(gan_output, kernel_scalar = True, minus_one_border = True, recalibrate = True, allow_plot = True,
               use_rint = False, cutoff_point = 0.85, bird_cutoff = 0.5):
    global img_counter
    stack_decoder = MultiLayerStackDecoder()
    level_visualizer = LevelVisualizer()

    # Move Gan img into positive realm
    test_output = (gan_output[0] + 1) / 2

    level_blocks = []

    plt_title = f"{'Kernel With Scaler' if kernel_scalar else 'Uniform Kernel'} with {'-1 border' if minus_one_border else 'Normal'}"

    for layer_idx in range(1, test_output.shape[-1] - 1):

        layer = test_output[:, :, layer_idx]
        layer = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))
        if use_rint:
            layer = np.rint(layer)
        # layer = testing_img

        if allow_plot:
            plot_img(layer, title = f'Layer {layer_idx}', plot_always = True)

        layer = np.flip(layer, axis = 0)

        highest_lowest_value = get_cutoff_point(layer)

        if allow_plot and plot_stuff:
            plt.hist(layer, bins = np.linspace(0, 1, 100), histtype = 'step', log = True)
            plt.title("Matrix histogram before low values are uniformed")
            if plot_to_file:
                plt.savefig(config.get_conv_debug_img_file(f'{img_counter}_before_uniformed'))
                img_counter += 1
                plt.close()
            else:
                plt.show()

        layer[layer <= highest_lowest_value] = 0

        if allow_plot and plot_stuff:
            plt.hist(layer, bins = np.linspace(0, 1, 100), histtype = 'step', log = True)
            plt.title("Matrix histogram after low values are uniformed")
            if plot_to_file:
                plt.savefig(config.get_conv_debug_img_file(f'{img_counter}_after_uniformed'))
                img_counter += 1
                plt.close()
            else:
                plt.show()

        trimmed_img, trim_data = DecoderUtils.trim_img(layer, ret_trims = True)

        if not recalibrate:
            trimmed_img = (trimmed_img * 2) - 1

        avg_results = []
        sum_results = []
        for idx, possible_block in enumerate(block_data.values()):
            sum_convolution_kernel = np.ones((possible_block['height'] + 1, possible_block['width'] + 1))

            if kernel_scalar:
                sum_convolution_kernel = sum_convolution_kernel * possible_block['scalar']

            avg_convolution_kernel = sum_convolution_kernel / np.sum(sum_convolution_kernel)

            if minus_one_border:
                sum_convolution_kernel = np.pad(sum_convolution_kernel, 1, 'constant', constant_values = -1)

            pad_value = -1 if recalibrate else 0

            pad_size = np.max(sum_convolution_kernel.shape)
            padded_img = np.pad(trimmed_img, pad_size, 'constant', constant_values = pad_value)

            sum_result = cv2.filter2D(padded_img, -1, sum_convolution_kernel)[pad_size:-pad_size, pad_size:-pad_size]
            avg_result = cv2.filter2D(padded_img, -1, avg_convolution_kernel)[pad_size:-pad_size, pad_size:-pad_size]

            avg_results.append(avg_result)
            sum_results.append(sum_result)

        # Create a matrix of block layer,
        # Start with a high hit rate 0.99 or something and iterate down with 0.01 (set prev position to 0)
        # Go from the bigger blocks that hit that "hit rate" and clear the remaining layers from that block.
        # Maybe recalibrate the hit rate? if only shit remains then return

        blocks: List[Dict] = list(block_data.values())

        hit_probabilities = np.stack(avg_results, axis = -1)
        size_ranking = np.stack(sum_results, axis = -1)
        stop_condition = np.sum(trimmed_img[trimmed_img > 0]).item()

        if allow_plot:
            plot_matrix_complete(hit_probabilities, blocks, title = 'Hit Confidence', block = False, plot_always = True,
                                 flipped = True)
            plot_matrix_complete(size_ranking, blocks, title = 'Size Ranking', block = False, plot_always = True,
                                 flipped = True)

        def delete_blocks(_block_rankings, center_block, _position):
            ret_block_ranking = np.copy(_block_rankings)

            left_extension = (center_block['width'] + 1) // 2
            right_extension = (center_block['width'] + 1) // 2
            top_extension = (center_block['height'] + 1) // 2
            bottom_extension = (center_block['height'] + 1) // 2
            y, x = _position

            max_height, max_width = ret_block_ranking.shape[:2]

            delete_rectangles = []

            for block_idx, outside_block in enumerate(blocks):
                start = x - (((outside_block['width'] + 1) // 2) + left_extension)
                end = x + ((outside_block['width'] + 1) // 2) + right_extension - outside_block['width'] % 2

                top = y - (((outside_block['height'] + 1) // 2) + top_extension)
                bottom = y + ((outside_block['height'] + 1) // 2) + bottom_extension - outside_block['height'] % 2

                start = start if start > 0 else 0
                end = end if end < max_width else max_width - 1
                top = top if top > 0 else 0
                bottom = bottom if bottom < max_height else max_height - 1

                x_cords = np.arange(start, end + 1, 1)
                y_cords = np.arange(top, bottom + 1, 1)
                x_pos, y_pos = np.meshgrid(y_cords, x_cords)
                ret_block_ranking[x_pos, y_pos, block_idx] = 0

                delete_rectangles.append((start, end, top, bottom))

            return ret_block_ranking, delete_rectangles

        def _select_blocks(_block_rankings, _selected_blocks: List, _stop_condition: float, _covered_area: float = 0):
            print(_covered_area, _stop_condition)

            # select the most probable block that is also the biggest
            next_block = np.unravel_index(np.argmax(_block_rankings), _block_rankings.shape)

            # Extract the block
            selected_block = blocks[next_block[-1]]
            block_position = list(next_block[0:2])
            description = f"Selected Block: {selected_block['name']} with {_block_rankings[next_block]} area with {len(_selected_blocks)} selected"
            print(description)

            # Remove all blocks that cant work with that blocks together
            next_block_rankings, delete_rectangles = delete_blocks(_block_rankings, selected_block, block_position)

            if allow_plot:
                plot_matrix_complete(_block_rankings, blocks, title = description, add_max = True, block = True,
                                     position = block_position, delete_rectangles = delete_rectangles,
                                     selected_block = next_block[-1], save_name = f'{selected_block["name"]}_selected')

            if np.all(next_block_rankings < 0.00001):
                print("No position available")
                return _selected_blocks

            next_blocks = _selected_blocks.copy()
            next_blocks.append(dict(
                position = block_position,
                block = selected_block,
            ))
            next_covered_area = _covered_area + ((selected_block['width'] + 1) * (selected_block['height'] + 1))
            return _select_blocks(next_block_rankings, next_blocks, _stop_condition, next_covered_area)

        percentage_cut = np.copy(hit_probabilities)
        percentage_cut[hit_probabilities <= cutoff_point] = 0

        current_size_ranking = percentage_cut * size_ranking
        if allow_plot:
            plot_matrix_complete(current_size_ranking, blocks, "Current Size Rankings", add_max = True, block = False,
                                 plot_always = True, flipped = True)

        rounded_block_rankings = np.around(current_size_ranking, 5)  # Round to remove floating point errors
        selected_blocks = _select_blocks(rounded_block_rankings, [], stop_condition, _covered_area = 0)

        top_space, bottom_space, left_space, right_space = trim_data
        if selected_blocks is not None:
            for block in selected_blocks:
                ic(block)
                block['position'][0] += top_space
                block['position'][1] += left_space
                block['material'] = layer_idx
                level_blocks.append(block)

        if allow_plot and plot_stuff:
            fig, ax = plt.subplots()
            created_level_elements = stack_decoder.create_level_elements(selected_blocks, [])
            # created_level_elements = recalibrate_blocks(created_level_elements)
            level_visualizer.create_img_of_structure(created_level_elements, title = plt_title, ax = ax)
            ax.set_title(f'Encoded Layer {layer_idx}')
            if plot_to_file:
                fig.savefig(config.get_conv_debug_img_file(f'{img_counter}_finished_structure'))
                img_counter += 1
                plt.close(fig)
            else:
                plt.show()

    bird_positions = _get_pig_position(test_output[:, :, -1], bird_cutoff = bird_cutoff)

    created_level_elements = stack_decoder.create_level_elements(level_blocks, bird_positions)
    # created_level_elements = recalibrate_blocks(created_level_elements)
    created_level = Level.create_level_from_structure(created_level_elements)

    if allow_plot:
        fig, ax = plt.subplots()
        level_visualizer.create_img_of_structure(created_level_elements, title = plt_title, ax = ax)
        if plot_to_file:
            ax.set_title('Finished Structure')
            fig.savefig(config.get_conv_debug_img_file(f'{img_counter}_finished_structure'))
            img_counter += 1
            plt.close(fig)
        else:
            plt.show()

    return created_level


if __name__ == '__main__':

    test_environment = TestEnvironment()
    test_outputs = test_environment.load_test_outputs_of_model('multilayer_with_air')
    test_output, image_name = test_environment.return_loaded_gan_output_by_idx(0)

    norm_img = (test_output[0] + 1) / 2

    norm_img[norm_img[:, :, 0] < 0.1, 0] = 0
    img = np.argmax(norm_img, axis = 2)

    plt.imshow(img)
    plt.show()

    plot_stuff = True
    allow_plot = True
    plot_to_file = False

    decoded_level = decode_gan(
        test_output,
        kernel_scalar = True,
        minus_one_border = True,
        recalibrate = False,
        allow_plot = allow_plot,
        use_rint = False,
        cutoff_point = 0.5,
        bird_cutoff = 0.1
    )

    fig, ax = plt.subplots()
    level_visualizer = LevelVisualizer()
    created_level_elements = recalibrate_blocks(decoded_level.get_used_elements())
    level_visualizer.create_img_of_structure(created_level_elements, title = "Finished Structure")
    plt.show()