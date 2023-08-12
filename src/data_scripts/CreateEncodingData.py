import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from tabulate import tabulate

from converter.to_img_converter.LevelImgDecoder import LevelImgDecoder
from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from game_management.GameManager import GameManager
from level import Constants
from level.LevelElement import LevelElement
from level.LevelVisualizer import LevelVisualizer
from util import Utils
from util.Config import Config

config = Config.get_instance()


def create_element_for_each_block(direction = 'vertical', stacked = 1, x_offset = 0, y_offset = 0, diff_materials = False):
    """
        Creates structure list of each block type
    """
    elements = []
    start_x = 0
    sizes = Constants.get_sizes(print_data = False)

    if diff_materials:
        materials = Constants.materials * (int(stacked / 3) + 1)
    else:
        materials = ['wood'] * stacked

    for idx, block in enumerate(sizes):
        index = list(Constants.block_names.values()).index(block['name']) + 1
        if block['rotated']:
            index += 1
        size = Constants.block_sizes[index]

        start_x += size[0] / 2

        vertical_stacked = 0
        horizontal_stacked = 0
        for stack_idx, stack in enumerate(range(stacked)):
            block_attribute = dict(
                type = block['name'],
                material = materials[stack],
                x = start_x + horizontal_stacked,
                y = size[1] / 2 + y_offset + vertical_stacked,
                rotation = 90 if block['rotated'] else 0
            )
            element = LevelElement(id = idx + stack_idx, **block_attribute)
            element.shape_polygon = element.create_geometry()
            elements.append(element)

            if stack_idx != stacked - 1:
                if direction == 'vertical':
                    vertical_stacked += size[1]
                elif direction == 'horizontal':
                    horizontal_stacked += size[0]

        start_x += size[0] / 2 + Constants.resolution * 2 + x_offset * idx + horizontal_stacked

    return elements, sizes


def get_debug_level():
    """
    Returns the img of the debug level either from a pickle or newly created
    """
    pickle_data = config.get_pickle_file("block_data")
    if os.path.isfile(pickle_data):
        with open(pickle_data, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        level_img_encoder = LevelImgEncoder()
        elements, sizes = create_element_for_each_block()
        level_img = level_img_encoder.create_calculated_img(elements)
        with open(pickle_data, 'wb') as handle:
            pickle.dump(level_img[0], handle, protocol = pickle.HIGHEST_PROTOCOL)
        return level_img[0]


def test_offesets(test_dot = False):
    """
    Creates the testing level with more and more space between the blocks.
    Searches for the rectangles in the img representation and compares
    the width difference between offsets.
    """
    level_img_encoder = LevelImgEncoder()
    level_img_decoder = LevelImgDecoder()

    offset_data = dict()

    subplot_amount = 15

    fig, axs = plt.subplots(subplot_amount, 1, dpi = 600, figsize = (8, 15))

    for x_offset, ax in zip(np.linspace(0, 0.15, num = subplot_amount), axs.flatten()):
        elements, sizes = create_element_for_each_block(x_offset = x_offset)

        # Create the images
        if test_dot:
            level_rep = level_img_encoder.create_dot_img(elements)
        else:
            level_rep = level_img_encoder.create_calculated_img(elements)

        ax.imshow(level_rep)
        ax.axis('off')

        recs = level_img_decoder.get_rectangles(level_rep)
        recs = sorted(recs, key = lambda x: x['min_x'])

        for block_idx, block in enumerate(sizes):
            for key, value in block.items():
                recs[block_idx][f'block_{key}'] = value

        offset_data[x_offset] = recs

    plt.tight_layout()
    plt.show()

    data_list = []
    for key_1, rect_list_1 in offset_data.items():
        c_list = [key_1]
        for key_2, rect_list_2 in offset_data.items():
            zipped = list(zip(rect_list_1, rect_list_2))
            max_width_difference = np.max(list(map(lambda pair: pair[0]['width'] - pair[1]['width'], zipped)))
            max_height_difference = np.max(list(map(lambda pair: pair[0]['height'] - pair[1]['height'], zipped)))
            average_width_difference = np.average(list(map(lambda pair: pair[0]['width'] - pair[1]['width'], zipped)))
            average_height_difference = np.average(
                list(map(lambda pair: pair[0]['height'] - pair[1]['height'], zipped)))

            max_width_difference = round(max_width_difference * 100) / 100
            max_height_difference = round(max_height_difference * 100) / 100
            average_width_difference = round(average_width_difference * 100) / 100
            average_height_difference = round(average_height_difference * 100) / 100

            c_list.append(f"({max_width_difference}, {max_height_difference}) \n"
                          f"({average_width_difference}, {average_height_difference})")
            # c_list.append(f"{list(map(lambda pair: pair[0]['width'] - pair[1]['width'], zipped))} \n"
            #               f"{list(map(lambda pair: pair[0]['height'] - pair[1]['height'], zipped))}")

        data_list.append(c_list)

    print(tabulate(data_list, headers = [' '] + list(offset_data.keys())))


def visualize_encoding_data(viz_recs = True, direction = 'vertical', stacked = 1, x_offset = 0, y_offset = 0, create_screen_shot = False, diff_materials = True):

    level_img_encoder = LevelImgEncoder()
    level_img_decoder = LevelImgDecoder()
    level_visualizer = LevelVisualizer()
    elements, sizes = create_element_for_each_block(direction, stacked, x_offset, y_offset, diff_materials)

    # Create the images
    dot_img, dot_time = Utils.timeit(level_img_encoder.create_dot_img, args = {'element_list': elements})
    calc_img, calc_img_time = Utils.timeit(level_img_encoder.create_calculated_img, args = {'element_list': elements})
    calc_img_no_size_check, calc_img_no_size_time = Utils.timeit(level_img_encoder.create_calculated_img_no_size_check,
                                                                 args = {'element_list': elements})

    fig = plt.figure(figsize = (6, 7), dpi = 300)
    fig.suptitle('Encoding Data Level')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows = 4, ncols = 1)

    if create_screen_shot:
        axs = subfigs[0].subplots(nrows = 1, ncols = 2)
        game_manager = GameManager(conf = config)
        game_manager.start_game()

        axs[1].set_title('Screenshot')
        game_manager.switch_to_level_elements(elements)
        img = game_manager.get_img()
        axs[1].imshow(img)
        ax = axs[0]
    else:
        ax = subfigs[0].subplots(nrows = 1, ncols = 1)

    level_visualizer.create_img_of_structure(elements, scaled = False, ax = ax)
    ax.set_title(f'No Size Reduction')

    iter_data = zip(subfigs[1:], [
        {'img': dot_img, 'name': 'Dot Encoding', 'time': dot_time},
        {'img': calc_img, 'name': 'Calculated Encoding', 'time': calc_img_time},
        {'img': calc_img_no_size_check, 'name': 'Calculated Encoding No Size Checks', 'time': calc_img_no_size_time},
    ])

    for subfig, data in iter_data:
        axs = subfig.subplots(nrows = 1, ncols = 2 if viz_recs else 1).flatten()

        if viz_recs:
            subfig.suptitle(data['name'] + f' time: {data["time"]}')
        else:
            axs[0].set_title(data['name'] + f' time: {data["time"]}')

        current_img = np.pad(data['img'], 2)
        axs[0].imshow(current_img)
        if viz_recs:
            level_img_decoder.visualize_rectangle(level_img = current_img, material_id = 1, ax = axs[1])

    plt.subplots_adjust(
        left = 0.2,
        bottom = 0.1,
        right = 0.85,
        top = 0.85,
        wspace = 0.2,
        hspace = 0.35
    )
    plt.show()

    if create_screen_shot:
        game_manager.stop_game()


def create_decoding_data():
    level_img_encoder = LevelImgEncoder()
    level_img_decoder = LevelImgDecoder()

    data_dict = dict()

    elements, sizes = create_element_for_each_block()
    level_rep = level_img_encoder.create_calculated_img(elements)

    recs = level_img_decoder.get_rectangles(np.pad(level_rep, 0))
    recs = sorted(recs, key = lambda x: x['min_x'])
    for block_idx, block in enumerate(sizes):
        for key, value in block.items():
            recs[block_idx][f'block_{key}'] = value

    recs = sorted(recs, key = lambda x: (-x['area'], -x['width']))

    data_dict['resolution'] = Constants.resolution
    for rec_idx, rec_data in enumerate(recs):
        data_dict[rec_idx] = dict(
            name = rec_data['block_name'],
            rotated = rec_data['block_rotated'],
            area = rec_data['area'],
            poly_area = rec_data['contour_area'],
            width = rec_data['width'],
            height = rec_data['height'],
            dim = (rec_data['width'], rec_data['height']),
            orig_dim = (rec_data['block_orig_width'], rec_data['block_orig_height'])
        )

    pickle_data = config.get_encoding_data(f"encoding_res_{Constants.resolution}")

    if type(pickle_data) != str:
        ic(pickle_data)
    else:
        with open(pickle_data, 'w') as f:
            f.write(json.dumps(data_dict, indent=4))


def print_rect_data(recs):
    np.set_printoptions(threshold = sys.maxsize, linewidth = sys.maxsize)
    print(tabulate([[
        c_dict['block_name'],
        c_dict['area'],
        c_dict['contour_area'],
        c_dict['area'] - c_dict['contour_area'],
        c_dict['block_area'],
        c_dict['height'],
        c_dict['width'],
        c_dict['block_width'],
        c_dict['block_height'],
        c_dict['block_rounded_width'],
        c_dict['block_rounded_height'],
        c_dict['block_rotated']
    ] for c_dict in recs],
        headers = [
            'block_name',
            'area',
            'contour_area',
            'area',
            'block_area',
            'height',
            'width',
            'block_width',
            'block_height',
            'block_rounded_width',
            'block_rounded_height',
            'block_rotated'
        ])
    )

    ic(recs)


if __name__ == '__main__':
    visualize_encoding_data(
        viz_recs = True, direction = 'vertical', stacked = 2, x_offset = 0, y_offset = 0, create_screen_shot = False
    )
    # test_encoding_data(test_dot = False)
    # create_decoding_data()
