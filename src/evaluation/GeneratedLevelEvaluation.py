import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from icecream import ic

from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from level.LevelReader import LevelReader
from util.Config import Config

mpl.rcParams["savefig.format"] = 'pdf'


def create_screenshots():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output_2")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)

    filtered_levels = list(filter(lambda element: True if 'is_stable' in element['data'] else False, data_output_list))

    # Number of levels with heigehst block
    # stable_levels = list(filter(lambda element: element['data']['is_stable'], filtered_levels))
    block_destroyed = list(filter(lambda element: element['data']['totalBlocksDestroyed'] == 0, filtered_levels))
    # low_damage_levels = list(filter(lambda element: element['data']['damage'] < 20, filtered_levels))
    # unstable_levels = list(filter(lambda element: not element['data']['is_stable'] if 'is_stable' in element['data'] else False,data_output_list))

    to_be_sorted_levels = block_destroyed

    # block_destroyed = list(filter(lambda element: element['data']['height'] > 4 and element['data']['width'] < 3, to_be_sorted_levels))

    # sorted_by_block_amount = sorted(to_be_sorted_levels, key = lambda element: (-element['data']['total']), reverse = False)
    sorted_by_block_amount = sorted(to_be_sorted_levels, key = lambda element: (-element['data']['pig_amount']),
                                    reverse = False)
    # sorted_by_lowest_structure = sorted(to_be_sorted_levels, key = lambda element: (element['data']['height']), reverse = False)

    screenshot_levels = sorted_by_block_amount
    # random.shuffle(screenshot_levels)

    print(f"Level amount: {len(screenshot_levels)}")

    # Create pictures
    game_connection = GameConnection(conf = config, port = 9001)
    game_manager: GameManager = GameManager(config, game_connection = game_connection)
    game_manager.start_game()

    amount_of_pictures = 50

    for idx in range(amount_of_pictures):
        fig, ax = plt.subplots(1, 1)
        level = screenshot_levels[idx]['level']

        print(screenshot_levels[idx]['data'])

        game_manager.create_levels_xml_file([level], store_level_name = f"stable_level_{idx}")
        game_manager.copy_game_levels(
            level_path = config.get_data_train_path(folder = 'temp'),
            rescue_level = False
        )
        game_manager.select_level(4)
        screenshot = game_manager.get_img()
        ax.imshow(screenshot)

        file_name = config.get_quality_search_folder(f'{idx}_pig_amount_new')
        plt.savefig(file_name)

        orig_img = screenshot_levels[idx]['gan_output']
        orig_img = orig_img * 2 - 1
        if str(type(orig_img)) != '<class \'numpy.ndarray\'>':
            orig_img = orig_img.numpy()

        layer_amount = orig_img.shape[-1]
        if layer_amount > 1:
            for layer in range(1, layer_amount):
                orig_img[orig_img[:, :, layer] > 0, layer] += layer + 1

        viz_img = np.max(orig_img, axis = 2)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(viz_img)
        file_name = config.get_quality_search_folder(f'{idx}_pig_amount_new_orig')
        plt.savefig(file_name)

    game_manager.stop_game()

def create_stable_levels():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output_2")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)

    filtered_levels = list(filter(lambda element: True if 'is_stable' in element['data'] else False, data_output_list))

    block_destroyed = list(filter(lambda element: element['data']['totalBlocksDestroyed'] == 0, filtered_levels))

    game_connection = GameConnection(conf = config, port = 9001)
    game_manager: GameManager = GameManager(config, game_connection = game_connection)

    levels = []
    names = []
    for idx in range(200):
        levels.append(block_destroyed[idx]['level'])
        names.append(f"level_{idx + 4}")

    game_manager.create_levels_xml_file(levels, store_level_name = names)


def create_bar_chart():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output_2")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)

    filtered_levels = list(filter(lambda element: True if 'is_stable' in element['data'] else False, data_output_list))

    # Number of levels with heigehst block
    # stable_levels = list(
    #     filter(lambda element: element['data']['damage'] <= 0 if 'damage' in element['data'] else False,
    #            filtered_levels))
    # unstable_levels = list(
    #     filter(lambda element: element['data']['damage'] > 0 if 'damage' in element['data'] else False,
    #            data_output_list))

    # stable_levels = list(
    #     filter(lambda element: element['data']['totalBlocksDestroyed'] == 0 if 'totalBlocksDestroyed' in element[
    #         'data'] else False,
    #            filtered_levels))
    # unstable_levels = list(
    #     filter(lambda element: element['data']['totalBlocksDestroyed'] > 0 if 'totalBlocksDestroyed' in element[
    #         'data'] else False,
    #            data_output_list))

    # stable_levels = list(
    #     filter(lambda element: element['data']['is_stable'] if 'is_stable' in element['data'] else False,
    #            filtered_levels))
    # unstable_levels = list(
    #     filter(lambda element: not element['data']['is_stable'] if 'is_stable' in element['data'] else False,
    #            data_output_list))

    filtered_levels = list(filter(lambda element: True if 'is_stable' in element['data'] else False, data_output_list))
    stable_levels = list(filter(lambda element: element['data']['height'] <= 3, filtered_levels))
    unstable_levels = list(filter(lambda element: element['data']['height'] > 3, filtered_levels))

    print(len(stable_levels))
    print(len(unstable_levels))

    sim_data = ['damage', 'is_stable', 'woodBlockDestroyed', 'iceBlockDestroyed', 'stoneBlockDestroyed',
                'totalBlocksDestroyed']

    meta_data_options = [
        'min_x', 'max_x', 'min_y', 'max_y', 'height', 'width', 'block_amount',
        'height_width_ration', 'platform_amount', 'pig_amount', 'special_block_amount', 'total', 'ice_blocks',
        'stone_blocks', 'wood_blocks'
    ]
    collected_data = sim_data + meta_data_options
    show_in_graph = {k: True for k in collected_data}
    show_in_graph['platform_amount'] = False
    show_in_graph['is_stable'] = False
    show_in_graph['pig_amount'] = False
    show_in_graph['special_block_amount'] = False
    show_in_graph['woodBlockDestroyed'] = False
    show_in_graph['iceBlockDestroyed'] = False
    show_in_graph['stoneBlockDestroyed'] = False
    show_in_graph['ice_blocks'] = False
    show_in_graph['ice_blocks'] = False
    show_in_graph['stone_blocks'] = False
    show_in_graph['wood_blocks'] = False
    show_in_graph['min_x'] = False
    show_in_graph['max_x'] = False
    show_in_graph['min_y'] = False
    show_in_graph['max_y'] = False
    show_in_graph['total'] = False

    collected_data_label_map = dict(
        damage = 'Damage',
        is_stable = 'Is Stable',
        woodBlockDestroyed = '# Wood Blocks Destroyed',
        iceBlockDestroyed = '# Ice Blocks Destroyed',
        stoneBlockDestroyed = '# Stone Blocks Destroyed',
        totalBlocksDestroyed = '# Total Blocks Destroyed',
        height_width_ration = 'Height width ratio',
        min_x = 'Min X',
        max_x = 'Max X',
        min_y = 'Min Y',
        max_y = 'Max Y',
        height = 'Height',
        width = 'Width',
        block_amount = '# Block',
        platform_amount = '# Platform',
        pig_amount = '# Pig',
        special_block_amount = '# Special Block',
        total = '# Total Elements',
        ice_blocks = '# ice block',
        stone_blocks = '# stone block',
        wood_blocks = '# wood block'
    )

    data = []

    for data_name, data_source in [('Levels smaller than 3 ', stable_levels), ('Levels Taller than 3', unstable_levels)]:
        collected_data_labels = []
        y_data = []
        for collected_data_key in collected_data:
            if not show_in_graph[collected_data_key]:
                continue

            collected_data_labels.append(collected_data_label_map[collected_data_key])

            if collected_data_key == 'height_width_ration':
                avg_value_list = list(
                    map(lambda element: element['data']['height'] / element['data']['width'], data_source))
            else:
                avg_value_list = list(map(lambda element: element['data'][collected_data_key], data_source))

            cleared = [i for i in avg_value_list if i is not None and i != -1]
            if type(cleared[0]) == bool:
                print(collected_data_key)
                y_data.append(len([True for value in avg_value_list if value is True]) / len(
                    [True for value in avg_value_list if value is False]) * 100)
            else:
                y_data.append(np.average(cleared))

        data.append(
            go.Bar(name = data_name, x = collected_data_labels, y = y_data, text = np.round(y_data, decimals = 2)))

    current_figure = go.Figure(data = data)
    current_figure.update_layout(legend = dict(
        orientation = "h",
        yanchor = "bottom",
        y = 1.02,
        xanchor = "right",
        x = 1
    ))
    current_figure.update_layout(
        barmode = 'group',
        font = dict(
            size = 18,
        ))
    # current_figure.show()

    img_bytes = current_figure.to_image(format = "pdf", width = 1200, height = 600, scale = 2)
    f = open("height_compare.pdf", "wb")
    f.write(img_bytes)
    f.close()


def print_level_characteristics():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output_2")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)

    filtered_levels = list(filter(lambda element: True if 'is_stable' in element['data'] else False, data_output_list))
    low_profile = list(filter(lambda element: element['data']['height'] <= 3, filtered_levels))
    high_profile = list(filter(lambda element: element['data']['height'] > 3, filtered_levels))

    low_profile_stable_levels = list(filter(lambda element: element['data']['totalBlocksDestroyed'] == 0, low_profile))
    low_profile_unstable_levels = list(filter(lambda element: element['data']['totalBlocksDestroyed'] > 0, low_profile))

    high_profile_stable_levels = list(
        filter(lambda element: element['data']['totalBlocksDestroyed'] == 0, high_profile))
    high_profile_unstable_levels = list(
        filter(lambda element: element['data']['totalBlocksDestroyed'] > 0, high_profile))

    print(len(low_profile_stable_levels))
    print(len(low_profile_unstable_levels))
    print(len(low_profile_stable_levels) / (len(low_profile_stable_levels) + len(low_profile_unstable_levels)))

    print(len(high_profile_stable_levels))
    print(len(high_profile_unstable_levels))

    print(len(high_profile_stable_levels) / (len(high_profile_stable_levels) + len(high_profile_unstable_levels)))



def block_type_frequencies():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output_2")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)

    filtered_levels = list(filter(lambda element: True if 'is_stable' in element['data'] else False, data_output_list))

    stable_levels = list(filter(lambda element: element['data']['totalBlocksDestroyed'] == 0, filtered_levels))
    unstable_levels = list(filter(lambda element: element['data']['totalBlocksDestroyed'] > 0, filtered_levels))

    # go over each level and map to the blocks used in the level
    stable_blocks = defaultdict(list)
    unstable_blocks = defaultdict(list)
    generally_used_blocks = defaultdict(list)

    for level in stable_levels:
        block_distro = defaultdict(int)
        for block in level['level'].blocks:
            block_def = block.type
            if 'Rect' in block_def:
                block_def += '_vertical' if block.is_vertical else '_horizontal'

            block_distro[block_def] += 1

        for key in block_distro.keys():
            stable_blocks[key].append(block_distro[key])

    for level in unstable_levels:
        block_distro = defaultdict(int)
        for block in level['level'].blocks:
            block_def = block.type
            if 'Rect' in block_def:
                block_def += '_vertical' if block.is_vertical else '_horizontal'

            block_distro[block_def] += 1

        for key in block_distro.keys():
            unstable_blocks[key].append(block_distro[key])


    for level in filtered_levels:
        block_distro = defaultdict(int)
        for block in level['level'].blocks:
            block_def = block.type
            if 'Rect' in block_def:
                block_def += '_vertical' if block.is_vertical else '_horizontal'

            block_distro[block_def] += 1

        for key in block_distro.keys():
            generally_used_blocks[key].append(block_distro[key])

    stable_blocks_avg_std = dict()
    unstable_blocks_avg_std = dict()
    generally_used_blocks_avg_std = dict()

    for key in stable_blocks.keys():
        stable_blocks_avg_std[key] = dict( avg = np.average(stable_blocks[key]), std = np.std(stable_blocks[key]))

    for key in unstable_blocks.keys():
        unstable_blocks_avg_std[key] = dict( avg = np.average(unstable_blocks[key]), std = np.std(unstable_blocks[key]))

    for key in generally_used_blocks.keys():
        generally_used_blocks_avg_std[key] = dict( avg = np.average(generally_used_blocks[key]), std = np.std(generally_used_blocks[key]))

    # ic(stable_blocks)
    # ic(unstable_blocks)
    #
    # stable_blocks_normalized = dict()
    # unstable_blocks_normalized = dict()

    # # normalize the data by the total number of levels in each category
    # for key in stable_blocks.keys():
    #     stable_blocks_normalized[key] = stable_blocks[key] / len(stable_levels)
    #
    # for key in unstable_blocks.keys():
    #     unstable_blocks_normalized[key] = unstable_blocks[key] / len(unstable_levels)


    data_dicts = dict(
        stable_blocks = stable_blocks_avg_std,
        unstable_blocks = unstable_blocks_avg_std,
        generally_used_blocks = generally_used_blocks_avg_std
    )
    #
    # data_dicts['stable_blocks_normalized'] = stable_blocks_normalized
    # data_dicts['unstable_blocks_normalized'] = unstable_blocks_normalized

    print(
        json.dumps(data_dicts, indent = 4, sort_keys = True)
    )

    current_figure = go.Figure()

    current_figure.add_trace(go.Bar(
        x = list(stable_blocks_avg_std.keys()),
        y = [value['avg'] for value in stable_blocks_avg_std.values()],
        name = 'Stable Levels'
    ))
    current_figure.add_trace(go.Bar(
        x = list(unstable_blocks_avg_std.keys()),
        y = [value['avg'] for value in unstable_blocks_avg_std.values()],
        name = 'Unstable Levels'
    ))
    current_figure.add_trace(go.Bar(
        x = list(generally_used_blocks_avg_std.keys()),
        y = [value['avg'] for value in generally_used_blocks_avg_std.values()],
        name = 'All Levels'
    ))

    current_figure.update_layout(legend = dict(
        orientation = "h",
        yanchor = "bottom",
        y = 1.02,
        xanchor = "right",
        x = 1
    ))
    current_figure.update_layout(
        barmode = 'group',
        font = dict(
            size = 18,
        ))
    # current_figure.show()
    img_bytes = current_figure.to_image(format = "pdf", width = 1200, height = 600, scale = 2)
    f = open("block_type_distro.pdf", "wb")
    f.write(img_bytes)
    f.close()


def calculate_density():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output_2")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)

    filtered_levels = list(filter(lambda element: True if 'is_stable' in element['data'] else False, data_output_list))

    stable_levels = list(filter(lambda element: element['data']['totalBlocksDestroyed'] == 0, filtered_levels))
    unstable_levels = list(filter(lambda element: element['data']['totalBlocksDestroyed'] > 0, filtered_levels))


    def calc_density(level_list):
        density_list = []
        for level in level_list:
            width = level['data']['width']
            height = level['data']['height']
            area = width * height

            occupied_area = 0
            for block in level['level'].blocks:
                occupied_area += block.width * block.height

            density_list.append(occupied_area / area)

        return density_list

    stable_density = calc_density(stable_levels)
    unstable_density = calc_density(unstable_levels)
    overall_density = calc_density(filtered_levels)

    print(
        json.dumps(dict(
            stable_density = dict(avg = np.average(stable_density), std = np.std(stable_density)),
            unstable_density = dict(avg = np.average(unstable_density), std = np.std(unstable_density)),
            overall_density = dict(avg = np.average(overall_density), std = np.std(overall_density))
        ), indent = 4, sort_keys = True)
    )

    fig = go.Figure()
    fig.add_trace(go.Box(y = stable_density, quartilemethod = "inclusive", name = "Stable"))
    fig.add_trace(go.Box(y = unstable_density, quartilemethod = "inclusive", name = "Unstable"))
    fig.add_trace(go.Box(y = overall_density, quartilemethod = "inclusive", name = "Overall"))
    fig.show()


def print_time_data():
    config = Config.get_instance()
    time_data_output = config.get_grid_search_file(f"time_data_output_2")
    if Path(time_data_output).exists():
        with open(time_data_output, 'rb') as f:
            time_data = pickle.load(f)

    print(time_data)

    for c_time_data in time_data:
        print(c_time_data)


def save_to_file():
    config = Config.get_instance()
    quantitative_search_output = config.get_grid_search_file(f"quantitative_search_output_2")
    if Path(quantitative_search_output).exists():
        with open(quantitative_search_output, 'rb') as f:
            data_output_list = pickle.load(f)


    # extracted_data = {
    #     f'level_{i}' : data_output_list[i]['data'] for i in range(0, len(data_output_list))
    # }
    #
    # # write to json file
    # with open(config.get_fids_file(f"main_set_data"), 'wb') as f:
    #     created_data = json.dumps(extracted_data, indent = 4, sort_keys = True)
    #     f.write(created_data.encode('utf-8'))

    level_reader = LevelReader()

    for i in range(0, len(data_output_list)):
        elements = data_output_list[i]['level'].get_used_elements()
        level = level_reader.create_level_from_structure(elements, red_birds = True)
        level_path = config.good_generated_level(f'level_{i}')
        level_reader.write_xml_file(level, level_path)


if __name__ == '__main__':
    # print_level_characteristics()
    # block_type_frequencies()
    #calculate_density()
    save_to_file()
    # print_time_data()
    # create_bar_chart()
    # create_screenshots()
    # create_stable_levels()