import json
import os
import pickle
from copy import copy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from level.LevelReader import LevelReader
from level.LevelUtil import StructureMetaData
from util.Config import Config

def load_data(data_name):
    with open(data_name, 'rb') as f:
        data = pickle.load(f)
    return data


def get_level_from_data(data_key, data_example):
    level_path = Config.get_instance().get_data_train_path('generated/single_structure')
    level_names = list(Path(level_path).glob("*"))
    parsed_level_names = list(map(lambda x: x.name[:-4], level_names))
    key_name = data_key.split(os.sep)[-1][:-2]

    position = parsed_level_names.index(key_name)
    if position == -1:
        return None

    level_reader = LevelReader()
    level = level_reader.parse_level(path = str(level_names[position]), use_platform = True)

    return level


def strip_screenshot_from_data(data: dict, out_file):
    print(data.keys())
    for key in data.keys():
        level_data = data[key]

        del level_data['level_screenshot']

    with open(out_file, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)


def parse_data(data: dict, out_file: str):
    print(len(data.keys()))
    for key in tqdm(data.keys(), total = len(data.keys()), desc = 'Parsing data'):
        level_data = data[key]

        if type(level_data['img_data']) is list:
            level_idx = np.argmax(list(map(lambda x: x.shape[0] * x.shape[1], level_data['img_data'])))
            level_data['img_data'] = level_data['img_data'][level_idx]

        if 'game_data' in level_data:
            level_data['game_data'] = json.loads(level_data['game_data'][1]['data'])[0]

    with open(out_file, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)


def visualize_data(data, start_index = 0, end_index = -1, height_filter = -1, width_filter = -1):
    print(data.keys())
    for key in list(data.keys())[start_index:end_index if end_index != -1 else len(data)]:
        level_data = data[key]
        level_data_shape = level_data['img_data'][0].shape

        if height_filter != -1 and level_data_shape[0] < height_filter:
            continue

        if width_filter != -1 and level_data_shape[1] < width_filter:
            continue

        if 'meta_data' in level_data:
            print(level_data['meta_data'])

        if 'game_data' in level_data:
            print(level_data['game_data'])

        if 'level_screenshot' in level_data:
            plt.imshow(level_data['level_screenshot'])
            plt.show()

        print(level_data['img_data'][0].shape)
        plt.imshow(level_data['img_data'][0])
        plt.show()


def get_max_shape_size(data: dict):
    max_height = -10000
    max_width = -10000

    max_value = -10000
    min_value = 10000

    for key in data.keys():
        level_data = data[key]

        img_data = level_data['img_data']
        level_img_shape = img_data.shape

        max_height = max(max_height, level_img_shape[0])
        max_width = max(max_width, level_img_shape[1])

        max_value = max(max_value, img_data.max())
        min_value = min(min_value, img_data.min())

    print(f'Height {max_height} Width: {max_width}')
    print(f'MaxVal {max_value} MinVal: {min_value}')  #

    return max_height, max_width, max_value, min_value


def visualize_shape(data: dict, max_height = 86, max_width = 212):
    label_height = range(0, max_height + 1)
    label_width = range(0, max_width + 1)

    height_count_dict = {i: 0 for i in label_height}
    width_count_dict = {i: 0 for i in label_width}

    for key in data.keys():
        level_data = data[key]

        level_img_shape = level_data['img_data'].shape
        height_count_dict[level_img_shape[0]] += 1
        width_count_dict[level_img_shape[1]] += 1

    fig, axs = plt.subplots(1, 2, dpi = 300, figsize = (9, 4))

    axs[0].bar(label_height, list(height_count_dict.values()))
    axs[0].set_title('Height distribution')
    axs[0].set_ylabel('Amount of levels')
    axs[0].set_xlabel('Height of Levels')

    axs[1].bar(label_width, list(width_count_dict.values()))
    axs[1].set_title('Width distribution')
    axs[1].set_ylabel('Amount of levels')
    axs[1].set_xlabel('Width of Levels')

    fig.suptitle('Height distribution', fontsize = 16)
    plt.show()


def view_files_with_prop(data, amount = -1, min_width = -1, max_width = -1, min_height = -1, max_height = -1):
    counter = 0
    level_list = []
    data_idx = dict()
    for key in list(data.keys()):

        level_data = data[key]
        level_data_shape = level_data['img_data'].shape

        level = get_level_from_data(key, level_data)

        if min_width != -1 and not level_data_shape[1] >= min_width or \
                max_width != -1 and not level_data_shape[1] <= max_width or \
                min_height != -1 and not level_data_shape[0] >= min_height or \
                max_height != -1 and not level_data_shape[0] <= max_height:
            continue

        if level in level_list:
            pos = level_list.index(level)
            ref_level_data = data_idx[pos]

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(level_data['img_data'])
            axs[0].set_title("Current Level")

            axs[1].imshow(ref_level_data['img_data'])
            axs[1].set_title("Ref Level")

            plt.title("Found duplicate")
            plt.show()
        else:
            plt.imshow(level_data['img_data'])
            plt.title(level_data_shape)
            plt.show()

        level_list.append(level)
        data_idx[len(level_list) - 1] = level_data

        counter += 1
        if amount != -1 and counter > amount:
            break

    print(f'Showed {counter} files')


def filter_level(data, out_file, plt_show = 0, skip_value = 5):
    comp_data_list = []
    temp_data = []

    plt_counter = 0
    remove_counter = 0

    out_dict = dict()
    for key in tqdm(list(data.keys()), total = len(data.keys()), desc = 'Filtering data'):
        level_data = data[key]
        meta_data: StructureMetaData = level_data['meta_data']
        if meta_data.block_amount <= 0:
            remove_counter += 1
            continue

        continue_flag = False

        for idx, comp_data in enumerate(comp_data_list):

            if comp_data['meta_data'] == meta_data:
                remove_counter += 1
                if plt_counter < plt_show and plt_counter % skip_value == 0:
                    comp_level = temp_data[idx]
                    fig, axs = plt.subplots(1, 2, dpi = 100)
                    # fig.suptitle("Found same metadata")

                    level_data_shape = comp_level.shape
                    axs[0].set_title(str(level_data_shape))
                    axs[0].imshow(comp_level)

                    level_data_shape = level_data['img_data'].shape
                    axs[1].set_title(str(level_data_shape))
                    axs[1].imshow(level_data['img_data'])
                    plt.show()
                    plt_counter += 1

                continue_flag = True
                break

            if level_data['img_data'].shape == comp_data['level_rep'].shape:
                orig_only_ones = np.zeros_like(level_data['img_data'])
                comp_only_ones = np.zeros_like(comp_data['level_rep'])

                orig_only_ones[level_data['img_data'] > 0] = 1
                comp_only_ones[comp_data['level_rep'] > 0] = 1

                negativ = orig_only_ones - comp_only_ones
                if np.alltrue(negativ == 0):
                    remove_counter += 1
                    if plt_show and plt_counter < plt_show and plt_counter % skip_value == 0:
                        fig, axs = plt.subplots(1, 2, dpi = 100)
                        fig.suptitle("Same shape")
                        axs[0].imshow(level_data['img_data'])
                        axs[1].imshow(comp_data['level_rep'])
                        plt.show()
                    continue_flag = True
                    break

        if continue_flag:
            continue

        temp_data.append(level_data['img_data'])
        comp_data_list.append(dict(meta_data = meta_data, level_rep = level_data['img_data']))
        out_dict[key] = level_data

    print(f'Removed {remove_counter}')

    with open(out_file, 'wb') as handle:
        pickle.dump(out_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)


def unify_level(data_dict, out_file, show_plt = False):
    if show_plt:
        fig, axs = plt.subplots(1, 3, dpi = 100, figsize = (12, 4))

    height_groups = dict()
    # Group level by height
    for key in tqdm(data_dict.keys(), total = len(data_dict.keys()), desc = 'Unifying data'):
        level_data = data_dict[key]

        level_img_shape = level_data['img_data'].shape
        height = level_img_shape[0]
        if height in height_groups:
            height_groups[height].append((key, level_data))
        else:
            height_groups[height] = [(key, level_data)]

    if show_plt:
        # merge close groups
        axs[0].bar(list(height_groups.keys()), list(map(len, height_groups.values())))
        axs[0].set_title(f'Before merging: {sum(list(map(len, height_groups.values())))}')
        axs[0].set_ylabel('Amount of levels')
        axs[0].set_xlabel('Height of Levels')

    heights = sorted(list(height_groups.keys()))

    out_dict = dict()
    temp_list = []
    for height_idx in range(len(heights) - 1):
        height = heights[height_idx]

        temp_list += height_groups[height]

        if len(temp_list) > 50:
            out_dict[height] = copy(temp_list)
            temp_list = []

    if show_plt:
        axs[1].bar(list(out_dict.keys()), list(map(len, out_dict.values())))
        axs[1].set_title(f'After group merging: {sum(list(map(len, out_dict.values())))}')
        axs[1].set_ylabel('Amount of levels')
        axs[1].set_xlabel('Height of Levels')

    avg_height = round(np.average(list(map(len, map(list, out_dict.values())))))
    for key, amount in out_dict.items():
        out_dict[key] = out_dict[key][:avg_height]

    if show_plt:
        axs[2].bar(list(out_dict.keys()), list(map(len, out_dict.values())))
        axs[2].set_title(f'After Unifying: {sum(list(map(len, out_dict.values())))}')
        axs[2].set_ylabel('Amount of levels')
        axs[2].set_xlabel('Height of Levels')

        plt.show()

    save_dict = dict()
    for height, level_list in out_dict.items():
        for key, level in level_list:
            save_dict[key] = level

    print(f'Amount of levels after unifying heights: {len(save_dict.keys())}')

    with open(out_file, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)


def create_filtered_dataset(data_set_name, show_plt = False):
    config = Config.get_instance()
    file_name = config.get_data_set(folder_name = data_set_name, file_name = 'original')
    # data_dict = load_data("../resources/data/pickles/level_data_with_screenshot")
    # strip_screenshot_from_data(data_dict)
    data_dict = load_data(file_name)
    file_name = config.get_data_set(folder_name = data_set_name, file_name = "parsed")
    parse_data(data_dict, file_name)

    data_dict = load_data(file_name)
    out_file_filtered = config.get_data_set(
        folder_name = data_set_name,
        file_name = "filtered"
    )
    filter_level(data_dict, out_file = out_file_filtered, plt_show = show_plt)
    data_set = config.get_data_set(folder_name = data_set_name, file_name = f'filtered')
    data_dict = load_data(data_set)
    out_file_filtered = config.get_data_set(folder_name = data_set_name,
                                            file_name = "unified")
    unify_level(data_dict, out_file = out_file_filtered)


def filter_dataset(data_set_file):

    data_dict = load_data(data_set_file)
    parsed_file_name = data_set_file.replace('original', 'parsed')
    parse_data(data_dict, parsed_file_name)

    data_dict = load_data(parsed_file_name)
    out_file_filtered = data_set_file.replace('original', 'filtered')
    filter_level(data_dict, out_file = out_file_filtered, plt_show = 0)

    data_dict = load_data(out_file_filtered)
    unified_filtered = data_set_file.replace('original', 'unified')
    unify_level(data_dict, out_file = unified_filtered)

    return unified_filtered


if __name__ == '__main__':
    create_filtered_dataset(data_set_name = 'simple_encoding', show_plt = True)

    # root_pickle_file = 'original'
    # config = Config.get_instance()
    # file_name = config.get_data_set(folder_name = data_set_name, file_name = root_pickle_file)

    # file_filtered = config.get_pickle_file(f"{root_pickle_file}_unified")
    # data_dict = load_data(file_filtered)
    # max_height, max_width, max_value, min_value = get_max_shape_size(data_dict)
    # #
    # data_dict = load_data(file_filtered)
    # visualize_shape(data_dict, max_height, max_width)
    # #
    # data_dict = load_data(file_filtered)
    # view_files_with_prop(data_dict, amount = -1, min_width = 0, max_width = 30, min_height = 0, max_height = 13)

    # data_dict = load_data(file_name)
    # visualize_data(data_dict, start_index = 0, end_index = 10, width_filter = -1)

    # data_dict = load_data("../resources/data/pickles/level_data_with_screenshot")
    # strip_screenshot_from_data(data_dict)
