import matplotlib.pyplot as plt
import numpy as np

from converter.to_img_converter.LevelImgDecoder import LevelImgDecoder
from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from data_scripts.CreateEncodingData import create_element_for_each_block
from test.TestEnvironment import TestEnvironment


def img_encoding_test():
    test_environment = TestEnvironment('generated/single_structure')

    for level_idx, level in test_environment.iter_levels():
        compare_encodings(level, test_environment)
        if level_idx > 0:
            break


def create_encodings():
    test_environment = TestEnvironment('generated/single_structure')

    fig, axs = plt.subplots(2, 6, dpi = 300)

    level_img_encoder = LevelImgEncoder()
    for (level_idx, level), ax in zip(test_environment.iter_levels(0, 12), axs.flatten()):
        encoded_calculated_orig = level_img_encoder.create_calculated_img(level.get_used_elements())
        ax.imshow(encoded_calculated_orig)
        ax.axis('off')

    plt.show()


def create_img_encoding(multilayer = True):
    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(2)

    level_img_encoder = LevelImgEncoder()
    level_img = level_img_encoder.create_calculated_img(level.get_used_elements())

    if multilayer:
        # Plot figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        multilayer_level_img = level_img_encoder.create_multi_dim_img_from_picture(level_img)
        # surf = mlab.surf(multilayer_level_img, warp_scale = "auto")
        # mlab.show()
        ax.voxels(multilayer_level_img)
        plt.show()
    else:
        plt.imshow(level_img)
        plt.show()


def create_one_element_encoding(multilayer = True):
    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(2)

    level_img_encoder = LevelImgEncoder()
    encoded_calculated_orig = level_img_encoder.create_one_element_img(level.get_used_elements(), multilayer)

    if multilayer:
        # Plot figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        ax.voxels(encoded_calculated_orig, edgecolors = 'grey')
        plt.show()
    else:
        plt.imshow(encoded_calculated_orig)
        plt.show()


def compare_encodings(level, test_environment):
    level_img_encoder = LevelImgEncoder()

    elements = level.get_used_elements()

    dot_encoding = level_img_encoder.create_dot_img(elements)
    encoded_calculated_orig = level_img_encoder.create_calculated_img(elements)
    no_size_check = level_img_encoder.create_calculated_img_no_size_check(elements)

    fig = plt.figure(constrained_layout = True, dpi = 100)
    fig.suptitle('Compare different level encodings')

    subfigs = fig.subfigures(nrows = 1, ncols = 4)
    ax = subfigs[0].subplots(nrows = 1, ncols = 1)
    test_environment.level_visualizer.create_img_of_structure(elements, ax = ax)
    ax.set_title("Patches")

    encoded_calculated = encoded_calculated_orig
    if no_size_check.shape != encoded_calculated_orig.shape:
        paddig = (no_size_check.shape[0] - encoded_calculated_orig.shape[0], 0), (
        no_size_check.shape[1] - encoded_calculated_orig.shape[1], 0)
        encoded_calculated = np.pad(encoded_calculated_orig, paddig)

    axs_1 = subfigs[1].subplots(nrows = 2, ncols = 1)
    axs_2 = subfigs[2].subplots(nrows = 2, ncols = 1)
    axs_3 = subfigs[3].subplots(nrows = 2, ncols = 1)

    axs_1[0].imshow(encoded_calculated)
    axs_1[0].set_title("With Size Checks")
    test_environment.level_visualizer.remove_ax_ticks(axs_1[0])

    axs_2[0].imshow(no_size_check)
    axs_2[0].set_title("No Size Checks")
    test_environment.level_visualizer.remove_ax_ticks(axs_2[0])

    axs_3[0].imshow(no_size_check - encoded_calculated)
    axs_3[0].set_title("Calc Difference")
    test_environment.level_visualizer.remove_ax_ticks(axs_3[0])

    encoded_calculated = encoded_calculated_orig
    if dot_encoding.shape != encoded_calculated.shape:
        top_pad = dot_encoding.shape[0] - encoded_calculated_orig.shape[0]
        right_pad = dot_encoding.shape[1] - encoded_calculated_orig.shape[1]
        if top_pad < 0:
            dot_encoding = np.pad(dot_encoding, ((abs(top_pad), 0), (0, 0)))
        else:
            encoded_calculated = np.pad(encoded_calculated_orig, ((abs(top_pad), 0), (0, 0)))

        if right_pad < 0:
            dot_encoding = np.pad(dot_encoding, ((0, 0), (abs(right_pad), 0)))
        else:
            encoded_calculated = np.pad(encoded_calculated_orig, ((0, 0), (abs(right_pad), 0)))

    axs_1[1].imshow(encoded_calculated)
    axs_1[1].set_title("With Size Checks")
    test_environment.level_visualizer.remove_ax_ticks(axs_1[1])

    axs_2[1].imshow(dot_encoding)
    axs_2[1].set_title("Dot Img")
    test_environment.level_visualizer.remove_ax_ticks(axs_2[1])

    axs_3[1].imshow(dot_encoding - encoded_calculated)
    axs_3[1].set_title("Dot Difference")
    test_environment.level_visualizer.remove_ax_ticks(axs_3[1])

    plt.tight_layout()
    plt.show()


def compare_stack_recs(direction = 'vertical', stacked = 4, x_offset = 0, y_offset = 0):
    level_img_encoder = LevelImgEncoder()
    level_img_decoder = LevelImgDecoder()

    data = dict()

    for stack in range(stacked):

        elements, sizes = create_element_for_each_block(direction, stack + 1, x_offset, y_offset, diff_materials = True)

        # Create the images
        calc_img = level_img_encoder.create_calculated_img(elements)
        recs = level_img_decoder.get_rectangles(calc_img, material_id = 2)

        plt.imshow(calc_img)
        plt.show()

        recs = sorted(recs, key = lambda x: x['min_x'])
        rec_data = dict()
        for rec_idx, rec in enumerate(recs):
            rec_data[rec_idx] = (rec['height'], rec['width'])

        data[stack] = rec_data

    for stack in range(1, stacked):
        print_string = ""
        for rec_idx, rec in data[0].items():
            y1, x1 = rec
            y2, x2 = data[stack][rec_idx]

            print_string += f'({y2 - (y1)}, {x2 - x1}) '

        print(print_string)


def compare_material_recs(direction = 'vertical', stacked = 3, x_offset = 0, y_offset = 0):
    level_img_encoder = LevelImgEncoder()
    level_img_decoder = LevelImgDecoder()

    data = dict()

    elements, sizes = create_element_for_each_block(direction, stacked, x_offset, y_offset, diff_materials = True)
    calc_img = level_img_encoder.create_calculated_img(elements)

    for material in range(stacked):
        # Create the images
        recs = level_img_decoder.get_rectangles(calc_img, material_id = material + 1)

        recs = sorted(recs, key = lambda x: x['min_x'])
        rec_data = dict()
        for rec_idx, rec in enumerate(recs):
            rec_data[rec_idx] = (rec['height'], rec['width'])

        data[material] = rec_data

    for material in range(1, stacked):
        print_string = ""
        for rec_idx, rec in data[0].items():
            y1, x1 = rec
            y2, x2 = data[material][rec_idx]

            print_string += f'({y2 - y1}, {x2 - x1}) '

        print(print_string)


if __name__ == '__main__':
    img_encoding_test()
