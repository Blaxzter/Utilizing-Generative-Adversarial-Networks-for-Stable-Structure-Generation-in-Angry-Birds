import matplotlib.pyplot as plt

from converter.to_img_converter.LevelImgDecoder import LevelImgDecoder
from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from data_scripts.CreateEncodingData import create_element_for_each_block
from level.Level import Level
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from test.visualization.LevelImgDecoderVisualization import LevelImgDecoderVisualization


def decode_test_level(direction = 'vertical', stacked = 3, x_offset = 0, y_offset = 0):
    elements, sizes = create_element_for_each_block(direction, stacked, x_offset, y_offset, diff_materials = False)

    test_level = Level.create_level_from_structure(elements)
    img_rep = create_encoding(test_level)
    decoded_level = create_decoding(img_rep)

    visualize_encoding_decoding(decoded_level, img_rep, test_level)


def img_encoding_decoding_test(test_with_game = False):
    test_environment = TestEnvironment('generated/single_structure')

    for level_idx, level in test_environment.iter_levels():
        level.print_elements(as_table = True)

        img_rep = create_encoding(level)
        decoded_level = create_decoding(img_rep)

        visualize_encoding_decoding(
            decoded_level = decoded_level,
            img_rep = img_rep,
            original_level = level
        )
        break


def visualize_encoding_decoding(decoded_level, img_rep, original_level):
    fig = plt.figure(constrained_layout = True, figsize = (5, 12), dpi = 100)
    fig.suptitle('Multi Block encoding: Vertical Stacked')

    subfigs = fig.subfigures(nrows = 3, ncols = 1)
    axs = subfigs[0].subplots(1, 2)

    subfigs[0].suptitle('Blocks visualized')
    visualizer = LevelVisualizer(line_size = 1)
    visualizer.create_img_of_level(original_level, use_grid = False, add_dots = False, ax = axs[0])
    visualizer.create_img_of_level(decoded_level, use_grid = False, add_dots = False, ax = axs[1])
    axs[0].set_title('Original')
    axs[1].set_title('Recreated')

    axs = subfigs[1].subplots(1, 2)
    subfigs[1].suptitle('Encoded')
    axs[0].imshow(img_rep)
    re_encoded_elements = create_encoding(decoded_level)
    axs[1].imshow(re_encoded_elements)
    #
    ax = subfigs[2].subplots(1, 1)
    subfigs[1].suptitle('Difference in encoding')
    ax.imshow(img_rep - re_encoded_elements)
    fig.tight_layout()
    plt.show()


def get_rectangles():
    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(0)
    img_rep = create_encoding(level)
    create_rectangles(img_rep)
    visualize_decoding(img_rep)


def create_rectangles(img_rep):
    level_img_decoder = LevelImgDecoderVisualization()
    # decoded_level = level_img_decoder.visualize_contours(img_rep)
    # level_img_decoder.visualize_rectangles(img_rep, material_id = 1)
    level_img_decoder.visualize_rectangles(img_rep, material_id = 2)
    # level_img_decoder.visualize_rectangles(img_rep, material_id = 3)


def visualize_decoding(img_rep):
    level_img_decoder = LevelImgDecoderVisualization()
    # level_img_decoder.visualize_one_decoding(img_rep, material_id = 1)
    level_img_decoder.visualize_one_decoding(img_rep, material_id = 2)
    # level_img_decoder.visualize_rectangles(img_rep, material_id = 3)


def create_encoding(level):
    level_img_encoder = LevelImgEncoder()
    elements = level.get_used_elements()
    return level_img_encoder.create_calculated_img(elements)


def create_decoding(level_representation):
    level_img_decoder = LevelImgDecoder()
    return level_img_decoder.decode_level(level_representation)


def visualize_encoding(level):
    level_img_encoder = LevelImgEncoder()

    elements = level.get_used_elements()

    encoded_dots = level_img_encoder.create_dot_img(elements)
    encoded_calculated = level_img_encoder.create_calculated_img(elements)

    fig, axs = plt.subplots(1, 2, dpi = 300, figsize = (12, 6))
    axs[0].imshow(encoded_calculated)
    axs[0].set_title("Calculated")

    axs[1].imshow(encoded_dots)
    axs[1].set_title("Through Dots")

    plt.tight_layout()
    plt.show()

    return encoded_calculated


if __name__ == '__main__':
    decode_test_level()
    # img_encoding_decoding_test()
    # get_rectangles()
