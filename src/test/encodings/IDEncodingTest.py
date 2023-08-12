import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from converter.to_img_converter.LevelIdImgDecoder import LevelIdImgDecoder
from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from data_scripts.CreateEncodingData import create_element_for_each_block
from level.Level import Level
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment


def one_element_with_test_level(direction = 'vertical', stacked = 3, x_offset = 0, y_offset = 0):
    level_img_encoder = LevelImgEncoder()
    level_img_decoder = LevelIdImgDecoder()
    level_visualizer = LevelVisualizer()

    fig, axs = plt.subplots(1, 3, figsize = (12, 4), dpi = 100)

    elements, sizes = create_element_for_each_block(direction, stacked, x_offset, y_offset, diff_materials = True)
    meta_level = Level.create_level_from_structure(elements)

    level_visualizer.create_img_of_structure(elements, ax = axs[0])

    encoded_img = level_img_encoder.create_one_element_img(elements)

    axs[1].matshow(encoded_img)

    print(np.unique(encoded_img))

    decoded_level = level_img_decoder.decode_level(
        level_img = encoded_img
    )
    level_visualizer.create_img_of_level(decoded_level, ax = axs[2])

    if decoded_level != meta_level:
        print("Decoded level is not the same")

    plt.show()


def test_encoding_ids(direction = 'vertical', stacked = 3, x_offset = 0, y_offset = 0):
    level_img_encoder = LevelImgEncoder()
    level_img_decoder = LevelIdImgDecoder()

    test_environment = TestEnvironment()
    level = test_environment.get_level(0)

    fig, axs = plt.subplots(1, 2, figsize = (12, 4), dpi = 100)

    test_environment.level_visualizer.create_img_of_structure(level.get_used_elements(), ax = axs[0])
    axs[0].set_title('Original Level')

    encoded_img = level_img_encoder.create_one_element_img(level.get_used_elements())

    # encoded_img[encoded_img == 0] = -5
    im = axs[1].matshow(encoded_img)
    axs[1].set_title('One Element Encoding')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    plt.show()

    plt.show()

if __name__ == '__main__':
    test_encoding_ids()
