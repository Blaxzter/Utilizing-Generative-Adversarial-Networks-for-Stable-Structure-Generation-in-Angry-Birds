import numpy as np
from matplotlib import pyplot as plt

from converter.to_img_converter.MultiLayerStackDecoder import MultiLayerStackDecoder
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from test.TestUtils import plot_img


def load_gan_example():
    test_environment = TestEnvironment()
    test_environment.load_test_outputs_of_model('multilayer_with_air')
    gan_output, image_name = test_environment.return_loaded_gan_output_by_idx(0)
    norm_img = (gan_output[0] + 1) / 2
    norm_img[norm_img[:, :, 0] < 0.1, 0] = 0
    img = np.argmax(norm_img, axis = 2)
    plot_img(img)
    multilayer_stack_decoder = MultiLayerStackDecoder()
    created_level = multilayer_stack_decoder.decode(gan_output)
    fig, ax = plt.subplots()
    level_visualizer = LevelVisualizer()
    level_visualizer.create_img_of_structure(created_level.get_used_elements(), title = "Finished Structure")
    plt.show()


def decode_example_level():
    test_env = TestEnvironment()
    level = test_env.get_level(0)
    representation = test_env.level_img_encoder.create_calculated_img(level.get_used_elements())

    multilayer_stack_decoder = MultiLayerStackDecoder(plot_to_file = True)
    multilayer_stack_decoder.use_negative_air_value = False
    multilayer_stack_decoder.custom_kernel_scale = True
    multilayer_stack_decoder.cutoff_point = 0.98
    multilayer_stack_decoder.recalibrate_blocks = False
    multilayer_stack_decoder.combine_layers = False

    multilayer_stack_decoder.layer_to_level(representation)

if __name__ == '__main__':
    decode_example_level()
