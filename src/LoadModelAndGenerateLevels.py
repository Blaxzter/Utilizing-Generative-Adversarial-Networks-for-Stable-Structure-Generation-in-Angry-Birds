import os

import matplotlib

from converter.to_img_converter.MultiLayerStackDecoder import MultiLayerStackDecoder
from game_management.GameManager import GameManager
from level.LevelReader import LevelReader
from level.LevelVisualizer import LevelVisualizer

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from converter.gan_processing.DecodingFunctions import DecodingFunctions
from generator.gan.SimpleGans import SimpleGAN88212, SimpleGAN100112, SimpleGAN100116
from generator.gan.BigGans import WGANGP128128_Multilayer, WGANGP128128
from util.Config import Config


output_dir = './generated_level'
model_path = '../models/Multilayer With Air (AIIDE)/'
model_name = 'WGANGP128128_Multilayer'
last_layer_dim = 5
n_create_images = 5

simulate_levels = False
science_birds_game_folder = "resources/science_birds/win-slow/"


def load_model(model_name, last_dim = 5):
    if model_name == 'WGANGP128128':
        ret_model = WGANGP128128(data_augmentation = False)
    elif model_name == 'WGANGP128128_Multilayer':
        ret_model = WGANGP128128_Multilayer(data_augmentation = False, last_dim = last_dim)
    elif model_name == 'SimpleGAN88212':
        ret_model = SimpleGAN88212(data_augmentation = False)
    elif model_name == 'SimpleGAN100112':
        ret_model = SimpleGAN100112(data_augmentation = False)
    elif model_name == 'SimpleGAN100116':
        ret_model = SimpleGAN100116(data_augmentation = False)
    else:
        raise ValueError('Invalid model name: ' + model_name)

    ret_model.print_summary()
    return ret_model


def load_gan_model():
    gan = load_model(model_name = model_name, last_dim = last_layer_dim)
    checkpoint = tf.train.Checkpoint(
        generator_optimizer = tf.keras.optimizers.Adam(1e-4),
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
        generator = gan.generator,
        discriminator = gan.discriminator
    )
    manager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep = 1)
    checkpoint.restore(manager.latest_checkpoint)
    return gan


def load_level_decoder():
    multilayer_stack_decoder = MultiLayerStackDecoder()
    multilayer_stack_decoder.round_to_next_int = True
    multilayer_stack_decoder.custom_kernel_scale = True
    multilayer_stack_decoder.minus_one_border = False
    multilayer_stack_decoder.combine_layers = True
    multilayer_stack_decoder.negative_air_value = -1
    multilayer_stack_decoder.cutoff_point = 0.5
    multilayer_stack_decoder.display_decoding = False
    return multilayer_stack_decoder


if __name__ == '__main__':

    import tensorflow as tf

    config = Config.get_instance()

    level_reader = LevelReader()
    level_visualizer = LevelVisualizer()

    if simulate_levels:
        config.set_game_folder_props(science_birds_game_folder)
        game_manager = GameManager(config)
        game_manager.start_game()

    # functions to move the output from [-1, 1] to [0, 1] range
    decoding_functions = DecodingFunctions(threshold_callback = lambda: 0.5)
    decoding_functions.set_rescaling(rescaling = tf.keras.layers.Rescaling)
    decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
    rescale_function = decoding_functions.rescale

    # function to flatten the gan output to an image with 1 channel
    decoding_function = decoding_functions.argmax_multilayer_decoding_with_air

    multilayer_stack_decoder = load_level_decoder()

    gan = load_gan_model()

    # create n dimensional random vector and generate images with it
    seed = gan.create_random_vector_batch(batch = n_create_images)
    generated_images, predictions = gan.create_img(seed)
    # Move from [-1, 1] to [0, 1] range
    gan_outputs_reformatted = rescale_function(generated_images)

    # check if output_dir exists and create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output location: {output_dir}")

    try:
        # Go over the images
        for i in range(n_create_images):

            # Get the flattend image
            ref_img, _ = decoding_function(gan_outputs_reformatted[i])

            # save image trough matplotlib
            plt.imshow(ref_img)
            plt.savefig(os.path.join(output_dir, f'{i}_generated_level.png'))
            # clear plot
            plt.clf()

            # Create level from gan output
            level = multilayer_stack_decoder.decode(gan_outputs_reformatted[i])

            fig, ax = plt.subplots(1, 1, dpi = 100)
            level_visualizer.create_img_of_structure(
                level.get_used_elements(), use_grid = False, ax = ax, scaled = True
            )
            fig.savefig(os.path.join(output_dir, f'{i}_decoded_level.png'))
            plt.clf()

            # Save level to xml
            level_xml = level_reader.create_level_from_structure(level.get_used_elements(), red_birds = True, move_to_ground = True)
            level_reader.write_xml_file(level_xml, os.path.join(output_dir, f'{i}_generated_level.xml'))

            # send to opend game
            if simulate_levels:
                game_manager.switch_to_level(level)

                # add input to continue
                input("Press Enter to continue...")

    except Exception as e:
        print(e)

    if simulate_levels:
        game_manager.stop_game()
