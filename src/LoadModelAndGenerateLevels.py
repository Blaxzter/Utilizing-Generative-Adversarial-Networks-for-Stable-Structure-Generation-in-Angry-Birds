import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from converter.gan_processing.DecodingFunctions import DecodingFunctions
from generator.gan.SimpleGans import SimpleGAN88212, SimpleGAN100112, SimpleGAN100116
from generator.gan.BigGans import WGANGP128128_Multilayer, WGANGP128128
from util.Config import Config

output_dir = './generated_level'
model_path = 'trained_models/test_run_200/20230826-190051'
model_name = 'WGANGP128128_Multilayer'
n_create_images = 5


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

if __name__ == '__main__':

    import tensorflow as tf

    config = Config.get_instance()

    decoding_functions = DecodingFunctions(threshold_callback = lambda: 0.5)
    decoding_functions.set_rescaling(rescaling = tf.keras.layers.Rescaling)
    decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
    rescale_function = decoding_functions.rescale
    decoding_function = decoding_functions.argmax_multilayer_decoding_with_air
    grid_search_dataset = config.get_grid_search_file("generated_data_set")

    gan = load_model(model_name = model_name, last_dim = 5)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer = tf.keras.optimizers.Adam(1e-4),
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
        generator = gan.generator,
        discriminator = gan.discriminator
    )
    checkpoint_prefix = os.path.join(model_path, "ckpt")
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_prefix, max_to_keep = 2
    )
    checkpoint.restore(manager.latest_checkpoint)

    seed = gan.create_random_vector_batch(batch = n_create_images)
    generated_images, predictions = gan.create_img(seed)
    gan_outputs_reformatted = rescale_function(generated_images)

    # check if output_dir exists and create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output location: {output_dir}")

    # save gan_outputs_reformatted images
    for i in range(n_create_images):

        ref_img, _ = decoding_function(gan_outputs_reformatted[i])

        # save image trough matplotlib
        plt.imshow(ref_img)
        plt.savefig(os.path.join(output_dir, f'generated_level_{i}.png'))
