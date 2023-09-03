import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import tensorflow as tf

from converter.gan_processing.DecodingFunctions import DecodingFunctions
from generator.gan.BigGans import WGANGP128128_Multilayer
from util.Config import Config


def create_data():
    config = Config.get_instance()

    decoding_functions = DecodingFunctions(threshold_callback = lambda: 0.5)
    decoding_functions.set_rescaling(rescaling = tf.keras.layers.Rescaling)
    decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
    rescale_function = decoding_functions.rescale

    # load first gan
    checkpoint_dir = config.get_new_model_path('Multilayer With Air (AIIDE)')
    air_gan = WGANGP128128_Multilayer(last_dim = 5)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer = tf.keras.optimizers.Adam(1e-4),
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
        generator = air_gan.generator,
        discriminator = air_gan.discriminator
    )
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_prefix, max_to_keep = 2
    )
    checkpoint.restore(manager.latest_checkpoint)

    # Load second gan
    checkpoint_dir = config.get_new_model_path('Big Gan Multilayer')
    no_air_gan = WGANGP128128_Multilayer()

    checkpoint = tf.train.Checkpoint(
        generator_optimizer = tf.keras.optimizers.Adam(1e-4),
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
        generator = no_air_gan.generator,
        discriminator = no_air_gan.discriminator
    )
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_prefix, max_to_keep = 2
    )
    checkpoint.restore(manager.latest_checkpoint)

    created_air_levels, _ = air_gan.create_img(air_gan.create_random_vector_batch(300))
    created_no_air_levels, _ = no_air_gan.create_img(no_air_gan.create_random_vector_batch(300))

    created_air_levels = rescale_function(created_air_levels)
    created_no_air_levels = rescale_function(created_no_air_levels)

    # plt.hist(created_air_levels[:, :, :, 1:].flatten())
    # plt.show()
    #
    # plt.hist(created_no_air_levels.flatten())
    # plt.show()

    air_frequency, bins = np.histogram(created_air_levels[:, :, :, 1:], bins = np.linspace(0, 1, 100))
    noair_frequency, bins = np.histogram(created_no_air_levels, bins = np.linspace(0, 1, 100))

    print(air_frequency)
    print(noair_frequency)

    #
    # df = pd.DataFrame(dict(
    #     series = np.concatenate((["a"] * len(air_frequency), ["b"] * len(noair_frequency))),
    #     data = np.concatenate((air_frequency, noair_frequency))
    # ))
    #
    # fig = px.histogram(df, x = "data", color = "series", barmode = "overlay")
    # fig.show()


if __name__ == '__main__':
    create_data()
