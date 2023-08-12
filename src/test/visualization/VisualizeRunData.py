import pickle

from tqdm import tqdm

from converter.gan_processing.DecodingFunctions import DecodingFunctions
from util.Config import Config

import tensorflow as tf

from PIL import Image
import cv2

import matplotlib.pyplot as plt
import numpy as np

def load_run_data(run_name, img_index = 0):
    config = Config.get_instance()
    pickle_files = config.get_epoch_run_data_files(run_name)

    epoch_data = {}

    decoding_functions = DecodingFunctions(threshold_callback = lambda: 0)
    decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
    decoding_functions.set_rescaling(rescaling = tf.keras.layers.Rescaling)
    img_decoding = decoding_functions.argmax_multilayer_decoding_with_air

    for idx, pickle_file in tqdm(enumerate(pickle_files)):
        with open(pickle_file, 'rb') as f:
            loaded_outputs = pickle.load(f)

        for epoch, outputs in loaded_outputs.items():

            epoch_data[epoch] = img_decoding(outputs['imgs'][img_index])[0]

    keys = list(epoch_data.keys())
    keys.sort()

    def array_to_image(image_array, _idx):
        # Create a temporary file path to save the image
        temp_path = f'tmp/temp_{_idx}.png'

        # Save the array as an image using plt.imsave
        plt.imsave(temp_path, image_array)

        # Read the saved image using PIL and convert to a NumPy array
        img = Image.open(temp_path)
        img_array = np.array(img)

        return img_array

    image_list = [
        array_to_image(epoch_data[epoch], epoch) for idx, epoch in tqdm(enumerate(keys))
    ]

    height, width, layers = image_list[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{run_name}_{img_index}.mp4v", fourcc, 60, (width, height))

    for image in image_list:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    run_name = "wgan_gp_128_128_one_encoding_fixed"
    load_run_data(run_name, img_index = 0)
