import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util.Config import Config


class TensorflowDataCreation:

    def __init__(self, max_width = 128, max_height = 128, air_layer = True):
        self.config = Config.get_instance()

        # max_height = 86 + 2
        # max_width = 212

        # max_height = 99 + 1
        # max_width = 110 + 2

        # max_height = 99 + 1
        # max_width = 115 + 1

        self.max_width = max_width
        self.max_height = max_height

        self.air_layer = True

    # Take from tensorflow simple_gan tutorial
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value ist tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

    def _float_feature(self, value):
        """Returns a floast_list from a float / double."""
        return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

    def serialize_array(self, array):
        array = tf.io.serialize_tensor(array)
        return array

    def pad_image_to_size(self, image_data):
        pad_left = int((self.max_width - image_data.shape[1]) / 2)
        pad_right = int((self.max_width - image_data.shape[1]) / 2)
        pad_top = self.max_height - image_data.shape[0]

        if pad_left + image_data.shape[1] + pad_right < self.max_width:
            pad_right += 1

        if len(image_data.shape) == 3:
            padded_img = np.zeros((self.max_height, self.max_width, image_data.shape[-1]))
            for dim in range(padded_img.shape[-1]):
                if self.air_layer and dim == 0:
                    constant_values = 1
                else:
                    constant_values = 0

                padded_img[:, :, dim] = np.pad(
                    image_data[:, :, dim],
                    ((pad_top, 0), (pad_left, pad_right)),
                    'constant',
                    constant_values = constant_values
                )
        else:
            padded_img = np.pad(image_data, ((pad_top, 0), (pad_left, pad_right)), 'constant')
            new_shape = (padded_img.shape[0], padded_img.shape[1], 1)
            padded_img = padded_img.reshape(new_shape)
        # plt.imshow(padded_img)
        # plt.show()

        return padded_img.astype(dtype = np.int16)

    def parse_single_data_example(self, data_example):
        # define the dictionary -- the structure -- of our single example

        meta_data = data_example['meta_data']
        img_data = data_example['img_data']
        game_data = data_example['game_data'] if 'game_data' in data_example else None

        new_img = self.pad_image_to_size(img_data)

        data = {
            # Img data
            'height': self._int64_feature(new_img.shape[0]),
            'width': self._int64_feature(new_img.shape[1]),
            'depth': self._int64_feature(new_img.shape[2]),
            'raw_image': self._bytes_feature(self.serialize_array(new_img)),

            # Meta data
            'level_height': self._float_feature(meta_data.height),
            'level_width': self._float_feature(meta_data.width),
            'pixel_height': self._int64_feature(img_data.shape[0]),
            'pixel_width': self._int64_feature(img_data.shape[1]),
            'block_amount': self._int64_feature(meta_data.block_amount),
            'pig_amount': self._int64_feature(meta_data.pig_amount),
            'platform_amount': self._int64_feature(meta_data.platform_amount),
            'special_block_amount': self._int64_feature(meta_data.special_block_amount),
        }

        if game_data is not None:
            # level data from playing
            data['cumulative_damage'] = self._float_feature(game_data['cumulative_damage'])
            data['initial_damage'] = self._float_feature(game_data['initial_damage'])
            data['is_stable'] = self._int64_feature(game_data['is_stable'])
            data['death'] = self._int64_feature(game_data['death'])
            data['birds_used'] = self._int64_feature(game_data['birds_used'])
            data['won'] = self._int64_feature(game_data['won'])
            data['score'] = self._int64_feature(game_data['score'])

        # create an Example, wrapping the single features
        out = tf.train.Example(features = tf.train.Features(feature = data))

        return out

    def create_tensorflow_data(self):

        data_set = self.config.get_data_set(folder_name = 'multilayer_with_air', file_name = "unified")
        with open(data_set, 'rb') as f:
            data_dict = pickle.load(f)

        record_file = self.config.get_tf_records(
            dataset_name = f'multilayer_with_air_{self.max_width}_{self.max_height}'
        )

        with tf.io.TFRecordWriter(record_file) as writer:
            for date_name, data_example in tqdm(data_dict.items()):
                tf_example = self.parse_single_data_example(data_example)
                writer.write(tf_example.SerializeToString())


    def create_tensorflow_data_from_file(self, dataset_file_path, outfile_path):
        with open(dataset_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        if 'tfrecords' not in outfile_path:
            outfile_path += '.tfrecords'

        with tf.io.TFRecordWriter(outfile_path) as writer:
            for date_name, data_example in tqdm(data_dict.items(), desc = 'Creating tfrecords'):
                tf_example = self.parse_single_data_example(data_example)
                writer.write(tf_example.SerializeToString())

        return outfile_path

if __name__ == '__main__':
    data_creation = TensorflowDataCreation()
    data_creation.create_tensorflow_data()
