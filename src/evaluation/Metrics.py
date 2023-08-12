import json
import os
import pickle

import numpy as np
import scipy
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from tqdm.asyncio import tqdm

from data_scripts.LevelDataset import LevelDataset
from util.Config import Config


def calculate_fid(model, images1, images2):
    """
    calculate frechet inception distance
    SOURCE: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
    """

    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis = 0), np.cov(act1, rowvar = False)
    mu2, sigma2 = act2.mean(axis = 0), np.cov(act2, rowvar = False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def create_fid_metric():
    with tf.device('/GPU:0'):
        # prepare the inception v3 model
        model = InceptionV3(include_top = False, pooling = 'avg', input_shape = (128, 128, 3))
        dataset = LevelDataset(dataset_name = "new_encoding_multilayer_unified_128_128", batch_size = 9)
        dataset.load_dataset(normalize = False)
        data_amount = dataset.get_data_amount()

        divisions = 5
        dataset.batch_size = int(data_amount / divisions)

        config = Config.get_instance()
        img_files = config.get_epoch_run_data_files('multilayer')

        img_files = sorted(
            map(str, img_files),
            key = lambda path: int(
                path.replace('..\\resources\\data\\pickles\\run_data\\multilayer\\wgan_gp_128_128_one_encoding_fixed_',
                             '')
                    .replace('_20220728-172004.pickle', ''))
        )

        data_dict = dict(
            predictions = dict(),
            fids = dict(),
        )

        eval_file = config.get_fids_file('multilayer_fid')
        if os.path.isfile(eval_file):
            with open(eval_file, 'r') as f:
                data_dict = json.loads(''.join(f.readlines()))

        predictions = data_dict['predictions']
        fids = data_dict['fids']

        selects = list(map(lambda val: [int(val), 0], np.linspace(0, len(img_files) - 1, 120)))
        selects[-1][1] = -1

        control_stacks = []

        for i in range(divisions):
            control_img_stack = np.zeros((dataset.batch_size, 128, 128, 3), dtype = np.uint8)
            for image_batch, data in dataset.get_dataset():
                for i in range(dataset.batch_size):
                    control_stacked_img = np.dstack((np.zeros((128, 128)) + 0.5, image_batch[i].numpy()))
                    control_img_stack[i] = np.repeat(np.argmax(control_stacked_img, axis = 2)[:, :, np.newaxis], 3,
                                                     axis = 2)
            images2 = preprocess_input(control_img_stack)
            control_stacks.append(images2)

        data_tqdm = tqdm(range(len(selects)))

        # for img_file in img_files:
        for select, img_dict_sel in selects:
            img_file = img_files[select]

            with open(img_file, 'rb') as f:
                img_dict = pickle.load(f)

            epoch, img_data = list(img_dict.items())[img_dict_sel]
            if str(epoch) not in predictions:
                predictions[epoch] = str(np.average(img_data['predictions']))

            if str(epoch) not in fids:
                created_img_stack = np.zeros((dataset.batch_size, 128, 128, 3), dtype = np.uint8)
                for i in range(9):
                    created_stacked_img = np.dstack((np.zeros((128, 128)) + 0.5, img_data['imgs'][i % 9].numpy()))
                    created_img_stack[list(range(i, dataset.batch_size, 9))] = np.repeat(
                        np.argmax(created_stacked_img, axis = 2)[:, :, np.newaxis], 3, axis = 2)
                images1 = preprocess_input(created_img_stack)

                # image_batch, data = next(iter(dataset.get_dataset()))
                # control_img_stack = np.zeros((dataset.batch_size, 128, 128, 3), dtype = np.uint8)
                # for i in range(dataset.batch_size):
                #     control_stacked_img = np.dstack((np.zeros((128, 128)) + 0.5, image_batch[i].numpy()))
                #     control_img_stack[i] = np.repeat(np.argmax(control_stacked_img, axis = 2)[:, :, np.newaxis], 3, axis = 2)

                current_fid = []

                for image in control_stacks:
                    fid = calculate_fid(model, images1, images2)
                    current_fid.append(fid)

                fids[epoch] = str(np.average(current_fid))

            data_tqdm.update()

            store_dict = dict(
                predictions = predictions,
                fids = fids,
            )

            eval_file = config.get_eval_file('multilayer_fid')
            with open(eval_file, 'w') as f:
                f.write(json.dumps(store_dict, indent = 4))


if __name__ == '__main__':
    create_fid_metric()
