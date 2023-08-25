import argparse
import os
import sys

from data_scripts.CreateDataScript import create_data_set
from generator.gan.SimpleGans import SimpleGAN88212, SimpleGAN100112, SimpleGAN100116

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from trainer.TrainerWrapper import NetworkTrainer
from util.Config import Config
from data_scripts.LevelDataset import LevelDataset
from generator.gan.BigGans import WGANGP128128_Multilayer, WGANGP128128


def load_model(model_name, data_augmentation = True, last_dim = 5):
    if model_name == 'WGANGP128128':
        ret_model = WGANGP128128(data_augmentation = data_augmentation)
    elif model_name == 'WGANGP128128_Multilayer':
        ret_model = WGANGP128128_Multilayer(data_augmentation = data_augmentation, last_dim = last_dim)
    elif model_name == 'SimpleGAN88212':
        ret_model = SimpleGAN88212(data_augmentation = data_augmentation)
    elif model_name == 'SimpleGAN100112':
        ret_model = SimpleGAN100112(data_augmentation = data_augmentation)
    elif model_name == 'SimpleGAN100116':
        ret_model = SimpleGAN100116(data_augmentation = data_augmentation)
    else:
        raise ValueError('Invalid model name: ' + model_name)

    ret_model.print_summary()

    return ret_model


def main(args):
    import tensorflow as tf

    # check if cuda is available
    if tf.config.experimental.list_physical_devices('GPU'):
        device_name = '/GPU:0'
    else:
        device_name = '/CPU:0'

    with tf.device(device_name):

        config = Config.get_instance_noargs(args)
        config.tag = args.run_name
        config.one_encoding = False  # currently not supported
        config.multilayer = args.multi_layer_size > 1

        # create dataset args.dataset_save_location args.run_name join
        dataset_name = os.path.join(args.dataset_save_location, args.run_name)
        # check if dataset save location exists and create it if not
        if not os.path.exists(args.dataset_save_location):
            os.makedirs(args.dataset_save_location)
            print(f"Created dataset save location: {args.dataset_save_location}")

        create_data_set(
            data_file = dataset_name,
            orig_level_folder = args.dataset,
            multi_layer_size = args.multi_layer_size,
            _config = config
        )

        dataset = LevelDataset(dataset_name = "multilayer_with_air_128_128", batch_size = 32)
        dataset.load_dataset()

        gan = load_model(args.model, data_augmentation = args.augmentation, last_dim = args.multi_layer_size)
        run_name = args.run_name

        gan.print_summary()

        trainer = NetworkTrainer(run_name = run_name, dataset = dataset, model = gan, epochs = 10000)
        # trainer.continue_training(run_name = run_name, checkpoint_date = "20220623-015436")

        trainer.train()


if __name__ == "__main__":

    print("Hallo")

    parser = argparse.ArgumentParser(description = 'GAN Training Script')

    # Model choices
    available_models = ['WGANGP128128', 'WGANGP128128_Multilayer', 'SimpleGAN88212', 'SimpleGAN100112',
                        'SimpleGAN100116']
    parser.add_argument('-m', '--model', type = str, choices = available_models, required = False,
                        help = 'Name of the GAN model to use.', default = 'WGANGP128128_Multilayer')

    # Dataset location
    parser.add_argument('-d', '--dataset', type = str, required = False, help = 'Path to the dataset folder.',
                        default = '../train_datasets/single_structure')

    # Epoch
    parser.add_argument('-e', '--epoch', type = int, required = False, help = 'Number of epochs for training.',
                        default = 10000)

    # Batch size
    parser.add_argument('-b', '--batch_size', type = int, required = False, help = 'Batch size for training.',
                        default = 32)

    # Multi layer size
    parser.add_argument('-x', '--multi_layer_size', type = int, required = False,
                        help = 'If a multilayer model is selected the amount of output layers on the last level. 4 = No Air, 5 = With air',
                        default = 5)

    # Data augmentation
    parser.add_argument('-a', '--augmentation', action = 'store_true', help = 'Use data augmentation if set.')

    # Run name
    parser.add_argument('-r', '--run_name', type = str, required = False, help = 'Description/name of the run.',
                        default = 'test_run')

    # Model save location
    parser.add_argument('-s', '--save_location', type = str, required = False,
                        help = 'Location where the model will be saved.', default = './trained_models')

    # dataset save locations
    parser.add_argument('-ds', '--dataset_save_location', type = str, required = False,
                        help = 'Location where the created dataset will be saved.', default = './train_datasets')

    args = parser.parse_args()

    print(parser)
    main(args)
