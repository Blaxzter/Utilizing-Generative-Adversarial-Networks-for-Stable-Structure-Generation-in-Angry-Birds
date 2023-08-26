import os
import time

import tensorflow as tf
from loguru import logger
from tensorflow.core.util import event_pb2
from tqdm.auto import tqdm

from data_scripts.LevelDataset import LevelDataset
from generator.gan.IGAN import IGAN
from trainer.WGANGPTrainStepper import WGANGPTrainStepper
from util.Config import Config
from util.TrainVisualizer import TensorBoardViz


class NetworkTrainer:

    def __init__(self, run_name, dataset: LevelDataset, model, epochs = 50, checkpoint_dir: str = None):

        self.config: Config = Config.get_instance()
        self.run_name = run_name

        self.model: IGAN = model
        self.dataset: LevelDataset = dataset
        self.visualizer: TensorBoardViz = TensorBoardViz(model = model, dataset = dataset, current_run = self.run_name)
        self.train_stepper = WGANGPTrainStepper(self.model, self.dataset, self.visualizer)
        self.visualizer.create_aggregator(self.train_stepper.get_aggregated_parameters())

        self.overwrite_save_location = checkpoint_dir

        self.checkpoint = None
        self.checkpoint_dir = None
        self.checkpoint_prefix = None
        self.manager = None

        self.epochs = epochs
        self.continue_run = False
        self.outer_tqdm = self.config.outer_tqdm

    def train(self):
        if not self.continue_run:
            self.visualizer.create_summary_writer(self.run_name)
            self.create_checkpoint_manager(self.run_name, checkpoint_dir = self.overwrite_save_location)
            logger.debug(f'Start Training of {self.run_name} for {self.epochs} epochs')
        else:
            logger.debug(
                f'Continue Training of {self.run_name} for {self.epochs} epochs at {self.visualizer.global_step}')

        current_epoch = 0
        if self.outer_tqdm:
            iter_data = tqdm(range(self.epochs), total = self.epochs, desc = f"Training: {self.run_name}")
        else:
            iter_data = range(self.epochs)

        self.visualizer.visualize(0, 0)

        for _ in iter_data:
            start_time = time.time()

            self.train_stepper.train_batch()

            # Produce images for the GIF as you go
            self.visualizer.visualize(current_epoch + 1, start_time)

            # Save the model every 15 epochs
            if (current_epoch + 1) % self.config.save_checkpoint_every == 0:
                self.manager.save()
                self.visualizer.store_data(current_epoch)

            current_epoch += 1

        # Generate after the final epoch
        self.manager.save()

    def save(self):
        self.manager.save()

    def create_checkpoint_manager(self, run_name, run_time = None, checkpoint_dir = None):
        if checkpoint_dir is not None and checkpoint_dir[-1] != '/':
            checkpoint_dir += '/'

        if run_time is None:
            if checkpoint_dir is not None:
                self.checkpoint_dir = checkpoint_dir + '{current_run}/{timestamp}/'
                self.checkpoint_dir = self.checkpoint_dir.replace('{current_run}', run_name)
                self.checkpoint_dir = self.checkpoint_dir.replace('{timestamp}', self.config.strftime)
            else:
                self.checkpoint_dir = self.config.get_current_checkpoint_dir(run_name)
        else:
            if checkpoint_dir is not None:
                self.checkpoint_dir = checkpoint_dir + '{current_run}/{timestamp}/'
                self.checkpoint_dir = self.checkpoint_dir.replace('{current_run}', run_name)
                self.checkpoint_dir = self.checkpoint_dir.replace('{timestamp}', run_time)
            else:
                self.checkpoint_dir = self.config.get_checkpoint_dir(run_name, run_time)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.debug(f"Created checkpoint save location: {self.checkpoint_dir}")

        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer = self.train_stepper.generator_optimizer,
            discriminator_optimizer = self.train_stepper.discriminator_optimizer,
            generator = self.model.generator,
            discriminator = self.model.discriminator
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_prefix, max_to_keep = self.config.keep_checkpoints
        )

    def load(self, run_name = None, checkpoint_date = None):
        if checkpoint_date is None:
            raise Exception("Pls define the checkpoint folder")

        checkpoint_dir = self.config.get_checkpoint_dir(run_name, checkpoint_date)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        manager = tf.train.CheckpointManager(
            self.checkpoint, checkpoint_prefix, max_to_keep = 2
        )
        self.checkpoint.restore(manager.latest_checkpoint)

    def continue_training(self, run_name, checkpoint_date):

        log_dir = self.visualizer.create_summary_writer(run_name = run_name, run_time = checkpoint_date)

        log_file = self.config.get_event_file(log_dir)
        data_set = tf.data.TFRecordDataset(log_file)
        *_, last = iter(data_set)
        epoch = event_pb2.Event.FromString(last.numpy()).step
        self.visualizer.global_step = epoch

        self.create_checkpoint_manager(run_name, checkpoint_date)

        self.load(run_name = run_name, checkpoint_date = checkpoint_date)
        self.continue_run = True
