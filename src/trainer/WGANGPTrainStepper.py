from functools import partial

import tensorflow as tf
from tqdm.auto import tqdm

from data_scripts.LevelDataset import LevelDataset
from generator.gan.IGAN import IGAN
from util.Config import Config
from util.TrainVisualizer import TensorBoardViz


###################################################################################################
# Oriented on implementation of https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2 #
###################################################################################################

class WGANGPTrainStepper:

    def __init__(self, model: IGAN, dataset: LevelDataset, visualizer: TensorBoardViz):

        self.config: Config = Config.get_instance()

        self.inner_tqdm = self.config.inner_tqdm

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1 = 0, beta_2 = 0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1 = 0, beta_2 = 0.9)

        self.critic_iterations = 5
        self.gradient_penalty_weight = 10.0

        self.model: IGAN = model
        self.dataset: LevelDataset = dataset
        self.visualizer: TensorBoardViz = visualizer

        self.batch_size = dataset.batch_size

    def get_aggregated_parameters(self):
        return ['generator_loss',
                'discriminator_loss',
                'discriminator_real_loss',
                'discriminator_fake_loss',
                'discriminator_gp',
                'real_probabilities',
                'fake_probabilities']

    def train_batch(self):

        if self.inner_tqdm:
            iter_data = tqdm(self.dataset.get_dataset(), total = self.dataset.steps, leave = True, desc = "Batch")
        else:
            iter_data = self.dataset.get_dataset()

        for image_batch, data in iter_data:
            disc_data = self.train_discriminator(image_batch)
            self.visualizer.add_data(disc_data)

            if self.discriminator_optimizer.iterations.numpy() % self.critic_iterations == 0:
                gen_data = self.train_generator()
                self.visualizer.add_data(gen_data)

    @tf.function
    def train_generator(self):
        noise = self.model.create_random_vector_batch(self.batch_size)
        ret_data_dict = dict()
        with tf.GradientTape() as tape:
            # create levels
            fake_img = self.model.generator(noise, training = True)

            # predict real or fake
            fake_logit = self.model.discriminator(fake_img, training = True)

            # calculate generator loss
            g_loss = self.generator_loss(fake_logit)
            ret_data_dict['generator_loss'] = g_loss

        gradients = tape.gradient(g_loss, self.model.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.model.generator.trainable_variables))
        return ret_data_dict

    @tf.function
    def train_discriminator(self, content):
        # Create input vectores
        noise = self.model.create_random_vector_batch(self.batch_size)
        ret_data_dict = dict()
        with tf.GradientTape() as tape:
            fake_img = self.model.generator(noise, training = True)

            real_logit = self.model.discriminator(content, training = True)
            fake_logit = self.model.discriminator(fake_img, training = True)

            real_loss, fake_loss = self.discriminator_loss(real_logit, fake_logit)
            gp = self.gradient_penalty(partial(self.model.discriminator, training = True), content, fake_img)

            d_loss = (real_loss + fake_loss) + gp * self.gradient_penalty_weight

        discriminator_grad = tape.gradient(d_loss, self.model.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grad, self.model.discriminator.trainable_variables))

        ret_data_dict['discriminator_loss'] = d_loss
        ret_data_dict['discriminator_real_loss'] = real_loss
        ret_data_dict['discriminator_fake_loss'] = fake_loss
        ret_data_dict['discriminator_gp'] = gp
        ret_data_dict['real_probabilities'] = real_logit
        ret_data_dict['fake_probabilities'] = fake_logit

        return ret_data_dict

    def discriminator_loss(self, real_logit, fake_logit):
        r_loss = - tf.reduce_mean(real_logit)
        f_loss = tf.reduce_mean(fake_logit)
        return r_loss, f_loss

    def generator_loss(self, fake):
        f_loss = - tf.reduce_mean(fake)
        return f_loss

    def gradient_penalty(self, f, real, fake):
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape = shape, minval = 0., maxval = 1.)
            inter = (alpha * a) + ((1 - alpha) * b)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = f(x)
        grad = tape.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis = 1)
        gp = tf.reduce_mean((norm - 1.) ** 2)

        return gp


