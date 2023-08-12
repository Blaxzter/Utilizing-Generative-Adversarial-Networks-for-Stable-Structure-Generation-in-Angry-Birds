import tensorflow as tf
from matplotlib import pyplot as plt

from generator.gan.IGAN import IGAN

layers = tf.keras.layers
activations = tf.keras.activations


# https://de.mathworks.com/help/deeplearning/ug/trainwasserstein-gan-with-gradient-penalty-wgan-gp.html#:~:text=To%20train%20a%20WGAN%2DGP%20model%2C%20you%20must%20train%20the,64%20for%2010%2C000%20generator%20iterations.
# Example for a WGAN-GP Network

class WGANGP128128(IGAN):

    def __init__(self, data_augmentation = None):
        super().__init__()
        self.input_array_size = 128
        self.channel_amount = 64

        self.input_shape = (1, 1, 128)
        self.output_shape = (128, 128)

        self.data_augmentation = data_augmentation

        self.generator = None
        self.discriminator = None
        self.create_generator_model()
        self.create_discriminator_model()

    def create_random_vector(self):
        return tf.random.normal([1, 1, 1, self.input_array_size])

    def create_random_vector_batch(self, batch):
        """
        Returns a Tensor that has the input shape required for the generator model
        """
        return tf.random.normal([batch, 1, 1, self.input_array_size])

    def create_generator_model(self):
        model = tf.keras.Sequential(name = 'Generator')

        model.add(layers.InputLayer(input_shape = self.input_shape))

        model.add(layers.Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'valid',
                                         use_bias = False))
        model.add(layers.LayerNormalization())
        model.add(layers.ReLU())

        for i in range(4):
            d = min(self.channel_amount * 2 ** (4 - 2 - i), self.channel_amount * 8)
            model.add(layers.Conv2DTranspose(d, 4, strides = 2, padding = 'same', use_bias = False))
            model.add(layers.LayerNormalization())
            model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(1, 4, strides = 2, padding = 'same'))
        model.add(layers.Activation(activations.tanh))

        self.generator = model

    def create_discriminator_model(self):
        model = tf.keras.Sequential(name = 'Discriminator')

        model.add(layers.InputLayer(input_shape = [self.output_shape[0], self.output_shape[1], 1]))
        model.add(layers.Conv2D(self.channel_amount, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(layers.LeakyReLU(0.2))

        for i in range(4):
            d = min(self.channel_amount * 2 ** (i + 1), self.channel_amount * 8)
            model.add(layers.Conv2D(d, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
            model.add(layers.LayerNormalization())
            model.add(layers.LeakyReLU(alpha = 0.2))

        model.add(layers.Conv2D(1, (4, 4), strides = (1, 1), padding = 'valid', use_bias = False))

        self.discriminator = model


class WGANGP128128_Multilayer(IGAN):

    def __init__(self, data_augmentation = None, last_dim = 4, last_layer = 'tanh'):
        super().__init__()
        self.input_array_size = 128
        self.channel_amount = 64

        self.input_shape = (1, 1, 128)
        self.output_shape = (128, 128, last_dim)

        self.data_augmentation = data_augmentation
        self.last_layer = last_layer

        self.generator = None
        self.discriminator = None
        self.create_generator_model()
        self.create_discriminator_model()

    def create_random_vector(self):
        return tf.random.normal([1, 1, 1, self.input_array_size])

    def create_random_vector_batch(self, batch):
        """
        Returns a Tensor that has the input shape required for the generator model
        """
        return tf.random.normal([batch, 1, 1, self.input_array_size])

    def create_generator_model(self):
        model = tf.keras.Sequential(name = 'Generator')

        model.add(layers.InputLayer(input_shape = self.input_shape))

        model.add(layers.Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'valid',
                                         use_bias = False))
        model.add(layers.LayerNormalization())
        model.add(layers.ReLU())

        for i in range(4):
            d = min(self.channel_amount * 2 ** (4 - 2 - i), self.channel_amount * 8)
            model.add(layers.Conv2DTranspose(d, 4, strides = 2, padding = 'same', use_bias = False))
            model.add(layers.LayerNormalization())
            model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(self.output_shape[2], 4, strides = 2, padding = 'same'))
        if self.last_layer == 'tanh':
            model.add(layers.Activation(activations.tanh))
        else:
            model.add(layers.ReLU())

        self.generator = model

    def create_discriminator_model(self):
        model = tf.keras.Sequential(name = 'Discriminator')

        model.add(layers.InputLayer(input_shape = [self.output_shape[0], self.output_shape[1], self.output_shape[2]]))
        model.add(layers.Conv2D(self.channel_amount, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(layers.LeakyReLU(0.2))

        for i in range(4):
            d = min(self.channel_amount * 2 ** (i + 1), self.channel_amount * 8)
            model.add(layers.Conv2D(d, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
            model.add(layers.LayerNormalization())
            model.add(layers.LeakyReLU(alpha = 0.2))

        model.add(layers.Conv2D(1, (4, 4), strides = (1, 1), padding = 'valid', use_bias = False))

        self.discriminator = model



if __name__ == '__main__':
    gan = WGANGP128128()
    gan.print_summary()

    created_img, img_probability = gan.create_img()

    # plt.imshow(created_img)
    # plt.suptitle(f'Probability: {img_probability}')
    # plt.show()
