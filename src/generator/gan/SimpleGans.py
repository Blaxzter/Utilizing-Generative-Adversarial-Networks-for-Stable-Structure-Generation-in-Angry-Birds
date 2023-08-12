import tensorflow as tf
from matplotlib import pyplot as plt

from generator.gan.IGAN import IGAN

layers = tf.keras.layers


class SimpleGAN88212(IGAN):

    def __init__(self, data_augmentation = None):
        super().__init__()
        self.input_array_size = 256

        self.output_shape = (88, 212)

        self.data_augmentation = data_augmentation

        self.generator = None
        self.discriminator = None
        self.create_generator_model()
        self.create_discriminator_model()

    def create_random_vector(self):
        return tf.random.normal([1, self.input_array_size])

    def create_generator_model(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Dense(22 * 53 * self.input_array_size, use_bias = False, input_shape = (self.input_array_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((22, 53, self.input_array_size)))
        assert model.output_shape == (None, 22, 53, self.input_array_size)

        model.add(layers.Conv2DTranspose(filters = 128, kernel_size = (5, 5), strides = (1, 1), padding = 'same',
                                         use_bias = False))
        assert model.output_shape == (None, 22, 53, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = (2, 2), padding = 'same',
                                         use_bias = False))
        assert model.output_shape == (None, 44, 106, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', use_bias = False,
                                         activation = 'tanh'))
        assert model.output_shape == (None, self.output_shape[0], self.output_shape[1], 1)

        self.generator = model

    def create_discriminator_model(self):
        model = tf.keras.Sequential()
        if self.data_augmentation:
            model.add(self.data_augmentation)

        model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same',
                                input_shape = [self.output_shape[0], self.output_shape[1], 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        self.discriminator = model


class SimpleGAN100112(IGAN):

    def __init__(self, data_augmentation = None):
        super().__init__()
        self.input_array_size = 256

        self.output_shape = (100, 112)

        self.data_augmentation = data_augmentation

        self.generator = None
        self.discriminator = None
        self.create_generator_model()
        self.create_discriminator_model()

    def create_random_vector(self):
        return tf.random.normal([1, self.input_array_size])

    def create_generator_model(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Dense(25 * 28 * self.input_array_size, use_bias = False, input_shape = (self.input_array_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((25, 28, self.input_array_size)))
        assert model.output_shape == (None, 25, 28, self.input_array_size)

        model.add(layers.Conv2DTranspose(filters = 128, kernel_size = (5, 5), strides = (1, 1), padding = 'same',
                                         use_bias = False))
        assert model.output_shape == (None, 25, 28, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = (2, 2), padding = 'same',
                                         use_bias = False))
        assert model.output_shape == (None, 50, 56, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', use_bias = False,
                                         activation = 'tanh'))
        assert model.output_shape == (None, self.output_shape[0], self.output_shape[1], 1)

        self.generator = model

    def create_discriminator_model(self):
        model = tf.keras.Sequential()
        if self.data_augmentation:
            model.add(self.data_augmentation)

        model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same',
                                input_shape = [self.output_shape[0], self.output_shape[1], 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        self.discriminator = model


class SimpleGAN100116(IGAN):

    def __init__(self, data_augmentation = None):
        super().__init__()
        self.input_array_size = 256

        self.output_shape = (100, 116)

        self.data_augmentation = data_augmentation

        self.generator = None
        self.discriminator = None
        self.create_generator_model()
        self.create_discriminator_model()

    def create_random_vector(self):
        return tf.random.normal([1, self.input_array_size])

    def create_generator_model(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Dense(25 * 29 * self.input_array_size, use_bias = False, input_shape = (self.input_array_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((25, 29, self.input_array_size)))
        assert model.output_shape == (None, 25, 29, self.input_array_size)

        model.add(layers.Conv2DTranspose(filters = 128, kernel_size = (5, 5), strides = (1, 1), padding = 'same',
                                         use_bias = False))
        assert model.output_shape == (None, 25, 29, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = (2, 2), padding = 'same',
                                         use_bias = False))
        assert model.output_shape == (None, 50, 58, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', use_bias = False,
                                         activation = 'tanh'))
        assert model.output_shape == (None, self.output_shape[0], self.output_shape[1], 1)

        self.generator = model

    def create_discriminator_model(self):
        model = tf.keras.Sequential()
        if self.data_augmentation:
            model.add(self.data_augmentation)

        model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same',
                                input_shape = [self.output_shape[0], self.output_shape[1], 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        self.discriminator = model

if __name__ == '__main__':
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal")
    ])

    gan = SimpleGAN88212()

    gan.print_summary()

    # random_vec = gan.create_random_vector()
    # generated_img = gan.generator(random_vec)
    # plt.imshow(generated_img[0, :, :, 0])
    # plt.show()
