import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

batch_size = 32
epochs = 50
data_amount = 500

final_learning_rate = 1e-4
initial_learning_rate = 1e-2
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
steps_per_epoch = int(data_amount / batch_size)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps = steps_per_epoch,
    decay_rate = learning_rate_decay_factor
)

print(f'learning_rate_decay_factor {learning_rate_decay_factor}')
print(f'steps_per_epoch {steps_per_epoch}')

x = np.arange(0, steps_per_epoch * 50)
y = lr_schedule(x)

print(f'y[0] {y[0]} y[-1] {y[-1]}')

plt.plot(x, y)
plt.show()