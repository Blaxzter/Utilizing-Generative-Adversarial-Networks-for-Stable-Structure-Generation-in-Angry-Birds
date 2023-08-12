import tensorflow as tf


mean_metric = tf.keras.metrics.Mean('generator_loss', dtype = tf.float32)
mean_metric(4)
mean_metric(2)

print(mean_metric.result())

mean_metric.reset_state()
print(mean_metric.result())
mean_metric(4)
print(mean_metric.result())

