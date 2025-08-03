import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("Available devices: ", tf.config.experimental.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))