import cv2
import tensorflow as tf
from tensorflow.keras import backend


class LivenessNet:
    @staticmethod
    def build(width, height, depth, classes, batch_size=1):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = tf.keras.models.Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1

        model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization(axis=channel_dim))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same'))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization(axis=channel_dim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same'))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization(axis=channel_dim))
        model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same'))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization(axis=channel_dim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

        return model
