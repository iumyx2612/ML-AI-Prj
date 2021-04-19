import tensorflow as tf
import sklearn


class MTDMTnet:
    @staticmethod
    def build(height, width, classes, batch_size=1):
        model = tf.keras.models.Sequential()
        input_shape = (batch_size, height, width)

        model.add(tf.keras.layers.Conv2D(6, (5, 5), input_shape=input_shape))
        model.add(tf.keras.activations.relu)
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(tf.keras.layers.Conv2D(16, (5, 5)))
        model.add(tf.keras.activations.relu)
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(120))
        model.add(tf.keras.activations.relu)
        model.add(tf.keras.layers.Dense(84))
        model.add(tf.keras.activations.relu)
        model.add(tf.keras.layers.Dense(classes))
        model.add(tf.keras.activations.softmax)

        return model
