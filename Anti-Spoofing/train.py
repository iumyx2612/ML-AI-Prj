import os, sys
import cv2
import numpy as np
from LivenessNet import LivenessNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime

data = []
labels = []
image_size = None
le = LabelEncoder()


def load_data(result_dir, _image_size: int):
    global data, labels
    assert _image_size, "Please specify image size"
    global image_size
    image_size = _image_size
    for folder in os.listdir(result_dir):
        label_path = os.path.join(result_dir, folder)
        for file in os.listdir(label_path):
            try:
                image = cv2.imread(os.path.join(label_path, file))
                assert image is not None
                image = cv2.resize(image, (image_size, image_size))
                data.append(image)
                labels.append(folder)
            except AssertionError as e:
                print("%s is not an image" %file)
    data = np.array(data, dtype=np.float)/255.0
    labels = le.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)


def train(epochs: int, evaluate=False):
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2)
    model = LivenessNet.build(image_size, image_size, 3, len(le.classes_))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    model_timer = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("TensorboardLogs", "LivenessNet " + model_timer)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, y_train, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[tensorboard_callback])
    model.save(os.path.join("Saved Models", "LivenessNet " + model_timer))


if __name__ == '__main__':
    load_data("Data", _image_size=64)
    train(epochs=50)

