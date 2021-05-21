import os
import cv2
import numpy as np
from training.LivenessNet import LivenessNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
import pickle

image_size = None
le = LabelEncoder()


def load_data(result_dir, _image_size: int):
    datas = []
    labels = []
    assert _image_size, "Please specify image size"
    global image_size
    image_size = _image_size
    for folder in os.listdir(result_dir):
        label_path = os.path.join(result_dir, folder)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                try:
                    image = cv2.imread(os.path.join(label_path, file))
                    assert image is not None
                    image = cv2.resize(image, (image_size, image_size))
                    datas.append(image)
                    labels.append(folder)
                except AssertionError as e:
                    print("%s is not an image" %file)
    datas = np.array(datas, dtype=np.float)
    labels = le.fit_transform(labels)
    #labels = tf.keras.utillib.to_categorical(labels)
    f = open("../LabelEncoders", 'wb')
    f.write(pickle.dumps(le))
    f.close()
    return datas, labels


def train(epochs: int):
    datas, labels = load_data("../Data", _image_size=64)
    (X_train, X_test, y_train, y_test) = train_test_split(datas, labels, test_size=0.1)
    model = LivenessNet.build(image_size, image_size, 3, len(le.classes_))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    model_timer = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("TensorboardLogs", "LivenessNet " + model_timer)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    model.fit(X_train, y_train, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[tensorboard_callback, early_stopping])
    model.save(os.path.join("Saved Models", "LivenessNet " + model_timer))