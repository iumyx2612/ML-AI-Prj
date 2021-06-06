from MTDMTnet import MTDMTnet
from preprocess import DEFAULT_IMG_SIZE
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import datetime
import pickle

img_size = DEFAULT_IMG_SIZE
X_train = np.array([])
X_test = np.array([])
y_train = np.array([])
y_test = np.array([])

SAVED_MODEL_FOLDER = "SavedModel"

le = LabelEncoder()


def load_in_data(data_dir):
    global img_size
    global le
    datas = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        for file in os.listdir(label_path):
            image_path = os.path.join(label_path, file)
            try:
                stream = open(image_path, 'rb')
                bytes = bytearray(stream.read())
                np_array = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
                assert image is not None
                image = np.reshape(image, (img_size, img_size, 1))
                datas.append(image)
                labels.append(label)
            except AssertionError as error:
                error.args += ("%s is not an image file" % file,)
    datas = np.array(datas, dtype=np.float) / 255.0
    labels = le.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)
    f = open("LabelEncoders", 'wb')
    f.write(pickle.dumps(le))
    f.close()
    return datas, labels


def train(data_dir, validation_split: float, epochs=50,  batch_size = 64, tensor_board_log_dir=None, _evaluate=False):
    global X_train, X_test, y_train, y_test
    global SAVED_MODEL_FOLDER
    datas, labels = load_in_data(data_dir)
    (X_train, X_test, y_train, y_test) = train_test_split(datas, labels, test_size=0.2)
    model = MTDMTnet.build(img_size, img_size, len(le.classes_))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_timer = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if tensor_board_log_dir is not None and type(tensor_board_log_dir) == str:
        log_dir = os.path.join(tensor_board_log_dir, "MTDMTnet " + model_timer)
        tensor_board_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(X_train, y_train,
                            epochs=epochs, verbose=2,
                            validation_split=0.2,
                            callbacks=[tensor_board_callback, early_stopping])
    else:
        history = model.fit(X_train, y_train,
                            epochs=epochs, verbose=2,
                            validation_split=0.2,
                            callbacks=[early_stopping])
    model.save(os.path.join(SAVED_MODEL_FOLDER, "MTDMTnet " + model_timer))