import cv2
import numpy as np
import tensorflow as tf
from model_misc import img_size
from preprocess import processImage
import matplotlib.pyplot as plt
import sys, os
import pickle

img_size = img_size
le = pickle.loads(open("LabelEncoders", 'rb').read())


def _load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def predict_image(img_path, model):
    stream = open(img_path, 'rb')
    bytes = bytearray(stream.read())
    np_array = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    fake_img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    global img_size
    fake_img = processImage(fake_img, img_size)
    predictions = model.predict(fake_img)
    print(predictions)
    prediction = predictions[0]
    #fake_img = np.reshape(fake_img, (fake_img.shape[1], fake_img.shape[2]))
    j = np.argmax(prediction)
    label = le.classes_[j]

    plt.imshow(image)
    plt.title("Prediction: " + str(label))
    plt.show()