import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from preprocess import processImage
import matplotlib.pyplot as plt
import sys, os
import pickle
from tqdm import tqdm


img_size = 128
le = pickle.loads(open("LabelEncoders", 'rb').read())


def predict_single_image(img_path, model: Model, threshVal=120):
    stream = open(img_path, 'rb')
    bytes = bytearray(stream.read())
    np_array = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    fake_img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    global img_size
    fake_img = processImage(fake_img, img_size, threshVal)
    cv2.imshow("Fake img", fake_img)
    cv2.waitKey()
    fake_img = np.reshape(fake_img, ((1, ) + fake_img.shape + (1,))) / 255.0
    predictions = model.predict(fake_img)
    print(predictions)
    prediction = predictions[0]
    #fake_img = np.reshape(fake_img, (fake_img.shape[1], fake_img.shape[2]))
    j = np.argmax(prediction)
    label = le.classes_[j]

    plt.imshow(image)
    plt.title("Prediction: " + str(label))
    plt.show()
    
if __name__ == '__main__':
    model = load_model("SavedModel/MTDMTnet 20210621-142145")
    predict_single_image("Data/m/4.png", model)