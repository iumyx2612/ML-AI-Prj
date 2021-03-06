import cv2
import numpy as np
import os, sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


LABELS = ["a", "ă", "â", "b", "c", "d", "đ", "e", "ê", "g", "h",
          "i", "k", "l", "m", "n", "o", "ô", "ơ", "p", "q", "r",
          "s", "t", "u", "ư", "v", "x", "y"]

def prepareData(labels=LABELS, data_dir="Data", result_dir="ProcessedData", img_size=128):
    for label in labels:
        counter = 0
        data_path = os.path.join(data_dir, label)
        result_path = os.path.join(result_dir, label)
        try:
            for file in tqdm(os.listdir(data_path)):
                image_path = os.path.join(data_path, file)
                stream = open(image_path, 'rb')
                bytes = bytearray(stream.read())
                np_array = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (img_size, img_size))
                ret, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
                dilate = cv2.dilate(thresh, (5, 5), iterations=3)
                try:
                    os.makedirs(result_path, exist_ok=True)
                except OSError as e:
                    print("Directory '%s' can not be created" %result_path)
                is_success, im_buf_arr = cv2.imencode(".jpg", dilate)
                im_buf_arr.tofile(os.path.join(result_path, str(label) + str(counter) + ".jpg"))
                counter += 1
        except FileNotFoundError as e:
            pass


def data_augumentation(data_dir="ProcessedData", max_gen=10):
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=False, fill_mode="constant")
    for dir in os.listdir(data_dir):
        data_path = os.path.join(data_dir, dir)
        for file in os.listdir(data_path):
            total = 0
            img_path = os.path.join(data_path, file)
            stream = open(img_path, 'rb')
            bytes = bytearray(stream.read())
            np_array = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
            imageGen = aug.flow(image, batch_size=1, save_to_dir=data_path, save_format="jpg", save_prefix="aug")
            for img in imageGen:
                total += 1
                if total == max_gen:
                    break
            if "aug" in file:
                break


def processImage(img, size, threshVal=120):
    image = cv2.resize(img, (size, size))
    ret, thresh = cv2.threshold(image, threshVal, 255, cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh, (5, 5), iterations=4)
    return dilate


if __name__ == '__main__':
    prepareData()
    data_augumentation(max_gen=30)