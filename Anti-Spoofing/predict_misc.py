import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_detection
import pickle
import sys, os

le = pickle.loads(open("LabelEncoders", 'rb').read())
classifier = face_detection.classifier


def predict_image(image_path, size, model: Model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    input = np.reshape(image, (1, ) + image.shape)
    predictions = model.predict(input)
    print(predictions)
    prediction = predictions[0]
    if prediction < 0.5:
        label = le.classes_[0]
    else:
        label = le.classes_[1]
    bboxes = face_detection.face_detect(classifier, image, 1.1, 10)
    for box in bboxes:
        x, y, w, h = box
        label = "{}: {:.4f}".format(label, prediction)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(image, (x, y), (x+w, y+h),
                      (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey()

def webcam(size, model: Model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Can't find camera")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
        bboxes = face_detection.face_detect(classifier, frame, 1.1, 10)
        for box in bboxes:
            x, y, w, h = box
            ROI = frame[y:y + h, x:x + w]
            face = cv2.resize(ROI, (size, size))
            face = np.reshape(face, (1, ) + face.shape)
            predictions = model.predict(face)
            print(predictions)
            prediction = float(predictions[0])
            if prediction < 0.5:
                label = le.classes_[0]
            else:
                label = le.classes_[1]
            label = "{}: {:.4f}".format(label, prediction)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
        cv2.imshow("Cam", frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    model = load_model("Saved Models/LivenessNet 20210507-002857")
    webcam(64, model)
