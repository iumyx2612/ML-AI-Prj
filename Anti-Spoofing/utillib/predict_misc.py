from tensorflow.keras.models import Model
import cv2
import numpy as np
from utillib import face_detection
from utillib.utils import load_images_with_labels
import pickle
import sys

le = pickle.loads(open("LabelEncoders", 'rb').read())
classifier = face_detection.classifier


def map_output_prediction(predictions: np.ndarray):
    preds = []
    if predictions.ndim == 1:
        if predictions < 0.5:
            return le.classes_[0]
        else:
            return le.classes_[1]
    elif predictions.ndim == 2:
        for prediction in predictions:
            if prediction < 0.5:
                preds.append(le.classes_[0])
            else:
                preds.append(le.classes_[1])
    return preds


def predict_batch_image(dir, model: Model, size=64, batch_size=32):
    datas, labels = load_images_with_labels(dir, size)
    preds = model.predict(datas)
    map_preds = map_output_prediction(preds)
    return map_preds, labels


def predict_single_image(image_path, model: Model, size=64):
    image = cv2.imread(image_path)
    fake_img = cv2.resize(image, (size, size))
    input = np.reshape(fake_img, (1, ) + fake_img.shape)
    predictions = model.predict(input)
    print(predictions)
    prediction = float(predictions[0])
    if prediction < 0.5:
        label = le.classes_[0]
    else:
        label = le.classes_[1]
    bboxes = face_detection.face_detect(classifier, image, 1.1, 10)
    for box in bboxes:
        x, y, w, h = box
        print(x, y, w, h)
        label = "{}: {:.4f}".format(label, prediction)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(image, (x, y), (x+w, y+h),
                      (0, 0, 255), 2)
    cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)
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