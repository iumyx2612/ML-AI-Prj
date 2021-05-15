import cv2
import numpy as np
import os, sys


classifier_path = os.path.join("../Pre-trained", "haarcascade_frontalface_default.xml")
classifier = cv2.CascadeClassifier(classifier_path)


def face_detect(classifier, image, *args):
    if type(classifier) == cv2.CascadeClassifier:
        bboxes = classifier.detectMultiScale(image, *args)
    else:
        sys.exit("Classifer unknown")
    return bboxes


if __name__ == '__main__':
    image = cv2.imread("../Data/Spoof/1.jpg")
    bboxes = face_detect(classifier, image, 1.1, 4)
    for box in bboxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      (0, 0, 255), 2)
    cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)
    cv2.imshow("Image", image)
    cv2.waitKey()