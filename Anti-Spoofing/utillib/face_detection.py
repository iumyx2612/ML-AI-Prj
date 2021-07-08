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
