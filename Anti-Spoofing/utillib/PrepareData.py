import cv2
from utillib import face_detection
import sys, os

classifier = face_detection.classifier


def cropping_faces_loop(data_dir, *args):
    ROIs = []
    for file in os.listdir(data_dir):
        image_path = os.path.join(data_dir, file)
        image = cv2.imread(image_path)
        bboxes = face_detection.face_detect(classifier, image, *args)
        for box in bboxes:
            x, y, h, w = box
            ROI = image[y:y+h, x:x+w]
            ROIs.append(ROI)
    return ROIs


def main(data_dir, *args, _result_dir="Result", type="Real", extension=".jpg"):
    data_dir = os.path.join("Raw Data", data_dir)
    result_dir = os.path.join(_result_dir, type)
    try:
        os.makedirs(result_dir, exist_ok=True)
    except OSError as e:
        print("Can't create directory " + result_dir)
    ROIs = cropping_faces_loop(data_dir, *args)
    counter = 0
    for roi in ROIs:
        cv2.imwrite(os.path.join(result_dir, str(counter) + extension), roi)
        counter += 1


if __name__ == '__main__':
    main("TestData", 1.1, 10, _result_dir="../Data", type="Spoof")
