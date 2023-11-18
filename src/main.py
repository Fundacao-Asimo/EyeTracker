import os
import cv2
import sys

from src.model.FaceTracker import FaceTracker

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, root_dir)


def main():
    image_shape = (640, 480)
    model = os.path.join(root_dir, "res", "face_landmarker.task")
    face = FaceTracker(model, 1, 0.5, 0.75, image_shape)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit("ERROR: Unable to read from the webcam. Please verify your webcam settings.")
        cv2.flip(image, 1, image)
        try:
            image = face.detect(image, draw=True)
            image = face.draw_info(image)
        except Exception as e:
            print(e)
        cv2.imshow("EyeTracker", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    main()
