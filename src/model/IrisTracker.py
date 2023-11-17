import time
import math

import cv2 as cv
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


class IrisTracker:
    def __init__(self,
                 model: str,
                 num_iris: int,
                 min_iris_detection_confidence: float,
                 min_iris_presence_confidence: float,
                 min_tracking_confidence: float,
                 ):
        self.model = model
        self.detector = self.initialize_detector(num_iris,
                                                 min_iris_detection_confidence,
                                                 min_iris_presence_confidence,
                                                 min_tracking_confidence,
                                                 )
        self.mp_iris = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.fps_avg_frame_count = 30
        self.COUNTER = 0
        self.FPS = 0
        self.START_TIME = time.time()
        self.DETECTION_RESULT = None
        self.LEFT_IRIS_IDS = [474, 475, 476, 477]
        self.RIGHT_IRIS_IDS = [469, 470, 471, 472]

    def save_result(self,
                    result: landmark_pb2.NormalizedLandmarkList,
                    unused_output_image,
                    timestamp_ms: int,
                    ):
        pass

    def initialize_detector(self,
                            num_iris: int,
                            min_iris_detection_confidence: float,
                            min_iris_presence_confidence: float,
                            min_tracking_confidence: float,
                            ):
        base_options = python.BaseOptions(model_asset_path = self.model)
        options = vision.HandLandmarkerOptions(base_options = base_options,
                                               running_mode = vision.RunningMode.LIVE_STREAM,
                                               num_iris = num_iris,
                                               min_iris_detection_confidence = min_iris_detection_confidence,
                                               min_iris_presence_confidence = min_iris_presence_confidence,
                                               min_tracking_confidence = min_tracking_confidence,
                                               result_callback = self.save_result,
                                               )
        return vision.HandLandmarker.create_from_options(options)

    def draw_circle_landmarks(self,
                              image: np.ndarray,
                              text_color: tuple = (255, 255, 255),
                              font_size: int = 1,
                              font_thickness: int = 1,
                              ) -> np.ndarray:
        pass

    def detect(self,
               frame: np.ndarray,
               draw: bool = False,
               ) -> np.ndarray:
        pass
