import mediapipe as mp
import numpy as np

import time
import cv2

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python


class FaceTracker:
    """
        Class to track the face using MediaPipe FaceMesh

        Attributes:
            model_path (str): The path to the model file
            detector (vision.FaceLandmarker): The face detector
            image_shape (tuple): The resolution of the webcam
            mp_face_mesh (mp.solutions.face_mesh): The MediaPipe FaceMesh instance
            mp_drawing (mp.solutions.drawing_utils): The MediaPipe drawing utilities
            mp_drawing_styles (mp.solutions.drawing_styles): The MediaPipe drawing styles
            fps_avg_frame_counter (int): The frame counter for calculating the FPS
            COUNTER (int): The frame counter
            FPS (int): The FPS
            START_TIME (float): The start time
            DETECTION_RESULT (landmark_pb2.NormalizedLandmarkList): The detection result

        Methods:
            __init__(model_path, num_faces, min_detection_confidence, min_tracking_confidence, image_shape): Initialize the FaceTracker class
            save_result(result, timestamp_ms, unused_output_image): Save the detection result
            initialize_detector(num_faces, min_detection_confidence, min_tracking_confidence): Initialize the face detector
            get_face_landmarks(idxs): Get the face landmarks
            draw_landmarks(image, text_color, font_size, font_thickness): Draw the landmarks on the image
            detect(image, draw): Detect the face in the frame
            draw_info(image, rect_color, text_color): Draw a rectangle with the face information on the image
    """

    def __init__(self,
                 model_path: str,
                 num_faces: int,
                 min_detection_confidence: float,
                 min_tracking_confidence: float,
                 image_shape: tuple) -> None:
        """
            Initialize the FaceTracker class.

            Args:
                model_path (str): The path to the model file
                num_faces (int): The number of faces to detect
                min_detection_confidence (float): The minimum confidence to detect a face
                min_tracking_confidence (float): The minimum confidence to track a face
                image_shape (tuple): The resolution of the webcam

            Returns:
                None
        """

        self.model_path = model_path
        self.detector = self.initialize_detector(num_faces,
                                                 min_detection_confidence,
                                                 min_tracking_confidence
                                                 )
        self.image_shape = image_shape
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.fps_avg_frame_counter = 30
        self.COUNTER = 0
        self.FPS = 0
        self.START_TIME = time.time()
        self.DETECTION_RESULT = None

    # noinspection PyUnusedLocal
    def save_result(self,
                    result: landmark_pb2.NormalizedLandmarkList,
                    unused_timestamp_ms: int,
                    unused_output_image) -> None:
        """
            Save the detection result

            Args:
                result (landmark_pb2.NormalizedLandmarkList): The detection result
                unused_timestamp_ms (int): Unused timestamp of the detection
                unused_output_image: Unused output image

            Returns:
                None
        """

        if self.COUNTER % self.fps_avg_frame_counter == 0:
            self.FPS = self.fps_avg_frame_counter / (time.time() - self.START_TIME)
            self.START_TIME = time.time()

        self.DETECTION_RESULT = result if len(result.face_landmarks) else None
        self.COUNTER += 1

        return None

    def initialize_detector(self,
                            num_faces: int,
                            min_detection_confidence: float,
                            min_tracking_confidence: float) -> vision.FaceLandmarker:
        """
            Initialize the face detector

            Args:
                num_faces (int): The number of faces to detect
                min_detection_confidence (float): The minimum confidence to detect a face
                min_tracking_confidence (float): The minimum confidence to track a face

        Returns:
            mediapipe.FaceLandmarker: FaceLandmarker instance
        """

        base_options = python.BaseOptions(model_asset_path = self.model_path)
        options = vision.FaceLandmarkerOptions(base_options = base_options,
                                               running_mode = vision.RunningMode.LIVE_STREAM,
                                               num_faces = num_faces,
                                               min_face_detection_confidence = min_detection_confidence,
                                               min_tracking_confidence = min_tracking_confidence,
                                               output_face_blendshapes = True,
                                               result_callback = self.save_result
                                               )

        return vision.FaceLandmarker.create_from_options(options)

    def get_face_landmarks(self,
                           idxs: list = None) -> list:
        """
            Get the face landmarks

            Args:
                idxs (list): The indices of the landmarks to get

            Returns:
                list: The face landmarks
        """

        if self.DETECTION_RESULT:
            return [self.DETECTION_RESULT.face_landmarks[0][idx] for idx in idxs]
        return []

    def draw_landmarks(self,
                       image: np.ndarray,
                       text_color: tuple = (255, 255, 255),
                       font_size: int = 1,
                       font_thickness: int = 1) -> np.ndarray:
        """
            Draw the landmarks on the image

            Args:
                image (np.ndarray): The image to draw the landmarks on
                text_color (tuple): The color of the text
                font_size (int): The size of the font
                font_thickness (int): The thickness of the font

            Returns:
                np.ndarray: The image with the landmarks drawn
        """

        fps_text = "FPS: {:.1f}".format(self.FPS)
        cv2.putText(img = image,
                    text = fps_text,
                    org = (24, 30),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = text_color,
                    thickness = font_thickness,
                    lineType = cv2.LINE_AA
                    )

        if self.DETECTION_RESULT:

            for face_landmarks in self.DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x = landmark.x,
                                                                                      y = landmark.y,
                                                                                      z = landmark.z) for landmark in face_landmarks])
                self.mp_drawing.draw_landmarks(image,
                                               landmark_list = face_landmarks_proto,
                                               connections = self.mp_face_mesh.FACEMESH_FACE_OVAL,
                                               connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                                               landmark_drawing_spec = None
                                               )

        return image

    def detect(self,
               image: np.ndarray,
               draw: bool = False
               ) -> np.ndarray:
        """
            Detect the face in the frame

            Args:
                image (np.ndarray): The frame to detect the face in
                draw (bool): Whether to draw the landmarks on the frame

            Returns:
                np.ndarray: Image with the landmarks drawn if draw is True, else the original image
        """

        rgb_image = cv2.cvtColor(src = image,
                                 code = cv2.COLOR_BGR2RGB
                                 )
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB,
                            data = rgb_image
                            )
        self.detector.detect_async(image = mp_image,
                                   timestamp_ms = time.time_ns() // 1_000_000
                                   )

        return self.draw_landmarks(image = image) if draw else image

    def draw_info(self,
                  image: np.ndarray,
                  rect_color: tuple = (255, 51, 51),
                  text_color: tuple = (0, 128, 255),
                  ) -> np.ndarray:
        """
            Draw a rectangle with the face information on the image

            Args:
                image (np.ndarray): The image to draw on
                rect_color (tuple): The color of the rectangle
                text_color (tuple): The color of the text

            Returns:
                np.ndarray: The image with the rectangle and text drawn
        """

        height, width = self.image_shape

        # Define the rectangle coordinates
        w, h = 410, 60
        x1, y1 = int(width * 0.021), int(height * 0.63)
        x2, y2 = x1 + w, y1 + h

        # Draw the rectangle with rounded corners
        radius, offset, thickness = 10, 10, 3

        # Top left
        cv2.line(img = image,
                 pt1 = (x1 + radius, y1),
                 pt2 = (x1 + radius + offset, y1),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.line(img = image,
                 pt1 = (x1, y1 + radius),
                 pt2 = (x1, y1 + radius + offset),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.ellipse(img = image,
                    center = (x1 + radius, y1 + radius),
                    axes = (radius, radius),
                    angle = 180,
                    startAngle = 0,
                    endAngle = 90,
                    color = rect_color,
                    thickness = thickness
                    )

        # Top right
        cv2.line(img = image,
                 pt1 = (x2 - radius, y1),
                 pt2 = (x2 - radius - offset, y1),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.line(img = image,
                 pt1 = (x2, y1 + radius),
                 pt2 = (x2, y1 + radius + offset),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.ellipse(img = image,
                    center = (x2 - radius, y1 + radius),
                    axes = (radius, radius),
                    angle = 270,
                    startAngle = 0,
                    endAngle = 90,
                    color = rect_color,
                    thickness = thickness
                    )

        # Bottom left
        cv2.line(img = image,
                 pt1 = (x1 + radius, y2),
                 pt2 = (x1 + radius + offset, y2),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.line(img = image,
                 pt1 = (x1, y2 - radius),
                 pt2 = (x1, y2 - radius - offset),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.ellipse(img = image,
                    center = (x1 + radius, y2 - radius),
                    axes = (radius, radius),
                    angle = 90,
                    startAngle = 0,
                    endAngle = 90,
                    color = rect_color,
                    thickness = thickness
                    )

        # Bottom right
        cv2.line(img = image,
                 pt1 = (x2 - radius, y2),
                 pt2 = (x2 - radius - offset, y2),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.line(img = image,
                 pt1 = (x2, y2 - radius),
                 pt2 = (x2, y2 - radius - offset),
                 color = rect_color,
                 thickness = thickness
                 )
        cv2.ellipse(img = image,
                    center = (x2 - radius, y2 - radius),
                    axes = (radius, radius),
                    angle = 0,
                    startAngle = 0,
                    endAngle = 90,
                    color = rect_color,
                    thickness = thickness
                    )

        # Draw the text
        text_x, text_y = x1 + int(w * 0.04), y1 + int(h * 0.63)
        nose_position = self.DETECTION_RESULT.face_landmarks[0][168]
        cv2.putText(img = image,
                    text = f'Nose position: ({nose_position.x * self.image_shape[0]:.2f}, {nose_position.y * self.image_shape[1]:.2f})',
                    org = (text_x, text_y),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.75,
                    color = text_color,
                    thickness = 2
                    )

        return image
