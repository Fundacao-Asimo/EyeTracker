import time
import cv2

import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


class FaceTracker:
    def __init__(self,
                 model: str,
                 num_faces: int,
                 min_detection_confidence: float,
                 min_tracking_confidence: float,
                 image_shape: tuple):
        """
        Initialize a HandTracker instance.

        Args:
            model (str): The path to the model for hand tracking.
            num_faces (int): Maximum number of faces to detect.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for successful face detection.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for successful face landmark tracking.
            image_shape (tuple): The resolution of the webcam
        """
        self.model = model
        self.image_shape = image_shape
        self.detector = self.initialize_detector(num_faces,
                                                 min_detection_confidence,
                                                 min_tracking_confidence,
                                                 )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.fps_avg_frame_count = 30
        self.COUNTER = 0
        self.FPS = 0
        self.START_TIME = time.time()
        self.DETECTION_RESULT = None

    def save_result(self,
                    result: landmark_pb2.NormalizedLandmarkList,
                    unused_output_image,
                    timestamp_ms: int,
                    ):
        """
        Saves the result of the detection.

        Args:
            result (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Result of the detection.
            unused_output_image (mediapipe.framework.formats.image_frame.ImageFrame): Unused.
            timestamp_ms (int): Timestamp of the detection.

        Returns:
            None
        """
        if self.COUNTER % self.fps_avg_frame_count == 0:
            self.FPS = self.fps_avg_frame_count / (time.time() - self.START_TIME)
            self.START_TIME = time.time()
        self.DETECTION_RESULT = result
        self.COUNTER += 1

    def initialize_detector(self,
                            num_faces: int,
                            min_detection_confidence: float,
                            min_tracking_confidence: float,
                            ):
        """
        Initializes the HandLandmarker instance.

        Args:
            num_faces (int): Maximum number of faces to detect.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for the face landmarks to be considered tracked successfully.

        Returns:
            mediapipe.HandLandmarker: HandLandmarker instance.
        """
        base_options = python.BaseOptions(model_asset_path = self.model)
        options = vision.FaceLandmarkerOptions(base_options = base_options,
                                               running_mode = vision.RunningMode.LIVE_STREAM,
                                               num_faces = num_faces,
                                               min_face_detection_confidence = min_detection_confidence,
                                               min_tracking_confidence = min_tracking_confidence,
                                               output_face_blendshapes = True,
                                               result_callback = self.save_result)
        return vision.FaceLandmarker.create_from_options(options)

    def draw_landmarks(self,
                       image: np.ndarray,
                       text_color: tuple = (255, 255, 255),
                       font_size: int = 1,
                       font_thickness: int = 1,
                       ) -> np.ndarray:
        """
        Draws the landmarks and handedness on the image.

        Args:
            image (numpy.ndarray): Image on which to draw the landmarks.
            text_color (tuple, optional): Color of the text. Defaults to (0, 0, 0).
            font_size (int, optional): Size of the font. Defaults to 1.
            font_thickness (int, optional): Thickness of the font. Defaults to 1.

        Returns:
            numpy.ndarray: Image with the landmarks drawn.
        """
        fps_text = "FPS = {:.1f}".format(self.FPS)
        cv2.putText(image,
                    fps_text,
                    (24, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                    )
        if self.DETECTION_RESULT:
            for face_landmarks in self.DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x = landmark.x,
                                                    y = landmark.y,
                                                    z = landmark.z
                                                    ) for landmark in face_landmarks
                ])
                self.mp_drawing.draw_landmarks(image = image,
                                               landmark_list = face_landmarks_proto,
                                               connections = self.mp_face_mesh.FACEMESH_IRISES,
                                               landmark_drawing_spec = None,
                                               connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                                               # connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                                               )
        return image

    def detect(self,
               frame: np.ndarray,
               draw: bool = False,
               ) -> np.ndarray:
        """
        Detects hands in the image.

        Args:
            frame (numpy.ndarray): Image in which to detect the hands.
            draw (bool, optional): Whether to draw the landmarks on the image. Defaults to False.

        Returns:
            numpy.ndarray: Image with the landmarks drawn if draw is True, else the original image.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_image)
        self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        return self.draw_landmarks(frame) if draw else frame

    def draw_info(self,
                  image: np.ndarray,
                  rect_color: tuple = (255, 51, 51),
                  text_color: tuple = (0, 128, 255),
                  ) -> np.ndarray:
        """
        Draw a rectangle and servo angle information on the image.

        Args:
            image (np.ndarray): The image to draw on.
            rect_color (tuple): The color of the rectangle.
            text_color (tuple): The color of the text.
        """

        # Get the height and width of the image
        height, width = self.image_shape

        # Define origin point and rectangle size
        x1, y1 = int(width * 0.021), int(height * 0.5)
        w, h = 300, 150
        x2, y2 = x1 + w, y1 + h

        # Draw the rectangle with rounded corners
        r, d, thickness = 20, 20, 3
        # Top left
        cv2.line(image, (x1 + r, y1), (x1 + r + d, y1), rect_color, thickness)
        cv2.line(image, (x1, y1 + r), (x1, y1 + r + d), rect_color, thickness)
        cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, rect_color, thickness)
        # Top right
        cv2.line(image, (x2 - r, y1), (x2 - r - d, y1), rect_color, thickness)
        cv2.line(image, (x2, y1 + r), (x2, y1 + r + d), rect_color, thickness)
        cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, rect_color, thickness)
        # Bottom left
        cv2.line(image, (x1 + r, y2), (x1 + r + d, y2), rect_color, thickness)
        cv2.line(image, (x1, y2 - r), (x1, y2 - r - d), rect_color, thickness)
        cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90, 0, 90, rect_color, thickness)
        # Bottom right
        cv2.line(image, (x2 - r, y2), (x2 - r - d, y2), rect_color, thickness)
        cv2.line(image, (x2, y2 - r), (x2, y2 - r - d), rect_color, thickness)
        cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0, 0, 90, rect_color, thickness)

        # Set the initial position of the text inside the rectangle
        text_x, text_y = x1 + int(w * 0.09), y1 + int(h * 0.25)
        # Write each line of text on the image

        # First info
        left_iris_pos = self.DETECTION_RESULT.face_landmarks[0][473]
        cv2.putText(image,
                    f"Left iris: ({left_iris_pos.x * self.image_shape[0]:.2f}; {left_iris_pos.y * self.image_shape[1]:.2f})",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    2,
                    )

        # Second info
        right_iris_pos = self.DETECTION_RESULT.face_landmarks[0][468]
        cv2.putText(image,
                    f"Right iris: ({right_iris_pos.x * self.image_shape[0]:.2f}; {right_iris_pos.y * self.image_shape[1]:.2f})",
                    (text_x, text_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    2,
                    )

        # Third info
        left_iris_ratio = self.DETECTION_RESULT.face_blendshapes[0][9].score
        cv2.putText(image,
                    f"Left eye ratio: {left_iris_ratio:.2f}",
                    (text_x, text_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    2,
                    )

        # Fourth info
        right_iris_ratio = self.DETECTION_RESULT.face_blendshapes[0][10].score
        cv2.putText(image,
                    f"Right eye ratio: {right_iris_ratio:.2f}",
                    (text_x, text_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    2,
                    )
        if left_iris_ratio > 0.45 and right_iris_ratio > 0.45:
            fps_text = "Blinked!"
            cv2.putText(image,
                        fps_text,
                        (24, 60),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                        )
        return image
