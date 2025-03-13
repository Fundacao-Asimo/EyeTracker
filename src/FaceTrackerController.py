import numpy as np

import cv2
import sys

from src.control.AnimatronicEye import AnimatronicEye
from src.model.FaceTracker import FaceTracker
from pyfirmata2 import Arduino
from time import sleep


class FaceTrackerController:
    """
        Class to control the AnimatronicEye to follow the face

        Attributes:
            tracker (FaceTracker): FaceTracker object
            board (Arduino): Arduino object
            controller (AnimatronicEye): AnimatronicEye object
            servos_values (dict): Dictionary of servo names and their angles
            image_shape (tuple): Image resolution (width, height)
            cap (cv2.VideoCapture): VideoCapture object
            track_limits (list): List of the tracking limits
            loop_count (int): The loop count
            blink_delay_value (int): The blink value delay

        Methods:
            __init__(port, model_path, image_shape): Initialize the FaceTrackerController class
            clear(): Clear the servos
            draw_limits_rectangle(image): Draw the limits rectangle on the image
            draw_info(image, text_color): Draw the servo angle information on the image
            follow_face(landmark): Follow the face with the AnimatronicEye
            blink(image): Blink the eyelids
            loop(): Main loop for tracking and controlling the AnimatronicEye
    """

    def __init__(self,
                 port: str,
                 model_path: str,
                 image_shape: tuple = (480, 640)) -> None:
        """
            Initialize the FaceTrackerController class

            Args:
                port (str): The port of the Arduino board
                model_path (str): The path to the model
                image_shape (tuple): The image resolution

            Returns:
                None
        """

        self.tracker = FaceTracker(model_path = model_path,
                                   num_faces = 1,
                                   min_detection_confidence = 0.5,
                                   min_tracking_confidence = 0.5,
                                   image_shape = image_shape
                                   )

        try:
            self.board = Arduino(port)
            self.controller = AnimatronicEye(self.board)
        except Exception as ex:
            print('Error: ', ex)
            sys.exit('ERROR: Unable to connect to the Arduino board. Please verify your Arduino port.')

        self.servos_values = {
            'vertical': 0,
            'horizontal': 0,
            'eyelid_lower': 0,
            'eyelid_upper': 0
        }

        self.image_shape = image_shape

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.track_limits = [[40, self.image_shape[1] - 40], [50, self.image_shape[0] - 50]]

        self.loop_count = 0
        self.blink_delay_value = 25

    def clear(self) -> None:
        """
            Clear the servos.

            Args:

            Returns:
                None
        """

        self.tracker.detector.close()
        self.board.exit()
        self.cap.release()
        cv2.destroyAllWindows()

    def draw_limits_rectangle(self,
                              image: np.ndarray) -> None:
        """
            Draw the limits rectangle on the image

            Args:
                image (np.ndarray): The image to draw on

            Returns:
                None
        """
        def draw_rect_fancy(frame: np.ndarray,
                            pt1: tuple,
                            pt2: tuple,
                            color: tuple,
                            thickness: int,
                            radius: int,
                            offset: int) -> None:
            """
                Draw a fancy rectangle on the image

                Args:
                    frame (np.ndarray): The image to draw on
                    pt1 (tuple): The first point of the rectangle
                    pt2 (tuple): The second point of the rectangle
                    color (tuple): The color of the rectangle
                    thickness (int): The thickness of the rectangle
                    radius (int): The radius of the corners
                    offset (int): The offset of the corners

                Returns:
                    None
            """

            x1, y1 = pt1
            x2, y2 = pt2

            # Top left
            cv2.line(img = frame,
                     pt1 = (x1 + radius, y1),
                     pt2 = (x1 + radius + offset, y1),
                     color = color,
                     thickness = thickness
                     )
            cv2.line(img = frame,
                     pt1 = (x1, y1 + radius),
                     pt2 = (x1, y1 + radius + offset),
                     color = color,
                     thickness = thickness
                     )
            cv2.ellipse(img = frame,
                        center = (x1 + radius, y1 + radius),
                        axes = (radius, radius),
                        angle = 180,
                        startAngle = 0,
                        endAngle = 90,
                        color = color,
                        thickness = thickness
                        )

            # Top right
            cv2.line(img = frame,
                     pt1 = (x2 - radius, y1),
                     pt2 = (x2 - radius - offset, y1),
                     color = color,
                     thickness = thickness
                     )
            cv2.line(img = frame,
                     pt1 = (x2, y1 + radius),
                     pt2 = (x2, y1 + radius + offset),
                     color = color,
                     thickness = thickness
                     )
            cv2.ellipse(img = frame,
                        center = (x2 - radius, y1 + radius),
                        axes = (radius, radius),
                        angle = 270,
                        startAngle = 0,
                        endAngle = 90,
                        color = color,
                        thickness = thickness
                        )

            # Bottom left
            cv2.line(img = frame,
                     pt1 = (x1 + radius, y2),
                     pt2 = (x1 + radius + offset, y2),
                     color = color,
                     thickness = thickness
                     )
            cv2.line(img = frame,
                     pt1 = (x1, y2 - radius),
                     pt2 = (x1, y2 - radius - offset),
                     color = color,
                     thickness = thickness
                     )
            cv2.ellipse(img = frame,
                        center = (x1 + radius, y2 - radius),
                        axes = (radius, radius),
                        angle = 90,
                        startAngle = 0,
                        endAngle = 90,
                        color = color,
                        thickness = thickness
                        )

            # Bottom right
            cv2.line(img = frame,
                     pt1 = (x2 - radius, y2),
                     pt2 = (x2 - radius - offset, y2),
                     color = color,
                     thickness = thickness
                     )
            cv2.line(img = frame,
                     pt1 = (x2, y2 - radius),
                     pt2 = (x2, y2 - radius - offset),
                     color = color,
                     thickness = thickness
                     )
            cv2.ellipse(img = frame,
                        center = (x2 - radius, y2 - radius),
                        axes = (radius, radius),
                        angle = 0,
                        startAngle = 0,
                        endAngle = 90,
                        color = color,
                        thickness = thickness
                        )

            return None

        px1, py1 = self.track_limits[0][0], self.track_limits[1][0]
        px2, py2 = self.track_limits[0][1], self.track_limits[1][1]
        rect_color: tuple = (255, 51, 51)

        draw_rect_fancy(frame = image,
                        pt1 = (px1, py1),
                        pt2 = (px2, py2),
                        color = rect_color,
                        thickness = 2,
                        radius = 20,
                        offset = 20
                        )

        return None

    def draw_info(self,
                  image: np.ndarray,
                  text_color: tuple = (0, 128, 255)) -> None:
        """
            Draw the servo angle information on the image

            Args:
                image (np.ndarray): The image to draw on
                text_color (tuple): The color of the text

            Returns:
                None
        """

        text_x, text_y = 55, 325

        for index, (name, servo) in enumerate(self.controller.servos.items()):
            cv2.putText(img = image,
                        text = "{}: {}".format(name.title(), servo.read()),
                        org = (text_x, text_y + index * 30),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.8,
                        color = text_color,
                        thickness = 2
                        )

        return None

    def follow_face(self,
                    landmark: int = 0) -> None:
        """
            Follow the face with the AnimatronicEye

            Args:
                landmark (int): The landmark to follow (0 for the face)

            Returns:
                None
        """

        landmark = self.tracker.get_face_landmarks(idxs = [landmark])
        if len(landmark) > 0:

            x, y = landmark[0].x, landmark[0].y

            self.servos_values['vertical'] = int(
                np.interp(
                    y * self.image_shape[0],
                    self.track_limits[0],
                    [
                        self.controller.servos['vertical'].get_min_limit(),
                        self.controller.servos['vertical'].get_max_limit()
                    ]
                )
            )

            self.servos_values['horizontal'] = int(
                np.interp(
                    x * self.image_shape[1],
                    self.track_limits[1],
                    [
                        self.controller.servos['horizontal'].get_min_limit(),
                        self.controller.servos['horizontal'].get_max_limit()
                    ]
                )
            )

            self.servos_values['eyelid_lower'] = self.controller.servos['eyelid_lower'].get_min_limit()

            self.servos_values['eyelid_upper'] = self.controller.servos['eyelid_upper'].get_min_limit()

            self.controller.control_servos(**self.servos_values)

        else:
            self.controller.initialize_sensors()

    def blink(self) -> None:
        """
            Blink the eyelids

            Args:

            Returns:
                None
        """

        phase = (self.loop_count // self.blink_delay_value) % 6

        if phase % 2 == 0:
            self.servos_values['eyelid_lower'] = self.controller.servos['eyelid_lower'].get_max_limit()
            self.servos_values['eyelid_upper'] = self.controller.servos['eyelid_upper'].get_max_limit()

        else:
            self.servos_values['eyelid_lower'] = self.controller.servos['eyelid_lower'].get_min_limit()
            self.servos_values['eyelid_upper'] = self.controller.servos['eyelid_upper'].get_min_limit()

        return None

    def loop(self) -> None:
        """
            Main loop for tracking and controlling the AnimatronicEye

            Args:

            Returns:
                None
        """

        print('Initializing', end = '')
        for index in range(10):
            sleep(0.1)
            print('.', end = '')
        print('')

        while self.cap.isOpened():
            success, image = self.cap.read()

            if not success:
                sys.exit('ERROR: Unable to read from the webcam. Please verify your webcam settings.')

            cv2.flip(src = image,
                     flipCode = 1,
                     dst = image
                     )

            try:
                image = self.tracker.detect(image, draw = True)
                if self.loop_count < 150:
                    self.blink()
                elif self.loop_count == 600:
                    self.loop_count = 0
                self.loop_count += 1
                self.follow_face()
                self.draw_info(image)
                self.draw_limits_rectangle(image)
            except Exception as ex:
                print(ex)

            cv2.imshow(winname = 'FaceTracker',
                       mat = image
                       )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.clear()
