import os

from src.control.Servo import Servo
from pyfirmata2 import Arduino

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, root_dir)


class AnimatronicEye:
    """
        Class to control the animatronic eye

        Attributes:
            PIN_VERTICAL (int): The pin for the vertical servo
            PIN_HORIZONTAL (int): The pin for the horizontal servo
            PIN_EYELID_LOWER (int): The pin for the lower eyelid servo
            PIN_EYELID_UPPER (int): The pin for the upper eyelid servo
            SERVOS_LIMITS (dict): The limits of the servos
            servos (dict): The servos

        Methods:
            __init__(board): Initialize the AnimatronicEye class
            __str__(): Get the string representation of the animatronic eye
            initialize_sensors(): Initialize the sensors
            control_servo(name, angle): Control the servo by name
            control_servos(vertical, horizontal, eyelid_lower, eyelid_upper): Control the servos
            get_servo_info(name): Get the information of the servo
            close(): Close the servos
    """

    PIN_VERTICAL = 2
    PIN_HORIZONTAL = 3
    PIN_EYELID_LOWER = 4
    PIN_EYELID_UPPER = 5

    SERVOS_LIMITS = {
        'vertical': [0, 180],
        'horizontal': [0, 180],
        'eyelid_lower': [0, 90],
        'eyelid_upper': [0, 90]
    }

    def __init__(self,
                 board: Arduino) -> None:
        """
            Initialize the AnimatronicEye class

            Args:
                board (Arduino): The Arduino board

            Returns:
                None
        """

        self.board = board

        self.servos = {
            'vertical': Servo(self.board, AnimatronicEye.PIN_VERTICAL),
            'horizontal': Servo(self.board, AnimatronicEye.PIN_HORIZONTAL),
            'eyelid_lower': Servo(self.board, AnimatronicEye.PIN_EYELID_LOWER),
            'eyelid_upper': Servo(self.board, AnimatronicEye.PIN_EYELID_UPPER)
        }

    def __str__(self) -> str:
        """
            Get the string representation of the animatronic eye

            Args:

            Returns:
                str: The string representation of the animatronic eye
        """

        board_info = self.board.__name__
        servos_info = ', '.join([str(servo) for servo in self.servos.values()])
        return '{} {{ {} }}'.format(board_info, servos_info)

    def initialize_sensors(self) -> None:
        """
            Initialize the sensors

            Args:

            Returns:
                None
        """

        for name, limits in self.SERVOS_LIMITS.items():
            self.servos[name].set_min(limits[0])
            self.servos[name].set_max(limits[1])
            self.servos[name].attach(limits[0])                     \
                if name == 'eyelid_lower' or name == 'eyelid_upper' \
                else self.servos[name].attach()

        return None

    def control_servo(self,
                      name: str,
                      angle: int) -> None:
        """
            Control the servo by name

            Args:
                name (str): The name of the servo
                angle (int): The angle to set

            Returns:
                 None
        """

        if name in self.servos:
            self.servos[name].write(angle)

        return None

    def control_servos(self,
                       vertical: int,
                       horizontal: int,
                       eyelid_lower: int,
                       eyelid_upper: int) -> None:
        """
            Control the servos

            Args:
                vertical (int): The vertical angle
                horizontal (int): The horizontal angle
                eyelid_lower (int): The eyelid lower angle
                eyelid_upper (int): The eyelid upper angle

            Returns:
                None
        """

        self.control_servo(name = 'vertical',
                           angle = vertical
                           )
        self.control_servo(name = 'horizontal',
                           angle = horizontal
                           )
        self.control_servo(name = 'eyelid_lower',
                           angle = eyelid_lower
                           )
        self.control_servo(name = 'eyelid_upper',
                           angle = eyelid_upper
                           )

        return None

    def get_servo_info(self,
                       name: str) -> str:
        """
            Get the information of the servo

            Args:
                name (str): The name of the servo

            Returns:
                str: The information of the servo
        """

        return str(self.servos[name]) if name in self.servos else None

    def close(self) -> None:
        """
            Close the servos

            Args:

            Returns:
                None
        """

        for servo in self.servos.values():
            servo.detach()

        return None
