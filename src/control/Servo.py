from pyfirmata2 import Arduino


class Servo:
    """
        Class to control a servo motor

        Attributes:
            EYE_SERVO_MIN (int): Index for the minimum angle of the servo
            EYE_SERVO_MAX (int): Index for the maximum angle of the servo
            limits (list): The limits of the servo
            pin (Pin): The pin of the servo
            angle (int): The current angle of the servo

        Methods:
            __init__(board, pin): Initialize the Servo class
            __str__(): Get the string representation of the servo
            get_limit(index): Get the limit of the servo at the given index
            get_max_limit(): Get the maximum angle of the servo
            get_min_limit(): Get the minimum angle of the servo
            set_limit(index, angle): Set the limit of the servo at the given index
            set_max(angle): Set the maximum angle of the servo
            set_min(angle): Set the minimum angle of the servo
            write(angle): Write the given angle to the servo
            read(): Read the current angle of the servo
            attach(): Attach the servo
            detach(): Detach the servo
    """

    EYE_SERVO_MIN = 0
    EYE_SERVO_MAX = 1

    def __init__(self,
                 board: Arduino,
                 pin: int) -> None:
        """
            Initialize the Servo class.

            Args:
                board (Arduino): The Arduino board
                pin (int): The pin of the servo

            Returns:
                None
        """

        self.limits = [0, 180]
        self.pin = board.get_pin("d:{}:s".format(pin))
        self.angle = 0

    def __str__(self) -> str:
        """
            Get the string representation of the servo

            Args:

            Returns:
                str: The string representation of the servo
        """

        return "{} {{ {} ; {} ; {} }}".format(self.pin.is_output(), *self.limits)

    def get_limit(self,
                  index: int) -> int:
        """
            Get the limit of the servo at the given index

            Args:
                index (int): The index of the limit to get (0 for min, 1 for max)

            Returns:
                int: The limit of the servo at the given index
        """

        index = max(0, min(index, Servo.EYE_SERVO_MAX))

        return self.limits[index]

    def get_max_limit(self) -> int:
        """
            Get the maximum angle of the servo

            Args:

            Returns:
                int: The maximum angle of the servo
        """

        return self.get_limit(index = Servo.EYE_SERVO_MAX)

    def get_min_limit(self) -> int:
        """
            Get the minimum angle of the servo

            Args:

            Returns:
                int: The minimum angle of the servo
        """

        return self.get_limit(index = Servo.EYE_SERVO_MIN)

    def set_limit(self,
                  index: int,
                  angle: int) -> None:
        """
            Set the limit of the servo at the given index

            Args:
                index (int): The index of the limit to set (0 for min, 1 for max)
                angle (int): The angle of the limit to set

            Returns:
                None
        """

        index = max(0, min(index, Servo.EYE_SERVO_MAX))
        angle = max(0, min(angle, 180))
        self.limits[index] = angle

        if self.limits[Servo.EYE_SERVO_MAX] < self.limits[Servo.EYE_SERVO_MIN]:
            (self.limits[Servo.EYE_SERVO_MIN],
             self.limits[Servo.EYE_SERVO_MAX]) = (self.limits[Servo.EYE_SERVO_MAX],
                                                  self.limits[Servo.EYE_SERVO_MIN]
                                                  )

        return None

    def set_max(self,
                angle: int) -> None:
        """
            Set the maximum angle of the servo

            Args:
                angle (int): The maximum angle of the servo

            Returns:
                None
        """

        self.set_limit(index = Servo.EYE_SERVO_MAX, angle = angle)

        return None

    def set_min(self,
                angle: int) -> None:
        """
            Set the minimum angle of the servo

            Args:
                angle (int): The minimum angle of the servo

            Returns:
                None
        """

        self.set_limit(index = Servo.EYE_SERVO_MIN, angle = angle)

        return None

    def write(self,
              angle: int) -> None:
        """
            Write the angle to the servo

            Args:
                angle (int): The angle to write to the servo

            Returns:
                None
        """

        angle = max(self.get_min_limit(),
                    min(angle, self.get_max_limit())
                    )

        if self.angle != angle:
            self.pin.write(angle)
            self.angle = angle

        return None

    def read(self) -> int:
        """
            Read the angle of the servo

            Args:

            Returns:
                int: The angle of the servo
        """

        return self.angle

    def attach(self,
               angle: int = None) -> None:
        """
            Attach the servo with the mean of the limits as the initial angle

            Args:
                angle (int): The initial angle of the servo

            Returns:
                None
        """

        if angle is None:
            angle = sum(self.limits) // 2
        self.write(angle)

        return None

    def detach(self) -> None:
        """
            Detach the servo

            Args:

            Returns:
                None
        """

        self.pin.write(0)
        self.angle = 0

        return None
