from abc import ABCMeta, abstractmethod, abstractproperty

class Robot(metaclass = ABCMeta):
    """
    Abstract class for robot specific functions
    """

    def __init__(self):
        """
        The constructor of the abstract class
        """
        pass

    @abstractproperty
    def sync_sleep_time(self):
        """
        Time to sleep to allow the joints to reach their targets
        """
        pass

    @abstractproperty
    def robot_handle(self):
        """
        Stores the handle to the robot
        This handle is used to invoke methods on the robot
        """
        pass

    @abstractproperty
    def interpolation(self):
        """
        Flag to indicate if intermediate joint angles should be interpolated
        """
        pass

    @abstractproperty
    def fraction_max_speed(self):
        """
        Fraction of the maximum motor speed to use
        """
        pass

    @abstractproperty
    def wait(self):
        """
        Flag to indicate whether the control should wait for each angle to reach its target
        """
        pass

    @abstractmethod
    def set_angles(self, joint_angles, duration=None, joint_velocities=None):
        """
        Sets the joints to the specified angles
        :type joint_angles: dict
        :param joint_angles: Dictionary of joint_names: angles (in radians)
        :type duration: float
        :param duration: Time to reach the angular targets (in seconds)
        :type joint_velocities: dict
        :param joint_velocities: dict of joint angles and velocities
        :return: None
        """
        pass

    @abstractmethod
    def get_angles(self, joint_names):
        """
        Gets the angles of the specified joints and returns a dict of joint_names: angles (in radians)
        :type joint_names: list(str)
        :param joint_names: List of joint names
        :rtype: dict
        """
        pass


