from abc import ABCMeta, abstractmethod, abstractproperty
import pybullet as p
import time
import pybullet_data
from src.rl.constants import *


class Quadruped:
    def __init__(self, GUI = False):
        flag = p.DIRECT
        if GUI:
             flag = p.GUI
        self.physicsClient = p.connect(flag)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")
        self.cubeStartPos = [0, 0 ,0]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

    def step(self):
        return [tf.zeros(spec.shape, spec.dtype) for spec in observation_spec[:-1]]

    def sync_sleep_time(self):
        """
        Time to sleep to allow the joints to reach their targets
        """
        pass

    def robot_handle(self):
        """
        Stores the handle to the robot
        This handle is used to invoke methods on the robot
        """
        pass

    def interpolation(self):
        """
        Flag to indicate if intermediate joint angles should be interpolated
        """
        pass

    def fraction_max_speed(self):
        """
        Fraction of the maximum motor speed to use
        """
        pass

    def wait(self):
        """
        Flag to indicate whether the control should wait for each angle to reach its target
        """
        pass

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

    def get_angles(self, joint_names):
        """
        Gets the angles of the specified joints and returns a dict of joint_names: angles (in radians)
        :type joint_names: list(str)
        :param joint_names: List of joint names
        :rtype: dict
        """
        pass

    def render(self):
        self.robotId = p.loadURDF(
            "../quadruped_description/urdf/quadruped.urdf",
            self.cubeStartPos, 
            self.cubeStartOrientation,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_MERGE_FIXED_LINKS
        )
        for i in range (10000):
            p.stepSimulation()
            time.sleep(1./240.)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotId)
        print(cubePos,cubeOrn)
        p.disconnect()
