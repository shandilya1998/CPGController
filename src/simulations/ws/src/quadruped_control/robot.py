from abc import ABCMeta, abstractmethod, abstractproperty
import pybullet as p
import time
import pybullet_data
import tensorflow as tf
from rl.constants import *

"""
    Refer to the following link for pybullet related information
    https://github.com/moribots/plen_ml_walk/blob/master/plen_bullet/src/plen_bullet/plen_env.py
"""


class Quadruped:
    def __init__(self, params, GUI = False):
        self.params = params
        flag = p.DIRECT
        if GUI:
             flag = p.GUI
        
        self.driven_joints = [
            b'Rev17', # Leg 1
            b'Rev16', # Leg 2
            b'Rev15', # Leg 3
            b'Rev14', # Leg 4
            b'Rev29', # Leg 1
            b'Rev28', # Leg 2
            b'Rev27', # Leg 3
            b'Rev26'  # Leg 4
        ]
 
        self.physicsClient = p.connect(flag)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")
        self.cubeStartPos = [0, 0 ,0]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.robotId,
            posObj=self.cubeStartPos,
            ornObj=self.cubeStartOrientation
        )

    def step(self, action):
        action = [tf.make_ndarray(ac)[0] for ac in action]
        """
            The bullet physics engine can not be parallelised using pybullet, thus only a batch size of 1 is supported
        """

        joint_angles = action[0]

        for step in range(self.params['rnn_steps']):
            angles = joint_angles[step, :]

            a_ = []
            for i, j in enumerate(self.driven_joints):
                a_.append(i)

            self.set_angles(a_)

        return [tf.zeros(spec.shape, spec.dtype) for spec in observation_spec[:-1]]

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

        targetPositions = []
        indices = []
        for i in range(4):
            targetPositions.append(joint_angles[i])
            targetPositions.append(joint_angles[i+4])
            indices.append(self.j2i[self.driven_joints[i])
            indices.append(self.j2i[self.driven_joints[i+4])

        p.setJointMotorControlArray(
            self.robotId,
            list(self.i2j.keys())
            p.POSITION_CONTROL,
            targetPositions
        )

    def get_angles(self, joint_names):
        """
        Gets the angles of the specified joints and returns a dict of joint_names: angles (in radians)
        :type joint_names: list(str)
        :param joint_names: List of joint names
        :rtype: dict
        """
        indices = []
        for n in joint_names:
            indices.append(self.j2i[n])

        states = p.getJointStates(
            self.robotId,
            indices,
            self.physicsClient
        )

        angles = states[0]
        return angles

    def load_urdf(self, urdf_path):
        return p.loadURDF(
            urdf_path,
            self.cubeStartPos, 
            self.cubeStartOrientation,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_MERGE_FIXED_LINKS
        )
        self.j2i = {}
        self.i2j = {}
        for i in range(p.getNumJoints(self.robotID, q.physicsClient)):
            if p.getJointInfo(robotID, i)[1] in self.driven_joints:
                self.j2i[p.getJointInfo(robotID, i)[1]] = p.getJointInfo(robotID, i)[0] 
                self.i2j[p.getJointInfo(robotID, i)[0]] = p.getJointInfo(robotID, i)[1]


    def render(self, urdf_path):
        self.robotId = self.load_urdf(urdf_path)
        for i in range (10000):
            p.stepSimulation()
            time.sleep(1./240.)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotId)
        print(cubePos,cubeOrn)
        p.disconnect()
