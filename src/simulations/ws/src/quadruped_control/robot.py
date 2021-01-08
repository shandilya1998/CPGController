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
            'Rev17', # Leg 1
            'Rev16', # Leg 2
            'Rev15', # Leg 3
            'Rev14', # Leg 4
            'Rev29', # Leg 1
            'Rev28', # Leg 2
            'Rev27', # Leg 3
            'Rev26'  # Leg 4
        ]
 
        self.physicsClient = p.connect(flag)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")
        self.cubeStartPos = [0, 0 ,0]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.robotID,
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
            angles = joint_angles[step, :].tolist()
            self.set_angles(angles)
            p.stepSimulation()
            
        return [tf.zeros(spec.shape, spec.dtype) for spec in observation_spec[:-1]]

    def compute_observations(self):
        com = self.compute_com()
        linear_vel, angular_vel = p.getBaseVelocity(self.robotID, self.physicsClient)  
    
    def compute_com(self):

        COM = {
            link_name : p.getLinkState(
                self.robotID,
                self._link_name_to_index[link_name]
            )[0] for link_name in self._link_name_to_index.keys()
        }

        com = [0, 0, 0]
        for link in COM.keys():
            com[0] += COM[link][0]*self.masses[link]
            com[1] += COM[link][1]*self.masses[link]
            com[2] += COM[link][2]*self.masses[link]
        com[0] = com[0]/self.total_mass
        com[1] = com[1]/self.total_mass
        com[2] = com[2]/self.total_mass
        
        return com

    def set_angles(self, targetPositions, duration=None, joint_velocities=None):
        """
        Sets the joints to the specified angles
        :type targetPositions: list in same order as driven_joint_indices
        :param joint_angles: Dictionary of joint_names: angles (in radians)
        :type duration: float
        :param duration: Time to reach the angular targets (in seconds)
        :type joint_velocities: dict
        :param joint_velocities: dict of joint angles and velocities
        :return: None
        """

        p.setJointMotorControlArray(
            self.robotID,
            self.driven_joint_indices,
            p.POSITION_CONTROL,
            targetPositions
        )

    def get_angles(self):
        """
        Gets the angles of the specified joints and returns a dict of joint_names: angles (in radians)
        :type joint_names: list(str)
        :param joint_names: List of joint names
        :rtype: dict
        """
        states = p.getJointStates(
            self.robotID,
            self.joint_angle_indices,
            self.physicsClient
        )

        angles = states[0]
        return angles

    def load_urdf(self, urdf_path):
        self.robotID = p.loadURDF(
            urdf_path,
            self.cubeStartPos, 
            self.cubeStartOrientation,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_MERGE_FIXED_LINKS
        )
        self.j2i = {}
        self.i2j = {}
        for i in range(p.getNumJoints(self.robotID, self.physicsClient)):
            if p.getJointInfo(self.robotID, i)[1].decode('UTF-8') in self.driven_joints:
                self.j2i[
                    p.getJointInfo(
                        self.robotID, 
                        i
                    )[1].decode('UTF-8')
                ] = p.getJointInfo(self.robotID, i)[0] 
                self.i2j[
                    p.getJointInfo(
                        self.robotID, 
                        i
                    )[0]
                ] = p.getJointInfo(self.robotID, i)[1].decode('UTF-8')
        
        self.driven_joint_indices=[self.j2i[j] for j in self.driven_joints]
        
        self._link_name_to_index = {
            p.getBodyInfo(self.robotID)[0].decode('UTF-8') : -1
        }

        self._index_to_link_name = {
            -1 : p.getBodyInfo(self.robotID)[0].decode('UTF-8')
        }
        
        for _id in range(p.getNumJoints(self.robotID)):
            _name = p.getJointInfo(self.robotID, _id)[12].decode('UTF-8')
            self._index_to_link_name[_id] = _name
            self._link_name_to_index[_name] = _id
 
        self.masses = { 
            link_name : p.getDynamicsInfo(
                self.robotID,
                self._link_name_to_index[link_name]
            )[0] for link_name in self._link_name_to_index.keys()
        }

        self.total_mass = sum(list(self.masses.values()))

        return self.robotID


    def render(self, urdf_path):
        self.robotID = self.load_urdf(urdf_path)
        for i in range (10000):
            p.stepSimulation()
            time.sleep(1./240.)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotID)
        print(cubePos,cubeOrn)
        p.disconnect()

q = Quadruped(params)
