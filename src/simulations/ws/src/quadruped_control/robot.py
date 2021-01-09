from abc import ABCMeta, abstractmethod, abstractproperty
import pybullet as p
import time
import pybullet_data
import tensorflow as tf
from rl.constants import *
from loss import loss.FitnessFunction as loss
"""
    Refer to the following link for pybullet related information
    https://github.com/moribots/plen_ml_walk/blob/master/plen_bullet/src/plen_bullet/plen_env.py
"""


class Quadruped:
    def __init__(self, params, GUI = False):
        self.params = params
        self.GUI = GUI 
        self.driven_joints = [
            'Leg1Hip',
            'Leg2Hip',
            'Leg3Hip',
            'Leg4Hip',
            'Leg1Knee',
            'Leg2Knee',
            'Leg3Knee',
            'Leg4Knee'
        ]
 
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
            contacts = self._get_plane_contacts()
            p.stepSimulation()
            
        return [tf.zeros(spec.shape, spec.dtype) for spec in observation_spec[:-1]]

    def _calculate_reward(self):
        

    def _compute_observations(self):
        com = self._compute_com()
        linear_vel, angular_vel = p.getBaseVelocity(self.robotID, self.physicsClient)

    def _compute_com(self):
        COM = {}
        for link_name in self._link_name_to_index.keys():
            if self._link_name_to_index[link_name] == -1:
                com, _ = p.getBasePositionAndOrientation(
                    self.robotID,
                    self.physicsClient
                )
                COM['base_link'] = com
            else:
                COM[link_name] = p.getLinkState(
                    bodyUniqueId = self.robotID,
                    linkIndex = self._link_name_to_index[link_name],
                    physicsClientId = self.physicsClient
                )[0]

        com = [0, 0, 0]
        for link in COM.keys():
            com[0] += COM[link][0]*self.masses[link]
            com[1] += COM[link][1]*self.masses[link]
            com[2] += COM[link][2]*self.masses[link]
        com[0] = com[0]/self.total_mass
        com[1] = com[1]/self.total_mass
        com[2] = com[2]/self.total_mass
        
        return com

    def _get_plane_contacts(self):
        contacts = p.getContactPoints(
            bodyA = self.robotID, 
            bodyB = self.planeID,
            physicsClientId = self.physicsClient
        )
        out = []
        for contact in contacts:
            out.append({
                'position' : contact[5],
                'linkIndex' : contact[3],
                'normalForce' : contact[9],
                'contactNormalOnB' : contact[7],
            })
        return out

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

    def _load_urdf(self, urdf_path):
        flag = p.DIRECT
        if self.GUI:
             flag = p.GUI
        self.physicsClient = p.connect(flag)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.params['g'])
        self.planeID = p.loadURDF("plane.urdf")
        self.cubeStartPos = [0, 0 ,0] 
        self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.robotID = p.loadURDF(
            urdf_path,
            self.cubeStartPos, 
            self.cubeStartOrientation,
            flags=p.URDF_USE_INERTIA_FROM_FILE | \
                p.URDF_USE_SELF_COLLISION | \
                p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | \
                p.URDF_MERGE_FIXED_LINKS
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

        print('------------------')
        print('Joint     |Index')
        print('------------------')
        for key in self.j2i.keys():
            space = 10-len(key)
            space = ''.join([' ' for i in range(space)])
            print('{j}{s}'.format(j = key, s = space), end = '|')
            print('{i}'.format(i = self.j2i[key]))
        print('------------------')
        self.driven_joint_indices=[self.j2i[j] for j in self.driven_joints]
        
        self._link_name_to_index = {
            p.getBodyInfo(self.robotID)[0].decode('UTF-8') : -1
        }

        self._index_to_link_name = {
            -1 : p.getBodyInfo(self.robotID)[0].decode('UTF-8')
        }

        print('-------------------------------------')
        print('Link                          |Index')
        print('-------------------------------------') 
        for i in range(p.getNumJoints(self.robotID)):
            _name = p.getJointInfo(self.robotID, i)[12].decode('UTF-8')
            _id = p.getJointInfo(self.robotID, i)[0]
            space = 30-len(_name)
            space = ''.join([' ' for i in range(space)])
            self._index_to_link_name[_id] = _name
            self._link_name_to_index[_name] = _id
            print('{j}{s}'.format(j = _name, s = space), end = '|')
            print('{i}'.format(i = _id))
        print('-------------------------------------')
 
        self.masses = { 
            link_name : p.getDynamicsInfo(
                self.robotID,
                self._link_name_to_index[link_name]
            )[0] for link_name in self._link_name_to_index.keys()
        }

        self.total_mass = sum(list(self.masses.values()))

        return self.robotID

    def build(self, urdf_path, period = 250):
        self.robotID = self._load_urdf(urdf_path)
        self.loss = loss(
            self.total_mass,
            self.params['g'],
                     
        )
        if self.GUI:
            for i in range (period):
                p.stepSimulation()
                time.sleep(1./240.)
            cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotID)
            print(cubePos,cubeOrn)

    def disconnect(self):
        p.disconnect()

