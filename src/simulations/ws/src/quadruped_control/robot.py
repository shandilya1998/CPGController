from abc import ABCMeta, abstractmethod, abstractproperty
import pybullet as p
import time
import pybullet_data
import tensorflow as tf
fom rl.constants import *
from reward.reward import FitnessFunction
import numpy as np

"""
    Refer to the following link for pybullet related information
    https://github.com/moribots/plen_ml_walk/blob/master/plen_bullet/src/plen_bullet/plen_env.py
"""

class WalkingSpider:
    def __init__(self, params, GUI = False, debug = False):
        self.debug = debug 
        self.params = params
        self.GUI = GUI

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _load_urdf(self, urdf_path):
        flag = p.DIRECT
        if self.GUI:
             flag = p.GUI
        self.physicsClient = p.connect(flag)
        p.resetSimulation()
        p.setTimeStep(self.params['dt']) 
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
        self.movingJoints = [0, 2, 3, 5, 6, 8, 9, 11]

    def reset(self):
        self.vt = [0, 0, 0, 0, 0, 0, 0, 0]
        self.vd = 0
        self.maxV = 8.72  # 0.12sec/60 deg = 500 deg/s = 8.72 rad/s
        self.envStepCounter = 0
        p.resetBasePositionAndOrientation(
            self.robotID,
            posObj=self.cubeStartPos,
            ornObj=self.cubeStartOrientation
        )
        observation = self.compute_observation()
        return observation

    def compute_observation(self):
        baseOri = np.array(p.getBasePositionAndOrientation(self.robotID))
        JointStates = p.getJointStates(self.robotID, self.movingJoints)
        BaseAngVel = p.getBaseVelocity(self.robotID)
        ContactPoints = p.getContactPoints(self.robotID, self.plane)

        if (self.debug):
            print("\nBase Orientation \nPos( x= {} , y = {} , z = {} )\nRot Quaternion( x = {} , y = {} , z = {}, w = {} )\n\n".format(
                baseOri[0][0], baseOri[0][1], baseOri[0][2],
                baseOri[1][0], baseOri[1][1], baseOri[1][2], baseOri[1][3]
            ))
            print(
                "\nJointStates: (Pos,Vel,6 Forces [Fx, Fy, Fz, Mx, My, Mz], \
                    appliedJointMotorTorque)\n")
            for i, joint in enumerate(JointStates):
                print("Joint #{} State: Pos {}, Vel {} Fx {} Fy {} Fz {} \
                    Mx {} My {} Mz {}, ApliedJointTorque {} ".format(
                        i, 
                        joint[0], 
                        joint[1], 
                        joint[2][0], joint[2][1], 
                        joint[2][2], joint[2][3], 
                        joint[2][4], joint[2][5], 
                        joint[3]
                    )
                )
            print("\nBase Angular Velocity (Linear Vel( x = {} , y = {} , \
                z =  {} ) Algular Vel(wx= {} ,wy= {} ,wz= {} ) ".format(
                BaseAngVel[0][0], BaseAngVel[0][1], BaseAngVel[0][2], 
                BaseAngVel[1][0], BaseAngVel[1][1], BaseAngVel[1][2])
            )

        obs = np.array([
            baseOri[0][2],  # z (height) of the Torso -> 1
            # orientation (quarternion x,y,z,w) of the Torso -> 4
            baseOri[1][0],
            baseOri[1][1],
            baseOri[1][2],
            baseOri[1][3],
            JointStates[0][0],  # Joint angles(Pos) -> 8
            JointStates[1][0],
            JointStates[2][0],
            JointStates[3][0],
            JointStates[4][0],
            JointStates[5][0],
            JointStates[6][0],
            JointStates[7][0],
            # 3-dim directional velocity and 3-dim angular velocity -> 3+3=6
            BaseAngVel[0][0],
            BaseAngVel[0][1],
            BaseAngVel[0][2],
            BaseAngVel[1][0],
            BaseAngVel[1][1],
            BaseAngVel[1][2],
            JointStates[0][1],  # Joint Velocities -> 8
            JointStates[1][1],
            JointStates[2][1],
            JointStates[3][1],
            JointStates[4][1],
            JointStates[5][1],
            JointStates[6][1],
            JointStates[7][1]
        ])
        # External forces (force x,y,z + torque x,y,z) 
        # applied to the CoM of each link 
        # (Ant has 14 links: ground+torso+12(3links for 4legs) for legs 
        # -> (3+3)*(14)=84
      external_forces = np.array([np.array(joint[2])
        external_forces = np.array([np.array(joint[2])
                                  for joint in JointStates])
        obs = np.append(obs, external_forces.ravel())
        # print("Obs: ", obs.shape, obs)
        return np.expand_dims(obs, 1)

    def _get_plane_contacts(self):
        contacts = p.getContactPoints(
            bodyA = self.robotID,
            bodyB = self.planeID,
            physicsClientId = self.physicsClient
        )
        out = []
        for contact in contacts:
            print('A')
            print(contact[5])
            print('B')
            print(contact[6])
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
        :type targetPositions: list in same order as movingJoints
        :param joint_angles: Dictionary of joint_names: angles (in radians)
        :type duration: float
        :param duration: Time to reach the angular targets (in seconds)
        :type joint_velocities: dict
        :param joint_velocities: dict of joint angles and velocities
        :return: None
        """

        p.setJointMotorControlArray(
            self.robotID,
            self.movingJoints,
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
            self.movingJoints,
            self.physicsClient
        )

        angles = states[0]
        return angles

    def step(self, action, history):
        action = [tf.make_ndarray(ac)[0] for ac in action]
        

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

        self.legs = [
            'leg_rev_v14', # Leg 1
            'leg_rev_v13', # Leg 2
            'leg_rev_v12', # Leg 3
            'leg_rev_v11'  # Leg 4
        ]

        self.knee_links = [
            [
                'leg_servo_arm_v14',
                'leg_rev_v14',
                'leg_parallel_linkage_v14'
            ], # Leg 1
            [   
                'leg_servo_arm_v13',
                'leg_rev_v13',
                'leg_parallel_linkage_v13'
            ], # Leg 2
            [   
                'leg_servo_arm_v12',
                'leg_rev_v12',
                'leg_parallel_linkage_v12'
            ], # Leg 3
            [   
                'leg_servo_arm_v11',
                'leg_rev_v11',
                'leg_parallel_linkage_v11'
            ], # Leg 4
        ]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def build(self, urdf_path, period = 250):
        self.robotID = self._load_urdf(urdf_path)
        self.gamma = np.zeros((4, ))
        print('Building Reward Class........')
        self.reward = FitnessFunction(
            self.total_mass,
            self.params['g'],
            self.params['thigh'],
            self.params['base_breadth'],
            self.params['friction_constant'],
            self.params['mu'],
            self.params['m1'],
            self.params['m2'],
            self.params['m3'],
            self.params['L0'],
            self.params['L1'],
            self.params['L2'],
            self.params['L3']
        )
        if self.GUI:
            for i in range (period):
                p.stepSimulation()
                time.sleep(1./240.)
            cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotID)
            print(cubePos,cubeOrn)
 
    def reset(self):
        self._reset(
            self.cubeStartPos,
            self.cubeStartOrientation,
            [0, 0 ,0],
            [0, 0, 0],
            {}
        )

    def _reset(self, 
        startPos, 
        startOrientation, 
        startLinearVelocity, 
        startAngularVelocity,
        jStates
    ):
        p.resetBasePositionAndOrientation(
            self.robotID,
            posObj=startPos,
            ornObj=startOrientation
        )
        p.resetBaseVelocity(
            self.robotID,
            linearVelocity = startLinearVelocity,
            angularVelocity = startAngularVelocity,
            physicsClientId = self.physicsClient
        )
        for index in jStates.keys():
            p.resetJointState(
                self.robotID,
                index,
                jStates[index]['targetValue'],
                jStates[index]['targetVelocity'],
                self.physicsClient
            )

    def step(self, action, history, startState):
        """
            action : tf.Tensor shape : (None, rnn_steps, 8)
            history : tf.Tensor shape : (None, rnn_steps - 1, 8)
            startState : [
                startPos : list of float length : 3
                startOrientation : list of float length : 3
                startLinearVelocity : list of float length : 3
                startAngularVelocity : list of float length : 3
                jStates : dict of dict index : {
                    'targetValues' : float,
                    'targetVelocity' : float
                }
            ]
        """
        action = [tf.make_ndarray(ac)[0] for ac in action]
        history = tf.make_ndarray(history)[0]
        """
            startState is the kinematic state of the quadruped at timestep
            t - T + 1, that is the first step in history
        """
        startPos, \
        startOrientation, \
        startLinearVelocity, \
        startAngularVelocity,
        jStates = startState

        self._reset(
            startPos,
            startOrientation,
            startLinearyVelocity,
            startAngularVelocity,
            jStates
        )

        Af, Bf = self._step(history[0]) # get startState and AF and BF from here

        for step in range(1, self.params['rnn_steps'] - 1):
            _, _ = self._step(history[step])

        A, B = self._step(action[0]) # get A and B from here

        inertia = [
            [
                self._get_inertia(
                    self._link_name_to_index[name]
                ) for name in knee
            ] for knee in self.knee_links
        ]

        theta = np.array([angle + np.pi/2 for angle in action[0][:4]])

        vec = []

        for i in range(len(inertia)):
            vec.append(np.array([
                np.cos(theta[i]),
                np.sin(theta[i]),
                0
            ]))

        for i in range(len(inertia)):
            for j in range(len(inertia[i])):
                mat = np.zeros((3,3))
                for k in range(3):
                    mat[k][k] = inertia[i][j][k]

                Inn = np.matmul(np.matmul(vec[i], mat), vec[i].T)
                inertia[i][j] = sum(inertia[i][j]) - np.sum(Inn)

        for step in range(1, self.params['rnn_steps'] - 1):
            _, _ = self._step(action[step])

        Al, Bl = self._step(action[-1]) # get AL and BL from here

        self.reward.build(
            self.params['rnn_steps'] - 1, 
            A, B, Al, Af, Bf, Bl,
            inertia[:][0],
            inertia[:][1],
            inertia[:][2]
        )

        history = np.expand_dims(
            np.concatenate([history[1:], action[0]], 0), 
            0
        )

        return observation, reward, history, startState

    
    def _step(self, joint_angles):
        """
            This method must return all relevant information for reward
            computation
        """
        angles = joint_angles.tolist()
        self.set_angles(angles)
        p.stepSimulation()
        contacts = self._get_plane_contacts()
        points = []
        com = self._compute_com()
        for contact in contacts:
            if contact['linkIndex'] in self._index_to_leg_name.keys():
                self.gamma[self.legs.index(
                    self._index_to_link_name(contact['linkIndex'])
                )] = 1
                points.append([contact[position][i] - com[i] \
                    for i in range(3)])

        """
            Need to test whether taking the first two items here is the
            appropriate decision
        """
        if len(points) < 2:
            """
                Set a high penalty for the case when the atleast two legs are not in contact with the ground
            """
            raise NotImplementedError
        A = points[0]
        B = points[1]
        
        """
            After every iteration the simulation has to be reset to the 
            current state. Before resetting the state of the simulations the
            current state should be stored so that the simulation may start
            from the timestep t
            The methods to be used to implement such a transition are:
                resetJointState
                resetBasePositionAndOrientation
                resetBaseVelocity
            The system must also keep track of joint and base velocities 
            at the start of the simulation
        """
    
        return [A, B]


    def _compute_observations(self):
        com = self._compute_com()
        linear_vel, angular_vel = p.getBaseVelocity(self.robotID, self.physicsClient)
        raise NotImplementedError

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

    def _get_inertia(self, link_index):
        dyn = p.getDynamicsInfo(
            self.robotID,
            linkIndex,
            self.physicsClient
        )
        return dyn[2]
        
    def _get_plane_contacts(self):
        contacts = p.getContactPoints(
            bodyA = self.robotID, 
            bodyB = self.planeID,
            physicsClientId = self.physicsClient
        )
        out = []
        for contact in contacts:
            print('A')
            print(contact[5])
            print('B')
            print(contact[6])
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

        self._leg_name_to_index = {}
        self._index_to_leg_name = {}

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
            self._leg_name_to_index 
            if _name in self.legs:
                self._leg_name_to_index[_name] = _id
                self._index_to_leg_name[_id] = _name
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

    def disconnect(self):
        p.disconnect()

