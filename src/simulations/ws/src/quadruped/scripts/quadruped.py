#!/usr/bin/env python
import rospy
import copy
#from rl.constants import *
#from reward.reward import FitnessFunction
import numpy as np
#import tensorflow as tf
from control_msgs.msg import FollowJointTrajectoryAction, \
    FollowJointTrajectoryActionGoal, \
    FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, \
    JointTrajectoryPoint
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, \
    SetModelStateRequest, \
    SetModelConfiguration, \
    SetModelConfigurationRequest
from gazebo_msgs.srv import GetModelState, \
    GetModelStateRequest, \
    GetLinkProperties, \
    GetLinkState, \
    GetPhysicsProperties
from geometry_msgs.msg import PoseStamped, WrenchStamped
from gazebo_msgs.msg import ModelState, ContactsState
from sensor_msgs.msg import Imu
from simulations.ws.src.quadruped.scripts.kinematics import Kinematics
from tf import TransformListener
from reward import FitnessFunction

class Leg:
    def __init__(self, params, leg_name, joint_name_lst):
        self.params = params
        self.leg_name = leg_name
        self.joint_name_lst = joint_name_lst
        self.jtp_zeros = np.zeros(len(joint_name_lst))
        self.jtp = rospy.Publisher(
            '/quadruped/{leg_name}_controller/command'.format(
                leg_name = leg_name
            ),
            JointTrajectory,
            queue_size=1
        )
        self.contact_state = None
        self.contact_state_subscriber = rospy.Subscriber(
            '/quadruped/{leg_name}_tip_contact_sensor'.format(
                leg_name = leg_name
            ),
            ContactsState,
            self.contact_callback
        )

        self.leg1_joint_force_torque_subscriber = rospy.Subscriber(
            '/quadruped/{leg_name}1_joint_force_torque'.format(
                leg_name = leg_name
            ),
            WrenchStamped,
            self._leg1_joint_force_torque_callback
        )

        self.leg2_joint_force_torque_subscriber = rospy.Subscriber(
            '/quadruped/{leg_name}2_joint_force_torque'.format(
                leg_name = leg_name
            ),
            WrenchStamped,
            self._leg2_joint_force_torque_callback
        )

        self.leg3_joint_force_torque_subscriber = rospy.Subscriber(
            '/quadruped/{leg_name}3_joint_force_torque'.format(
                leg_name = leg_name
            ),
            WrenchStamped,
            self._leg3_joint_force_torque_callback
        )

    def get_processed_joint_force_troque(self):
        return {
            'leg1' : {
                'frame_id' : self.leg1_joint_force_torque.header.frame_id,
                'force' : np.array([
                    self.leg1_joint_force_torque.wrench.force.x,
                    self.leg1_joint_force_torque.wrench.force.y,
                    self.leg1_joint_force_torque.wrench.force.z
                ]),
                'torque' : np.array([
                    self.leg1_joint_force_torque.wrench.torque.x,
                    self.leg1_joint_force_torque.wrench.torque.y,
                    self.leg1_joint_force_torque.wrench.torque.z
                ])
            },
            'leg2' : {
                'frame_id' : self.leg2_joint_force_torque.header.frame_id,
                'force' : np.array([
                    self.leg2_joint_force_torque.wrench.force.x,
                    self.leg2_joint_force_torque.wrench.force.y,
                    self.leg2_joint_force_torque.wrench.force.z
                ]),
                'torque' : np.array([
                    self.leg2_joint_force_torque.wrench.torque.x,
                    self.leg2_joint_force_torque.wrench.torque.y,
                    self.leg2_joint_force_torque.wrench.torque.z
                ])
            },
            'leg3' : {
                'frame_id' : self.leg3_joint_force_torque.header.frame_id,
                'force' : np.array([
                    self.leg3_joint_force_torque.wrench.force.x,
                    self.leg3_joint_force_torque.wrench.force.y,
                    self.leg3_joint_force_torque.wrench.force.z
                ]),
                'torque' : np.array([
                    self.leg3_joint_force_torque.wrench.torque.x,
                    self.leg3_joint_force_torque.wrench.torque.y,
                    self.leg3_joint_force_torque.wrench.torque.z
                ])
            }
        }

    def _leg1_joint_force_torque_callback(self, wrench):
        self.leg1_joint_force_torque = wrench

    def _leg2_joint_force_torque_callback(self, wrench):
        self.leg2_joint_force_torque = wrench

    def _leg3_joint_force_torque_callback(self, wrench):
        self.leg2_joint_force_torque = wrench

    def get_processed_contact_state(self):
        contact_state = self.contact_state
        states = contact_state.states
        force = None
        position = None
        torque = None
        normal = None
        flag = False
        if states:
            flag = True
            force = np.mean(
                [[
                    state.total_wrench.force.x,
                    state.total_wrench.force.y,
                    state.total_wrench.force.z
                ] for state in states],
                0
            )
            torque = np.mean(
                [[
                    state.total_wrench.torque.x,
                    state.total_wrench.torque.y,
                    state.total_wrench.torque.z
                ] for state in states],
                0
            )
            position = np.mean(
                [[
                    state.contact_positions[0].x,
                    state.contact_positions[0].y,
                    state.contact_positions[0].z
                ] for state in states],
                0
            )
            normal = np.mean(
                [[
                    state.contact_normals[0].x,
                    state.contact_normals[0].y,
                    state.contact_normals[0].z
                ] for state in states],
                0
            )

        contact_state = {
            'force' : force,
            'torque' : torque,
            'position' : position,
            'normal' : normal,
            'flag' : flag,
            'leg_name' : self.leg_name
        }
        return contact_state

    def contact_callback(self, contact_state):
        self.contact_state = contact_state

    def move(self, pos):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        point.time_from_start = rospy.Duration(1.0/60.0)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

    def reset_move(self, pos):
        jtp_msg = JointTrajectory()
        self.jtp.publish(jtp_msg)
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        point.time_from_start = rospy.Duration(0.0001)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

class AllLegs:
    def __init__(self, params):
        self.params = params
        self.joint_name_lst = self.params['joint_name_lst']
        self.leg_name_lst = self.params['leg_name_lst']
        self.front_right = Leg(
            self.params,
            self.leg_name_lst[0],
            self.joint_name_lst[:3]
        )
        self.front_left = Leg(
            self.params,
            self.leg_name_lst[1],
            self.joint_name_lst[3:6]
        )
        self.back_right = Leg(
            self.params,
            self.leg_name_lst[2],
            self.joint_name_lst[6:9]
        )
        self.back_left = Leg(
            self.params,
            self.leg_name_lst[3],
            self.joint_name_lst[9:]
        )
        self.A = np.zeros((3,))
        self.B = np.zeros((3,))
        self.w = 1

    def get_all_torques(self):
        regex = r'leg\d{1,3}'
        torque = []
        for joint in self.joint_name_lst:
            leg = self.get_leg_handle(joint[:-7])
            out = leg.get_processed_joint_force_troque()
            joint = re.search(regex, joint).group()
            out = out[joint]
            torque.append(np.norm(out['torque']))
        return np.array(torque)

    def get_all_contacts(self):
        self.fr_contact = self.front_right.get_processed_contact_state()
        self.fl_contact = self.front_left.get_processed_contact_state()
        self.br_contact = self.back_right.get_processed_contact_state()
        self.bl_contact = self.back_left.get_processed_contact_state()
        return self.fr_contact, self.fl_contact, self.br_contact, self.bl_contact

    def get_AB(self):
        fr_contact, fl_contact, br_contact, bl_contact = self.get_all_contacts()
        contacts = []
        if fr_contact['flag']:
            contacts.append(fr_contact)
        if fl_contact['flag']:
            contacts.append(fl_contact)
        if br_contact['flag']:
            contacts.append(br_contact)
        if bl_contact['flag']:
            contacts.append(bl_contact)
        if len(contacts) == 1:
            self.w = 0.5
            return [contacts[0], contacts[0]]
        elif len(contacts) == 2:
            self.w = 1
            return contacts
        elif len(contacts) == 3:
            """
                Assumption being made that the two support legs will always 
                be parrallel to the forward direction or diagonal
            """
            A = None
            B = None
            self.w = 1
            names = [contact.leg_name for contact in contacts]
            if 'front' in names[0] and 'front' in names[1]:
                B = contacts[2]
                if 'right' in names[2]:
                    A = contacts[1]
                elif 'left' in names[2]:
                    A = contacts[0]
            elif 'front' in names[0] and 'back' in names[1]:
                A = contacts[0]
                if 'right' in names[0]:
                    B = contacts[2]
                elif 'left' in names[0]:
                    B = contacts[1]
            return A, B
        elif len(contacts) == 4:
            A = None
            B = None
            self.w = 1
            for contact in contacts:
                if contact.leg_name == self.leg_name_lst[0]:
                    A = contact
                elif contact.leg_name == self.leg_name_lst[-1]:
                    B = contact
            return A, B
        else:
            self.w = 0.2
            return []


    def get_leg_handle(self, leg):
        if leg == self.leg_name_lst[0]:
            return self.front_right
        elif leg == self.leg_name_lst[1]:
            return self.front_left
        elif leg == self.leg_name_lst[2]:
            return self.back_right
        elif leg == self.leg_name_lst[3]:
            return self.back_left
        else:
            raise ValueError('Leg can be one of `front_right_leg, \
                front_left_leg, \
                back_right_leg, \
                back_left_leg`, \
                but got {leg}'.format(leg = leg)
            )

    def reset_move(self, pos):
        self.front_right.reset_move(pos[:3])
        self.front_left.reset_move(pos[3:6])
        self.back_right.reset_move(pos[6:9])
        self.back_left.reset_move(pos[9:])

    def move(self, pos):
        self.front_right.move(pos[:3])
        self.front_left.move(pos[3:6])
        self.back_right.move(pos[6:9])
        self.back_left.move(pos[9:])


class Quadruped:
    def __init__(self, params):
        rospy.init_node('joint_position_node')
        self.nb_joints = params['action_dim']
        self.nb_links = params['action_dim'] + 1

        self.motion_state_shape = tuple(
            params['observation_spec'][0].shape
        )
        self.motion_state_dtype = params[
            'observation_spec'
        ][0].dtype.as_numpy_dtype
        self.motion_state = np.zeros(
            shape = self.motion_state_shape,
            dtype = self.motion_state_dtype
        )

        self.robot_state_shape = tuple(
            params['observation_spec'][1].shape
        )
        self.robot_state_dtype = params[
            'observation_spec'
        ][1].dtype.as_numpy_dtype
        self.robot_state = np.zeros(
            shape = self.robot_state_shape,
            dtype = self.robot_state_dtype
        )

        self.osc_state_shape = tuple(
            params['observation_spec'][2].shape
        )
        self.osc_state_dtype = params[
            'observation_spec'
        ][2].dtype.as_numpy_dtype
        self.osc_state = np.zeros(
            shape = self.osc_state_shape,
            dtype = self.osc_state_dtype
        )
        self.history_shape = tuple(
            params['observation_spec'][3].shape
        )
        self.robot_action_shape = tuple(
            params['action_spec'][0].shape
        )
        self.osc_action_shape = tuple(
            params['action_spec'][1].shape
        )
        self.params = params
        self.link_name_lst = self.params['link_name_lst']
        self.leg_name_lst = self.params['leg_name_lst']
        self.leg_link_name_lst = self.link_name_lst[1:]
        self.joint_name_lst = self.params['joint_name_lst']

        self.all_legs = AllLegs(
            self.params,
        )

        self.starting_pos = self.params['starting_pos']
        self.history = np.repeat(
            np.expand_dims(self.starting_pos, 0),
            self.params['rnn_steps'] - 1,
            0
        )


        self.link_prop_proxy = rospy.ServiceProxy(
            '/gazebo/get_link_properties',
            GetLinkProperties
        )
        self.pause_proxy = rospy.ServiceProxy(
            '/gazebo/pause_physics', 
            Empty
        )
        self.unpause_proxy = rospy.ServiceProxy(
            '/gazebo/unpause_physics', 
            Empty
        )
        self.model_config_proxy = rospy.ServiceProxy(
            '/gazebo/set_model_configuration',
            SetModelConfiguration
        )
        self.model_config_req = SetModelConfigurationRequest()
        self.model_config_req.model_name = 'quadruped'
        self.model_config_req.urdf_param_name = 'robot_description'
        self.model_config_req.joint_names = self.joint_name_lst
        self.model_config_req.joint_positions = self.starting_pos
        self.model_state_proxy = rospy.ServiceProxy(
            '/gazebo/set_model_state', 
            SetModelState
        )
        self.model_state_req = SetModelStateRequest()
        self.model_state_req.model_state = ModelState()
        self.model_state_req.model_state.model_name = 'quadruped'
        self.model_state_req.model_state.pose.position.x = 0.0
        self.model_state_req.model_state.pose.position.y = 0.0
        self.model_state_req.model_state.pose.position.z = 0.0
        self.model_state_req.model_state.pose.orientation.x = 0.0
        self.model_state_req.model_state.pose.orientation.y = 0.0
        self.model_state_req.model_state.pose.orientation.z = 0.0
        self.model_state_req.model_state.pose.orientation.w = 0.0
        self.model_state_req.model_state.twist.linear.x = 0.0
        self.model_state_req.model_state.twist.linear.y = 0.0
        self.model_state_req.model_state.twist.linear.z = 0.0
        self.model_state_req.model_state.twist.angular.x = 0.0
        self.model_state_req.model_state.twist.angular.y = 0.0
        self.model_state_req.model_state.twist.angular.z = 0.0
        self.model_state_req.model_state.reference_frame = 'world'

        self.get_model_state_proxy = rospy.ServiceProxy(
            '/gazebo/get_model_state', 
            GetModelState
        )
        self.get_model_state_req = GetModelStateRequest()
        self.get_model_state_req.model_name = 'quadruped'
        self.get_model_state_req.relative_entity_name = 'world'


        self.joint_state_subscriber = rospy.Subscriber(
            '/quadruped/joint_states',
            JointState,
            self.joint_state_subscriber_callback
        )

        self.orientation = np.zeros(4)
        self.angular_vel = np.zeros(3)
        self.linear_acc = np.zeros(3)
        self.imu_subscriber = rospy.Subscriber(
            '/quadruped/imu',
            Imu,
            self.imu_subscriber_callback
        )

        self.reward = 0.0
        self.episode_start_time = 0.0
        self.max_sim_time = 15.0
        self.pos_z_limit = 0.18

        self.kinematics = Kinematics(
            self.params
        )

        self.compute_reward = FitnessFunction(
            self.params
        )

        self._counter_1 = 0
        self.counter_1 = 0 # Counter for support line change
        self.dt = self.params['dt']
        self.A = {'leg_name' : None}
        self.B = {'leg_name' : None}
        self.AF = {'leg_name' : None}
        self.BF = {'leg_name' : None}
        self.AL = {'leg_name' : None}
        self.BL = {'leg_name' : None}
        self.mass = self.get_total_mass()
        self.tf_listener_ = TransformListener()
        self.force = np.zeros((3,))
        self.torque = np.zeros((3,))
        self.com = np.zeros((3,))
        self.v_exp = np.zeros((3,))
        self.v_real = np.zeros((3,))
        self.eta = 1e8

        self.history_joint_torque = np.zeros(self.history_shape)
        self.history_joint_vel = np.zeros(self.history_shape)
        self.history_pos = np.zeros((
            self.params['rnn_steps'] - 1,
            3
        ))
        self.history_pos[:, 2] = 0.25
        self.history_vel = np.zeros((
            self.params['rnn_steps'] - 1,
            3
        ))
        self.history_delta_motion = np.zeros((
            self.params['rnn_steps'] - 2,
            6
        ))

    def get_total_mass(self):
        mass = 0
        for link in self.params['link_name_lst']:
            msg = self.link_prop_proxy(link)
            mass += msg.mass
        return mass

    def get_com(self):
        reference = 'world'
        x, y, z = 0,0,0
        for link in self.params['link_name_lst']:
            prop = self.link_prop_proxy(link)
            mass = prop.mass
            pose = PoseStamped()
            pose.header.frame_id = link[11:]
            pose.position.x = prop.pose.position.x
            pose.position.y = prop.pose.position.y
            pose.position.z = prop.pose.position.z
            pose.orientation.x = prop.pose.orientation.x
            pose.orientation.y = prop.pose.orientation.y
            pose.orientation.z = prop.pose.orientation.z
            pose.orientation.w = prop.pose.orientation.w
            t_pose = self.tf_listener_.transformPose("world", pose)
            x += t_pose.position.x * mass
            y += t_pose.position.y * mass
            z += t_pose.position.z * mass
        x = x / self.mass
        y = y / self.mass
        z = z / self.mass
        return np.array([x, y, z])

    def get_moment(self):
        contacts = [
            self.all_legs.fr_contact,
            self.all_legs.fl_contact,
            self.all_legs.br_contact,
            self.all_legs.bl_contact
        ]
        positions = [contact['position'] for contact in contacts]
        forces = [contact['force'] \
                if contact['force'] is not None \
                else np.zeros((3,)) \
                for contact in contacts
        ]
        r = [position - self.com \
                if position is not None \
                else np.zeros((3)) \
                for position in positions
        ]
        torque = [np.cross(r, f) for r, f in zip(r, forces)]
        torque = np.sum(np.array(torque), 0)
        return torque

    def get_contact(self):
        contacts = {
            leg : {
                self.all_legs.get_leg_handle(leg).contact_state
            } for i, leg in enumerate(self.leg_name_lst)
        }
        return contacts

    def joint_state_subscriber_callback(self, joint_state):
        state = [st for st in joint_state.position]
        vel = [st for st in joint_state.velocity]
        for i, name in enumerate(joint_state.name):
            index = self.joint_name_lst.index(name)
            state[index] = joint_state.position[i]
            vel[index] = joint_state.velocity[i]
        self.joint_position = np.array(state)
        self.joint_velocity = np.array(vel)

    def imu_subscriber_callback(self, imu):
        self.orientation = np.array(
            [
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
                imu.orientation.w
            ]
        )
        self.angular_vel = np.array(
            [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z
            ]
        )
        self.linear_acc = np.array(
            [
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z
            ]
        )

    def reset(self):
        #pause physics
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException:
            print('/gazebo/pause_physics service call failed')
        #set models pos from world
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.model_state_proxy(self.model_state_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_state call failed')
        #set model's joint config
        rospy.wait_for_service('/gazebo/set_model_configuration')
        try:
            self.model_config_proxy(self.model_config_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_configuration call failed')

        self.joint_pos = self.starting_pos
        self.all_legs.reset_move(self.starting_pos)
        #unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except rospy.ServiceException:
            print('/gazebo/unpause_physics service call failed')

        rospy.sleep(0.5)
        self.reward = 0.0

        self.osc_state = np.zeros(
            shape = self.osc_state_shape,
            dtype = self.osc_state_dtype
        )
        self.history = np.repeat(
            np.expand_dims(self.starting_pos, 0),
            self.params['rnn_steps'] - 1,
            0
        )

        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        pos = np.array([
            model_state.pose.position.x, 
            model_state.pose.position.y, 
            model_state.pose.position.z
        ])
        self.last_joint = self.joint_position
        self.last_pos = pos
        diff_joint = np.zeros(self.nb_joints)
        #print(self.joint_position.shape)
        #print(diff_joint.shape)
        #print(self.orientation.shape)
        #print(self.angular_vel.shape)
        #print(self.linear_acc.shape)
        self.robot_state = np.concatenate([
            self.joint_position,
            diff_joint,
            self.orientation,
            self.angular_vel,
            self.linear_acc
        ]).reshape(self.robot_state_shape)
        self.motion_state = np.append(
            pos,
            [0],
            0
        )
        self.episode_start_time = rospy.get_time()
        self.last_action = np.zeros(self.nb_joints)
        self.reward = 0.0
        return [
            self.motion_state,
            self.robot_state,
            self.osc_state,
            self.history
        ], self.reward

    def set_support_lines(self, action):
        AB = self.all_legs.get_AB()
        if AB:
            self._counter_1 += 1
            if self.A['leg_name'] != AB[0]['leg_name']:
                self.AF.update(self.A)
                self.counter_1 = copy.deepcopy(self._counter_1)
                self_counter_1 = 0
            if self.B['leg_name'] != AB[1]['leg_name']:
                self.BF.update(self.B)
                self.counter_1 = copy.deepcopy(self._counter_1)
                self_counter_1 = 0
            self.A.update(AB[0])
            self.B.update(AB[1])
            A_name = self.A['leg_name']
            B_name = self.B['leg_name']
            _A_name = None
            _B_name = None
            if 'right' in A_name:
                _A_name = self.leg_name_lst[1]
            else:
                _A_name = self.leg_name_lst[0]
            if 'right' in B_name:
                _B_name = self.leg_name_lst[3]
            else:
                _B_name = self.leg_name_lst[2]
            current_pose = self.kinematics.get_current_end_effector_fk()
            pose = None
            c = 0
            t = 0
            for step in range(self.params['rnn_steps']):
                pose=self.kinematics.get_end_effector_fk(action[step].tolist())
                if pose[A_name]['position']['z'] - \
                        current_pose[A_name]['position']['z'] > 0:
                    temp = A_name
                    A_name = _A_name
                    _A_name = temp
                    t = copy.deepcopy(c)
                    c = 0
                    break
                if pose[B_name]['position']['z'] - \
                        current_pose[B_name]['position']['z'] > 0:
                    temp = B_name
                    B_name = _B_name
                    _B_name = temp
                    t = copy.deepcopy(c)
                    c = 0
                    break
            """
            delta = {
                leg : {
                    'x' : pose[leg]['position']['x'] - \
                        current_pose[leg]['position']['x'],
                    'y' : pose[leg]['position']['y'] - \
                        current_pose[leg]['position']['y'],
                    'z' : pose[leg]['position']['z'] - \
                        current_pose[leg]['position']['z']
                }
            }
            """
            m = {
                0 : 'x',
                1 : 'y',
                2 : 'z'
            }
            self.Tb = (self.counter_1 + t) * self.dt
            self.AL.update({
                'torque' : None,
                'force' : None,
                'position' : np.array(
                    [
                        self.A['position'][i] + \
                            pose[A_name]['position'][m[i]] - \
                            current_pose[A_name]['position'][m[i]] \
                            for i in range(3)
                    ]
                ),
                'normal' : self.A['normal'],
                'flag' : True,
                'leg_name' : A_name
            })
            self.BL.update({
                'torque' : None,
                'force' : None,
                'position' : np.array(
                    [
                        self.B['position'][i] + \
                            pose[B_name]['position'][m[i]] - \
                            current_pose[B_name]['position'][m[i]] \
                            for i in range(3)
                    ]
                ),
                'normal' : self.B['normal'],
                'flag' : True,
                'leg_name' : B_name
            })
        else:
            raise NotImplementedError


    def get_reward(self, action):
        """
            Need to complete A, B, AF, BF, AL and BL computation
        """
        self.set_support_lines(action)
        self.com = self.get_com()
        self.moment = self.get_moment()
        self.force = self.mass * self.linear_acc
        self.compute_reward.build(
            self.counter_1 * self.dt,
            self.Tb,
            self.A,
            self.B,
            self.AF,
            self.BF,
            self.AL,
            self.BL,
        )
        vd = np.norm(self.v_exp)
        if vd == 0:
            vd = 1e-8
        self.eta = (self.params['L'] + self.params['W'])/(2*vd)
        self.joint_torque = self.all_legs.get_all_torques()
        self.history_joint_torque = np.concatenate(
            [
                self.history_joint_torque[1:],
                self.joint_torque
            ],
            0
        )
        self.history_joint_vel = np.concatenate(
            [
                self.history_joint_vel[1:],
                self.joint_velocity
            ],
            0
        )
        return self.compute_reward(
            self.com,
            self.force,
            self.torque,
            self.v_real,
            self.v_exp,
            self.eta,
            self.all_legs.w,
            self.history_joint_vel,
            self.history_joint_torque,
            self.history_pos,
            self.history_vel,
            self.history_delta_motion
        )

    def step(self, action, delta_motion):
        action = [
            tf.make_ndarray(
                tf.make_tensor_proto(a)
            ) for a in action
        ]
        print('action:', action)
        self.joint_pos = action[0][0][0].tolist()
        self.osc_state = action[1][0].tolist()
        self.all_legs.move(self.joint_pos)
        print('joint pos:', self.joint_pos)
        #rospy.sleep(15.0/60.0)
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        pos = np.array([
            model_state.pose.position.x,
            model_state.pose.position.y,
            model_state.pose.position.z
        ])
        self.history_pos = np.concatenate([
            self.history_pos[1:],
            pos
        ])
        self.v_real = np.array([
            model_state.twist.linear.x,
            model_state.twist.linear.y,
            model_state.twist.linear.z
        ])
        self.history_vel = np.concatenate([
            self.history_vel[1:],
            self.v_real
        ])
        self.history_delta_motion = np.concatenate([
            self.history_delta_motion[1:],
            delta_motion
        ])
        self.v_exp = self.v_exp + delta_motion[3:6]
        vself.action = action[0][0]
        self.reward = self.get_reward(action[0][0])

        diff_joint = self.joint_position - self.last_joint

        self.robot_state = np.concatenate([
            self.joint_position,
            diff_joint,
            self.orientation,
            self.angular_vel,
            self.linear_acc
        ]).reshape(self.robot_state_shape)

        self.motion_state = np.concatenate(
            [
                pos,
                delta_motion
            ],
            0
        )

        self.history = np.concatenate(
            [
                self.history[1:, :],
                self.joint_pos
            ],
            0
        )

        self.last_joint = self.joint_position
        self.last_pos = pos

        curr_time = rospy.get_time()
        print('time:', curr_time - self.episode_start_time)

        return [
            self.motion_state,
            self.robot_state,
            self.osc_state,
            self.history
        ], self.reward

    def start(self, count):
        rospy.spin()

    def stop(self, reason):
        rospy.signal_shutdown(reason)
