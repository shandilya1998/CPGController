#!/usr/bin/env python
import rospy
import re
import copy
import actionlib
import numpy as np
import tensorflow as tf
from control_msgs.msg import FollowJointTrajectoryAction, \
    FollowJointTrajectoryActionGoal, \
    FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, \
    JointTrajectoryPoint
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, \
    SetModelStateRequest, \
    SetModelConfiguration, \
    SetModelConfigurationRequest, \
    GetModelState, \
    GetModelStateRequest, \
    GetLinkProperties, \
    GetLinkState, \
    GetPhysicsProperties
from geometry_msgs.msg import Pose, PoseStamped, WrenchStamped
import tf2_geometry_msgs
from gazebo_msgs.msg import ModelState, ContactsState
from sensor_msgs.msg import Imu
from simulations.ws.src.quadruped.scripts.kinematics import Kinematics
from reward import FitnessFunction
import tf2_ros
import time
import matplotlib.pyplot as plt
from moveit_commander import MoveGroupCommander, roscpp_initialize
import sys
import moveit_commander
import math

class Leg:
    def __init__(self, params, leg_name, joint_name_lst):
        self.params = params
        self.leg_name = leg_name
        self.joint_name_lst = joint_name_lst
        self.jta = actionlib.SimpleActionClient(
            '/quadruped/{leg_name}_controller/follow_joint_trajectory'.format(
                leg_name = self.leg_name
            ),
            FollowJointTrajectoryAction
        )
        print('[DDPG] Waiting for joint trajectory action')
        self.jta.wait_for_server()
        print('[DDPG] Found joint trajectory action!')
        self.jtp_zeros = np.zeros((len(joint_name_lst),)).tolist()
        self.jtp_vel = np.zeros((len(joint_name_lst),))
        self.jtp_vel.fill(8.7772)
        self.jtp_effort = np.zeros((len(joint_name_lst),))
        self.jtp_effort.fill(100)
        self.jtp = rospy.Publisher(
            '/quadruped/{leg_name}_controller/command'.format(
                leg_name = leg_name
            ),
            JointTrajectory,
            queue_size=1
        )
        self.vel = 8
        self.contact_state = ContactsState()
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
                ], dtype = np.float32),
                'torque' : np.array([
                    self.leg1_joint_force_torque.wrench.torque.x,
                    self.leg1_joint_force_torque.wrench.torque.y,
                    self.leg1_joint_force_torque.wrench.torque.z
                ], dtype = np.float32)
            },
            'leg2' : {
                'frame_id' : self.leg2_joint_force_torque.header.frame_id,
                'force' : np.array([
                    self.leg2_joint_force_torque.wrench.force.x,
                    self.leg2_joint_force_torque.wrench.force.y,
                    self.leg2_joint_force_torque.wrench.force.z
                ], dtype = np.float32),
                'torque' : np.array([
                    self.leg2_joint_force_torque.wrench.torque.x,
                    self.leg2_joint_force_torque.wrench.torque.y,
                    self.leg2_joint_force_torque.wrench.torque.z
                ], dtype = np.float32)
            },
            'leg3' : {
                'frame_id' : self.leg3_joint_force_torque.header.frame_id,
                'force' : np.array([
                    self.leg3_joint_force_torque.wrench.force.x,
                    self.leg3_joint_force_torque.wrench.force.y,
                    self.leg3_joint_force_torque.wrench.force.z
                ], dtype = np.float32),
                'torque' : np.array([
                    self.leg3_joint_force_torque.wrench.torque.x,
                    self.leg3_joint_force_torque.wrench.torque.y,
                    self.leg3_joint_force_torque.wrench.torque.z
                ], dtype = np.float32)
            }
        }

    def _leg1_joint_force_torque_callback(self, wrench):
        self.leg1_joint_force_torque = wrench

    def _leg2_joint_force_torque_callback(self, wrench):
        self.leg2_joint_force_torque = wrench

    def _leg3_joint_force_torque_callback(self, wrench):
        self.leg3_joint_force_torque = wrench

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
                0, dtype = np.float32
            )
            torque = np.mean(
                [[
                    state.total_wrench.torque.x,
                    state.total_wrench.torque.y,
                    state.total_wrench.torque.z
                ] for state in states],
                0, dtype = np.float32
            )
            position = np.mean(
                [[
                    state.contact_positions[0].x,
                    state.contact_positions[0].y,
                    state.contact_positions[0].z
                ] for state in states],
                0, dtype = np.float32
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

    def _move_jta(self, pos):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_vel
        point.time_from_start = rospy.Duration(1.0)
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)

    def move_jta(self, pos):
        self._move_jta(pos)

    def reset_move_jta(self, pos):
        self._move_jta(pos)

    def move(self, pos):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        pos = np.array(pos)
        duration = rospy.Duration(1.0)
        point.velocities = self.jtp_vel.tolist()
        #point.accelerations = self.jtp_zeros
        #point.effort = self.jtp_zeros
        point.time_from_start = rospy.Duration(1.0/60.0)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

    def reset_move(self, pos):
        self.last_pos = pos
        jtp_msg = JointTrajectory()
        self.jtp.publish(jtp_msg)
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        #point.accelerations = self.jtp_zeros
        #point.effort = self.jtp_zeros
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
        self.A = np.zeros((3,), dtype = np.float32)
        self.B = np.zeros((3,), dtype = np.float32)
        self.w = 1

    def get_all_torques(self):
        regex = r'leg\d{1,3}'
        torque = []
        for joint in self.joint_name_lst:
            leg = self.get_leg_handle(joint[:-7])
            out = leg.get_processed_joint_force_troque()
            joint = re.search(regex, joint).group()
            out = out[joint]
            torque.append(np.linalg.norm(out['torque']))
        return np.array(torque, dtype = np.float32)

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
            B = copy.deepcopy(contacts[0])
            B['position'] = B['position'] + 1e-8
            return contacts + [B]
        elif len(contacts) == 2:
            return contacts
        elif len(contacts) == 3:
            """
                Assumption being made that the two support legs will always 
                be parrallel to the forward direction or diagonal
            """
            A = None
            B = None
            names = [contact['leg_name'] for contact in contacts]
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
            return [A, B]
        elif len(contacts) == 4:
            A = None
            B = None
            for contact in contacts:
                if contact['leg_name'] == self.leg_name_lst[0]:
                    A = contact
                elif contact['leg_name'] == self.leg_name_lst[-1]:
                    B = contact
            return [A, B]
        else:
            return contacts

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
    def __init__(self, params, experiment):
        self.params = params
        rospy.init_node('joint_position_node_exp{exp}'.format(
            exp = experiment
        ))
        roscpp_initialize(sys.argv)
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
        self.motion_state_set = False

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
        self.osc_state = self.create_init_osc_state()
        self.osc_state_set = False
        self.history_shape = tuple(
            params['history_spec'].shape
        )
        self.output_action_shape = tuple(
            params['action_spec'][0].shape
        )
        self.output_action_dtype = params[
            'action_spec'
        ][0].dtype.as_numpy_dtype
        self.osc_action_shape = tuple(
            params['action_spec'][1].shape
        )
        self.osc_action_dtype = params[
            'action_spec'
        ][1].dtype.as_numpy_dtype
        self.params = params
        self.link_name_lst = self.params['link_name_lst']
        self.leg_name_lst = self.params['leg_name_lst']
        self.leg_link_name_lst = self.link_name_lst[1:]
        self.joint_name_lst = self.params['joint_name_lst']

        self.starting_pos = self.params['starting_pos']
        self.history_osc = np.repeat(
            np.expand_dims(self.osc_state, 0),
            2 * self.params['rnn_steps'] - 1,
            0
        )
        self.history = np.repeat(
            np.expand_dims(self.starting_pos, 0),
            2 * self.params['rnn_steps'] - 1,
            0
        )

        self.all_legs = AllLegs(
            self.params,
        )

        self.kinematics = Kinematics(
            self.params
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
        self.get_physics_params_proxy = rospy.ServiceProxy(
            '/gazebo/get_physics_properties',
            GetPhysicsProperties
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

        self.dt = self.params['dt']
        self.orientation = np.zeros(4, dtype = np.float32)
        self.angular_vel = np.zeros(3, dtype = np.float32)
        self.linear_acc = np.zeros(3, dtype = np.float32)
        self.imu_subscriber = rospy.Subscriber(
            '/quadruped/imu',
            Imu,
            self.imu_subscriber_callback
        )

        self.reward = 0.0
        self.d1 = 0.0
        self.d2 = 0.0
        self.d3 = 0.0
        self.stability = 0.0
        self.COT = 0.0
        self.r_motion = 0.0
        self.episode_start_time = 0.0
        self.max_sim_time = 15.0
        self.pos_z_limit = 0.18

        self.compute_reward = FitnessFunction(
            self.params
        )

        self.mass = self.get_total_mass()
        self.force = np.zeros((3,))
        self.force.fill(1e-8)
        self.torque = np.zeros((3,))
        self.torque.fill(1e-8)
        self.com = np.zeros((3,))
        self.com.fill(1e-8)
        self.v_exp = np.zeros((3,))
        self.v_exp.fill(1e-8)
        self.v_real = np.zeros((3,))
        self.v_real.fill(1e-8)
        self.pos = np.zeros((3,))
        self.pos.fill(1e-8)
        self.last_pos = self.pos
        self.eta = 1e8

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.Tb = self.params['dt']
        self.upright = True
        self.COT = 0.0
        self.r_motion = 0.0
        self.stability = 0.0

        self._reset()
        rospy.sleep(10.0)
        current_pose = self.kinematics.get_current_end_effector_fk()
        A, B = self.all_legs.get_AB()
        self.A_init = A
        self.B_init = B
        A = self.get_contact_ob(A['leg_name'], current_pose)
        B = self.get_contact_ob(B['leg_name'], current_pose)
        def get_other_leg(name):
            if 'front' in name:
                if 'left' in name:
                    return 'front_right_leg'
                else:
                    return 'front_left_leg'
            else:
                if 'left' in name:
                    return 'back_right_leg'
                else:
                    return 'back_left_leg'
        A_ = self.get_contact_ob(get_other_leg(A['leg_name']), current_pose)
        B_ = self.get_contact_ob(get_other_leg(B['leg_name']), current_pose)
        self.A = [A_, A, A_]
        self.A_time = [-1e-3, 0.0, 1e-3]
        self.B = [B_, B, B_]
        self.B_time = [-1e-3, 0.0, 1e-3]
        self.last_joint = np.zeros((self.params['action_dim']))
        self.t = 1e-8
        self.delta = self.dt
        rospy.sleep(2.0)
        self.com = self.get_com()
        self.counter = 0
        params = self.get_physics_params_proxy()
        self.gravity = np.array([
            params.gravity.x,
            params.gravity.y,
            params.gravity.z
        ], dtype = np.float32)
        self.time = rospy.get_rostime().to_sec()

    def set_motion_state(self, desired_motion):
        self.motion_state = desired_motion
        self.motion_state_set = True

    def create_init_osc_state(self):
        r = np.ones((self.params['units_osc'],), dtype = np.float32)
        phi = np.zeros((self.params['units_osc'],), dtype = np.float32)
        z = r * np.exp(1j * phi)
        x = np.real(z)
        y = np.imag(z)
        return np.concatenate([x, y], -1)

    def _hopf_oscillator(self, omega, mu, b):
        rng = np.arange(1, self.params['units_osc'] + 1)
        x = self.osc_state[:self.params['units_osc']]
        y = self.osc_state[self.params['units_osc']:]
        z = x + 1j * y
        rng = omega * rng
        mod = (mu - np.square(np.abs(z)))
        r = np.abs(z)
        phi = np.angle(z)
        r = r + (mu-r**2)*r*self.dt
        phi = phi + rng*self.dt
        z = r*np.exp(1j*phi) + b
        x = np.real(z)
        y = np.imag(z)
        self.osc_state = np.concatenate([x, y], -1)

    def set_osc_state(self, osc):
        self.osc_state = osc
        self.osc_state_set = True

    def get_state_tensor(self):
        diff_joint = self.joint_position - self.last_joint

        self.robot_state = np.concatenate([
            np.sin(self.joint_position),
            np.sin(diff_joint),
            self.orientation,
            self.angular_vel,
            self.linear_acc
        ]).reshape(self.robot_state_shape)

        return [
            np.expand_dims(self.motion_state, 0),
            np.expand_dims(self.robot_state, 0),
            np.expand_dims(self.osc_state, 0)
        ]

    def get_total_mass(self):
        mass = 0
        for link in self.params['link_name_lst']:
            msg = self.link_prop_proxy(link)
            mass += msg.mass
        return mass

    def transform_pose(self, input_pose, from_frame, to_frame):
        # **Assuming /tf2 topic is being broadcasted
        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time.now()

        try:
            # ** It is important to wait for the listener to start listening. 
            # Hence the rospy.Duration(1.0/60.0)
            output_pose_stamped = self.tf_buffer.transform(
                pose_stamped,
                to_frame,
                rospy.Duration(0.1)
            )
            return output_pose_stamped.pose

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):
            raise

    def get_com(self):
        reference = 'dummy_link'
        x, y, z = 0,0,0
        link = 'quadruped::base_link'
        prop = self.link_prop_proxy(link)
        mass = prop.mass
        pose = Pose()
        pose.position.x = prop.com.position.x
        pose.position.y = prop.com.position.y
        pose.position.z = prop.com.position.z
        pose.orientation.x = prop.com.orientation.x
        pose.orientation.y = prop.com.orientation.y
        pose.orientation.z = prop.com.orientation.z
        pose.orientation.w = prop.com.orientation.w
        t_pose = self.transform_pose(pose, link[11:], reference)
        x += t_pose.position.x * mass
        y += t_pose.position.y * mass
        z += t_pose.position.z * mass
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
        self.joint_position = np.array(state, dtype = np.float32)
        self.joint_velocity = np.array(vel, dtype = np.float32)

    def imu_subscriber_callback(self, imu):
        self.orientation = np.array(
            [
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
                imu.orientation.w
            ], dtype = np.float32
        )
        self.angular_vel = np.array(
            [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z
            ], dtype = np.float32
        )
        self.linear_acc = np.array(
            [
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z
            ], dtype = np.float32
        )

    def _reset(self):
        #pause physics
        start = time.time()
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException:
            print('[Gazebo] /gazebo/pause_physics service call failed')
        #set models pos from world
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.model_state_proxy(self.model_state_req)
        except rospy.ServiceException:
            print('[Gazebo] /gazebo/set_model_state call failed')
        #set model's joint config
        rospy.wait_for_service('/gazebo/set_model_configuration')
        try:
            self.model_config_proxy(self.model_config_req)
        except rospy.ServiceException:
            print('[Gazebo] /gazebo/set_model_configuration call failed')
        self.action = self.starting_pos
        self.all_legs.reset_move(self.starting_pos)
        #unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except rospy.ServiceException:
            print('[Gazebo] /gazebo/unpause_physics service call failed')
        end = time.time()

    def reset(self):
        self._reset()
        rospy.sleep(1.0)
        start = time.time()
        self.reward = 0.0
        if not self.osc_state_set:
            self.osc_state = self.create_init_osc_state()
        else:
            self.osc_state_set = False
        self.history = np.repeat(
            np.expand_dims(self.starting_pos, 0),
            2 * self.params['rnn_steps'] - 1,
            0
        )

        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)

        self.pos = np.array([
            model_state.pose.position.x,
            model_state.pose.position.y,
            model_state.pose.position.z
        ], dtype = np.float32)
        self.history_pos = np.zeros(
            shape = (2 * self.params['rnn_steps'] - 1, 3),
            dtype = np.float32
        )
        self.history_pos = np.concatenate(
            [
                self.history_pos[1:, :],
                np.expand_dims(self.pos, 0)
            ], 0
        )

        self.last_joint = self.joint_position
        diff_joint = np.zeros(self.nb_joints, dtype = np.float32)
        self.robot_state = np.concatenate([
            np.sin(self.joint_position),
            np.sin(diff_joint),
            self.orientation,
            self.angular_vel,
            self.linear_acc
        ])
        if not self.motion_state_set:
            self.motion_state = np.zeros(
                shape = self.motion_state_shape,
                dtype = self.motion_state_dtype
            )
        else:
            self.motion_state_set = False
        self.joint_torque = self.all_legs.get_all_torques()
        self.history_joint_torque = np.zeros(
            shape = (2 * self.params['rnn_steps'] - 1, self.nb_joints),
            dtype = np.float32
        )
        self.history_joint_torque = np.concatenate(
            [
                self.history_joint_torque[1:, :],
                np.expand_dims(self.joint_torque, 0)
            ]
        )

        self.history_joint_vel = np.zeros(
            shape = (2 * self.params['rnn_steps'] - 1, self.nb_joints),
            dtype = np.float32
        )
        self.history_joint_vel = np.concatenate(
                [
                    self.history_joint_vel[1:],
                    np.expand_dims(self.joint_velocity,0)
                ],
                0
            )

        self.v_real = np.array([
            model_state.twist.linear.x,
            model_state.twist.linear.y,
            model_state.twist.linear.z
        ], dtype = np.float32)
        self.history_vel = np.zeros(
            shape = (2 * self.params['rnn_steps'] - 1, 3),
            dtype = np.float32
        )
        self.history_vel = np.concatenate([
            self.history_vel[1:],
            np.expand_dims(self.v_real, 0)
        ])
        self.history_desired_motion = np.zeros(
            shape = (
                2 * self.params['rnn_steps'] - 1,
                self.params['motion_state_size']
            ),
            dtype = np.float32
        )

        self.starting_pos = self.params['starting_pos']
        self.history = np.zeros(
            shape = (2 * self.params['rnn_steps'] - 1, self.nb_joints),
            dtype = np.float32
        )

        rospy.sleep(2.0)
        self.com = self.get_com()
        current_pose = self.kinematics.get_current_end_effector_fk()
        AB = self.all_legs.get_AB()
        if AB:
            self.upright = False
            AB = [self.A_init, self.B_init]
        A, B = AB
        A = self.get_contact_ob(A['leg_name'], current_pose)
        B = self.get_contact_ob(B['leg_name'], current_pose)
        def get_other_leg(name):
            if 'front' in name:
                if 'left' in name:
                    return 'front_right_leg'
                else:
                    return 'front_left_leg'
            else:
                if 'left' in name:
                    return 'back_right_leg'
                else:
                    return 'back_left_leg'
        A_ = self.get_contact_ob(get_other_leg(A['leg_name']), current_pose)
        B_ = self.get_contact_ob(get_other_leg(B['leg_name']), current_pose)
        self.A = [A_, A, A_]
        self.A_time = [-1e-3, 0.0, 1e-3]
        self.B = [B_, B, B_]
        self.B_time = [-1e-3, 0.0, 1e-3]
        end = time.time()
        self.episode_start_time = rospy.get_time()
        self.counter = 0
        self.time = rospy.get_rostime().to_sec()
        self.COT = 0.0
        self.r_motion = 0.0
        self.stability = 0.0
        return [
            self.motion_state,
            self.robot_state,
            self.osc_state,
        ], self.reward

    def get_contact_ob(self, leg_name, pose):
        m = {
            0 : 'x',
            1 : 'y',
            2 : 'z'
        }
        return {
            'position' : np.array(
                [
                    pose[leg_name]['position'][m[i]] for i in range(3)
                ], dtype = np.float32
            ),
            'leg_name' : leg_name,
            'force' : self.force,
            'torque' : self.torque,
            'v_real' : self.v_real,
            'v_exp' : self.v_exp,
            'eta' : self.eta,
        }

    def set_support_lines(self):
        AB = self.all_legs.get_AB()
        if len(AB) == 2:
            self.upright = True
            current_pose = self.kinematics.get_end_effector_fk(
                self.action.tolist()
            )
            if self.A[-1]['leg_name'] != AB[0]['leg_name']:
                self.A_time.pop(0)
                self.A.pop(0)
                self.A_time.append(self.time)
                self.A.append(
                    self.get_contact_ob(AB[0]['leg_name'],current_pose)
                )
            else:
                self.A[-1] = self.get_contact_ob(AB[0]['leg_name'],current_pose)
            if self.B[-1]['leg_name'] != AB[1]['leg_name']:
                self.B_time.pop(0)
                self.B.pop(0)
                self.B_time.append(self.time)
                self.B.append(
                    self.get_contact_ob(AB[1]['leg_name'],current_pose)
                )
            else:
                self.B[-1] = self.get_contact_ob(AB[1]['leg_name'],current_pose)

        else:
            #raise NotImplementedError
            self.upright = False
            self.reward += -5.0

    def set_last_pos(self):
        self.last_pos = self.pos

    def set_observation(self, action, desired_motion):
        self.action, self.osc_state = action
        self.action = self.action[0]
        self.osc_state = self.osc_state[0].astype('float32')
        self.all_legs.move(self.action.tolist())
        self.delta = rospy.get_rostime().to_sec() - self.time
        self.time = rospy.get_rostime().to_sec()
        if self.delta == 0.0:
            self.delta = self.dt
        diff_joint = self.joint_position - self.last_joint

        self.robot_state = np.concatenate([
            np.sin(self.joint_position),
            np.sin(diff_joint),
            self.orientation,
            self.angular_vel,
            self.linear_acc
        ]).astype('float32')

        self.motion_state = desired_motion.astype('float32')

        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        self.pos = np.array([
            model_state.pose.position.x,
            model_state.pose.position.y,
            model_state.pose.position.z
        ], dtype = np.float32)

        self.joint_torque = self.all_legs.get_all_torques()
        self.v_real = np.array([
            model_state.twist.linear.x,
            model_state.twist.linear.y,
            model_state.twist.linear.z
        ], dtype = np.float32)
        self.v_exp =  desired_motion[3:6]

    def set_history(self, desired_motion):
        self.history = np.concatenate(
            [self.history[1:], np.expand_dims(self.action, 0)], 0
        )
        self.history_osc = np.concatenate(
            [self.history_osc[1:], np.expand_dims(self.osc_state, 0)], 0
        )
        self.history_pos = np.concatenate(
            [
                self.history_pos[1:, :],
                np.expand_dims(self.pos, 0)
            ], 0
        )
        self.history_joint_torque = np.concatenate(
            [
                self.history_joint_torque[1:, :],
                np.expand_dims(self.joint_torque, 0)
            ]
        )
        self.history_joint_vel = np.concatenate(
            [
                self.history_joint_vel[1:],
                np.expand_dims(self.joint_velocity,0)
            ],
            0
        )
        self.history_desired_motion = np.concatenate([
            self.history_desired_motion[1:],
            np.expand_dims(desired_motion, 0)
        ])
        self.history_vel = np.concatenate([
            self.history_vel[1:],
            np.expand_dims(self.v_real, 0)
        ])

    def get_COT(self):
        self.COT = self.compute_reward.COT(
            self.joint_torque,
            self.joint_velocity,
            self.v_real,
            self.mass,
            self.gravity,
            self.delta
        )
        return self.COT

    def get_motion_reward(self):
        self.r_motion = self.compute_reward.motion_reward(
            self.history_pos,
            self.history_vel,
            self.history_desired_motion,
            self.pos,
            self.last_pos,
            self.motion_state
        )
        return self.r_motion

    def get_stability_reward(self):
        if self.upright:
            t_1 = max(self.A_time[0], self.B_time[0])
            t_2 = max(self.A_time[-1], self.B_time[-1])
            self.Tb = t_2 - t_1
            self.t = max(self.A_time[1], self.B_time[1])
            eta = self.A[1]['eta']
            force = self.A[1]['force']
            torque = self.A[1]['torque']
            v_real = self.A[1]['v_real']
            v_exp = self.A[1]['v_exp']
            if self.A_time[1] < self.B_time[1]:
                eta = self.B[1]['eta']
                force = self.A[1]['force']
                torque = self.A[1]['torque']
                v_real = self.A[1]['v_real']
                v_exp = self.A[1]['v_exp']
            if self.Tb == 0:
                self.reward += -5.0
                return
            self.compute_reward.build(
                self.t,
                self.Tb,
                self.A[1],
                self.B[1],
                self.A[0],
                self.B[0],
                self.A[-1],
                self.B[-1],
            )
            self.d1, self.d2, self.d3, self.stability = \
                self.compute_reward.stability_reward(
                self.com,
                force,
                torque,
                v_real,
                v_exp,
                eta,
                self.mass,
                self.gravity
            )
            self.reward += np.nan_to_num(self.stability)
            if self.compute_reward.zmp.support_plane.flag:
                self.reward += -5.0
            if math.isnan(self.reward):
                self.reward += -5.0
        else:
            self.reward += -5.0
        return self.reward

    def step(self, action, desired_motion):
        now = rospy.get_rostime().to_sec() 
        self.reward = 0.0
        action = [
            a.numpy() for a in action
        ]
        self.set_observation(action, desired_motion)
        rospy.sleep(15.0/60.0)
        self.force = self.mass * self.linear_acc
        self.torque = self.get_moment()
        vd = np.linalg.norm(self.v_exp)
        if (self.action > np.pi/3).any() or (self.action < -np.pi/3).any():
            self.reward += -5.0
        if vd == 0:
            vd = 1e-8
        self.eta = (self.params['L'] + self.params['W'])/(2*vd)
        self.set_support_lines()
        self.set_history(desired_motion)
        self.delta = rospy.get_rostime().to_sec() - now
        return [
            self.motion_state,
            self.robot_state,
            self.osc_state,
        ]

    def get_state(self):
        return [
            self.motion_state,
            self.robot_state,
            self.osc_state,
        ]

    def get_history(self):
        return tf.convert_to_tensor(np.expand_dims(self.history, 0))

    def get_osc_history(self):
        return tf.convert_to_tensor(np.expand_dims(self.history_osc, 0))

    def start(self, count):
        rospy.spin()

    def stop(self, reason):
        rospy.signal_shutdown(reason)
