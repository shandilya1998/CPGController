#!/usr/bin/env python
import rospy
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
    GetModelStateRequest
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Imu

class Quadruped:
    def __init__(self, params):
        rospy.init_node('joint_position_node')
        self.nb_joints = params['action_dim']
        self.nb_links = params['action_dim'] + 1
        self.robot_state_shape = (params['robot_state_size'],)
        self.motion_state_shape = (params['motion_state_size'],)
        self.params = params
        self.link_name_lst = [
            'quadruped::base_link',
            'quadruped::front_right_leg1',
            'quadruped::front_right_leg2',
            'quadruped::front_right_leg3',
            'quadruped::front_left_leg1',
            'quadruped::front_left_leg2',
            'quadruped::front_left_leg3',
            'quadruped::back_right_leg1',
            'quadruped::back_right_leg2',
            'quadruped::back_right_leg3',
            'quadruped::back_left_leg1',
            'quadruped::back_left_leg2',
            'quadruped::back_left_leg3'
        ]
        self.leg_link_name_lst = self.link_name_lst[1:]
        self.joint_name_lst = [
            'front_right_leg1_joint',
            'front_right_leg2_joint',
            'front_right_leg3_joint',
            'front_left_leg1_joint',
            'front_left_leg2_joint',
            'front_left_leg3_joint',
            'back_right_leg1_joint',
            'back_right_leg2_joint',
            'back_right_leg3_joint',
            'back_left_leg1_joint',
            'back_left_leg2_joint',
            'back_left_leg3_joint'
        ]

        self.joint_pos_pub_lst = [
            rospy.Publisher(
                '/quadruped/{joint}/command'.format(joint = j), 
                Float64,
                queue_size = 10
            ) for j in self.joint_name_lst
        ]

        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.model_config_proxy = rospy.ServiceProxy('/gazebo/set_model_configuration', \
            SetModelConfiguration)
        self.model_config_req = SetModelConfigurationRequest()
        self.model_config_req.model_name = 'quadruped'
        self.model_config_req.urdf_param_name = 'robot_description'
        self.model_config_req.joint_names = self.joint_name_lst
        self.model_config_req.joint_positions = self.starting_pos
        self.model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.model_state_req = SetModelStateRequest()
        self.model_state_req.model_state = ModelState()
        self.model_state_req.model_state.model_name = 'quadruped'
        self.model_state_req.model_state.pose.position.x = 0.0
        self.model_state_req.model_state.pose.position.y = 0.0
        self.model_state_req.model_state.pose.position.z = 0.25
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

        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.get_model_state_req = GetModelStateRequest()
        self.get_model_state_req.model_name = 'quadruped'
        self.get_model_state_req.relative_entity_name = 'world'

        self.robot_state = np.zeros(self.robot_state_shape)
        self.joint_state = np.zeros(self.nb_joints)
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

    def set_all_joint_pos(self, pos):
        """
            param: 
                name: pos
                type: list
                description: list of joint positions in the same order as the joints in 
                    self.joint_name_lst
        """
        for i in range(len(self.joint_name_lst)):
            self._set_joint_pos(i, pos[i])

    def _set_joint_pos(self, joint_index, pos):
        self.joint_pos_pub_lst[joint_index].publish(pos)

    def joint_state_subscriber_callback(self, joint_state):
        state = [st for st in joint_state.position]
        for i, name in enumerate(joint_state.name):
            index = self.joint_name_list.index(name)
            state[index] = joint_state.position[i]
        self.joint_state = np.array(state)

    def imu_subscriber_callback(self,imu):
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

    def step(self, action, history):
        action = [tf.make_ndarray(ac)[0] for ac in action]
        history = tf.make_ndarray(history)[0]
        joint_pos = action[0]
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        pos = np.array(
            [
                model_state.pose.position.x,
                model_state.pose.position.y,
                model_state.pose.position.z
            ]
        )


    def stop(self, reason):
        rospy.signal_shutdown(reason)
