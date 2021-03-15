import moveit_commander
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, \
    GetPositionFKRequest, \
    GetPositionIK
from std_msgs.msg import Header
import sys
# refer to the following link for creation of a service to get fk
# https://groups.google.com/g/moveit-users/c/Wb7TqHuf-ig

class Kinematics:
    def __init__(self, params):
        self.params = params
        self.joint_name_lst = self.params['joint_name_lst']
        self.leg_name_lst = self.params['leg_name_lst']
        self.quadruped = moveit_commander.RobotCommander()
        self.group_names = self.quadruped.get_group_names()

        self.end_effector_link_name_lst = []
        self.front_right_leg = moveit_commander.MoveGroupCommander(
            self.leg_name_lst[0]
        )
        self.end_effector_link_name_lst.append(
            self.front_right_leg.get_end_effector_link()
        )
        self.front_left_leg = moveit_commander.MoveGroupCommander(
            self.leg_name_lst[1]
        )
        self.end_effector_link_name_lst.append(
            self.front_left_leg.get_end_effector_link()
        )
        self.back_right_leg = moveit_commander.MoveGroupCommander(
            self.leg_name_lst[2]
        )
        self.end_effector_link_name_lst.append(
            self.back_right_leg.get_end_effector_link()
        )
        self.back_left_leg = moveit_commander.MoveGroupCommander(
            self.leg_name_lst[3]
        )
        self.end_effector_link_name_lst.append(
            self.back_left_leg.get_end_effector_link()
        )

        self.compute_fk_proxy = rospy.ServiceProxy(
            'compute_fk', 
            GetPositionFK
        )

        self.compute_ik_proxy = rospy.ServiceProxy(
            'compute_ik',
            GetPositionIK
        )

    def get_current_end_effector_fk(self):
        msg = []
        msg.append(self.front_right_leg.get_current_pose())
        msg.append(self.front_left_leg.get_current_pose())
        msg.append(self.back_right_leg.get_current_pose())
        msg.append(self.back_left_leg.get_current_pose())
        pose = {
            self.leg_name_lst[i] : {
                'position' : {
                    'x' : item.pose.position.x,
                    'y' : item.pose.position.y,
                    'z' : item.pose.position.z
                },
                'orientation' : {
                    'x' : item.pose.orientation.x,
                    'y' : item.pose.orientation.y,
                    'z' : item.pose.orientation.z,
                    'w' : item.pose.orientation.w
                }
            } for i, item in enumerate(msg)
        }
        return pose

    def get_end_effector_fk(self, pos):
        header = Header()
        rs = RobotState()
        rs.joint_state.name = self.joint_name_lst
        rs.joint_state.position = pos
        msg = self.compute_fk_proxy(
            header,
            self.end_effector_link_name_lst,
            rs
        )
        pose = {
            leg : {
                'position' : {
                    'x' : msg.pose_stamped[i].pose.position.x,
                    'y' : msg.pose_stamped[i].pose.position.y,
                    'z' : msg.pose_stamped[i].pose.position.z
                },
                'orientation' : {
                    'x' : msg.pose_stamped[i].pose.orientation.x,
                    'y' : msg.pose_stamped[i].pose.orientation.y,
                    'z' : msg.pose_stamped[i].pose.orientation.z,
                    'w' : msg.pose_stamped[i].pose.orientation.w
                }
            } for i, leg in enumerate(
                self.leg_name_lst
            )
        }
        return pose
