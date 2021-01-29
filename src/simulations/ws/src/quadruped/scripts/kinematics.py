import moveit_commander
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from std_msgs.msg import Header

# refer to the following link for creation of a service to get fk
# https://groups.google.com/g/moveit-users/c/Wb7TqHuf-ig

class Kinematics:
    def __init__(self, params, joint_name_lst):
        self.params = params
        rospy.init_node('get_fk', anonymous = True)
        self.joint_name_lst = joint_name_lst
        self.quadruped = moveit_commander.RobotCommander()
        self.group_names = self.quadruped.get_group_names()
        
        self.end_effector_link_name_lst = []
        self.front_right_leg = moveit_commander.MoveGroupCommander(
            'front_right_leg'
        )
        self.end_effector_link_name_lst.append(
            self.front_right_leg.get_end_effector_link()
        )
        self.front_left_leg = moveit_commander.MoveGroupCommander(
            'front_left_leg'
        )
        self.end_effector_link_name_lst.append(
            self.front_left_leg.get_end_effector_link()
        )
        self.back_right_leg = moveit_commander.MoveGroupCommander(
            'back_right_leg'
        )
        self.end_effector_link_name_lst.append(
            self.back_right_leg.get_end_effector_link()
        )
        self.back_left_leg = moveit_commander.MoveGroupCommander(
            'back_left_leg'
        )
        self.end_effector_link_name_lst.append(
            self.back_left_leg.get_end_effector_link()
        )
        
        self.compute_fk_proxy = rospy.ServiceProxy(
            'compute_fk', 
            GetPositionFK
        )

    def get_fk(self, pos):
        header = Header()
        rs = RobotState()
        rs.joint_state.name = joint_name_lst
        rs.joint_state.position = pos
        msg = self.compute_fk_proxy(
            header,
            self.end_effector_link_name_lst,
            rs
        )
        return msg
