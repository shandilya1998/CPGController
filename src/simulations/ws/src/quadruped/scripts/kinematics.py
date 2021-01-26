import moveit_commander
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from std_msgs.msg import Header

class Kinematics:
    def __init__(self, params):
        self.params = params
        rospy.init('get_fk', anonymous = True)
        self.quadruped = moveit_commander.MoveGroupCommander('')
