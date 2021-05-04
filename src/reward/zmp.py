from reward.support_plane import SupportPlane
import numpy as np
import rospy
from gazebo_msgs.srv import GetPhysicsProperties

class ZMP:
    def __init__(self, params):
        self.params = params
        self.support_plane = SupportPlane(params)
        self.get_physics_prop_proxy = rospy.ServiceProxy(
            '/gazebo/get_physics_properties',
            GetPhysicsProperties
        )
        g = self.get_physics_prop_proxy().gravity
        self.g = np.array([
            g.x,
            g.y,
            g.z
        ])
        self.zmp_s = np.zeros((3,))
        self.zmp = np.zeros((3,))
        self.inertial_plane = np.eye(N = 3)
        self.plane = self.inertial_plane

    def update_g(self):
        g = self.get_physics_prop_proxy().gravity
        self.g = np.array([
            g.x,
            g.y,
            g.z
        ])

    def build(self, t, Tb, A, B, AL, BL, AF, BF):
        self.support_plane.build(t, Tb, A, B, AL, BL, AF, BF)

    def _transform(self, vec, cs1, cs2):
            return self.support_plane.transform(vec, cs1, cs2)

    def get_ZMP_s(self, com, force, torque):
        self.plane = self.support_plane()
        com_s = self._transform(com, self.plane, self.inertial_plane)
        force_s = self._transform(force, self.plane, self.inertial_plane)
        torque_s = self._transform(torque, self.plane, self.inertial_plane)
        g_s = self._transform(self.g, self.plane, self.inertial_plane)
        zmp_s = np.zeros((3,))
        zmp_s[1] = com_s[1] - (
            com_s[0] * (
                force_s[1] + g_s[1]
            ) + torque_s[2]
        ) / (force_s[0] + g_s[0])
        zmp_s[2] = com_s[2] - (
            com_s[0] * (
                force_s[2] + g_s[2]
            ) - torque_s[1]
        ) / (force_s[0] + g_s[0])
        return zmp_s

    def __call__(self, com, force, torque, v_real, v_exp, eta):
        self.zmp_s = self.get_ZMP_s(com, force, torque)
        self.zmp = self.zmp_s + eta*(v_real - v_exp)
        self.zmp[0] = 0
        return self._transform(self.zmp, self.inertial_plane, self.plane)
