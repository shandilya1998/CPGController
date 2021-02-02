from reward.support_plane import SupportPlane
import numpy as np
import rospy

class ZMP:
    def __init__(self, params):
        self.params = params
        self.support_plane = SupportPlane(params)
        self.get_physics_prop_proxy = rospy.ServiceProxy(
            '/gazebo/get_world_properties',
            GetPhysicsProperties
        )
        g = self.get_physics_prop_proxy().gravity
        self.g = np.array([
            g.x,
            g.y,
            g.z
        ])

    def update_g(self):
        g = self.get_physics_prop_proxy().gravity
        self.g = np.array([
            g.x,
            g.y,
            g.z
        ])

    def build(self, t, Tb, A, B, AL, BL, AF, BF):
        self.support_plane.build(t, Tb, A, B, AL, BL, AF, BF)

    def __call__(self, com, force, torque):
        plane = self.support_plane()
        inertial_plane = np.eye(N = 3, k = 1)
        com_s = transform(com, plane, inertial_plane)
        force_s = transform(force, plane, inertial_plane)
        torque_s = transform(torque, plane, inertial_plane)
        g_s = transform(self.g, plane, inertial_plane)
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
