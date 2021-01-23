import pybullet as p
from gait_generation import gait_generator
import time
import numpy as np
import pybullet_data

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotID = p.loadURDF("simulations/ws/src/quadruped_description/urdf/spider_simple.urdf",cubeStartPos, cubeStartOrientation, 
                   # useMaximalCoordinates=1, ## New feature in Pybullet
                   flags=p.URDF_USE_INERTIA_FROM_FILE | \
                        p.URDF_USE_SELF_COLLISION | \
                        p.URDF_MERGE_FIXED_LINKS | \
                        p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

def get_joint_index(robotID):
    print('Robot ID:')
    print(robotID)
    print('Number of Joints:')
    num_j = p.getNumJoints(robotID)
    print(num_j)

    joints = ['Leg1Hip', 'Leg2Hip', 'Leg3Hip', 'Leg4Hip', 'Leg1Knee', 'Leg2Knee', 'Leg3Knee', 'Leg4Knee']

    joint_dct = {}

    for i in range(num_j):
        info = p.getJointInfo(robotID, i)
        if info[1].decode() in joints:
            joint_dct[info[1].decode()] = info[0]

    return joint_dct

print(get_joint_index(robotID))
print(p.getContactPoints(robotID))

driven_joints = [ 
    'Leg1Hip',
    'Leg2Hip',
    'Leg3Hip',
    'Leg4Hip',
    'Leg1Knee',
    'Leg2Knee',
    'Leg3Knee',
    'Leg4Knee'
]

j2i = {}
i2j = {}
for i in range(p.getNumJoints(robotID, physicsClient)):
    if p.getJointInfo(robotID, i)[1].decode('UTF-8') in driven_joints:
        j2i[
            p.getJointInfo(
                robotID,
                i
            )[1].decode('UTF-8')
        ] = p.getJointInfo(robotID, i)[0]
        i2j[
            p.getJointInfo(
                robotID,
                i
            )[0]
        ] = p.getJointInfo(robotID, i)[1].decode('UTF-8')

print('------------------')
print('Joint     |Index')
print('------------------')
for key in j2i.keys():
    space = 10-len(key)
    space = ''.join([' ' for i in range(space)])
    print('{j}{s}'.format(j = key, s = space), end = '|')
    print('{i}'.format(i = j2i[key]))
    print('------------------')
    driven_joint_indices=[j2i[j] for j in driven_joints]

print(driven_joint_indices)

N = 10000
Tst = 200
Tsw = 50
theta = 30
dt = 0.001
out, _ = gait_generator.get_signal(dt, Tsw, Tst, N, theta, 5)
print(out.shape)
for i in range (0, N, 10):
    angles = np.zeros(8).tolist()
    for j in range(4):
        angles[j] = out[i][2*j+1]*np.pi/180
        angles[j+4] = out[i][2*j+2]*np.pi/180
    #print(len(angles))
    p.setJointMotorControlArray(
        robotID,
        driven_joint_indices,
        p.POSITION_CONTROL,
        angles
    )
    #print(angles[0])
    p.stepSimulation()
    time.sleep(1./240.)
    #print(p.getContactPoints(robotID, planeId))
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotID)
print(cubePos,cubeOrn)


p.disconnect()

