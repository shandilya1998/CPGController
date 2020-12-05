import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("../quadruped_description/urdf/quadruped.urdf",cubeStartPos, cubeStartOrientation, 
                   # useMaximalCoordinates=1, ## New feature in Pybullet
                   flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_MERGE_FIXED_LINKS)

def get_joint_index(robotId):
    print('Robot ID:')
    print(robotId)
    print('Number of Joints:')
    num_j = p.getNumJoints(robotId)
    print(num_j)

    joints = ['Rev14', 'Rev15', 'Rev16', 'Rev17', 'Rev26', 'Rev27', 'Rev28', 'Rev29']

    joint_dct = {}

    for i in range(num_j):
        info = p.getJointInfo(robotId, i)
        if info[1].decode() in joints:
            joint_dct[info[1].decode()] = info[0]

    return joint_dct

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)

p.disconnect()

