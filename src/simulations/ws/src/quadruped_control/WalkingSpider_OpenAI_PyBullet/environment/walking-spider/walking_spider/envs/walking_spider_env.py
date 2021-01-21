import gym
from gym import spaces
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data

import time
import math
import numpy as np


class WalkingSpiderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init__(self, render=True):
        super(WalkingSpiderEnv, self).__init__()
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(75,), dtype=np.float32)

        self._observation = []
        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # used by loadURDF
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        self._seed()

        p.resetSimulation()
        p.setGravity(0, 0, -10)  # m/s^2
        # p.setTimeStep(1./60.)   # sec
        p.setTimeStep(0.01)   # sec
        self.plane = p.loadURDF("plane.urdf")

        self.cubeStartPos = [0, 0, 0.06]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.robotId = p.loadURDF(
            os.path.join(path, "spider.xml"),
            self.cubeStartPos,
            self.cubeStartOrientation
        )
        self.movingJoints = [0, 2, 3, 5, 6, 8, 9, 11]

    def reset(self):
        self.vt = [0, 0, 0, 0, 0, 0, 0, 0]
        self.vd = 0
        self.maxV = 8.72  # 0.12sec/60 deg = 500 deg/s = 8.72 rad/s
        self.envStepCounter = 0
        p.resetBasePositionAndOrientation(
            self.robotId,
            posObj=self.cubeStartPos,
            ornObj=self.cubeStartOrientation
        )
        observation = self.compute_observation()
        return observation

    def step(self, action):
        self.assign_throttle(action)
        observation = self.compute_observation()
        reward = self.compute_reward()
        done = self.compute_done()
        self.envStepCounter += 1
        return observation, reward, done, {}

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
    def moveLeg(self, robot, id, target):
      if(robot is None):
          return
      p.setJointMotorControl2(
          bodyUniqueId=robot,
          jointIndex=id,
          controlMode=p.POSITION_CONTROL,  # controlMode   = p.VELOCITY_CONTROL,        #p.POSITION_CONTROL,
          targetPosition=target # targetVelocity=target # targetVelocity= target                     #targetPosition=position,
      )    

    def assign_throttle(self, action):
      for i, key in enumerate(self.movingJoints):
        self.vt[i] = self.clamp(self.vt[i] + action[i], -2, 2)
        self.moveLeg(robot=self.robotId, id=key,  target=self.vt[i])


    def compute_observation(self):
      baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
      JointStates = p.getJointStates(self.robotId, self.movingJoints)
      BaseAngVel = p.getBaseVelocity(self.robotId)
      ContactPoints = p.getContactPoints(self.robotId, self.plane)

      debug = False
      if (debug):
        print("\nBase Orientation \nPos( x= {} , y = {} , z = {} )\nRot Quaternion( x = {} , y = {} , z = {}, w = {} )\n\n".format(
            baseOri[0][0], baseOri[0][1], baseOri[0][2],
            baseOri[1][0], baseOri[1][1], baseOri[1][2], baseOri[1][3]
        ))
        print(
            "\nJointStates: (Pos,Vel,6 Forces [Fx, Fy, Fz, Mx, My, Mz], appliedJointMotorTorque)\n")
        for i, joint in enumerate(JointStates):
            print("Joint #{} State: Pos {}, Vel {} Fx {} Fy {} Fz {} Mx {} My {} Mz {}, ApliedJointTorque {} ".format(
                i, joint[0], joint[1], joint[2][0], joint[2][1], joint[2][2], joint[2][3], joint[2][4], joint[2][5], joint[3]))
        print("\nBase Angular Velocity (Linear Vel( x= {} , y= {} , z=  {} ) Algular Vel(wx= {} ,wy= {} ,wz= {} ) ".format(
            BaseAngVel[0][0], BaseAngVel[0][1], BaseAngVel[0][2], BaseAngVel[1][0], BaseAngVel[1][1], BaseAngVel[1][2]))
      #print( "\n\nContact Points" ,ContactPoints if len(ContactPoints) > 0 else None )
      # print("\nContactPoints (contactFlag, idA, idB, linkidA, linkidB, posA, posB, contactnormalonb, contactdistance, normalForce, lateralFriction1, lateralFrictionDir1, lateralFriction2, lateralFrictionDir2)\n\n".format(
      # ))
      # print("Contact Point ", len(ContactPoints), ContactPoints[0] if len(ContactPoints) > 0 else None )

      obs = np.array([
          baseOri[0][2],  # z (height) of the Torso -> 1
          # orientation (quarternion x,y,z,w) of the Torso -> 4
          baseOri[1][0],
          baseOri[1][1],
          baseOri[1][2],
          baseOri[1][3],
          JointStates[0][0],  # Joint angles(Pos) -> 8
          JointStates[1][0],
          JointStates[2][0],
          JointStates[3][0],
          JointStates[4][0],
          JointStates[5][0],
          JointStates[6][0],
          JointStates[7][0],
          # 3-dim directional velocity and 3-dim angular velocity -> 3+3=6
          BaseAngVel[0][0],
          BaseAngVel[0][1],
          BaseAngVel[0][2],
          BaseAngVel[1][0],
          BaseAngVel[1][1],
          BaseAngVel[1][2],
          JointStates[0][1],  # Joint Velocities -> 8
          JointStates[1][1],
          JointStates[2][1],
          JointStates[3][1],
          JointStates[4][1],
          JointStates[5][1],
          JointStates[6][1],
          JointStates[7][1]
      ])
      # External forces (force x,y,z + torque x,y,z) applied to the CoM of each link (Ant has 14 links: ground+torso+12(3links for 4legs) for legs -> (3+3)*(14)=84
      external_forces = np.array([np.array(joint[2])
                                  for joint in JointStates])
      # print("Joint State Shape" , external_forces.shape, external_forces.flatten().shape )
      obs = np.append(obs, external_forces.ravel())
      # print("Obs: ", obs.shape, obs)
      return obs.tolist()

    def compute_reward(self):
      baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
      xposbefore = baseOri[0][0]

      BaseAngVel = p.getBaseVelocity(self.robotId)
      xvelbefore = BaseAngVel[0][0]

      p.stepSimulation()
      
      baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
      xposafter = baseOri[0][0]

      BaseAngVel = p.getBaseVelocity(self.robotId)
      xvelafter  = BaseAngVel[0][0]

      # forward_reward = (xposafter - xposbefore)
      forward_reward = 20 * (xvelbefore - xvelafter)


      JointStates = p.getJointStates(self.robotId, self.movingJoints)
      torques = np.array([np.array(joint[3]) for joint in JointStates])
      ctrl_cost = 1.0 * np.square(torques).sum()

      ContactPoints = p.getContactPoints(self.robotId, self.plane)
      contact_cost = 5 * 1e-1 * len(ContactPoints)
      # survive_reward = 1.0
      survive_reward = 0.0
      reward = forward_reward - ctrl_cost - contact_cost + survive_reward
      # print("Reward ", reward , "Contact Cost ", contact_cost, "forward reward ",forward_reward, "Control Cost ", ctrl_cost)
      # print("Reward ", reward)
      reward = reward if reward > 0 else 0

      p.addUserDebugLine(lineFromXYZ=(0, 0, 0), lineToXYZ=(
          0.3, 0, 0), lineWidth=5, lineColorRGB=[0, 255, 0], parentObjectUniqueId=self.robotId)
      p.addUserDebugText("Rewards {}".format(
          reward), [0, 0, 0.3],lifeTime=0.25, textSize=2.5, parentObjectUniqueId=self.robotId)
    #   print("Forward Reward", reward )
    #   print("Forward Reward", forward_reward, ctrl_cost, contact_cost , xvelbefore, xvelafter, xposbefore, xposafter )

      return reward 

    def compute_done(self):
      return False
    
    def render(self, mode='human', close=False):
      pass