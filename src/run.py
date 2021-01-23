from simulations.ws.src.quadruped_control import robot
from rl.constants import params

spider = robot.WalkingSpider(params)
robotID = spider._load_urdf('simulations/ws/src/quadruped_description/urdf/spider_simple.urdf')
