from simulations.ws.src.quadruped.scripts.quadruped import Quadruped
from rl.constants import params


class Learner():
    def __init__(self):
        self.params = params
        self.quadruped = Quadruped(params)
