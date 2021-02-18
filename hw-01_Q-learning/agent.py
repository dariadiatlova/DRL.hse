import random
import numpy as np
import os
from .train import transform_state


class Agent:
    def __init__(self):
        self.Q = np.load(__file__[:-8] + "/agent.npz.npy")

    def act(self, state):
        state = transform_state(state)
        action = np.argmax(self.Q[state])
        return action

    def reset(self):
        pass

