import random
import numpy as np
import os
import torch
from torch import nn

class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")


    def act(self, state):
        with torch.no_grad():
            action = self.model(torch.tensor(state))
            return action.argmax(-1).numpy()


    def reset(self):
        pass
