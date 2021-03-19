import random
import numpy as np
import os
import torch

device = torch.device("cpu")


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.to(device)
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float().to(device)
            return self.model(state).numpy()

    def reset(self):
        pass

