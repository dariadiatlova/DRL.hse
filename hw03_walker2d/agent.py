from torch import nn
from torch.distributions import Normal
import random
import numpy as np
import os
import torch

device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim)
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions (use it to compute entropy loss)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).expand_as(mu)
        distr = Normal(mu, sigma)
        return torch.exp(distr.log_prob(action).sum(-1)), distr

    def act(self, state):
        # Returns an action, not-transformed action and distribution
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).expand_as(mu)
        distr = Normal(mu, sigma)
        pure_action = distr.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, distr


class Agent:
    def __init__(self):
        self.model = Actor(22, 6).to(device)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float().to(device)
            action, _, _ = self.model.act(state)
        return action

    def reset(self):
        pass
