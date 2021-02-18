from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

GAMMA = 0.98
GRID_SIZE_X = 30
GRID_SIZE_Y = 30
SEED = 1000
rs = RandomState(MT19937(SeedSequence(SEED)))

# Simple discretization
def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X - 1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y - 1)
    return x + GRID_SIZE_X * y


class QLearning:
    def __init__(self, state_dim, action_dim, alpha=0.1):
        self.Q = np.zeros((state_dim, action_dim)) + 2.
        self.alpha = alpha
        self.best_score = -200


    def update(self, transition):
        state, action, next_state, reward, done = transition
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] +\
                                      self.alpha * (reward + GAMMA *
                                                    np.amax(self.Q[next_state]) -\
                                                    self.Q[state][action])

    def act(self, state):
        values = self.Q[state]
        action = np.argmax(values)
        return action

    def save(self, score):
        if self.best_score <= score:
            self.best_score = score
            print(self.best_Q - self.Q)
            self.best_Q = self.Q
            np.save("agent.npz", self.Q)



def evaluate_policy(agent, episodes=5):
    env = make("MountainCar-v0")
    env.seed(SEED)
    env.action_space.seed(SEED)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            action = agent.act(transform_state(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("MountainCar-v0")
    env.seed(SEED)
    env.action_space.seed(SEED)
    ql = QLearning(state_dim=GRID_SIZE_X * GRID_SIZE_Y, action_dim=3)
    eps = 0.2
    transitions = 4000000
    trajectory = []
    state = transform_state(env.reset())
    for i in range(transitions):
        total_reward = 0
        steps = 0
        eps = eps - 2/transitions if eps > 0.01 else 0.01
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = ql.act(state)

        next_state, reward, done, _ = env.step(action)
        reward += abs(next_state[1]) / 0.07
        next_state = transform_state(next_state)

        trajectory.append((state, action, next_state, reward, done))

        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []
        state = next_state if not done else transform_state(env.reset())

        if (i + 1) % (transitions // 100) == 0:
            rewards = evaluate_policy(ql, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)},\
            Reward std: {np.std(rewards)}")
            score = np.mean(rewards)
            ql.save(score)


