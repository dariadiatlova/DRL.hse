from gym import make
import numpy as np
import torch
from collections import deque
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
SEED = 1000
rs = RandomState(MT19937(SeedSequence(SEED)))


class DQN():
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.steps = 0  # Do not change
        channels = 256
        linear1 = nn.Linear(state_dim, channels)
        model = [nn.ReLU()]
        linear2 = nn.Linear(channels, channels)
        linear3 = nn.Linear(channels, action_dim)
        model = [linear1] + model + [linear2] + model + [linear3]
        self.model = nn.Sequential(*model) # Torch model
        self.target = copy.deepcopy(self.model)
        self.gamma = GAMMA
        self.bs = BATCH_SIZE
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = deque(maxlen=TRANSITIONS)
        self.best_score = -200


    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = random.sample(self.buffer, self.bs)
        return list(zip(*batch))

    def compute_loss(self, batch):
        state, action, next_state, reward, done = batch
        state = torch.tensor(np.array(state), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).view(-1)
        done = torch.tensor(np.array(done), dtype=torch.bool)
        action = torch.tensor(np.array(action)).view(-1, 1)

        with torch.no_grad():
            target_Q = self.target(next_state).max(dim=-1)[0]
            target_Q[done] = 0
            target_Q = reward + self.gamma * target_Q
        Q = self.model(state)
        Q = Q.gather(1, action.reshape(-1, 1))[:, 0]
        loss = F.mse_loss(Q, target_Q)
        return loss


    def train_step(self, batch):
        # Use batch to update DQN's network.
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        with torch.no_grad():
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32)
            action = self.model(state)
        return np.argmax(action.numpy())


    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self, score):
        if score > self.best_score:
            torch.save(self.model, "agent.pkl")
            self.best_score = score


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    env.seed(SEED)
    env.action_space.seed(SEED)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    env.seed(SEED)
    env.action_space.seed(SEED)
    dqn = DQN(state_dim=env.observation_space.shape[0],
               action_dim=env.action_space.n)
    eps = 0.3
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(
                f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            score = np.mean(rewards)
            dqn.save(score)
