import pybullet_envs
import gym
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


SEED = 129
random = RandomState(MT19937(SeedSequence(SEED)))
torch.manual_seed(SEED)


GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 128
ENV_NAME = "AntBulletEnv-v0"
NOISE_CLIP = 0.5
POLICY_NOISE = 0.2
NOISE = 0.2
TRANSITIONS = 4000000


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=ACTOR_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=ACTOR_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = deque(maxlen=200000)
        self.best_reward = 0

    def critic_loss(self, state, action, Q_T):
        Q1 = self.critic_1(state, action)
        Q2 = self.critic_2(state, action)
        return F.mse_loss(Q1, Q_T), F.mse_loss(Q2, Q_T)

    def update(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in
                           range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            with torch.no_grad():
                noise = NOISE * torch.clip(torch.randn_like(self.target_actor(next_state)), -NOISE_CLIP,
                                           NOISE_CLIP)
                next_action = torch.clip(self.target_actor(next_state) + noise, -1, 1)

                Q_target = reward + (
                            GAMMA * (1 - done) * (torch.minimum(self.target_critic_1(next_state, next_action),
                                                                self.target_critic_2(next_state,
                                                                                     next_action))))

            # Update critic
            critic_1_loss, critic_2_loss = self.critic_loss(state, action, Q_target)

            self.critic_1_optim.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optim.step()

            self.critic_2_optim.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optim.step()

            # Update actor
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save(self.actor.model, "agent.pkl")

def evaluate_policy(env, agent, episodes=5):
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
    env = make(ENV_NAME)
    env.seed(SEED)
    env.action_space.seed(SEED)

    test_env = make(ENV_NAME)
    test_env.seed(SEED)
    test_env.action_space.seed(SEED)

    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    eps = 1

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + 0.2 * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            td3.save(np.mean(rewards))
