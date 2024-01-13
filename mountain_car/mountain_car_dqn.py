# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435

# Main concepts implemented for DQN:
# 1. Use Neural Network to approximate the Q values
# 2. Different nets for stablizing the training. Syncing every n steps
# 3. Batch train with experience replay


import matplotlib.pyplot as plt
import numpy as np
import math
import random

import torch
import torch.nn as nn

from pathlib import Path
from typing import NamedTuple

import seaborn as sns
from tqdm import tqdm

from collections import deque

import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme()

class Params(NamedTuple):
    total_episodes: int
    layer_sizes: int
    exp_replay_size: int
    batch_size: int
    learning_rate: float
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    tau: float
    action_size: int
    state_size: int
    savefig_folder: Path
    model_path: Path

params = Params(
    total_episodes=1000,
    layer_sizes=[],
    exp_replay_size = 10000,
    batch_size = 128,
    learning_rate=1e-4,
    gamma=0.99,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=1000,
    tau=5e-2,
    action_size=3,
    state_size=2,
    savefig_folder=Path("mountain_car/dqn_results"),
    model_path=Path('mountain_car/mountain_car_model'),
)
params = params._replace(layer_sizes=[params.state_size, 128, 128, params.action_size])
print(params)
# Create the figure folder if it doesn't exists
# params.savefig_folder.mkdir(parents=True, exist_ok=True)

class DQNlearning:
    def __init__(self, tau, layer_sizes, learning_rate, gamma, state_size, action_size, exp_replay_size, batch_size):
        
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exp_replay_size = exp_replay_size
        self.experience_replay = deque(maxlen=exp_replay_size)
        self.batch_size = batch_size
        
        self.net = self.build_nn(layer_sizes).to(device)
        self.target_net = self.build_nn(layer_sizes).to(device)

        self.target_net.load_state_dict(self.net.state_dict())
        self.tau = tau

        # self.loss_fn = torch.nn.SmoothL1Loss() # With Huber loss the model is not learning
        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, amsgrad=True)
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)
    
    def choose_action(self, state, epsilon):
        explor_exploit_tradeoff = random.random()

        if explor_exploit_tradeoff < epsilon:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.net(state).max(1).indices.view(1,1)
    
    def sample_from_experience(self):
        sample_exp = random.sample(self.experience_replay, self.batch_size)
        
        state = torch.cat([exp[0] for exp in sample_exp])
        action = torch.cat([exp[1] for exp in sample_exp])
        reward = torch.cat([exp[2] for exp in sample_exp])
        new_state = torch.cat([exp[3] for exp in sample_exp])
        terminated = torch.cat([exp[4] for exp in sample_exp])
        return state, action, reward, new_state, terminated

    def train(self):
        """Update Q_expected(s,a) = [R(s,a) + gamma * max Q(s',a')] if not terminated state
                  Q_expected(s,a) = 0 if terminal state"""
        if len(self.experience_replay) < self.batch_size:
            # Not enough experiences collected
            return
        
        state, action, reward, new_state, terminated = self.sample_from_experience()

        # predicted Q
        q_pre = self.net(state).gather(1, action) # Q for the (state, action)

        # expected Q: 0  or R + gamma*max(Q(s',a'))
        with torch.no_grad():
            q_exp = torch.logical_not(terminated)*(reward + self.gamma * self.target_net(new_state).max(1).values)

        loss = self.loss_fn(q_pre, q_exp.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

def run_env():
    rewards = np.zeros((params.total_episodes, 1))
    epsilons = np.zeros((params.total_episodes, 1))
    lossess = np.zeros((params.total_episodes, 1))
    episodes = np.arange(params.total_episodes)

    step = 0
    for episode in tqdm(
        episodes, desc=f"Run Episodes", leave=False
    ):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_rewards = 0
        total_losses = 0

        while not done:
            epsilon = params.eps_end + (params.eps_start-params.eps_end)*math.exp(-1.0*step/params.eps_decay)

            action = agent.choose_action(
                state=state,
                epsilon=epsilon,
            )

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            terminated = torch.tensor([terminated], device=device, dtype=torch.bool)
            done = terminated or truncated
            agent.experience_replay.append([state, action, reward, next_state, terminated])

            loss = agent.train()
            if loss is not(None):
                total_losses += loss
                # Soft Update of the target network
                # theta' = tau * theta + ( 1 - tau ) * theta'
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*agent.tau + \
                                                 target_net_state_dict[key]*(1-agent.tau)
                agent.target_net.load_state_dict(target_net_state_dict)

            state = next_state
            total_rewards += reward.item()
            step += 1
        
        rewards[episode] = total_rewards
        epsilons[episode] = epsilon
        lossess[episode] = total_losses/step
        
    return rewards, epsilons, lossess

def run_model():
    state = env.reset()[0]
    done = False
    total_rewards = 0
    epsilon = 0.0
    while not done:
        action = agent.choose_action(
            state=state,
            epsilon=epsilon,
        )
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = new_state
        total_rewards += reward
        
    return total_rewards

agent = DQNlearning(
    tau=params.tau,
    layer_sizes=params.layer_sizes,
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size,
    exp_replay_size=params.exp_replay_size,
    batch_size=params.batch_size,
)

mode = input("Train or Replay (T/R)?: ")
if (mode == "T"):
    env = gym.make(
    "MountainCar-v0")
    rewards, epsilons, lossess = run_env()
    print(rewards[-10:])
    save = "Y"#input("Save model (Y/N)?: ")
    if save == "Y":
        torch.save(agent.net.state_dict(), params.model_path)
        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(rewards)
        ax[0].set_title("rewards")
        ax[1].plot(epsilons)
        ax[1].set_title("epsilons")
        ax[2].plot(lossess)
        ax[2].set_title("lossess")
        fig.savefig(params.savefig_folder, bbox_inches="tight")
        # plt.show()
    
else:
    env = gym.make(
    "MountainCar-v0", 
    render_mode='human')
    agent.net.load_state_dict(torch.load(str(params.model_path)+"_dqn_baseline"))
    rewards = run_model()
    print(rewards)

env.close()