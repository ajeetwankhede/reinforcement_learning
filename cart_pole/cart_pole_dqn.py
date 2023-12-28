# https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435

# Main concepts implemented for DQN:
# 1. Use Neural Network to approximate the Q values
# 2. Different nets for stablizing the training. Syncing every n steps
# 3. Batch train with experience replay

# Issues observed:
# 1. Hard to get the same results for every run
# 2. Very sensitive to random seed and other hyper parameters
# 3. Difficult to generalize the agent

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from collections import deque
import copy
from random import *

import gymnasium as gym

sns.set_theme()

class Params(NamedTuple):
    total_episodes: int
    layer_sizes: int
    sync_freq: int
    exp_replay_size: int
    batch_size: int
    learning_rate: float
    gamma: float
    seed: int
    action_size: int
    state_size: int
    savefig_folder: Path
    model_path: Path

params = Params(
    total_episodes=7000,
    layer_sizes=[],
    sync_freq = 5,
    exp_replay_size = 256,
    batch_size = 16,
    learning_rate=1e-3,
    gamma=0.95,
    seed=1423,
    action_size=2,
    state_size=4,
    savefig_folder=Path("cart_pole/"),
    model_path=Path('cart_pole/cart_pole_model'),
)
params = params._replace(layer_sizes=[params.state_size, 64, params.action_size])
print(params)
# Create the figure folder if it doesn't exists
params.savefig_folder.mkdir(parents=True, exist_ok=True)

class DQNlearning:
    def __init__(self, seed, layer_sizes, learning_rate, gamma, state_size, action_size, sync_freq, exp_replay_size, batch_size):
        torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = torch.tensor(gamma).float()
        self.exp_replay_size = exp_replay_size
        self.experience_replay = deque(maxlen=exp_replay_size)
        self.batch_size = batch_size
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.net = self.build_nn(layer_sizes)
        self.target_net = self.build_nn(layer_sizes)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.Tanh() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)
    
    def get_q_next(self, state):
        with torch.no_grad():
            Qp = self.target_net(torch.tensor(state))
        
        Q, _ = torch.max(Qp, axis=1)
        return Q
    
    def sample_from_experience(self):
        sample_exp = sample(self.experience_replay, self.batch_size)
        state = torch.tensor([exp[0] for exp in sample_exp]).float()
        action = torch.tensor([exp[1] for exp in sample_exp]).float()
        reward = torch.tensor([exp[2] for exp in sample_exp]).float()
        new_state = torch.tensor([exp[3] for exp in sample_exp]).float()
        return state, action, reward, new_state

    def train(self):
        """Update Q(s,a):= Q(s,a) + alpha [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        state, action, reward, new_state = self.sample_from_experience()

        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.net.state_dict())
            self.network_sync_counter = 0

        # predicted Q
        Qp = self.net(torch.tensor(state))
        q_pre, _ = torch.max(Qp, axis=1)

        # expected Q
        q_next = self.get_q_next(new_state)
        q_exp = reward + self.gamma * q_next

        loss = self.loss_fn(q_pre, q_exp)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()

    def choose_action(self, state, epsilon):
        explor_exploit_tradeoff = torch.rand(1,).item()

        if explor_exploit_tradeoff < epsilon:
            action = torch.randint(0,self.action_size,(1,))
        else:
            with torch.no_grad():
                Qp = self.net(torch.from_numpy(state).float())
            Q, action = torch.max(Qp, axis=0)
        return action.item()

def run_env():
    rewards = np.zeros((params.total_episodes, 1))
    steps = np.zeros((params.total_episodes, 1))
    epsilons = np.zeros((params.total_episodes, 1))
    lossess = np.zeros((params.total_episodes, 1))
    episodes = np.arange(params.total_episodes)

    index = 0
    epsilon = 1
    for episode in tqdm(
        episodes, desc=f"Run Episodes", leave=False
    ):
        state = env.reset()[0]
        step = 0
        done = False
        total_rewards = 0
        total_losses = 0

        while not done:
            action = agent.choose_action(
                state=state,
                epsilon=epsilon,
            )

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.experience_replay.append([state, action, reward, new_state])

            if index > 128:
                index = 0
                for j in range(4):
                    loss = agent.train()
                    total_losses += loss

            state = new_state
            total_rewards += reward
            step += 1
            index +=1

        if epsilon > 0.05:
            epsilon -= (1/5000)
        
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
    seed=params.seed,
    layer_sizes=params.layer_sizes,
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size,
    sync_freq=params.sync_freq,
    exp_replay_size=params.exp_replay_size,
    batch_size=params.batch_size,
)

mode = input("Train or Replay (T/R)?: ")
if (mode == "T"):
    env = gym.make(
    "CartPole-v1")
    rewards, epsilons, lossess = run_env()
    print(rewards[-10:])
    save = input("Save model (Y/N)?: ")
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
        plt.show()
    
else:
    env = gym.make(
    "CartPole-v1", 
    render_mode='human')
    agent.net.load_state_dict(torch.load(str(params.model_path)+"_dqn_baseline"))
    rewards = run_model()
    print(rewards)

env.close()