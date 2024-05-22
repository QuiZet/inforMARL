import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_tag_v3
from torch.distributions import Categorical

class MAPPO(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(MAPPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        value = self.critic(obs)
        probs = self.actor(obs)
        return value, probs

    def get_action(self, obs):
        _, probs = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_actions(self, obs, action):
        value, probs = self.forward(obs)
        dist = Categorical(probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return value, action_log_probs, dist_entropy

def make_env():
    return simple_tag_v3.parallel_env()

def compute_returns(next_value, rewards, masks, gamma=0.99): 
#Masks used to handle variable episode lengths (premature ends due to truncation or done)
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns