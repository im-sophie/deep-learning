import operator
from functools import reduce

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import gym

import sophiedl as S

class GenericNetwork(nn.Module):
    def __init__(
        self, 
        learning_rate,
        observation_space_shape,
        action_space_shape_flat_size,
        layer_dimensions):
        assert len(layer_dimensions) > 0, "at least one layer dimension must be specified"

        super().__init__()

        self.learning_rate = learning_rate
        self.observation_space_shape = observation_space_shape
        self.action_space_shape_flat_size = action_space_shape_flat_size
        self.layer_dimensions = layer_dimensions

        self.input_layer = nn.Linear(*observation_space_shape, layer_dimensions[0])
        
        self.hidden_layers = []

        for i in range(len(layer_dimensions) - 1):
            hidden_layer = nn.Linear(layer_dimensions[i], layer_dimensions[i + 1])
            self.add_module("hidden_layers[{0}]".format(i), hidden_layer)
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(layer_dimensions[-1], self.action_space_shape_flat_size)

        self.optimizer = O.Adam(self.parameters(), lr = learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")
        self.to(self.device)
    
    def forward(self, observation):
        x = T.Tensor(observation).to(self.device)
        x = F.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        return self.output_layer(x)

class MemoryEntry(object):
    def __init__(self, action):
        self.action = action
        self.reward = None

class ActorCriticAgent(object):
    def __init__(self,
        actor_network,
        critic_network,
        gamma,
        writer = None):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.gamma = gamma
        self.memory = []
        self.writer = writer
    
    def act(self, observation):
        assert observation.shape == self.actor_network.observation_space_shape, "observation must match expected shape"
        assert observation.shape == self.critic_network.observation_space_shape, "observation must match expected shape"

        action_probabilities = T.distributions.Categorical(
            F.softmax(self.actor_network.forward(observation))
        )

        action = action_probabilities.sample()

        self.memory.append(
            MemoryEntry(
                action_probabilities.log_prob(action)
            )
        )

        return action.item()
    
    def reward(self, reward):
        assert len(self.memory) > 0, "cannot reward agent that has done no actions so far"
        assert self.memory[-1].reward is None, "agent has already been rewarded"
        self.memory[-1].reward = reward
    
    def learn(self, episode_index):
        self.actor_network.optimizer.zero_grad()
        self.critic_network.optimizer.zero_grad()

        critic_value = self.critic_network.forward()

if __name__ == "__main__":
    writer = SummaryWriter(
        log_dir = "./runs"
    )

    env = S.EnvironmentGymWrapper(
        gym.make("LunarLander-v2")
    )

    agent = PGOAgent(
        policy_network = PolicyNetwork(
            learning_rate = 0.001,
            observation_space_shape = env.observation_space_shape,
            action_space_shape_flat_size = env.action_space_shape.flat_size,
            layer_dimensions = [128, 128]
        ),
        gamma = 0.99,
        writer = writer
    )

    reward_sum = 0

    for i in range(2500):
        writer.add_scalar("Reward sum", reward_sum, i)

        print("Episode {0}, reward sum {1:.3f}".format(i, reward_sum))

        done = False
        observation = env.reset()
        reward_sum = 0

        while not done:
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.reward(reward)
            reward_sum += reward
        
        agent.learn(i)
