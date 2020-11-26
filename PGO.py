import operator
from functools import reduce

import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import gym

import sophiedl as S

class PGOAgent(S.AgentBase):
    def __init__(self,
        policy_network,
        memory_cleanup_schedule,
        hyperparameter_set,
        tensorboard_summary_writer = None):
        super().__init__(memory_cleanup_schedule, hyperparameter_set, tensorboard_summary_writer)
        self.policy_network = policy_network
    
    def on_act(self, observation):
        assert observation.shape == self.policy_network.observation_space_shape, "observation must match expected shape"

        action_probabilities = T.distributions.Categorical(
            F.softmax(self.policy_network.forward(observation))
        )

        action = action_probabilities.sample()

        return action, action_probabilities.log_prob(action)
    
    def on_learn(self):
        self.policy_network.optimizer.zero_grad()

        discounted_future_rewards = T.zeros(len(self.memory_buffer), dtype = T.float, device = self.policy_network.device)

        for i in range(len(self.memory_buffer)):
            discounted_future_reward_sum = 0
            discount_factor = 1

            for j in range(i, len(self.memory_buffer)):
                if self.memory_buffer[j].reward:
                    discounted_future_reward_sum += self.memory_buffer[j].reward * discount_factor
                discount_factor *= self.hyperparameter_set["gamma"]
            
            discounted_future_rewards[i] = discounted_future_reward_sum

        discounted_future_rewards -= discounted_future_rewards.mean()

        std = discounted_future_rewards.std().item()
        if std != 0:
            discounted_future_rewards /= std

        loss = 0
        for discounted_future_reward, memory in zip(discounted_future_rewards, self.memory_buffer):
            loss -= discounted_future_reward * memory.action_log_probabilities

        loss.backward()
        self.policy_network.optimizer.step()

if __name__ == "__main__":
    env = S.EnvironmentGymWrapper(
        gym.make("LunarLander-v2")
    )

    hyperparameter_set = S.HyperparameterSet()
    hyperparameter_set.add("gamma", 0.99)
    hyperparameter_set.add("learning_rate", 0.001)
    hyperparameter_set.add("layer_dimensions", [128, 128])

    S.Runner(
        env,
        PGOAgent(
            policy_network = S.ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = env.observation_space_shape,
                output_feature_count = env.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            memory_cleanup_schedule = S.MemoryCleanupScheduleMonteCarlo(),
            hyperparameter_set = hyperparameter_set
        ),
        episode_count = 2500,
        hyperparameter_set = hyperparameter_set,
        tensorboard_output_dir = "./runs"
    ).run()
