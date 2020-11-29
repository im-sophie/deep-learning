import operator
from functools import reduce

import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import gym

import sophiedl as S

class ActorCriticAgent(S.AgentBase):
    def __init__(self,
        actor_network,
        critic_network,
        hyperparameter_set,
        tensorboard_summary_writer = None):
        super().__init__(
            hyperparameter_set,
            tensorboard_summary_writer
        )

        self.actor_network = actor_network
        self.critic_network = critic_network
    
    def on_act(self, observation):
        assert observation.shape == self.actor_network.observation_space_shape, "observation must match expected shape"

        mu, sigma = self.actor_network.forward(observation)

        action_probabilities = T.distributions.Normal(
            mu,
            T.exp(sigma)
        )

        action = action_probabilities.sample(
            sample_shape = (1,)
        )

        return T.tanh(action).cpu().numpy(), action_probabilities.log_prob(action)

    def on_should_learn(self, runner_context):
        return len(self.memory_buffer) > 0

    def on_learn(self):
        assert self.memory_buffer[-1].observation_current.shape == self.critic_network.observation_space_shape, "observation must match expected shape"
        assert self.memory_buffer[-1].observation_next.shape == self.critic_network.observation_space_shape, "observation must match expected shape"

        self.actor_network.optimizer.zero_grad()
        self.critic_network.optimizer.zero_grad()

        critic_value = self.critic_network.forward(self.memory_buffer[-1].observation_current)
        critic_value_next = self.critic_network.forward(self.memory_buffer[-1].observation_next)

        delta = (
            self.memory_buffer[-1].reward
            + (0 if self.memory_buffer[-1].done else self.hyperparameter_set["gamma"] * critic_value_next)
            - critic_value
        )

        actor_loss = -self.memory_buffer[-1].action_log_probabilities * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()

        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()

if __name__ == "__main__":
    env = S.EnvironmentGymWrapper(
        gym.make("MountainCarContinuous-v0")
    )

    hyperparameter_set = S.HyperparameterSet()
    hyperparameter_set.add("learning_rate_actor", 5e-6)
    hyperparameter_set.add("learning_rate_critic", 1e-5)
    hyperparameter_set.add("layer_dimensions_actor", [256, 256])
    hyperparameter_set.add("layer_dimensions_critic", [256, 256])
    hyperparameter_set.add("gamma", 0.99)

    S.Runner(
        env,
        ActorCriticAgent(
            actor_network = S.ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate_actor"],
                observation_space_shape = env.observation_space_shape,
                output_feature_count = 2,
                layer_dimensions = hyperparameter_set["layer_dimensions_actor"]
            ),
            critic_network = S.ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate_critic"],
                observation_space_shape = env.observation_space_shape,
                output_feature_count = 1,
                layer_dimensions = hyperparameter_set["layer_dimensions_critic"]
            ),
            hyperparameter_set = hyperparameter_set
        ),
        episode_count = 100,
        hyperparameter_set = hyperparameter_set,
        tensorboard_output_dir = "./runs/ActorCritic"
    ).run()
