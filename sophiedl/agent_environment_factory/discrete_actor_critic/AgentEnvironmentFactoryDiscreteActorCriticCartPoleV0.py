import torch.nn as nn

import gym

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentDiscreteActorCritic
from ...network import ParameterizedLinearNetwork, OptimizedSequential
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryDiscreteActorCriticCartPoleV0(AgentEnvironmentFactoryBase):
    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("learning_rate_actor", 1e-5)
        hyperparameter_set.add("learning_rate_critic", 5e-4)
        hyperparameter_set.add("gamma", 0.99)
        return hyperparameter_set
    
    def on_create_environment(self):
        return EnvironmentGymWrapper(
            gym.make("CartPole-v0")
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentDiscreteActorCritic(
            actor_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    32
                ),
                nn.ReLU(),
                nn.Linear(
                    32,
                    32
                ),
                nn.ReLU(),
                nn.Linear(
                    32,
                    environment.action_space_shape.flat_size
                ),
                optimizer_factory = OptimizedSequential.optimizer_factory_adam,
                learning_rate = hyperparameter_set["learning_rate_actor"]
            ),
            critic_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    32
                ),
                nn.ReLU(),
                nn.Linear(
                    32,
                    32
                ),
                nn.ReLU(),
                nn.Linear(
                    32,
                    1
                ),
                optimizer_factory = OptimizedSequential.optimizer_factory_adam,
                learning_rate = hyperparameter_set["learning_rate_critic"]
            ),
            hyperparameter_set = hyperparameter_set
        )
