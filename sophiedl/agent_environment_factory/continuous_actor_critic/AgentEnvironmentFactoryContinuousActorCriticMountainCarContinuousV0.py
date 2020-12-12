import gym

import torch.nn as nn

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentContinuousActorCritic
from ...network import ParameterizedLinearNetwork, OptimizedSequential
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryContinuousActorCriticMountainCarContinuousV0(AgentEnvironmentFactoryBase):
    def __init__(self):
        super().__init__(2500)

    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("learning_rate_actor", 5e-6)
        hyperparameter_set.add("learning_rate_critic", 1e-5)
        # hyperparameter_set.add("layer_dimensions_actor", [256, 256])
        # hyperparameter_set.add("layer_dimensions_critic", [256, 256])
        hyperparameter_set.add("gamma", 0.99)
        return hyperparameter_set
    
    def on_create_environment(self):
        return EnvironmentGymWrapper(
            gym.make("MountainCarContinuous-v0")
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentContinuousActorCritic(
            actor_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    256
                ),
                nn.ReLU(),
                nn.Linear(
                    256,
                    256
                ),
                nn.ReLU(),
                nn.Linear(
                    256,
                    2 # for mu and sigma
                ),
                optimizer_factory = OptimizedSequential.optimizer_factory_adam,
                learning_rate = hyperparameter_set["learning_rate_actor"]
            ),
            critic_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    256
                ),
                nn.ReLU(),
                nn.Linear(
                    256,
                    256
                ),
                nn.ReLU(),
                nn.Linear(
                    256,
                    1
                ),
                optimizer_factory = OptimizedSequential.optimizer_factory_adam,
                learning_rate = hyperparameter_set["learning_rate_critic"]
            ),
            hyperparameter_set = hyperparameter_set
        )
