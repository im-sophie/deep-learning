import torch.nn as nn

import gym

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentPGO
from ...network import ParameterizedLinearNetwork, OptimizedSequential
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryPGOLunarLanderV2(AgentEnvironmentFactoryBase):
    def __init__(self):
        super().__init__(2500)

    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.99)
        hyperparameter_set.add("learning_rate", 0.001)
        return hyperparameter_set
    
    def on_create_environment(self):
        return EnvironmentGymWrapper(
            gym.make("LunarLander-v2")
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentPGO(
            policy_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    128
                ),
                nn.ReLU(),
                nn.Linear(
                    128,
                    128
                ),
                nn.ReLU(),
                nn.Linear(
                    128,
                    environment.action_space_shape.flat_size
                ),
                optimizer_factory = OptimizedSequential.optimizer_factory_adam,
                learning_rate = hyperparameter_set["learning_rate"]
            ),
            hyperparameter_set = hyperparameter_set
        )
