# Typing
from typing import cast, Union

# NumPy
import numpy as np # type: ignore

# PyTorch
import torch as T
import torch.nn as nn

# Gym
import gym # type: ignore

# Internal
from ...agent.AgentBase import AgentBase
from ...agent.AgentPGO import AgentPGO
from ...environment.EnvironmentBase import EnvironmentBase
from ...environment.EnvironmentGymWrapper import EnvironmentGymWrapper
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...network.OptimizedSequential import OptimizedSequential
from ..base.RunnerRLFactoryBase import RunnerRLFactoryBase

class RunnerRLFactoryPGOLunarLanderV2(RunnerRLFactoryBase):
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.99)
        hyperparameter_set.add("learning_rate", 0.001)
        hyperparameter_set.add("episode_count", 2500)
        return hyperparameter_set
    
    def on_create_environment(self) -> EnvironmentBase:
        return EnvironmentGymWrapper(
            gym.make("LunarLander-v2"),
            np.ndarray,
            int
        )
    
    def on_create_agent(self, environment: EnvironmentBase, hyperparameter_set: HyperparameterSet) -> AgentBase:
        return AgentPGO(
            policy_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    128
                ), # type: ignore
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
