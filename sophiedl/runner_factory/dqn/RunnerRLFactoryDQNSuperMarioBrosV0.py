# Typing
from typing import Optional

# PyTorch
import torch.nn as nn

# Gym Super Mario Bros
from nes_py.wrappers import JoypadSpace # type: ignore
import gym_super_mario_bros # type: ignore
import gym_super_mario_bros.actions # type: ignore

# Internal
from ...agent.AgentBase import AgentBase
from ...agent.AgentDQN import AgentDQN
from ...agent.EpsilonGreedyStrategy import EpsilonGreedyStrategy
from ...environment.EnvironmentBase import EnvironmentBase
from ...environment.EnvironmentGymWrapper import EnvironmentGymWrapper
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...network.CNNCell import CNNCell
from ...network.OptimizedSequential import OptimizedSequential
from ...running.ObservationPreprocessorBase import ObservationPreprocessorBase
from ...running.ObservationPreprocessorSuperMarioBrosV0 import ObservationPreprocessorSuperMarioBrosV0
from ..RunnerRLFactoryBase import RunnerRLFactoryBase

class RunnerRLFactoryDQNSuperMarioBrosV0(RunnerRLFactoryBase):
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.99)
        hyperparameter_set.add("learning_rate", 1e-3)
        hyperparameter_set.add("epsilon_start", 0.9)
        hyperparameter_set.add("epsilon_end", 0.05)
        hyperparameter_set.add("epsilon_decay", 0.005)
        hyperparameter_set.add("memory_batch_size", 10)
        hyperparameter_set.add("target_update_interval", 2)
        hyperparameter_set.add("episode_count", 100)
        hyperparameter_set.add("episode_max_length", 5000)
        return hyperparameter_set

    def on_create_environment(self) -> EnvironmentBase:
        return EnvironmentGymWrapper(
            JoypadSpace(
                gym_super_mario_bros.make(
                    "SuperMarioBros-v0"
                ),
                gym_super_mario_bros.actions.SIMPLE_MOVEMENT
            )
        )
    
    def _create_network(self, environment: EnvironmentBase, hyperparameter_set: HyperparameterSet) -> OptimizedSequential:
        return OptimizedSequential(
            CNNCell(
                in_channels = 1,
                out_channels = 8,
                kernel_size = 9
            ),
            CNNCell(
                in_channels = 8,
                out_channels = 8,
                kernel_size = 9
            ),
            CNNCell(
                in_channels = 8,
                out_channels = 8,
                kernel_size = 9
            ),
            nn.MaxPool2d(
                kernel_size = 5
            ),
            CNNCell(
                in_channels = 8,
                out_channels = 16,
                kernel_size = 3
            ),
            CNNCell(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 3
            ),
            CNNCell(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 3
            ),
            nn.MaxPool2d(
                kernel_size = 2
            ),
            nn.Flatten(),
            nn.Linear(
                in_features = 8096,
                out_features = 100
            ),
            nn.ReLU(),
            nn.Linear(
                in_features = 100,
                out_features = environment.action_space_shape.flat_size
            ),
            optimizer_factory = OptimizedSequential.optimizer_factory_adam,
            learning_rate = hyperparameter_set["learning_rate"]
        )

    def on_create_agent(self, environment: EnvironmentBase, hyperparameter_set: HyperparameterSet) -> AgentBase:
        return AgentDQN(
            policy_network = self._create_network(environment, hyperparameter_set),
            target_network = self._create_network(environment, hyperparameter_set),
            epsilon_greedy_strategy = EpsilonGreedyStrategy(
                start = hyperparameter_set["epsilon_start"],
                end = hyperparameter_set["epsilon_end"],
                decay = hyperparameter_set["epsilon_decay"]
            ),
            hyperparameter_set = hyperparameter_set
        )

    def on_create_observation_preprocessor(self) -> Optional[ObservationPreprocessorBase]:
        return ObservationPreprocessorSuperMarioBrosV0()
