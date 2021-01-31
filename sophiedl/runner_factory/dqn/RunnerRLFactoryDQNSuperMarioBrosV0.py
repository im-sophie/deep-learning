# Typing
from typing import cast, Optional, Union

# NumPy
import numpy as np # type: ignore

# PyTorch
import torch as T
import torch.nn as nn
import torchvision.transforms as transforms # type: ignore

# Gym Super Mario Bros
from nes_py.wrappers import JoypadSpace # type: ignore
import gym_super_mario_bros # type: ignore
import gym_super_mario_bros.actions # type: ignore

# Internal
from ...agent.AgentBase import AgentBase
from ...agent.AgentDQN import AgentDQN
from ...agent.EpsilonGreedyStrategy import EpsilonGreedyStrategy
from ...domain.Shape import Shape
from ...environment.EnvironmentBase import EnvironmentBase
from ...environment.EnvironmentGymWrapper import EnvironmentGymWrapper
from ...environment.EnvironmentTransformCopyNDArray import EnvironmentTransformCopyNDArray
from ...environment.EnvironmentTransformPyTorchTransforms import EnvironmentTransformPyTorchTransforms
from ...environment.EnvironmentTransformSkipFrames import EnvironmentTransformSkipFrames
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...network.CNNCell import CNNCell
from ...network.OptimizedSequential import OptimizedSequential
from ..base.RunnerRLFactoryBase import RunnerRLFactoryBase

class RunnerRLFactoryDQNSuperMarioBrosV0(RunnerRLFactoryBase):
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.9)
        hyperparameter_set.add("learning_rate", 0.00025)
        hyperparameter_set.add("epsilon_start", 0.9)
        hyperparameter_set.add("epsilon_end", 0.05)
        hyperparameter_set.add("epsilon_decay", 0.005)
        hyperparameter_set.add("memory_batch_size", 32)
        hyperparameter_set.add("target_update_interval", 2)
        hyperparameter_set.add("episode_count", 100)
        hyperparameter_set.add("episode_max_length", 1000)
        return hyperparameter_set

    def on_create_environment(self) -> EnvironmentBase:
        return EnvironmentTransformPyTorchTransforms(
            EnvironmentTransformCopyNDArray(
                EnvironmentTransformSkipFrames(
                    EnvironmentGymWrapper(
                        JoypadSpace(
                            gym_super_mario_bros.make(
                                "SuperMarioBros-v0"
                            ),
                            gym_super_mario_bros.actions.RIGHT_ONLY
                        ),
                        np.ndarray,
                        int
                    ),
                    skip_count = 4
                )
            ),
            T.Tensor,
            Shape((1, 84, 89)),
            int,
            transforms.ToPILImage(),
            transforms.Resize(84),
            transforms.Grayscale(),
            transforms.ToTensor()#,
            # transforms.Lambda(
            #     lambda x: x / 255.
            # )
        )
    
    def _create_network(self, environment: EnvironmentBase, hyperparameter_set: HyperparameterSet) -> OptimizedSequential:
        return OptimizedSequential(
            # CNNCell(
            #     in_channels = 1,
            #     out_channels = 8,
            #     kernel_size = 9
            # ),
            # CNNCell(
            #     in_channels = 8,
            #     out_channels = 8,
            #     kernel_size = 9
            # ),
            # CNNCell(
            #     in_channels = 8,
            #     out_channels = 8,
            #     kernel_size = 9
            # ),
            # nn.MaxPool2d(
            #     kernel_size = 5
            # ),
            # CNNCell(
            #     in_channels = 8,
            #     out_channels = 16,
            #     kernel_size = 3
            # ),
            # CNNCell(
            #     in_channels = 16,
            #     out_channels = 16,
            #     kernel_size = 3
            # ),
            # CNNCell(
            #     in_channels = 16,
            #     out_channels = 16,
            #     kernel_size = 3
            # ),
            # nn.MaxPool2d(
            #     kernel_size = 2
            # ),
            # nn.Flatten(),
            # nn.Linear(
            #     in_features = 144,
            #     out_features = 100
            # ),
            # nn.ReLU(),
            # nn.Linear(
            #     in_features = 100,
            #     out_features = environment.action_space_shape.flat_size
            # ),
            CNNCell(
                in_channels = 1,
                out_channels = 32,
                kernel_size = 8,
                stride = 4
            ),
            CNNCell(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 4,
                stride = 2
            ),
            CNNCell(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride = 1
            ),
            nn.Flatten(),
            nn.Linear(
                in_features = 3136,
                out_features = 512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features = 512,
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
