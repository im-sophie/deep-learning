import torch.nn as nn

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gym_super_mario_bros.actions

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentDQN, EpsilonGreedyStrategy
from ...network import ParameterizedLinearNetwork, OptimizedSequential
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class RunnerRLFactoryDQNSuperMarioBrosV0(AgentEnvironmentFactoryBase):
    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.99)
        hyperparameter_set.add("learning_rate", 1e-3)
        # hyperparameter_set.add("layer_dimensions", [20, 10])
        hyperparameter_set.add("epsilon_start", 0.9)
        hyperparameter_set.add("epsilon_end", 0.05)
        hyperparameter_set.add("epsilon_decay", 0.005)
        hyperparameter_set.add("memory_batch_size", 100)
        hyperparameter_set.add("target_update_interval", 2)
        return hyperparameter_set

    def on_create_environment(self):
        return EnvironmentGymWrapper(
            JoypadSpace(
                gym_super_mario_bros.make(
                    "SuperMarioBros-v0"
                ),
                gym_super_mario_bros.actions.SIMPLE_MOVEMENT
            )
        )
    
    def _create_network(self):
        return OptimizedSequential(
            nn.
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentDQN(
            policy_network = ParameterizedLinearNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = environment.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            target_network = ParameterizedLinearNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = environment.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            epsilon_greedy_strategy = EpsilonGreedyStrategy(
                start = hyperparameter_set["epsilon_start"],
                end = hyperparameter_set["epsilon_end"],
                decay = hyperparameter_set["epsilon_decay"]
            ),
            hyperparameter_set = hyperparameter_set
        )
