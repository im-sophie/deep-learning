# Gym
import gym # type: ignore

# PyTorch
import torch.nn as nn

# Internal
from ...agent.AgentBase import AgentBase
from ...agent.AgentDQN import AgentDQN
from ...agent.EpsilonGreedyStrategy import EpsilonGreedyStrategy
from ...environment.EnvironmentBase import EnvironmentBase
from ...environment.EnvironmentGymWrapper import EnvironmentGymWrapper
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...network.OptimizedSequential import OptimizedSequential
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryDQNCartPoleV0(AgentEnvironmentFactoryBase):
    def __init__(self) -> None:
        super().__init__(500)

    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.99)
        hyperparameter_set.add("learning_rate", 1e-3)
        hyperparameter_set.add("epsilon_start", 0.9)
        hyperparameter_set.add("epsilon_end", 0.05)
        hyperparameter_set.add("epsilon_decay", 0.005)
        hyperparameter_set.add("memory_batch_size", 100)
        hyperparameter_set.add("target_update_interval", 2)
        return hyperparameter_set
    
    def on_create_environment(self) -> EnvironmentBase:
        return EnvironmentGymWrapper(
            gym.make("CartPole-v0")
        )
    
    def on_create_agent(self, environment: EnvironmentBase, hyperparameter_set: HyperparameterSet) -> AgentBase:
        return AgentDQN(
            policy_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    20
                ), # type: ignore
                nn.ReLU(),
                nn.Linear(
                    20,
                    10
                ),
                nn.ReLU(),
                nn.Linear(
                    10,
                    environment.action_space_shape.flat_size
                ),
                optimizer_factory = OptimizedSequential.optimizer_factory_adam,
                learning_rate = hyperparameter_set["learning_rate"]
            ),
            target_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    20
                ), # type: ignore
                nn.ReLU(),
                nn.Linear(
                    20,
                    10
                ),
                nn.ReLU(),
                nn.Linear(
                    10,
                    environment.action_space_shape.flat_size
                ),
                optimizer_factory = OptimizedSequential.optimizer_factory_adam,
                learning_rate = hyperparameter_set["learning_rate"]
            ),
            epsilon_greedy_strategy = EpsilonGreedyStrategy(
                start = hyperparameter_set["epsilon_start"],
                end = hyperparameter_set["epsilon_end"],
                decay = hyperparameter_set["epsilon_decay"]
            ),
            hyperparameter_set = hyperparameter_set
        )
