# Gym
import gym # type: ignore

# PyTorch
import torch.nn as nn

# Internal
from ...agent.AgentBase import AgentBase
from ...agent.AgentDiscreteActorCritic import AgentDiscreteActorCritic
from ...environment.EnvironmentBase import EnvironmentBase
from ...environment.EnvironmentGymWrapper import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...network.OptimizedSequential import OptimizedSequential
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryDiscreteActorCriticCartPoleV0(AgentEnvironmentFactoryBase):
    def __init__(self) -> None:
        super().__init__(2500)

    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("learning_rate_actor", 1e-5)
        hyperparameter_set.add("learning_rate_critic", 5e-4)
        hyperparameter_set.add("gamma", 0.99)
        return hyperparameter_set
    
    def on_create_environment(self) -> EnvironmentBase:
        return EnvironmentGymWrapper(
            gym.make("CartPole-v0")
        )
    
    def on_create_agent(self, environment: EnvironmentBase, hyperparameter_set: HyperparameterSet) -> AgentBase:
        return AgentDiscreteActorCritic(
            actor_network = OptimizedSequential(
                nn.Linear(
                    *environment.observation_space_shape,
                    32
                ), # type: ignore
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
                ), # type: ignore
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
