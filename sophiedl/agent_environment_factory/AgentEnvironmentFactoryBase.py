# Standard library
import abc
from collections import namedtuple

# Typing
from typing import Optional

# Internal
from ..agent.AgentBase import AgentBase
from ..environment.EnvironmentBase import EnvironmentBase
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..running.RunnerRL import RunnerRL

AgentEnvironmentFactoryResult = namedtuple(
    "AgentEnvironmentFactoryResult",
    [
        "environment",
        "agent",
        "hyperparameter_set"
    ]
)

class AgentEnvironmentFactoryBase(abc.ABC):
    def __init__(self, default_episode_count: int) -> None:
        self.default_episode_count = default_episode_count

    @abc.abstractmethod
    def on_create_environment(self) -> EnvironmentBase:
        pass

    @abc.abstractmethod
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        pass

    @abc.abstractmethod
    def on_create_agent(self, environment: EnvironmentBase, hyperparameter_set: HyperparameterSet) -> AgentBase:
        pass

    def create_agent(self, hyperparameter_set: Optional[HyperparameterSet] = None) -> AgentEnvironmentFactoryResult:
        environment = self.on_create_environment()

        if hyperparameter_set is None:
            hyperparameter_set = self.on_create_default_hyperparameter_set()
        
        return AgentEnvironmentFactoryResult(
            environment,
            self.on_create_agent(environment, hyperparameter_set),
            hyperparameter_set
        )

    def create_runner(self, episode_count: int, hyperparameter_set: Optional[HyperparameterSet] = None, tensorboard_output_dir: Optional[str] = None) -> RunnerRL:
        result = self.create_agent(
            hyperparameter_set = hyperparameter_set
        )

        return RunnerRL(
            result.environment,
            result.agent,
            episode_count = episode_count,
            hyperparameter_set = result.hyperparameter_set,
            tensorboard_output_dir = tensorboard_output_dir
        )
