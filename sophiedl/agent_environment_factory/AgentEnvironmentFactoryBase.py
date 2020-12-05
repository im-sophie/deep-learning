import abc
from collections import namedtuple

from ..running import Runner

AgentEnvironmentFactoryResult = namedtuple(
    "AgentEnvironmentFactoryResult",
    [
        "environment",
        "agent",
        "hyperparameter_set"
    ]
)

class AgentEnvironmentFactoryBase(abc.ABC):
    @abc.abstractmethod
    def on_create_environment(self):
        pass

    @abc.abstractmethod
    def on_create_default_hyperparameter_set(self):
        pass

    @abc.abstractmethod
    def on_create_agent(self, environment, hyperparameter_set):
        pass

    def create_agent(self, hyperparameter_set = None):
        environment = self.on_create_environment()

        if not hyperparameter_set:
            hyperparameter_set = self.on_create_default_hyperparameter_set()
        
        return AgentEnvironmentFactoryResult(
            environment,
            self.on_create_agent(environment, hyperparameter_set),
            hyperparameter_set
        )

    def create_runner(self, episode_count, hyperparameter_set = None, tensorboard_output_dir = None):
        result = self.create_agent(
            hyperparameter_set = hyperparameter_set
        )

        return Runner(
            result.environment,
            result.agent,
            episode_count = episode_count,
            hyperparameter_set = result.hyperparameter_set,
            tensorboard_output_dir = tensorboard_output_dir
        )
