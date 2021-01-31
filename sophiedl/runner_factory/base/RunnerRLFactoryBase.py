# Standard library
import abc

# Typing
from typing import Optional, Union

# PyTorch
import torch as T

# Internal
from ...agent.AgentBase import AgentBase
from ...environment.EnvironmentBase import EnvironmentBase
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...running.RunnerRL import RunnerRL
from .RunnerFactoryBase import RunnerFactoryBase

class RunnerRLFactoryBase(RunnerFactoryBase):
    @abc.abstractmethod
    def on_create_environment(self) -> EnvironmentBase:
        pass

    @abc.abstractmethod
    def on_create_agent(
        self,
        environment: EnvironmentBase,
        hyperparameter_set: HyperparameterSet) -> AgentBase:
        pass

    def on_create_runner(
        self,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str]) -> RunnerRL:
        environment = self.on_create_environment()

        return RunnerRL(
            environment,
            self.on_create_agent(environment, hyperparameter_set),
            hyperparameter_set = hyperparameter_set,
            tensorboard_output_dir = tensorboard_output_dir
        )
