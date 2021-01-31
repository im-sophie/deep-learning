# Standard library
import abc

# Typing
from typing import Optional

# Internal
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...running.RunnerBase import RunnerBase

class RunnerFactoryBase(abc.ABC):
    @abc.abstractmethod
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        pass

    def create_default_hyperparameter_set(self) -> HyperparameterSet:
        return self.on_create_default_hyperparameter_set()

    @abc.abstractmethod
    def on_create_runner(
        self,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str]) -> RunnerBase:
        pass

    def create_runner(
        self,
        hyperparameter_set: Optional[HyperparameterSet] = None,
        tensorboard_output_dir: Optional[str] = None) -> RunnerBase:
        if hyperparameter_set is None:
            hyperparameter_set = self.on_create_default_hyperparameter_set()

        return self.on_create_runner(
            hyperparameter_set,
            tensorboard_output_dir = tensorboard_output_dir
        )
