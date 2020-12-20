# Standard library
import abc

# Typing
from typing import Optional

# PyTorch
import torch as T

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..network.OptimizedModule import OptimizedModule
from ..running.RunnerBase import RunnerBase
from ..running.RunnerClassifierTorchDataLoader import RunnerClassifierTorchDataLoader
from .RunnerFactoryBase import RunnerFactoryBase

class RunnerClassifierTorchDataLoaderFactoryBase(RunnerFactoryBase):
    @abc.abstractmethod
    def on_create_network(self, hyperparameter_set: HyperparameterSet) -> OptimizedModule:
        pass

    @abc.abstractmethod
    def on_create_dataset_training(self) -> T.utils.data.Dataset[T.Tensor]:
        pass

    @abc.abstractmethod
    def on_create_dataset_testing(self) -> T.utils.data.Dataset[T.Tensor]:
        pass

    def on_create_runner(self,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str]) -> RunnerBase:
        return RunnerClassifierTorchDataLoader(
            hyperparameter_set,
            self.on_create_network(hyperparameter_set),
            T.utils.data.DataLoader(
                self.on_create_dataset_training(),
                batch_size = hyperparameter_set["batch_size_training"],
                shuffle = True,
                num_workers = 4
            ),
            T.utils.data.DataLoader(
                self.on_create_dataset_testing(),
                batch_size = hyperparameter_set["batch_size_testing"],
                shuffle = True,
                num_workers = 4
            )
        )
