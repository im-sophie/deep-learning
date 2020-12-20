# Typing
from typing import Iterable

# PyTorch
import torch as T

# Internal
from ..network.OptimizedModule import OptimizedModule
from .RunnerBase import RunnerBase

class RunnerClassifier(RunnerBase):
    def __init__(self,
        network: OptimizedModule,
        training_data: Iterable[T.Tensor],
        testing_data: Iterable[T.Tensor]) -> None:
        self.network = network
        self.training_data = training_data
        self.testing_data = testing_data

    def run(self) -> None:
        raise NotImplementedError()
