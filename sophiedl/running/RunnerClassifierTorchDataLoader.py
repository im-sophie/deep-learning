# PyTorch
import torch as T
import torch.nn as nn

# Typing
from typing import cast, Sequence, Tuple

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..network.OptimizedModule import OptimizedModule
from .RunnerClassifierBase import RunnerClassifierBase

class RunnerClassifierTorchDataLoader(RunnerClassifierBase):
    data_loader_training: T.utils.data.DataLoader[T.Tensor]
    data_loader_testing: T.utils.data.DataLoader[T.Tensor]

    def __init__(self,
        hyperparameter_set: HyperparameterSet,
        network: OptimizedModule,
        data_loader_training: T.utils.data.DataLoader[T.Tensor],
        data_loader_testing: T.utils.data.DataLoader[T.Tensor]) -> None:
        super().__init__(
            hyperparameter_set,
            network
        )
        self.data_loader_training = data_loader_training
        self.data_loader_testing = data_loader_testing
    
    def on_get_training_data(self) -> Sequence[Tuple[T.Tensor, T.Tensor]]:
        return cast(Sequence[Tuple[T.Tensor, T.Tensor]], self.data_loader_training)
    
    def on_get_testing_data(self) -> Sequence[Tuple[T.Tensor, T.Tensor]]:
        return cast(Sequence[Tuple[T.Tensor, T.Tensor]], self.data_loader_testing)
    
    def on_get_loss(self, predictions: T.Tensor, labels: T.Tensor) -> T.Tensor:
        return nn.CrossEntropyLoss()(predictions, labels) # type: ignore
