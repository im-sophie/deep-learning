# PyTorch
import torch as T
import torch.nn as nn
from torchvision.datasets import MNIST # type: ignore
from torchvision.transforms import ToTensor # type: ignore

# Internal
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...network.CNNCell import CNNCell
from ...network.OptimizedModule import OptimizedModule
from ...network.OptimizedSequential import OptimizedSequential
from ..base.RunnerNetworkTorchDataLoaderFactoryBase import RunnerNetworkTorchDataLoaderFactoryBase

class RunnerNetworkTorchDataLoaderFactoryMNIST(RunnerNetworkTorchDataLoaderFactoryBase):
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("batch_size_training", 32)
        hyperparameter_set.add("batch_size_testing", 32)
        hyperparameter_set.add("epoch_count", 25)
        hyperparameter_set.add("learning_rate", 0.001)
        return hyperparameter_set

    def on_create_network(self, hyperparameter_set: HyperparameterSet) -> OptimizedModule:
        return OptimizedSequential(
            CNNCell(
                in_channels = 1,
                out_channels = 32
            ),
            CNNCell(
                in_channels = 32,
                out_channels = 32
            ),
            CNNCell(
                in_channels = 32,
                out_channels = 32
            ),
            nn.MaxPool2d(
                kernel_size = 2
            ),
            CNNCell(
                in_channels = 32,
                out_channels = 64
            ),
            CNNCell(
                in_channels = 64,
                out_channels = 64
            ),
            CNNCell(
                in_channels = 64,
                out_channels = 64
            ),
            nn.MaxPool2d(
                kernel_size = 2
            ),
            nn.Flatten(),
            nn.Linear(
                in_features = 256,
                out_features = 10
            ),
            optimizer_factory = OptimizedSequential.optimizer_factory_adam,
            learning_rate = hyperparameter_set["learning_rate"]
        )

    def on_create_dataset_training(self) -> T.utils.data.Dataset[T.Tensor]:
        return MNIST( # type: ignore
            "datasets/mnist",
            train = True,
            download = True,
            transform = ToTensor()
        )
    
    def on_create_dataset_testing(self) -> T.utils.data.Dataset[T.Tensor]:
        return MNIST( # type: ignore
            "datasets/mnist",
            train = False,
            download = True,
            transform = ToTensor()
        )
