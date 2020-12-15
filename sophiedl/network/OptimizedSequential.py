# Typing
from typing import Callable, cast, Iterable, Optional

# PyTorch
import torch as T
import torch.nn as nn
import torch.optim as O

# Internal
from .OptimizedModule import OptimizedModule

class OptimizedSequential(nn.Sequential, OptimizedModule):
    learning_rate: float
    optimizer: O.Optimizer
    device: T.device

    @staticmethod
    def optimizer_factory_adam(parameters: Iterable[T.Tensor], learning_rate: float) -> O.Adam:
        return O.Adam(parameters, lr = learning_rate)

    def __init__(self,
        *args: nn.Module,
        optimizer_factory: Optional[Callable[[Iterable[T.Tensor], float], O.Adam]] = None,
        learning_rate: float = 0):
        super().__init__(
            *args
        )

        assert optimizer_factory is not None

        self.learning_rate = learning_rate
        self.optimizer = optimizer_factory(self.parameters(), learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")
        self.to(self.device)
