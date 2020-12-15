import torch as T
import torch.nn as nn
import torch.optim as O

class OptimizedModule(nn.Module):
    learning_rate: float
    optimizer: O.Optimizer
    device: T.device

    def forward(self, _: T.Tensor) -> T.Tensor: ...
