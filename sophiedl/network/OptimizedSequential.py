import torch as T
import torch.nn as nn
import torch.optim as O

class OptimizedSequential(nn.Sequential):
    @staticmethod
    def optimizer_factory_adam(parameters, learning_rate):
        return O.Adam(parameters, lr = learning_rate)

    def __init__(self, *args, optimizer_factory = None, learning_rate = 0):
        super().__init__(
            *args
        )

        self.learning_rate = learning_rate
        self.optimizer = optimizer_factory(self.parameters(), learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")
        self.to(self.device)
