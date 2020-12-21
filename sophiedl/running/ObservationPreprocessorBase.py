# Standard library
import abc

# Typing
from typing import Any

# PyTorch
import torch as T

class ObservationPreprocessorBase(abc.ABC):
    @abc.abstractmethod
    def on_preprocess(self, observation: T.Tensor) -> T.Tensor:
        pass

    def __call__(self, observation: Any) -> T.Tensor:
        return self.on_preprocess(observation)