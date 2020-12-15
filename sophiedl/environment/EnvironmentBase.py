# Standard library
from functools import reduce
import abc
import operator

# Typing
from typing import cast, Iterable, Union

# PyTorch
import torch as T

# Internal
from ..domain.Shape import Shape

class EnvironmentBase(abc.ABC):
    @staticmethod
    def _get_shape_flat_size(shape: Shape) -> int:
        return reduce(operator.mul, cast(Iterable[int], shape), 1)

    @abc.abstractmethod
    def _on_get_observation_space_shape(self) -> Union[Shape, Iterable[int]]:
        pass

    @abc.abstractmethod
    def _on_get_action_space_shape(self) -> Union[Shape, Iterable[int]]:
        pass
    
    # TODO: Cleanup these next two methods to be less Gym-specific.

    @abc.abstractmethod
    def reset(self) -> T.Tensor:
        pass

    @abc.abstractmethod
    def step(self, action: Union[T.Tensor, float]) -> T.Tensor:
        pass

    @property
    def observation_space_shape(self) -> Shape:
        return Shape.to_shape(
            self._on_get_observation_space_shape()
        )
    
    @property
    def action_space_shape(self) -> Shape:
        return Shape.to_shape(
            self._on_get_action_space_shape()
        )
