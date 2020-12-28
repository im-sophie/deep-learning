# Future
from __future__ import annotations

# Standard library
from collections.abc import Iterable
from functools import reduce
import copy as C
import operator

# Typing
from typing import Any, List

# NumPy
import numpy as np # type: ignore

# PyTorch
import torch as T

# Gym
import gym # type: ignore

class Shape(List[int]):
    @staticmethod
    def to_shape(value: Any, copy: bool = True) -> Shape:
        if isinstance(value, Shape):
            if copy:
                return C.deepcopy(value)
            else:
                return value
        else:
            try:
                return Shape(value)
            except TypeError:
                raise TypeError("cannot convert non-iterable value to Shape")
    
    @staticmethod
    def get_shape(value: Any) -> Shape:
        if isinstance(value, np.ndarray):
            return Shape.to_shape(value.shape)
        elif isinstance(value, T.Tensor):
            return Shape.to_shape(value.shape)
        elif isinstance(value, gym.spaces.Discrete):
            return Shape.to_shape((value.n,))
        elif isinstance(value, gym.spaces.Box):
            return Shape.to_shape(value.shape)
        elif isinstance(value, gym.Space):
            raise TypeError("unexpected subtype of Space: {0}".format(type(value).__name__))
        else:
            raise TypeError("unexpected type for value: {0}".format(type(value).__name__))

    @property
    def flat_size(self) -> int:
        return reduce(operator.mul, self, 1)
