# Standard library
from collections import namedtuple
from functools import reduce
import abc
import operator

# Typing
from typing import Any, cast, Iterable, List, TypeVar, Union

# NumPy
import numpy as np # type: ignore

# PyTorch
import torch as T

# Internal
from ..domain.Shape import Shape

EnvironmentStepResult = namedtuple(
    "EnvironmentStepResult",
    [
        "observation",
        "reward",
        "done",
        "info"
    ]
)

class EnvironmentBase(abc.ABC):
    observation_type: type
    action_type: type

    def __init__(self, observation_type: type, action_type: type):
        self.observation_type = observation_type
        self.action_type = action_type

    def __str__(self) -> str:
        return "{0}:\n  Observation type: {1}\n  Observation space shape: {2}\n  Action type: {3}\n  Action space shape: {4} (flat size: {5})".format(
            type(self).__name__,
            self.observation_type,
            self.observation_space_shape,
            self.action_type,
            self.action_space_shape,
            self.action_space_shape.flat_size
        )

    @abc.abstractmethod
    def on_get_observation_space_shape(self) -> Union[Shape, Iterable[int]]:
        pass

    @abc.abstractmethod
    def on_get_action_space_shape(self) -> Union[Shape, Iterable[int]]:
        pass
    
    # TODO: Cleanup these next two methods to be less Gym-specific.

    @abc.abstractmethod
    def on_reset(self) -> Any:
        pass

    @abc.abstractmethod
    def on_step(self, action: Any) -> EnvironmentStepResult:
        pass

    def reset(self) -> Any:
        observation = self.on_reset()

        if not isinstance(observation, self.observation_type):
            raise TypeError(
                "expected observation to be of type {0}, not {1}".format(
                    self.observation_type.__name__,
                    type(observation).__name__
                )
            )
        
        if Shape.get_shape(observation) != self.observation_space_shape:
            raise Exception(
                "expected observation to have shape {0}, not {1}".format(
                    self.observation_space_shape,
                    Shape.get_shape(observation)
                )
            )
        
        return observation

    def step(self, action: Any) -> EnvironmentStepResult:
        if not isinstance(action, self.action_type):
            raise TypeError(
                "expected action to be of type {0}, not {1}".format(
                    self.action_type.__name__,
                    type(action).__name__
                )
            )

        observation, reward, done, info = self.on_step(action)

        if isinstance(observation, tuple):
            observation = np.array(observation)

        if not isinstance(observation, self.observation_type):
            raise TypeError(
                "expected observation to be of type {0}, not {1}".format(
                    self.observation_type.__name__,
                    type(observation).__name__
                )
            )
        
        if Shape.get_shape(observation) != self.observation_space_shape:
            raise Exception(
                "expected observation to have shape {0}, not {1}".format(
                    self.observation_space_shape,
                    Shape.get_shape(observation)
                )
            )
        
        return EnvironmentStepResult(
            observation,
            reward,
            done,
            info
        )

    @property
    def observation_space_shape(self) -> Shape:
        return Shape.to_shape(
            self.on_get_observation_space_shape()
        )

    @property
    def action_space_shape(self) -> Shape:
        return Shape.to_shape(
            self.on_get_action_space_shape()
        )
