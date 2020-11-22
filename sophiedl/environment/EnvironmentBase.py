from functools import reduce
import operator
import abc

from ..domain import Shape

class EnvironmentBase(abc.ABC):
    @staticmethod
    def _get_shape_flat_size(shape):
        return reduce(operator.mul, shape, 1)

    def __init__(self):
        pass

    @abc.abstractmethod
    def _on_get_observation_space_shape(self):
        return Shape()

    @abc.abstractmethod
    def _on_get_action_space_shape(self):
        return Shape()
    
    # TODO: Cleanup these next two methods to be less Gym-specific.

    @abc.abstractmethod
    def reset(self):
        return None

    @abc.abstractmethod
    def step(self, action):
        return None

    @property
    def observation_space_shape(self):
        return Shape.to_shape(
            self._on_get_observation_space_shape()
        )
    
    @property
    def action_space_shape(self):
        return Shape.to_shape(
            self._on_get_action_space_shape()
        )
