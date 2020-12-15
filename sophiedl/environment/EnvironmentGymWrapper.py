import torch as T
import gym # type: ignore
from typing import Any, Union, Iterable, cast
from .EnvironmentBase import EnvironmentBase
from ..domain import Shape

class EnvironmentGymWrapper(EnvironmentBase):
    gym_environment: gym.Env

    @staticmethod
    def _get_space_shape(space: Any) -> Iterable[int]:
        if isinstance(space, gym.spaces.Discrete):
            return (space.n,)
        elif isinstance(space, gym.spaces.Box):
            return cast(Iterable[int], space.shape)
        elif isinstance(space, gym.Space):
            raise TypeError("unexpected subtype of Space: {0}".format(type(space).__name__))
        else:
            raise TypeError("unexpected type for space: {0}".format(type(space).__name__))

    def __init__(self, gym_environment: gym.Env) -> None:
        super().__init__()

        self.gym_environment = gym_environment
    
    def _on_get_observation_space_shape(self) -> Iterable[int]:
        return EnvironmentGymWrapper._get_space_shape(
            self.gym_environment.observation_space
        )

    def _on_get_action_space_shape(self) -> Iterable[int]:
        return EnvironmentGymWrapper._get_space_shape(
            self.gym_environment.action_space
        )

    def reset(self) -> T.Tensor:
        return cast(T.Tensor, self.gym_environment.reset())

    def step(self, action: Union[T.Tensor, float]) -> T.Tensor:
        return cast(T.Tensor, self.gym_environment.step(action))
