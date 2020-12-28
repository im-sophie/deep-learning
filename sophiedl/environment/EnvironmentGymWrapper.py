# Typing
from typing import Any, cast, Generic, get_args, Iterable, TypeVar, Union

# NumPy
import numpy as np # type: ignore

# PyTorch
import torch as T

# Gym
import gym # type: ignore

# Internal
from ..domain.Shape import Shape
from .EnvironmentBase import EnvironmentBase, EnvironmentStepResult

class EnvironmentGymWrapper(EnvironmentBase):
    gym_environment: gym.Env

    def __init__(self, gym_environment: gym.Env, observation_type: type, action_type: type) -> None:
        super().__init__(observation_type, action_type)

        self.gym_environment = gym_environment
    
    def on_get_observation_space_shape(self) -> Union[Shape, Iterable[int]]:
        return Shape.get_shape(
            self.gym_environment.observation_space
        )

    def on_get_action_space_shape(self) -> Union[Shape, Iterable[int]]:
        return Shape.get_shape(
            self.gym_environment.action_space
        )

    def on_reset(self) -> Any:
        return self.gym_environment.reset()

    def on_step(self, action: Any) -> EnvironmentStepResult:
        observation, reward, done, info = self.gym_environment.step(action)

        if isinstance(reward, np.generic):
            reward = reward.item()
        
        if isinstance(done, np.generic):
            done = done.item()

        return EnvironmentStepResult(
            observation,
            reward,
            done,
            info
        )
