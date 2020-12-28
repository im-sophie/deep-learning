# Typing
from typing import Any, Iterable, TypeVar, Union

# PyTorch
import torch as T

# Internal
from ..domain.Shape import Shape
from .EnvironmentBase import EnvironmentBase, EnvironmentStepResult
from .EnvironmentTransformBase import EnvironmentTransformBase

class EnvironmentTransformSkipFrames(EnvironmentTransformBase):
    skip_count: int

    def __init__(self, pre_environment: EnvironmentBase, skip_count: int) -> None:
        super().__init__(pre_environment, pre_environment.observation_type, pre_environment.action_type)
        self.skip_count = skip_count
    
    def on_get_observation_space_shape(self) -> Union[Shape, Iterable[int]]:
        return self.pre_environment.observation_space_shape

    def on_get_action_space_shape(self) -> Union[Shape, Iterable[int]]:
        return self.pre_environment.action_space_shape

    def on_reset(self) -> Any:
        return self.pre_environment.reset()
    
    def on_step(self, action: Any) -> EnvironmentStepResult:
        assert self.skip_count > 1

        result = None

        for i in range(self.skip_count - 1):
            result = self.pre_environment.step(action)

            assert result is not None
            
            if result.done:
                break
        
        assert result is not None

        if result.done:
            return result
        else:
            return self.pre_environment.step(action)
