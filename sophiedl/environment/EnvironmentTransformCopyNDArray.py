# Typing
from typing import Any, Iterable, Union

# NumPy
import numpy as np # type: ignore

# Internal
from .EnvironmentBase import EnvironmentBase, EnvironmentStepResult
from .EnvironmentTransformBase import EnvironmentTransformBase
from ..domain.Shape import Shape

class EnvironmentTransformCopyNDArray(EnvironmentTransformBase):
    def __init__(self, pre_environment: EnvironmentBase) -> None:
        super().__init__(pre_environment, pre_environment.observation_type, pre_environment.action_type)
    
    def on_get_observation_space_shape(self) -> Union[Shape, Iterable[int]]:
        return self.pre_environment.observation_space_shape

    def on_get_action_space_shape(self) -> Union[Shape, Iterable[int]]:
        return self.pre_environment.action_space_shape

    def on_reset(self) -> Any:
        return self.pre_environment.reset().copy()
    
    def on_step(self, action: Any) -> EnvironmentStepResult:
        pre_result = self.pre_environment.step(action)
        return EnvironmentStepResult(
            pre_result.observation.copy(),
            pre_result.reward,
            pre_result.done,
            pre_result.info
        )
