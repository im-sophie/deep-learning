# Typing
from typing import Any, Iterable, Union

# PyTorch
import torch as T
import torchvision.transforms as transforms # type: ignore

# Internal
from .EnvironmentBase import EnvironmentBase, EnvironmentStepResult
from .EnvironmentTransformBase import EnvironmentTransformBase
from ..domain.Shape import Shape

class EnvironmentTransformPyTorchTransforms(EnvironmentTransformBase):
    transforms: transforms.Compose
    observation_space_shape_post: Shape

    def __init__(self, pre_environment: EnvironmentBase, observation_type: type, observation_space_shape_post: Shape, action_type: type, *transforms_: Any) -> None:
        super().__init__(pre_environment, observation_type, action_type)
        self.observation_space_shape_post = observation_space_shape_post
        self.transforms = transforms.Compose(list(transforms_))
    
    def on_get_observation_space_shape(self) -> Union[Shape, Iterable[int]]:
        return self.observation_space_shape_post

    def on_get_action_space_shape(self) -> Union[Shape, Iterable[int]]:
        return self.pre_environment.action_space_shape

    def on_reset(self) -> Any:
        return self.transforms(self.pre_environment.reset())
    
    def on_step(self, action: Any) -> EnvironmentStepResult:
        pre_result = self.pre_environment.step(action)
        return EnvironmentStepResult(
            self.transforms(pre_result.observation),
            pre_result.reward,
            pre_result.done,
            pre_result.info
        )
