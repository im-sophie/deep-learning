# Typing
from typing import Generic, Iterable, TypeVar, Union

# Internal
from ..domain.Shape import Shape
from .EnvironmentBase import EnvironmentBase

class EnvironmentTransformBase(EnvironmentBase):
    pre_environment: EnvironmentBase

    def __init__(self, pre_environment: EnvironmentBase, observation_type: type, action_type: type) -> None:
        super().__init__(observation_type, action_type)
        self.pre_environment = pre_environment
    
    def __str__(self) -> str:
        return "{0}\n{1}".format(
            str(self.pre_environment),
            super().__str__()
        )
