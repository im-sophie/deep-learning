# Standard library
import abc

# Typing
from typing import Optional

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from .RunnerContextBase import RunnerContextBase

class RunnerBase(abc.ABC):
    hyperparameter_set: HyperparameterSet
    context: Optional[RunnerContextBase]

    def __init__(self,
        hyperparameter_set: HyperparameterSet):
        self.hyperparameter_set = hyperparameter_set
        self.context = None
    
    @abc.abstractmethod
    def run(self) -> None:
        pass
