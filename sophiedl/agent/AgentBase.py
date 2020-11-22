import abc

from ..memory import MemoryBuffer

class AgentBase(abc.ABC):
    def __init__(
        self,
        hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        self.memory_buffer = MemoryBuffer()
