import torch as T
from .Memory import Memory
from typing import List, Union, overload

class MemoryBuffer(object):
    memories: List[Memory]
    memory_staging: Memory

    def __init__(self) -> None:
        self.memories = []
        self.memory_staging = Memory()
    
    def add_observation_current(self, value: T.Tensor) -> None:
        if self.memory_staging.observation_current:
            raise Exception("current observation already added for current memory")
    
        self.memory_staging.observation_current = value
    
    def add_observation_next(self, value: T.Tensor) -> None:
        if self.memory_staging.observation_next:
            raise Exception("next observation already added for next memory")
    
        self.memory_staging.observation_next = value

    def add_done(self, value: bool) -> None:
        if self.memory_staging.done:
            raise Exception("done flag already added for next memory")
    
        self.memory_staging.done = value

    def add_action(self, value: Union[T.Tensor, float]) -> None:
        if self.memory_staging.action:
            raise Exception("action already added for current memory")
    
        self.memory_staging.action = value
    
    def add_action_log_probabilities(self, value: T.Tensor) -> None:
        if self.memory_staging.action_log_probabilities:
            raise Exception("action log probabilities already added for current memory")
    
        self.memory_staging.action_log_probabilities = value
    
    def add_reward(self, value: float) -> None:
        if self.memory_staging.reward:
            raise Exception("reward already added for current memory")
    
        self.memory_staging.reward = value

    def push(self) -> None:
        self.memories.append(self.memory_staging)
        self.memory_staging = Memory()

    def __len__(self) -> int:
        return len(self.memories)
    
    @overload
    def __getitem__(self, key: int) -> Memory: ...

    @overload
    def __getitem__(self, key: slice) -> List[Memory]: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[Memory, List[Memory]]:
        return self.memories[key]
    
    def __delitem__(self, key: Union[int, slice]) -> None:
        del self.memories[key]
