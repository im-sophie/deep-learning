import torch as T
from ..domain.Repr import Repr
from typing import Optional, Union

class Memory(Repr):
    observation_current: Optional[T.Tensor]
    observation_next: Optional[T.Tensor]
    done: bool
    action: Optional[Union[T.Tensor, float]]
    action_log_probabilities: Optional[T.Tensor]
    reward: Optional[float]
    
    def __init__(
        self,
        observation_current: Optional[T.Tensor] = None,
        observation_next: Optional[T.Tensor] = None,
        done: bool = False,
        action: Optional[Union[T.Tensor, float]] = None,
        action_log_probabilities: Optional[T.Tensor] = None,
        reward: Optional[float] = None) -> None:
        self.observation_current = observation_current
        self.observation_next = observation_next
        self.done = done
        self.action = action
        self.action_log_probabilities = action_log_probabilities
        self.reward = reward

if __name__ == "__main__":
    print(Memory(observation_current = T.as_tensor([1]), reward = 5))
