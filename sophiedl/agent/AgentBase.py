import abc
import torch as T
from ..memory.MemoryBuffer import MemoryBuffer
from ..running.RunnerRLContext import RunnerRLContext
from ..HyperparameterSet import HyperparameterSet
from typing import Union, Tuple, Optional, cast

class AgentBase(abc.ABC):
    hyperparameter_set: HyperparameterSet
    memory_buffer: MemoryBuffer

    def __init__(
        self,
        hyperparameter_set: HyperparameterSet):
        self.hyperparameter_set = hyperparameter_set
        self.memory_buffer = MemoryBuffer()
    
    @abc.abstractmethod
    def on_act(self,
        runner_context: RunnerRLContext,
        observation: T.Tensor) -> Tuple[Union[T.Tensor, float], Optional[T.Tensor]]:
        pass

    @abc.abstractmethod
    def on_should_learn(self,
        runner_context: RunnerRLContext) -> bool:
        pass

    @abc.abstractmethod
    def on_learn(self,
        runner_context: RunnerRLContext) -> None:
        pass

    def act(self,
        runner_context: RunnerRLContext,
        observation: T.Tensor) -> Union[T.Tensor, float]:
        action, action_log_probabilities = self.on_act(runner_context, observation)

        self.memory_buffer.add_observation_current(observation)
        self.memory_buffer.add_action(action)

        if action_log_probabilities is not None:
            self.memory_buffer.add_action_log_probabilities(
                action_log_probabilities
            )
        
        return action
    
    def reward(self,
        reward: float,
        observation: T.Tensor,
        done: bool) -> None:
        self.memory_buffer.add_observation_next(observation)
        self.memory_buffer.add_reward(reward)
        self.memory_buffer.add_done(done)
        self.memory_buffer.push()

    def learn(self,
        runner_context: RunnerRLContext) -> None:
        if self.on_should_learn(runner_context):
            self.on_learn(runner_context)

    def clear_memory(self) -> None:
        del self.memory_buffer[:]
