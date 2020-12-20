# Typing
from typing import Optional

# PyTorch
from torch.utils.tensorboard import SummaryWriter

# Internal
from ..domain.Repr import Repr
from ..environment.EnvironmentBase import EnvironmentBase
from .RunnerContextBase import RunnerContextBase

class RunnerRLContext(RunnerContextBase, Repr):
    environment: EnvironmentBase
    tensorboard_summary_writer: Optional[SummaryWriter]
    episode_index: int
    step_index_episode: int
    step_index_total: int
    reward_sum: float
    done: bool

    def __init__(self, environment: EnvironmentBase, tensorboard_summary_writer: Optional[SummaryWriter]) -> None:
        self.environment = environment
        self.tensorboard_summary_writer = tensorboard_summary_writer
        self.episode_index = 0
        self.step_index_episode = 0
        self.step_index_total = 0
        self.reward_sum = 0
        self.done = False
    
    def reset_episode(self) -> None:
        self.step_index_episode = 0
        self.reward_sum = 0
        self.done = False
    
    def add_scalar(self, name: str, value: float, timestamp: float) -> None:
        if self.tensorboard_summary_writer:
            self.tensorboard_summary_writer.add_scalar(name, value, timestamp) # type: ignore

    def cowabunga(self) -> None:
        raise Exception("it is")
