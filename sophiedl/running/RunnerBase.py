# Standard library
import abc
import glob
import os
import shutil

# Typing
from typing import Optional

# Torch
from torch.utils.tensorboard import SummaryWriter

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from .RunnerContextBase import RunnerContextBase

class RunnerBase(abc.ABC):
    hyperparameter_set: HyperparameterSet
    context: Optional[RunnerContextBase]
    tensorboard_output_dir: Optional[str]
    clear_tensorboard_output_dir: bool
    moving_average_window: int

    def __init__(
        self,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str] = None,
        clear_tensorboard_output_dir: bool = True,
        moving_average_window: int = 30):
        self.hyperparameter_set = hyperparameter_set
        self.context = None
        self.tensorboard_output_dir = tensorboard_output_dir
        self.clear_tensorboard_output_dir = clear_tensorboard_output_dir
        self.moving_average_window = moving_average_window
    
    @abc.abstractmethod
    def on_run(self, tensorboard_summary_writer: Optional[SummaryWriter]) -> None:
        pass

    def run(self) -> None:
        tensorboard_summary_writer: Optional[SummaryWriter] = None

        if self.tensorboard_output_dir:
            if self.clear_tensorboard_output_dir:
                for i in glob.glob(os.path.join(os.path.abspath(self.tensorboard_output_dir), "*")):
                    if os.path.isdir(i):
                        shutil.rmtree(i)
                    else:
                        os.remove(i)

            tensorboard_summary_writer = SummaryWriter(
                log_dir = os.path.join(os.path.abspath(self.tensorboard_output_dir), str(self.hyperparameter_set))
            ) # type: ignore
        
        self.on_run(tensorboard_summary_writer)
