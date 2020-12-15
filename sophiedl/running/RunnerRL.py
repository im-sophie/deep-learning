import os
import glob
import shutil
from typing import Optional, List

from torch.utils.tensorboard import SummaryWriter

import tqdm # type: ignore

from .RunnerRLContext import RunnerRLContext
from ..environment.EnvironmentBase import EnvironmentBase
from ..agent.AgentBase import AgentBase
from ..HyperparameterSet import HyperparameterSet

class RunnerRL(object):
    environment: EnvironmentBase
    agent: AgentBase
    episode_count: int
    hyperparameter_set: HyperparameterSet
    tensorboard_output_dir: Optional[str]
    clear_tensorboard_output_dir: bool
    moving_average_window: int
    context: Optional[RunnerRLContext]

    def __init__(
        self,
        environment: EnvironmentBase,
        agent: AgentBase,
        episode_count: int,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str] = None,
        clear_tensorboard_output_dir: bool = True,
        moving_average_window: int = 30):
        self.environment = environment
        self.agent = agent
        self.episode_count = episode_count
        self.hyperparameter_set = hyperparameter_set
        self.tensorboard_output_dir = tensorboard_output_dir
        self.clear_tensorboard_output_dir = clear_tensorboard_output_dir
        self.moving_average_window = moving_average_window
        self.context = None

    def _run_episode(self) -> None:
        assert self.context is not None
        self.context.reset_episode()

        done = False
        observation = self.environment.reset()

        while not done:
            action = self.agent.act(self.context, observation)
            
            observation, reward, done, _ = self.environment.step(action)
            
            self.agent.reward(reward, observation, done)
            
            self.context.done = done
            self.context.reward_sum += reward
            
            self.agent.learn(self.context)
            
            self.context.step_index_episode += 1
            self.context.step_index_total += 1

        self.context.add_scalar("Reward Sum", self.context.reward_sum, self.context.episode_index)
        self.context.add_scalar("Episode Length", self.context.step_index_episode, self.context.episode_index)

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
        
        self.context = RunnerRLContext(self.environment, tensorboard_summary_writer)

        reward_sum_history: List[float] = []

        with tqdm.tqdm(total = self.episode_count, bar_format="{percentage:.1f}% {bar} Reward sum: {postfix[0]:8.3f}, moving average ({postfix[1]}): {postfix[2]:8.3f}, elapsed: {postfix[3]}/{postfix[4]}", postfix = [0, self.moving_average_window, 0, 0, self.episode_count]) as t:
            for _ in range(self.episode_count):
                self._run_episode()
                self.context.episode_index += 1
                t.postfix[0] = self.context.reward_sum
                t.postfix[2] = sum(reward_sum_history) / len(reward_sum_history) if len(reward_sum_history) > 0 else 0
                t.postfix[3] = self.context.episode_index
                reward_sum_history.append(self.context.reward_sum)
                while len(reward_sum_history) > self.moving_average_window:
                    del reward_sum_history[0]
                t.update()
