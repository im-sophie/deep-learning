# Typing
from typing import List, Optional

# Torch
from torch.utils.tensorboard import SummaryWriter

# TQDM
import tqdm # type: ignore

# Internal
from ..agent.AgentBase import AgentBase
from ..environment.EnvironmentBase import EnvironmentBase
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from .RunnerBase import RunnerBase
from .RunnerRLContext import RunnerRLContext

class RunnerRL(RunnerBase):
    environment: EnvironmentBase
    agent: AgentBase
    hyperparameter_set: HyperparameterSet
    context: Optional[RunnerRLContext]

    def __init__(
        self,
        environment: EnvironmentBase,
        agent: AgentBase,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str] = None,
        clear_tensorboard_output_dir: bool = True,
        moving_average_window: int = 30):
        super().__init__(
            hyperparameter_set,
            tensorboard_output_dir = tensorboard_output_dir,
            clear_tensorboard_output_dir = clear_tensorboard_output_dir,
            moving_average_window = moving_average_window
        )
        self.environment = environment
        self.agent = agent
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

    def on_run(self, tensorboard_summary_writer: Optional[SummaryWriter]) -> None:
        self.context = RunnerRLContext(self.environment, tensorboard_summary_writer)

        reward_sum_history: List[float] = []

        with tqdm.tqdm(total = self.hyperparameter_set["episode_count"], bar_format="{percentage:.1f}% {bar} Reward sum: {postfix[0]:8.3f}, moving average ({postfix[1]}): {postfix[2]:8.3f}, elapsed: {postfix[3]}/{postfix[4]}", postfix = [0, self.moving_average_window, 0, 0, self.hyperparameter_set["episode_count"]]) as t:
            for _ in range(self.hyperparameter_set["episode_count"]):
                self._run_episode()
                self.context.episode_index += 1
                t.postfix[0] = self.context.reward_sum
                t.postfix[2] = sum(reward_sum_history) / len(reward_sum_history) if len(reward_sum_history) > 0 else 0
                t.postfix[3] = self.context.episode_index
                reward_sum_history.append(self.context.reward_sum)
                while len(reward_sum_history) > self.moving_average_window:
                    del reward_sum_history[0]
                t.update()
