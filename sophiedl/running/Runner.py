import os
import glob
import shutil

from torch.utils.tensorboard import SummaryWriter

from .RunnerContext import RunnerContext

class Runner(object):
    def __init__(
        self,
        environment,
        agent,
        episode_count,
        hyperparameter_set,
        tensorboard_output_dir = None,
        clear_tensorboard_output_dir = True):
        self.environment = environment
        self.agent = agent
        self.episode_count = episode_count
        self.hyperparameter_set = hyperparameter_set
        self.tensorboard_output_dir = tensorboard_output_dir
        self.clear_tensorboard_output_dir = clear_tensorboard_output_dir
        self.tensorboard_summary_writer = None
        self.context = RunnerContext()

    def _run_episode(self):
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
            
            self.context.step_index += 1

        print("Episode {0}, reward sum {1:.3f}".format(self.context.episode_index, self.context.reward_sum))

        if self.tensorboard_summary_writer:
            self.tensorboard_summary_writer.add_scalar("Reward Sum", self.context.reward_sum, self.context.episode_index)

    def run(self):
        if self.tensorboard_output_dir:
            if self.clear_tensorboard_output_dir:
                for i in glob.glob(os.path.join(os.path.abspath(self.tensorboard_output_dir), "*")):
                    if os.path.isdir(i):
                        shutil.rmtree(i)
                    else:
                        os.remove(i)

            self.tensorboard_summary_writer = SummaryWriter(
                log_dir = os.path.join(os.path.abspath(self.tensorboard_output_dir), str(self.hyperparameter_set))
            )

        for _ in range(self.episode_count):
            self._run_episode()
            self.context.episode_index += 1
