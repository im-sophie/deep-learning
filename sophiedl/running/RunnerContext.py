from ..domain import Repr

class RunnerContext(Repr):
    def __init__(self, environment, tensorboard_summary_writer):
        self.environment = environment
        self.tensorboard_summary_writer = tensorboard_summary_writer
        self.episode_index = 0
        self.step_index_episode = 0
        self.step_index_total = 0
        self.reward_sum = 0
        self.done = False
    
    def reset_episode(self):
        self.step_index_episode = 0
        self.reward_sum = 0
        self.done = False
    
    def add_scalar(self, name, value, timestamp):
        if self.tensorboard_summary_writer:
            self.tensorboard_summary_writer.add_scalar(name, value, timestamp)

    def cowabunga(self):
        raise Exception("it is")
