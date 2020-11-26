from ..domain import Repr

class RunnerContext(Repr):
    def __init__(self):
        self.episode_index = 0
        self.step_index = 0
        self.reward_sum = 0
    
    def reset_episode(self):
        self.step_index = 0
        self.reward_sum = 0
