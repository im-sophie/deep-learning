# Standard library
import math
import random

# Internal
from ..running.RunnerRLContext import RunnerRLContext

class EpsilonGreedyStrategy(object):
    start: float
    end: float
    decay: float

    def __init__(self, start: float, end: float, decay: float) -> None:
        self.start = start
        self.end = end
        self.decay = decay
    
    def should_explore(self, runner_context: RunnerRLContext) -> bool:
        epsilon = self.end + (self.start - self.end) * math.exp(-runner_context.step_index_total * self.decay)

        runner_context.add_scalar("Epsilon", epsilon, runner_context.step_index_total)

        x = random.random()

        runner_context.add_scalar("Epsilon Random Value", x, runner_context.step_index_total)

        return x <= epsilon
