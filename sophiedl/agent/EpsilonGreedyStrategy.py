import random
import math

class EpsilonGreedyStrategy(object):
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    def should_explore(self, runner_context):
        epsilon = self.end + (self.start - self.end) * math.exp(-runner_context.step_index_total * self.decay)

        runner_context.add_scalar("Epsilon", epsilon, runner_context.step_index_total)

        x = random.random()

        runner_context.add_scalar("Epsilon Random Value", x, runner_context.step_index_total)

        return x <= epsilon
