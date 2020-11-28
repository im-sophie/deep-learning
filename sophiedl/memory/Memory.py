from ..domain import Repr

class Memory(Repr):
    def __init__(
        self,
        observation_current = None,
        observation_next = None,
        done = False,
        action = None,
        action_log_probabilities = None,
        reward = None):
        self.observation_current = observation_current
        self.observation_next = observation_next
        self.done = done
        self.action = action
        self.action_log_probabilities = action_log_probabilities
        self.reward = reward

if __name__ == "__main__":
    print(Memory(observation_current = 1, reward = 5))
