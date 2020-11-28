from .Memory import Memory

class MemoryBuffer(object):
    def __init__(self):
        self.memories = []
        self.memory_staging = Memory()
    
    def add_observation_current(self, value):
        if self.memory_staging.observation_current:
            raise Exception("current observation already added for current memory")
    
        self.memory_staging.observation_current = value
    
    def add_observation_next(self, value):
        if self.memory_staging.observation_next:
            raise Exception("next observation already added for next memory")
    
        self.memory_staging.observation_next = value

    def add_done(self, value):
        if self.memory_staging.done:
            raise Exception("done flag already added for next memory")
    
        self.memory_staging.done = value

    def add_action(self, value):
        if self.memory_staging.action:
            raise Exception("action already added for current memory")
    
        self.memory_staging.action = value
    
    def add_action_log_probabilities(self, value):
        if self.memory_staging.action_log_probabilities:
            raise Exception("action log probabilities already added for current memory")
    
        self.memory_staging.action_log_probabilities = value
    
    def add_reward(self, value):
        if self.memory_staging.reward:
            raise Exception("reward already added for current memory")
    
        self.memory_staging.reward = value

    def push(self):
        self.memories.append(self.memory_staging)
        self.memory_staging = Memory()

    def __len__(self):
        return len(self.memories)
    
    def __getitem__(self, key):
        return self.memories[key]
    
    def __delitem__(self, key):
        del self.memories[key]
