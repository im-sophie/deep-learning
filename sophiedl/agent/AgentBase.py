import abc

from ..memory import MemoryBuffer

class AgentBase(abc.ABC):
    def __init__(
        self,
        hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        self.memory_buffer = MemoryBuffer()
    
    @abc.abstractmethod
    def on_act(self, runner_context, observation):
        pass

    @abc.abstractmethod
    def on_should_learn(self, runner_context):
        pass

    @abc.abstractmethod
    def on_learn(self, runner_context):
        pass

    def act(self, runner_context, observation):
        action, action_log_probabilities = self.on_act(runner_context, observation)

        self.memory_buffer.add_observation_current(observation)
        self.memory_buffer.add_action(action)

        if type(action_log_probabilities) != type(None):
            self.memory_buffer.add_action_log_probabilities(action_log_probabilities)
        
        return action
    
    def reward(self, reward, observation, done):
        self.memory_buffer.add_observation_next(observation)
        self.memory_buffer.add_reward(reward)
        self.memory_buffer.add_done(done)
        self.memory_buffer.push()

    def learn(self, runner_context):
        if self.on_should_learn(runner_context):
            self.on_learn(runner_context)

    def clear_memory(self):
        del self.memory_buffer[:]
