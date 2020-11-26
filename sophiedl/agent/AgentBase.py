import abc

from ..memory import MemoryBuffer

class AgentBase(abc.ABC):
    def __init__(
        self,
        memory_cleanup_schedule,
        hyperparameter_set,
        tensorboard_summary_writer = None):
        self.hyperparameter_set = hyperparameter_set
        self.tensorboard_summary_writer = tensorboard_summary_writer
        self.memory_buffer = MemoryBuffer(
            memory_cleanup_schedule
        )
    
    @abc.abstractmethod
    def on_act(self, observation):
        pass

    @abc.abstractmethod
    def on_learn(self):
        pass

    def act(self, runner_context, observation):
        self.memory_buffer.cleanup(runner_context)

        action, action_log_probabilities = self.on_act(observation)

        self.memory_buffer.add_observation_current(observation)
        self.memory_buffer.add_action(action)

        if action_log_probabilities:
            self.memory_buffer.add_action_log_probabilities(action_log_probabilities)
        
        return action.item()
    
    def reward(self, reward, observation):
        self.memory_buffer.add_observation_next(observation)
        self.memory_buffer.add_reward(reward)
        self.memory_buffer.push()

    def learn(self):
        self.on_learn()
