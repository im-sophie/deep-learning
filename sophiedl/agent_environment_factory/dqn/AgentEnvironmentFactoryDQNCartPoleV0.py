import gym

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentDQN, EpsilonGreedyStrategy
from ...network import ParameterizedNetwork
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryDQNCartPoleV0(AgentEnvironmentFactoryBase):
    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.99)
        hyperparameter_set.add("learning_rate", 1e-3)
        hyperparameter_set.add("layer_dimensions", [20, 10])
        hyperparameter_set.add("epsilon_start", 0.9)
        hyperparameter_set.add("epsilon_end", 0.05)
        hyperparameter_set.add("epsilon_decay", 0.005)
        hyperparameter_set.add("memory_batch_size", 100)
        hyperparameter_set.add("target_update_interval", 2)
        return hyperparameter_set
    
    def on_create_environment(self):
        return EnvironmentGymWrapper(
            gym.make("CartPole-v0")
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentDQN(
            policy_network = ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = environment.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            target_network = ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = environment.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            epsilon_greedy_strategy = EpsilonGreedyStrategy(
                start = hyperparameter_set["epsilon_start"],
                end = hyperparameter_set["epsilon_end"],
                decay = hyperparameter_set["epsilon_decay"]
            ),
            hyperparameter_set = hyperparameter_set
        )
