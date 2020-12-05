import gym

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentPGO
from ...network import ParameterizedNetwork
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryPGOLunarLanderV2(AgentEnvironmentFactoryBase):
    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("gamma", 0.99)
        hyperparameter_set.add("learning_rate", 0.001)
        hyperparameter_set.add("layer_dimensions", [128, 128])
        return hyperparameter_set
    
    def on_create_environment(self):
        return EnvironmentGymWrapper(
            gym.make("LunarLander-v2")
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentPGO(
            policy_network = ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = environment.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            hyperparameter_set = hyperparameter_set
        )
