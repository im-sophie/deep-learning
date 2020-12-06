import gym

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentContinuousActorCritic
from ...network import ParameterizedLinearNetwork
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryContinuousActorCriticMountainCarContinuousV0(AgentEnvironmentFactoryBase):
    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("learning_rate_actor", 5e-6)
        hyperparameter_set.add("learning_rate_critic", 1e-5)
        hyperparameter_set.add("layer_dimensions_actor", [256, 256])
        hyperparameter_set.add("layer_dimensions_critic", [256, 256])
        hyperparameter_set.add("gamma", 0.99)
        return hyperparameter_set
    
    def on_create_environment(self):
        return EnvironmentGymWrapper(
            gym.make("MountainCarContinuous-v0")
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentContinuousActorCritic(
            actor_network = ParameterizedLinearNetwork(
                learning_rate = hyperparameter_set["learning_rate_actor"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = 2,
                layer_dimensions = hyperparameter_set["layer_dimensions_actor"]
            ),
            critic_network = ParameterizedLinearNetwork(
                learning_rate = hyperparameter_set["learning_rate_critic"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = 1,
                layer_dimensions = hyperparameter_set["layer_dimensions_critic"]
            ),
            hyperparameter_set = hyperparameter_set
        )
