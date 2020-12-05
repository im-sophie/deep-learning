import gym

from ...environment import EnvironmentGymWrapper
from ...HyperparameterSet import HyperparameterSet
from ...agent import AgentDiscreteActorCritic
from ...network import ParameterizedNetwork
from ..AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase

class AgentEnvironmentFactoryDiscreteActorCriticCartPoleV0(AgentEnvironmentFactoryBase):
    def on_create_default_hyperparameter_set(self):
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("learning_rate_actor", 1e-5)
        hyperparameter_set.add("learning_rate_critic", 5e-4)
        hyperparameter_set.add("layer_dimensions_actor", [32, 32])
        hyperparameter_set.add("layer_dimensions_critic", [32, 32])
        hyperparameter_set.add("gamma", 0.99)
        return hyperparameter_set
    
    def on_create_environment(self):
        return EnvironmentGymWrapper(
            gym.make("CartPole-v0")
        )
    
    def on_create_agent(self, environment, hyperparameter_set):
        return AgentDiscreteActorCritic(
            actor_network = ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate_actor"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = environment.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions_actor"]
            ),
            critic_network = ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate_critic"],
                observation_space_shape = environment.observation_space_shape,
                output_feature_count = 1,
                layer_dimensions = hyperparameter_set["layer_dimensions_critic"]
            ),
            hyperparameter_set = hyperparameter_set
        )
