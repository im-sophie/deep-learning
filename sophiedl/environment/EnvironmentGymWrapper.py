import gym

from .EnvironmentBase import EnvironmentBase

class EnvironmentGymWrapper(EnvironmentBase):
    @staticmethod
    def _get_space_shape(space):
        if isinstance(space, gym.spaces.Discrete):
            return (space.n,)
        elif isinstance(space, gym.spaces.Box):
            return space.shape
        elif isinstance(space, gym.Space):
            raise TypeError("unexpected subtype of Space: {0}".format(type(space).__name__))
        else:
            raise TypeError("unexpected type for space: {0}".format(type(space).__name__))

    def __init__(self, gym_environment):
        super().__init__()

        self.gym_environment = gym_environment
    
    def _on_get_observation_space_shape(self):
        return EnvironmentGymWrapper._get_space_shape(
            self.gym_environment.observation_space
        )

    def _on_get_action_space_shape(self):
        return EnvironmentGymWrapper._get_space_shape(
            self.gym_environment.action_space
        )

    def reset(self):
        return self.gym_environment.reset()

    def step(self, action):
        return self.gym_environment.step(action)
