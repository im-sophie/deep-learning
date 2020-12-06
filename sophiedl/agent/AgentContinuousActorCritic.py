import torch as T
import torch.nn.functional as F

from .AgentBase import AgentBase

class AgentContinuousActorCritic(AgentBase):
    def __init__(self,
        actor_network,
        critic_network,
        hyperparameter_set):
        super().__init__(
            hyperparameter_set
        )

        self.actor_network = actor_network
        self.critic_network = critic_network
    
    def on_act(self, runner_context, observation):
        mu, sigma = self.actor_network.forward(
            T.as_tensor(observation, dtype = T.float32, device = self.actor_network.device)
        )

        action_probabilities = T.distributions.Normal(
            mu,
            T.exp(sigma)
        )

        action = action_probabilities.sample(
            sample_shape = (1,)
        )

        return T.tanh(action).cpu().numpy(), action_probabilities.log_prob(action)

    def on_should_learn(self, runner_context):
        return len(self.memory_buffer) > 0

    def on_learn(self, runner_context):
        self.actor_network.optimizer.zero_grad()
        self.critic_network.optimizer.zero_grad()

        critic_value = self.critic_network.forward(
            T.as_tensor(self.memory_buffer[-1].observation_current, dtype = T.float32, device = self.critic_network.device)
        )
        critic_value_next = self.critic_network.forward(
            T.as_tensor(self.memory_buffer[-1].observation_next, dtype = T.float32, device = self.critic_network.device)
        )

        delta = (
            self.memory_buffer[-1].reward
            + (0 if self.memory_buffer[-1].done else self.hyperparameter_set["gamma"] * critic_value_next)
            - critic_value
        )

        actor_loss = -self.memory_buffer[-1].action_log_probabilities * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()

        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()
