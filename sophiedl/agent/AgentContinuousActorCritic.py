# Typing
from typing import Tuple

# PyTorch
import torch as T
import torch.nn.functional as F

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..network.OptimizedModule import OptimizedModule
from ..running.RunnerRLContext import RunnerRLContext
from .AgentBase import AgentBase

class AgentContinuousActorCritic(AgentBase):
    actor_network: OptimizedModule
    critic_network: OptimizedModule

    def __init__(
        self,
        actor_network: OptimizedModule,
        critic_network: OptimizedModule,
        hyperparameter_set: HyperparameterSet):
        super().__init__(
            hyperparameter_set
        )

        self.actor_network = actor_network
        self.critic_network = critic_network
    
    def on_act(self, runner_context: RunnerRLContext, observation: T.Tensor) -> Tuple[float, T.Tensor]:
        mu, sigma = self.actor_network.forward(
            T.as_tensor(observation, dtype = T.float32, device = self.actor_network.device)
        )

        action_probabilities = T.distributions.Normal(
            mu,
            T.exp(sigma)
        ) # type: ignore

        action = action_probabilities.sample(
            sample_shape = (1,)
        ) # type: ignore
        action_log_probabilities = action_probabilities.log_prob(action) # type: ignore

        return T.tanh(action).cpu().numpy(), action_log_probabilities

    def on_should_learn(self, _: RunnerRLContext) -> bool:
        return len(self.memory_buffer) > 0

    def on_learn(self, _: RunnerRLContext) -> None:
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

        assert self.memory_buffer[-1].action_log_probabilities is not None
        actor_loss = -self.memory_buffer[-1].action_log_probabilities * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()

        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()
