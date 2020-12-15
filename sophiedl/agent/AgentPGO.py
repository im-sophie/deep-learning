# Typing
from typing import cast, Iterable, Tuple, Union

# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..memory.Memory import Memory
from ..network.OptimizedModule import OptimizedModule
from ..running.RunnerRLContext import RunnerRLContext
from .AgentBase import AgentBase

class AgentPGO(AgentBase):
    policy_network: OptimizedModule

    def __init__(self,
        policy_network: OptimizedModule,
        hyperparameter_set: HyperparameterSet) -> None:
        super().__init__(
            hyperparameter_set
        )

        self.policy_network = policy_network
    
    def on_act(self, runner_context: RunnerRLContext, observation: T.Tensor) -> Tuple[int, T.Tensor]:
        action_probabilities = T.distributions.Categorical(
            F.softmax(
                self.policy_network.forward(
                    T.as_tensor(observation, dtype = T.float32, device = self.policy_network.device)
                )
            )
        ) # type: ignore

        action = action_probabilities.sample() # type: ignore
        action_log_probabilities = action_probabilities.log_prob(action) # type: ignore

        return action.item(), action_log_probabilities
    
    def on_should_learn(self, runner_context: RunnerRLContext) -> bool:
        return runner_context.done

    def on_learn(self, runner_context: RunnerRLContext) -> None:
        self.policy_network.optimizer.zero_grad()

        discounted_future_rewards = T.zeros(len(self.memory_buffer), dtype = T.float, device = self.policy_network.device)

        for i in range(len(self.memory_buffer)):
            discounted_future_reward_sum = 0.
            discount_factor = 1.

            for j in range(i, len(self.memory_buffer)):
                if self.memory_buffer[j].reward is not None:
                    discounted_future_reward_sum += cast(float, self.memory_buffer[j].reward) * discount_factor
                discount_factor *= self.hyperparameter_set["gamma"]
            
            discounted_future_rewards[i] = discounted_future_reward_sum

        discounted_future_rewards -= discounted_future_rewards.mean()

        std = discounted_future_rewards.std().item()
        if std != 0:
            discounted_future_rewards /= std

        assert self.memory_buffer[0].action_log_probabilities is not None
        loss = T.zeros_like(self.memory_buffer[0].action_log_probabilities)

        for discounted_future_reward, memory in zip(cast(Iterable[float], discounted_future_rewards), cast(Iterable[Memory], self.memory_buffer)):
            assert memory.action_log_probabilities is not None
            loss -= discounted_future_reward * memory.action_log_probabilities

        loss.backward() # type: ignore
        self.policy_network.optimizer.step()

        self.clear_memory()
