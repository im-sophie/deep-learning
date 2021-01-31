# Standard library
import random

# Typing
from typing import Tuple

# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..network.OptimizedModule import OptimizedModule
from ..running.RunnerRLContext import RunnerRLContext
from .AgentBase import AgentBase
from .EpsilonGreedyStrategy import EpsilonGreedyStrategy

class AgentDQN(AgentBase):
    policy_network: OptimizedModule
    target_network: OptimizedModule
    epsilon_greedy_strategy: EpsilonGreedyStrategy
    
    def __init__(
        self,
        policy_network: OptimizedModule,
        target_network: OptimizedModule,
        epsilon_greedy_strategy: EpsilonGreedyStrategy,
        hyperparameter_set: HyperparameterSet):
        super().__init__(
            hyperparameter_set
        )

        self.policy_network = policy_network
        self.target_network = target_network
        self.epsilon_greedy_strategy = epsilon_greedy_strategy
        self._explore_count = 0
        self._exploit_count = 0

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
    
    def on_act(self, runner_context: RunnerRLContext, observation: T.Tensor) -> Tuple[int, None]:
        if self.epsilon_greedy_strategy.should_explore(runner_context):
            self._explore_count += 1
            return random.randint(0, runner_context.environment.action_space_shape.flat_size - 1), None
        else:
            self._exploit_count += 1
            with T.no_grad(): # type: ignore
                return self.policy_network(
                    T.as_tensor(observation, dtype = T.float32).to(self.policy_network.device).unsqueeze(dim = 0)
                ).argmax().item(), None
    
    def on_should_learn(self, _: RunnerRLContext) -> bool:
        return True
    
    def on_learn(self, runner_context: RunnerRLContext) -> None:
        if runner_context.done:
            runner_context.add_scalar("Explore #", self._explore_count, runner_context.episode_index)
            runner_context.add_scalar("Exploit #", self._exploit_count, runner_context.episode_index)
            runner_context.add_scalar("Explore %/Episode", self._explore_count / runner_context.step_index_episode, runner_context.episode_index)
            runner_context.add_scalar("Exploit %/Episode", self._exploit_count / runner_context.step_index_episode, runner_context.episode_index)

            self._explore_count = 0
            self._exploit_count = 0
        
        if runner_context.step_index_total >= self.hyperparameter_set["memory_batch_size"]:
            self.policy_network.optimizer.zero_grad()

            memory_batch = random.sample(self.memory_buffer.memories, self.hyperparameter_set["memory_batch_size"])

            t_observations_current = T.stack(
                [T.as_tensor(i.observation_current, dtype = T.float32) for i in memory_batch]
            ).to(self.policy_network.device)

            t_observations_next = T.stack(
                [T.as_tensor(i.observation_next, dtype = T.float32) for i in memory_batch]
            ).to(self.policy_network.device)

            t_actions = T.as_tensor(
                [i.action for i in memory_batch]
            ).to(self.policy_network.device)

            t_rewards = T.as_tensor(
                [i.reward for i in memory_batch]
            ).to(self.policy_network.device)

            t_nonterminal_indecies = T.as_tensor(
                [i for i in range(len(memory_batch)) if not memory_batch[i].done]
            ).to(self.policy_network.device)

            q_values_current = self.policy_network(
                t_observations_current
            ).gather(
                dim = 1,
                index = t_actions.unsqueeze(
                    dim = 1
                )
            ).squeeze(
                dim = 1
            )

            q_values_next = T.zeros(
                self.hyperparameter_set["memory_batch_size"]
            ).to(self.policy_network.device)

            q_values_next[t_nonterminal_indecies] = self.target_network(
                t_observations_next[t_nonterminal_indecies]
            ).max(dim = 1).values.detach()

            q_values_target = (q_values_next * self.hyperparameter_set["gamma"]) + t_rewards

            loss = F.mse_loss(q_values_current, q_values_target)

            runner_context.add_scalar("Loss", loss.item(), runner_context.step_index_total)

            loss.backward() # type: ignore
            self.policy_network.optimizer.step()

            if runner_context.done and runner_context.episode_index > 0 and runner_context.episode_index % self.hyperparameter_set["target_update_interval"] == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
