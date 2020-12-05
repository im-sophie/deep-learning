import torch as T
import torch.nn.functional as F

from .AgentBase import AgentBase

class AgentPGO(AgentBase):
    def __init__(self,
        policy_network,
        hyperparameter_set):
        super().__init__(
            hyperparameter_set
        )

        self.policy_network = policy_network
    
    def on_act(self, runner_context, observation):
        assert observation.shape == self.policy_network.observation_space_shape, "observation must match expected shape"

        action_probabilities = T.distributions.Categorical(
            F.softmax(self.policy_network.forward(observation))
        )

        action = action_probabilities.sample()

        return action.item(), action_probabilities.log_prob(action)
    
    def on_should_learn(self, runner_context):
        return runner_context.done

    def on_learn(self, runner_context):
        self.policy_network.optimizer.zero_grad()

        discounted_future_rewards = T.zeros(len(self.memory_buffer), dtype = T.float, device = self.policy_network.device)

        for i in range(len(self.memory_buffer)):
            discounted_future_reward_sum = 0
            discount_factor = 1

            for j in range(i, len(self.memory_buffer)):
                if self.memory_buffer[j].reward:
                    discounted_future_reward_sum += self.memory_buffer[j].reward * discount_factor
                discount_factor *= self.hyperparameter_set["gamma"]
            
            discounted_future_rewards[i] = discounted_future_reward_sum

        discounted_future_rewards -= discounted_future_rewards.mean()

        std = discounted_future_rewards.std().item()
        if std != 0:
            discounted_future_rewards /= std

        loss = 0
        for discounted_future_reward, memory in zip(discounted_future_rewards, self.memory_buffer):
            loss -= discounted_future_reward * memory.action_log_probabilities

        loss.backward()
        self.policy_network.optimizer.step()

        self.clear_memory()
