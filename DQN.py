import random

import torch as T
import torch.nn.functional as F

import gym

import sophiedl as S

class DQNAgent(S.AgentBase):
    def __init__(self,
        policy_network,
        target_network,
        epsilon_greedy_strategy,
        hyperparameter_set):
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
    
    def on_act(self, runner_context, observation):
        assert observation.shape == self.policy_network.observation_space_shape, "observation must match expected shape"

        if self.epsilon_greedy_strategy.should_explore(runner_context):
            self._explore_count += 1
            return random.randint(0, runner_context.environment.action_space_shape.flat_size - 1), None
        else:
            self._exploit_count += 1
            with T.no_grad():
                return self.policy_network(
                    T.as_tensor(observation, dtype = T.float32).to(self.policy_network.device).unsqueeze(dim = 0)
                ).argmax().item(), None
    
    def on_should_learn(self, runner_context):
        return True
    
    def on_learn(self, runner_context):
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

            loss.backward()
            self.policy_network.optimizer.step()

            if runner_context.done and runner_context.episode_index > 0 and runner_context.episode_index % self.hyperparameter_set["target_update_interval"] == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

if __name__ == "__main__":
    env = S.EnvironmentGymWrapper(
        gym.make("CartPole-v0")
    )

    hyperparameter_set = S.HyperparameterSet()
    hyperparameter_set.add("gamma", 0.99)
    hyperparameter_set.add("learning_rate", 1e-3)
    hyperparameter_set.add("layer_dimensions", [20, 10])
    hyperparameter_set.add("epsilon_start", 0.9)
    hyperparameter_set.add("epsilon_end", 0.05)
    hyperparameter_set.add("epsilon_decay", 0.005)
    hyperparameter_set.add("memory_batch_size", 100)
    hyperparameter_set.add("target_update_interval", 2)

    S.Runner(
        env,
        DQNAgent(
            policy_network = S.ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = env.observation_space_shape,
                output_feature_count = env.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            target_network = S.ParameterizedNetwork(
                learning_rate = hyperparameter_set["learning_rate"],
                observation_space_shape = env.observation_space_shape,
                output_feature_count = env.action_space_shape.flat_size,
                layer_dimensions = hyperparameter_set["layer_dimensions"]
            ),
            epsilon_greedy_strategy = S.EpsilonGreedyStrategy(
                start = hyperparameter_set["epsilon_start"],
                end = hyperparameter_set["epsilon_end"],
                decay = hyperparameter_set["epsilon_decay"]
            ),
            hyperparameter_set = hyperparameter_set
        ),
        episode_count = 500,
        hyperparameter_set = hyperparameter_set,
        tensorboard_output_dir = "./runs/DQN"
    ).run()
