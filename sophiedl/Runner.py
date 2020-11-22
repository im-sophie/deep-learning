class Runner(object):
    def __init__(
        self,
        environment,
        agent,
        episode_count,
        tensorboard_output_dir = None):
        self.environment = environment
        self.agent = agent
        self.episode_count = episode_count
        self.tensorboard_output_dir = tensorboard_output_dir
    
    def _create_tensorboard_summary_writer(self):
        return SummaryWriter(
            log_dir = self.tensorboard_output_dir
        ) if self.tensorboard_output_dir else None

    def _run_episode(self, episode_index, tensorboard_summary_writer):
        reward_sum = 0

        done = False
        observation = self.environment.reset()

        while not done:
            action = self.agent.act(observation)
            observation, reward, done, _ = self.environment.step(action)
            agent.reward(reward)
            reward_sum += reward
        
        agent.learn(episode_index)

        print("Episode {0}, reward sum {1:.3f}".format(episode_index, reward_sum))

        if tensorboard_summary_writer:
            tensorboard_summary_writer.add_scalar("Reward Sum", reward_sum, episode_index)

    def run(self):
        tensorboard_summary_writer = self._create_tensorboard_summary_writer()

        for episode_index in range(self.episode_count):
            self._run_episode(episode_index)
