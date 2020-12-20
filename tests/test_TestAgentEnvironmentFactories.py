import os
import sys

sys.path.append(os.path.abspath("."))

import sophiedl as S

class TestAgentEnvironmentFactories(object):
    def test_all_two_episodes(self):
        for runner_factory_type in S.list_runner_factories():
            runner_factory = runner_factory_type()
            hyperparameter_set = runner_factory.create_default_hyperparameter_set()
            hyperparameter_set["episode_count"] = 2
            runner_factory.create_runner(
                hyperparameter_set = hyperparameter_set
            ).run()
