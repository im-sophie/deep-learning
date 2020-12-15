import os
import sys

sys.path.append(os.path.abspath("."))

import sophiedl as S

class TestAgentEnvironmentFactories(object):
    def test_all_two_episodes(self):
        for i in S.list_agent_environment_factories():
            i().create_runner(2).run()
