import os
import sys

sys.path.append(os.path.abspath("."))

import sophiedl as S

class TestRunnerFactories(object):
    def _test_runner_factory_helper(self, runner_factory_type):
        runner_factory = runner_factory_type()

        hyperparameter_set = runner_factory.create_default_hyperparameter_set()

        if "episode_count" in hyperparameter_set:
            hyperparameter_set["episode_count"] = 2
            
        if "epoch_count" in hyperparameter_set:
            hyperparameter_set["epoch_count"] = 2
        
        runner_factory.create_runner(
            hyperparameter_set = hyperparameter_set
        ).run()

    def test_network_RunnerNetworkTorchDataLoaderFactoryMNIST(self):
        self._test_runner_factory_helper(S.RunnerNetworkTorchDataLoaderFactoryMNIST)

    def test_continuous_actor_critic_RunnerRLFactoryContinuousActorCriticMountainCarContinuousV0(self):
        self._test_runner_factory_helper(S.RunnerRLFactoryContinuousActorCriticMountainCarContinuousV0)
    
    def test_discrete_actor_critic_RunnerRLFactoryDiscreteActorCriticCartPoleV0(self):
        self._test_runner_factory_helper(S.RunnerRLFactoryDiscreteActorCriticCartPoleV0)

    def test_dqn_RunnerRLFactoryDQNCartPoleV0(self):
        self._test_runner_factory_helper(S.RunnerRLFactoryDQNCartPoleV0)
    
    def test_pgo_RunnerRLFactoryPGOLunarLanderV2(self):
        self._test_runner_factory_helper(S.RunnerRLFactoryPGOLunarLanderV2)
