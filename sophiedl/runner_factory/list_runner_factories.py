# Typing
from typing import cast, List

# Internal
from .base.RunnerFactoryBase import RunnerFactoryBase
from .continuous_actor_critic.RunnerRLFactoryContinuousActorCriticMountainCarContinuousV0 import RunnerRLFactoryContinuousActorCriticMountainCarContinuousV0
from .discrete_actor_critic.RunnerRLFactoryDiscreteActorCriticCartPoleV0 import RunnerRLFactoryDiscreteActorCriticCartPoleV0
from .dqn.RunnerRLFactoryDQNCartPoleV0 import RunnerRLFactoryDQNCartPoleV0
from .dqn.RunnerRLFactoryDQNSuperMarioBrosV0 import RunnerRLFactoryDQNSuperMarioBrosV0
from .network.RunnerNetworkTorchDataLoaderFactoryMNIST import RunnerNetworkTorchDataLoaderFactoryMNIST
from .network.RunnerNetworkTreeToNLPresentationFactory import RunnerNetworkTreeToNLPresentationFactory
from .pgo.RunnerRLFactoryPGOLunarLanderV2 import RunnerRLFactoryPGOLunarLanderV2
from .rnn.RunnerRNNTorchDataLoaderFactoryText import RunnerRNNTorchDataLoaderFactoryText

def list_runner_factories() -> List[RunnerFactoryBase]:
    return list(
        filter(
            lambda i: isinstance(i, type)
                and issubclass(i, RunnerFactoryBase)
                and i != cast(type, RunnerFactoryBase),
            globals().values()
        )
    )
