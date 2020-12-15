from typing import List, cast
from .AgentEnvironmentFactoryBase import AgentEnvironmentFactoryBase
from .continuous_actor_critic.AgentEnvironmentFactoryContinuousActorCriticMountainCarContinuousV0 import AgentEnvironmentFactoryContinuousActorCriticMountainCarContinuousV0
from .discrete_actor_critic.AgentEnvironmentFactoryDiscreteActorCriticCartPoleV0 import AgentEnvironmentFactoryDiscreteActorCriticCartPoleV0
from .dqn.AgentEnvironmentFactoryDQNCartPoleV0 import AgentEnvironmentFactoryDQNCartPoleV0
from .pgo.AgentEnvironmentFactoryPGOLunarLanderV2 import AgentEnvironmentFactoryPGOLunarLanderV2

def list_agent_environment_factories() -> List[AgentEnvironmentFactoryBase]:
    return list(
        filter(
            lambda i: isinstance(i, type)
                and issubclass(i, AgentEnvironmentFactoryBase)
                and i != cast(type, AgentEnvironmentFactoryBase),
            globals().values()
        )
    )
