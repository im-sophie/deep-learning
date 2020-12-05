from . import *

def list_agent_factories():
    return list(
        filter(
            lambda i: isinstance(i, type)
                and issubclass(i, AgentFactoryBase)
                and i != AgentFactoryBase,
            globals().values()
        )
    )
