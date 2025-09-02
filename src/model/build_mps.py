from .depictqa import DepictQA
from .mps_agent import MPSAgent


def build_agent(args, training=True):
    """Build the DepictQA agent with MPS support."""
    model = DepictQA(args)
    
    if training:
        agent = MPSAgent(model, args)
        return agent
    else:
        return model
