from enum import Enum
from pydantic import BaseModel


class AgentType(Enum):
    """
    Enumerated type for agent types.
    """
    REACT = "react"
    REWOO = "rewoo"
    DIRECT = "direct"


class AgentOutput(BaseModel):
    """
    Pydantic model for agent output.
    """
    output: str
    cost: float
    token_usage: int
