from dataclasses import dataclass
from typing import Union


@dataclass
class AgentAction:
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str
