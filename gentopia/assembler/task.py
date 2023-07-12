from dataclasses import dataclass
from typing import Union, NamedTuple


@dataclass
class AgentAction:
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str