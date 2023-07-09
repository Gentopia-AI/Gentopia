from __future__ import annotations
from enum import Enum
from pydantic import BaseModel

#from gentopia.agent.base_agent import BaseAgent


class AgentType(Enum):
    """
    Enumerated type for agent types.
    """
    openai = "openai"
    react = "react"
    rewoo = "rewoo"
    vanilla = "vanilla"
    openai_memory = "openai_memory"

    @staticmethod
    def get_agent_class(_type: AgentType):
        """
        Get agent class from agent type.
        :param _type: agent type
        :return: agent class
        """
        if _type == AgentType.react:
            from gentopia.agent.react import ReactAgent
            return ReactAgent
        elif _type == AgentType.rewoo:
            from gentopia.agent.rewoo import RewooAgent
            return RewooAgent
        elif _type == AgentType.vanilla:
            from gentopia.agent.vanilla import VanillaAgent
            return VanillaAgent
        elif _type == AgentType.openai:
            from gentopia.agent.openai import OpenAIFunctionChatAgent
            return OpenAIFunctionChatAgent
        elif _type == AgentType.openai_memory:
            from gentopia.agent.openai_memory import OpenAIMemoryChatAgent
            return OpenAIMemoryChatAgent
        else:
            raise ValueError(f"Unknown agent type: {_type}")


class AgentOutput(BaseModel):
    """
    Pydantic model for agent output.
    """
    output: str
    cost: float
    token_usage: int