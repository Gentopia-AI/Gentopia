from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any

from langchain import PromptTemplate
from pydantic import BaseModel

from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType, AgentOutput


class BaseAgent(ABC, BaseModel):
    name: str
    type: AgentType
    version: str
    description: str
    target_tasks: List[str]
    llm: Union[BaseLLM, Dict[str, BaseLLM]]
    prompt_template: Union[PromptTemplate, Dict[str, PromptTemplate]]
    plugins: List[Any]


    @abstractmethod
    def run(self, input) -> AgentOutput:
        pass
