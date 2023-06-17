from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional, Type

from langchain import PromptTemplate
from pydantic import BaseModel, create_model

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
    args_schema: Optional[Type[BaseModel]] = create_model("ArgsSchema", input=(str, ...))

    @abstractmethod
    def run(self, *args, **kwargs) -> AgentOutput:
        pass

    def __str__(self):
        return f"""
        {f'Agent: {self.name}'.center(50, '-')}\n
        Type: {self.type}\n
        Version: {self.version}\n
        Description: {self.description}\n
        Target Tasks: {self.target_tasks}\n
        LLM: {self.llm.model_name if isinstance(self.llm, BaseLLM) else
        {k: v.model_name for k, v in self.llm.items()} }\n
        Plugins: {[i.name for i in self.plugins]}\n
        """
