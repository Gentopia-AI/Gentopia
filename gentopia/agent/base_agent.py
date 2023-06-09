import io
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional, Type, Callable

from gentopia import PromptTemplate
from pydantic import BaseModel, create_model

from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType, AgentOutput
from gentopia.memory.api import MemoryWrapper
from rich import print as rprint

from gentopia.tools import BaseTool


class BaseAgent(ABC, BaseModel):
    name: str
    type: AgentType
    version: str
    description: str
    target_tasks: List[str]
    llm: Union[BaseLLM, Dict[str, BaseLLM]]
    prompt_template: Union[PromptTemplate, Dict[str, PromptTemplate]]
    plugins: List[Any]
    args_schema: Optional[Type[BaseModel]] = create_model("ArgsSchema", instruction=(str, ...))
    memory: Optional[MemoryWrapper]

    @abstractmethod
    def run(self, *args, **kwargs) -> AgentOutput:
        pass

    @abstractmethod
    def stream(self, *args, **kwargs) -> AgentOutput:
        pass

    # def __str__(self):
    #     return f"""
    #     {f'Agent: {self.name}'.center(50, '-')}\n
    #     Type: {self.type}\n
    #     Version: {self.version}\n
    #     Description: {self.description}\n
    #     Target Tasks: {self.target_tasks}\n
    #     LLM: {self.llm.model_name if isinstance(self.llm, BaseLLM) else
    #     {k: v.model_name for k, v in self.llm.items()} }\n
    #     Plugins: {[i.name for i in self.plugins]}\n
    #     """

    def __str__(self):
        result = io.StringIO()
        rprint(self, file=result)
        return result.getvalue()

    def _format_function_map(self) -> Dict[str, Callable]:
        # Map the function name to the real function object.
        function_map = {}
        for plugin in self.plugins:
            if isinstance(plugin, BaseTool):
                function_map[plugin.name] = plugin._run
            else:
                function_map[plugin.name] = plugin.run
        return function_map
