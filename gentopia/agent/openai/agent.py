import logging
import re
from typing import List, Union, Optional, Type, Tuple, Dict, Callable

from langchain import PromptTemplate
from langchain.schema import AgentFinish

from gentopia.prompt import ZeroShotVanillaPrompt
from gentopia.tools import BaseTool
from pydantic import create_model, BaseModel

from gentopia.agent.base_agent import BaseAgent
from gentopia.assembler.task import AgentAction
from gentopia.llm.client.openai import OpenAIGPTClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType, AgentOutput
from gentopia.util.cost_helpers import calculate_cost


class OpenAIFunctionChatAgent(BaseAgent):
    name: str = "OpenAIAgent"
    type: AgentType = AgentType.openai
    version: str = "NA"
    description: str = "OpenAI Function Call Agent"
    target_tasks: list[str] = []
    llm: OpenAIGPTClient
    prompt_template: PromptTemplate = ZeroShotVanillaPrompt
    plugins: List[Union[BaseTool, BaseAgent]] = []
    examples: Union[str, List[str]] = None
    message_scratchpad: List[Dict] = [{"role": "system", "content": "You are a helpful AI assistant."}]

    def _format_plugin_schema(self, plugin: Union[BaseTool, BaseAgent]) -> Dict:
        """Format tool into the open AI function API."""
        if isinstance(plugin, BaseTool):
            if plugin.args_schema:
                parameters = plugin.args_schema.schema()
            else:
                parameters = {
                    # This is a hack to get around the fact that some tools
                    # do not expose an args_schema, and expect an argument
                    # which is a string.
                    # And Open AI does not support an array type for the
                    # parameters.
                    "properties": {
                        "__arg1": {"title": "__arg1", "type": "string"},
                    },
                    "required": ["__arg1"],
                    "type": "object",
                }

            return {
                "name": plugin.name,
                "description": plugin.description,
                "parameters": parameters,
            }
        else:
            parameters = plugin.args_schema.schema()
            return {
                "name": plugin.name,
                "description": plugin.description,
                "parameters": parameters,
            }

    def _format_function_map(self) -> Dict[str, Callable]:
        # Map the function name to the real function object.
        function_map = {}
        for plugin in self.plugins:
            function_map[plugin.name] = plugin.run
        return function_map

    def _format_function_schema(self) -> List[Dict]:
        # List the function schema.
        function_schema = []
        for plugin in self.plugins:
            function_schema.append(self._format_plugin_schema(plugin))
        return function_schema
    def run(self, instruction:str) -> AgentOutput:
        self.message_scratchpad.append({"role": "user", "content": instruction})

        function_map = self._format_function_map()
        function_schema = self._format_function_schema()

        #TODO: stream output, cost and token usage
        response = self.llm.function_chat_completion(self.message_scratchpad, function_map, function_schema)
        if response.state == "success":
            # Update message history
            self.message_scratchpad.append(response.message_scratchpad)
            print(self.message_scratchpad)
            return AgentOutput(
                output=response.content,
                cost=0,
                token_usage=0
            )




