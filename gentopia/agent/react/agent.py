import logging
import re
from typing import List, Union, Optional, Type, Tuple

from langchain import PromptTemplate
from langchain.schema import AgentFinish
from gentopia.tools.basetool import BaseTool
from pydantic import create_model, BaseModel

from gentopia.agent.base_agent import BaseAgent
from gentopia.assembler.task import AgentAction
from gentopia.llm.client.openai import OpenAIGPTClient
from gentopia.model.agent_model import AgentType, AgentOutput
from gentopia.utils.cost_helpers import calculate_cost

FINAL_ANSWER_ACTION = "Final Answer:"


class ReactAgent(BaseAgent):
    name: str = "ReactAgent"
    type: AgentType = AgentType.react
    version: str
    description: str
    target_tasks: list[str]
    llm: OpenAIGPTClient
    prompt_template: PromptTemplate
    plugins: List[Union[BaseTool, BaseAgent]]
    examples: Union[str, List[str]] = None
    args_schema: Optional[Type[BaseModel]] = create_model("ReactArgsSchema", instruction=(str, ...))

    intermediate_steps: List[Tuple[AgentAction, str]] = []

    def _compose_plugin_description(self) -> str:
        """
        Compose the worker prompt from the workers.

        Example:
        toolname1[input]: tool1 description
        toolname2[input]: tool2 description
        """
        prompt = ""
        try:
            for plugin in self.plugins:
                prompt += f"{plugin.name}[input]: {plugin.description}\n"
        except Exception:
            raise ValueError("Worker must have a name and description.")
        return prompt

    def _construct_scratchpad(
            self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought:"
        return thoughts

    def _parse_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise Exception(
                    "Parsing LLM output produced both a final answer "
                    f"and a parse-able action: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise Exception(
                f"Could not parse LLM output: `{text}`",
            )
        elif not re.search(
                r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise Exception(
                f"Could not parse LLM output: `{text}`"
            )
        else:
            raise Exception(f"Could not parse LLM output: `{text}`")

    def _compose_prompt(self, instruction) -> str:
        """
        Compose the prompt from template, worker description, examples and instruction.
        """
        agent_scratchpad = self._construct_scratchpad(self.intermediate_steps)
        tool_description = self._compose_plugin_description()
        tool_names = ", ".join([plugin.name for plugin in self.plugins])
        if self.prompt_template is None:
            from gentopia.prompt.react import ZeroShotReactPrompt
            self.prompt_template = ZeroShotReactPrompt
        return self.prompt_template.format(
            instruction=instruction,
            agent_scratchpad=agent_scratchpad,
            tool_description=tool_description,
            tool_names=tool_names
        )

    def run(self, instruction):
        logging.info(f"Running {self.name + ':' + self.version} with instruction: {instruction}")
        total_cost = 0.0
        total_token = 0

        prompt = self._compose_prompt(instruction)
        logging.info(f"Prompt: {prompt}")
        response = self.llm.completion(prompt)
        if response.state == "error":
            print("Planner failed to retrieve response from LLM")
            raise ValueError("Planner failed to retrieve response from LLM")

        print(f"Planner run successful.")
        total_cost += calculate_cost(self.llm.model_name, response.prompt_token,
                                     response.completion_token)
        total_token += response.prompt_token + response.completion_token
        self.intermediate_steps.append(self._parse_output(response.content))
        return AgentOutput(output=response.content, cost=total_cost, token_usage=total_token)

    def stream(self, *args, **kwargs) -> AgentOutput:
        pass
