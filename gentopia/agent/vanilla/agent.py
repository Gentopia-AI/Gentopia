from typing import List, Union, Optional

from gentopia.agent.base_agent import BaseAgent
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType
from gentopia.output.base_output import BaseOutput
from gentopia.prompt.vanilla import *
from gentopia.utils.cost_helpers import *
from gentopia.utils.text_helpers import *


class VanillaAgent(BaseAgent):
    name: str = "VanillaAgent"
    type: AgentType = AgentType.vanilla
    version: str
    description: str
    target_tasks: list[str]
    llm: BaseLLM
    prompt_template: PromptTemplate = None
    examples: Union[str, List[str]] = None

    def _compose_fewshot_prompt(self) -> str:
        if self.examples is None:
            return ""
        if isinstance(self.examples, str):
            return self.examples
        else:
            return "\n\n".join([e.strip("\n") for e in self.examples])

    def _compose_prompt(self, instruction: str) -> str:
        fewshot = self._compose_fewshot_prompt()
        if self.prompt_template is not None:
            if "fewshot" in self.prompt_template.input_variables:
                return self.prompt_template.format(fewshot=fewshot, instruction=instruction)
            else:
                return self.prompt_template.format(instruction=instruction)
        else:
            if self.examples is None:
                return ZeroShotVanillaPrompt.format(instruction=instruction)
            else:
                return FewShotVanillaPrompt.format(fewshot=fewshot, instruction=instruction)

    def run(self, instruction: str, output: Optional[BaseOutput] = None) -> AgentOutput:
        prompt = self._compose_prompt(instruction)
        if output is None:
            output = BaseOutput()
        output.thinking(self.name)
        response = self.llm.completion(prompt)
        output.done()
        output.print(response.content)
        total_cost = calculate_cost(self.llm.model_name, response.prompt_token,
                                    response.completion_token)
        total_token = response.prompt_token + response.completion_token

        return AgentOutput(
            output=response.content,
            cost=total_cost,
            token_usage=total_token)

    def stream(self, instruction: str, output: Optional[BaseOutput] = None):
        prompt = self._compose_prompt(instruction)
        if output is None:
            output = BaseOutput()
        output.thinking(self.name)
        response = self.llm.stream_chat_completion([{"role": "user", "content": prompt}])
        output.done()
        output.print(f"[blue]{self.name}: ")
        for i in response:
            output.panel_print(i.content, self.name, True)
        output.clear()
