from typing import List, Union, Optional

from pydantic import BaseModel

from gentopia.agent.base_agent import BaseAgent
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.completion_model import BaseCompletion
from gentopia.output.base_output import BaseOutput
from gentopia.prompt.rewoo import *
from gentopia.tools import BaseTool
import logging


class Planner(BaseModel):
    model: BaseLLM
    prompt_template: PromptTemplate = None
    examples: Union[str, List[str]] = None
    workers: List[Union[BaseTool, BaseAgent]]
    # logger = logging.getLogger('application')

    def _compose_worker_description(self) -> str:
        """
        Compose the worker prompt from the workers.

        Example:
        toolname1[input]: tool1 description
        toolname2[input]: tool2 description
        """
        prompt = ""
        try:
            for worker in self.workers:
                prompt += f"{worker.name}[input]: {worker.description}\n"
        except Exception:
            raise ValueError("Worker must have a name and description.")
        return prompt

    def _compose_fewshot_prompt(self) -> str:
        if self.examples is None:
            return ""
        if isinstance(self.examples, str):
            return self.examples
        else:
            return "\n\n".join([e.strip("\n") for e in self.examples])

    def _compose_prompt(self, instruction) -> str:
        """
        Compose the prompt from template, worker description, examples and instruction.
        """
        worker_desctription = self._compose_worker_description()
        fewshot = self._compose_fewshot_prompt()
        if self.prompt_template is not None:
            if "fewshot" in self.prompt_template.input_variables:
                return self.prompt_template.format(tool_description=worker_desctription, fewshot=fewshot,
                                                   task=instruction)
            else:
                return self.prompt_template.format(tool_description=worker_desctription, task=instruction)
        else:
            if self.examples is not None:
                return FewShotPlannerPrompt.format(tool_description=worker_desctription, fewshot=fewshot,
                                                   task=instruction)
            else:
                return ZeroShotPlannerPrompt.format(tool_description=worker_desctription, task=instruction)

    def run(self, instruction: str, output: BaseOutput = BaseOutput()) -> BaseCompletion:

        output.info("Running Planner")
        prompt = self._compose_prompt(instruction)
        output.debug(f"Prompt: {prompt}")
        response = self.model.completion(prompt)
        if response.state == "error":
            output.error("Planner failed to retrieve response from LLM")
            raise ValueError("Planner failed to retrieve response from LLM")
        else:
            output.info(f"Planner run successful.")
            return response

    def stream(self, instruction: str, output: BaseOutput = BaseOutput()):
        prompt = self._compose_prompt(instruction)
        output.debug(f"Prompt: {prompt}")
        response = self.model.stream_chat_completion([{"role": "user", "content": prompt}])
        for i in response:
            yield i.content
