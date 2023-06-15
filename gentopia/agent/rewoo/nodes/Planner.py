from typing import List, Union

from langchain.tools import BaseTool
from pydantic import BaseModel

from gentopia.agent.base_agent import BaseAgent
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.completion_model import BaseCompletion
from gentopia.prompt.rewoo import *


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
            return self.prompt_template.format(tool_description=worker_desctription, fewshot=fewshot, task=instruction)
        else:
            if self.examples is not None:

                return FewShotPlannerPrompt.format(tool_description=worker_desctription, fewshot=fewshot,
                                                   task=instruction)
            else:
                return ZeroShotPlannerPrompt.format(tool_description=worker_desctription, task=instruction)

    def run(self, instruction: str) -> BaseCompletion:
        self.logger.info("Running Planner")
        prompt = self._compose_prompt(instruction)
        response = self.model.completion(prompt)
        if response.state == "error":
            self.logger.error("Planner failed to retrieve response from LLM")
            raise ValueError("Planner failed to retrieve response from LLM")
        else:
            self.logger.info(f"Planner run successful.")
            return response
