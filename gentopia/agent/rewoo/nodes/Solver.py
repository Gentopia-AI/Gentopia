from typing import List, Union

from pydantic import BaseModel

from gentopia.llm.base_llm import BaseLLM
from gentopia.model.completion_model import BaseCompletion
from gentopia.prompt.rewoo import *


class Solver(BaseModel):
    model: BaseLLM
    prompt_template: PromptTemplate = None
    examples: Union[str, List[str]] = None
    # logger = logging.getLogger('application')

    def _compose_fewshot_prompt(self) -> str:
        if self.examples is None:
            return ""
        if isinstance(self.examples, str):
            return self.examples
        else:
            return "\n\n".join([e.strip("\n") for e in self.examples])

    def _compose_prompt(self, instruction, plan_evidence) -> str:
        """
        Compose the prompt from template, plan&evidence, examples and instruction.
        """
        fewshot = self._compose_fewshot_prompt()
        if self.prompt_template is not None:
            return self.prompt_template.format(plan_evidence=plan_evidence, fewshot=fewshot, task=instruction)
        else:
            if self.examples is not None:
                return FewShotSolverPrompt.format(plan_evidence=plan_evidence, fewshot=fewshot, task=instruction)
            else:
                return ZeroShotSolverPrompt.format(plan_evidence=plan_evidence, task=instruction)

    def run(self, instruction: str, plan_evidence: str) -> BaseCompletion:
        self.logger.info("Running Solver")
        prompt = self._compose_prompt(instruction, plan_evidence)
        response = self.model.completion(prompt)
        if response.state == "error":
            self.logger.error("Solver failed to retrieve response from LLM")
        else:
            self.logger.info(f"Solver run successful.")

            return response
