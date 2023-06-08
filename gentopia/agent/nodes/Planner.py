from gentopia.agent.nodes.LLMNode import LLMNode
from gentopia.prompt.planner import *
from gentopia.util.util import LLAMA_WEIGHTS


class Planner(LLMNode):
    def __init__(self, workers, prefix=DEFAULT_PREFIX, suffix=DEFAULT_SUFFIX, fewshot=DEFAULT_FEWSHOT,
                 model_name="text-davinci-003", stop=None):
        super().__init__("Planner", model_name, stop, input_type=str, output_type=str)
        self.workers = workers
        self.prefix = prefix
        self.worker_prompt = workers.generate_worker_prompt()
        self.suffix = suffix
        self.fewshot = fewshot

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        prompt = self.prefix + self.worker_prompt + self.fewshot + self.suffix + input + '\n'
        if self.model_name in LLAMA_WEIGHTS:
            prompt = [self.prefix + self.worker_prompt, input]
        response = self.call_llm(prompt, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion
