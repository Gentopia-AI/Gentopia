from typing import Generator

from gentopia.llm.base_llm import BaseLLM
from gentopia.model.completion_model import BaseCompletion, ChatCompletion
from gentopia.model.param_model import BaseParamModel


class TmpLLM(BaseLLM):
    model_name: str
    model_description: str
    model_param: BaseParamModel

    def get_model_name(self) -> str:
        pass

    def get_model_param(self) -> BaseParamModel:
        pass

    # Get direct completion as string.
    def completion(self, prompt) -> BaseCompletion:
        pass

    def chat_completion(self, message) -> ChatCompletion:
        pass

    def stream_chat_completion(self, prompt) -> Generator:
        pass
