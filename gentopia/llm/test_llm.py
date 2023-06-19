from typing import Generator

from gentopia.llm.base_llm import BaseLLM
from gentopia.model.completion_model import ChatCompletion, BaseCompletion
from gentopia.model.param_model import BaseParamModel


class TestLLM(BaseLLM):
    model_name: str = "test"
    model_param: BaseParamModel = BaseParamModel()

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_param(self) -> BaseParamModel:
        pass

    def completion(self, prompt) -> BaseCompletion:
        return BaseCompletion(state="success", content="test")

    def chat_completion(self, message) -> ChatCompletion:
        pass

    def stream_chat_completion(self, prompt) -> Generator:
        pass

    model_name: str = "test"
    model_param: BaseParamModel
