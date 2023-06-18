from abc import ABC, abstractmethod
from typing import Generator
from pydantic import BaseModel
from gentopia.model.completion_model import BaseCompletion, ChatCompletion
from gentopia.model.param_model import BaseParamModel


class BaseLLM(ABC, BaseModel):

    model_name: str
    params: BaseParamModel

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_model_param(self) -> BaseParamModel:
        pass

    @abstractmethod
    # Get direct completion as string.
    def completion(self, prompt) -> BaseCompletion:
        pass

    @abstractmethod
    def chat_completion(self, message) -> ChatCompletion:
        pass

    @abstractmethod
    def stream_chat_completion(self, prompt) -> Generator:
        pass

