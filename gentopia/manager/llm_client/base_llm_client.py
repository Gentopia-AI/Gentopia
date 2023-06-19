from abc import ABC, abstractmethod

from fastapi import APIRouter

from gentopia.llm.base_llm import BaseLLM
from gentopia.manager.base_llm_manager import BaseServerInfo


class BaseLLMClient(ABC):
    server: BaseServerInfo
    llm: BaseLLM
    router: APIRouter = APIRouter()

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def completion(self, prompt):
        pass

    @abstractmethod
    def chat_completion(self, message):
        pass

    @abstractmethod
    def stream_chat_completion(self, prompt):
        pass
