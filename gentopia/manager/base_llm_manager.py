from abc import ABC, abstractmethod
from typing import AnyStr, List, Dict, Any, Optional

from pydantic import BaseModel

from gentopia.manager.server_info import BaseServerInfo
from gentopia.model.param_model import BaseParamModel


class BaseLLMManager(ABC):
    server: List[BaseServerInfo]
    _type: AnyStr = "BaseLLMManager"

    @abstractmethod
    def _get_server(self, llm_name: AnyStr) -> BaseServerInfo:
        pass

    @property
    def type(self):
        return self._type

    def find(self, llm_name: AnyStr, model_param: BaseParamModel, **kwargs) -> Optional[BaseServerInfo]:
        for server in self.server:
            if server.llm_name == llm_name:
                if model_param == server.model_param and kwargs == server.kwargs:
                    return server
        return None

    def wrap_llm(self, llm):
        pass
