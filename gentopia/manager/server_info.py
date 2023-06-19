from abc import ABC
from typing import Dict, Any

from pydantic import BaseModel

from gentopia.model.param_model import BaseParamModel


class BaseServerInfo(ABC, BaseModel):
    host: str
    port: int
    llm_name: str
    model_param: BaseParamModel
    kwargs: Dict[Any, Any] = dict()


class LocalServerInfo(BaseServerInfo):
    log_level: str = "info"
