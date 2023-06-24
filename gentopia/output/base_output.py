import json
from abc import ABC
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from gentopia.model.completion_model import BaseCompletion


class BaseOutput:

    def stop(self):
        pass

    def update_status(self, output: str, **kwargs):
        pass

    def thinking(self, name: str):
        pass

    def done(self, _all=False):
        pass

    def stream_print(self, item: str):
        pass

    def json_print(self, item: Dict[str, Any]):
        pass

    def panel_print(self, item: Any, title: str = "Output", stream: bool = False):
        pass

    def clear(self):
        pass

    def print(self, content: str, **kwargs):
        pass

    def format_json(self, json_obj: str):
        formatted_json = json.dumps(json_obj, indent=2)
        return formatted_json
