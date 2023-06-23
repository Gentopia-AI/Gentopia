import json
from abc import ABC
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from gentopia.model.completion_model import BaseCompletion


class BaseOutput:

    def __init__(self):
        self.status_stack: List[str] = []

    def stop(self):
        pass

    def update_status(self, output: str, **kwargs):
        self.status_stack.append(output)
        print(self.status_stack, flush=True)

    def thinking(self, name: str):
        self.status_stack.append(name)
        print(f"{name} is thinking...", flush=True)

    def done(self, _all=False):
        if _all:
            self.status_stack = []

        else:
            if len(self.status_stack) > 0:
                self.status_stack.pop()
            if len(self.status_stack) > 0:
                print(f"{self.status_stack[-1]} is thinking...")

    def stream_print(self, item: str):
        print(item, end="", flush=True)

    def json_print(self, item: Dict[str, Any]):
        print(item, end="", flush=True)

    def panel_print(self, item: Any, title: str = "Output", stream: bool = False):
        print(title.center(80, "-"), flush=True)
        print(item, flush=True)

    def clear(self):
        pass

    def print(self, content: str, **kwargs):
        print(content, **kwargs)

    def format_json(self, json_obj: str):
        formatted_json = json.dumps(json_obj, indent=2)
        return formatted_json
