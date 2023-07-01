import json
import logging
from abc import ABC
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from gentopia.model.completion_model import BaseCompletion
from gentopia.output import check_log


class BaseOutput:

    def __init__(self):
        self.logger = logging

    def stop(self):
        pass

    def update_status(self, output: str, **kwargs):
        if check_log():
            self.logger.info(output)


    def thinking(self, name: str):
        if check_log():
            self.logger.info(f"{name} is thinking...")


    def done(self, _all=False):
        if check_log():
            self.logger.info("Done")


    def stream_print(self, item: str):
        pass

    def json_print(self, item: Dict[str, Any]):
        if check_log():
            self.logger.info(json.dumps(item, indent=2))

    def panel_print(self, item: Any, title: str = "Output", stream: bool = False):
        if check_log():
            self.logger.info('-'*20)
            self.logger.info(item)
            self.logger.info('-' * 20)

    def clear(self):
        pass

    def print(self, content: str, **kwargs):
        if check_log():
            self.logger.info(content)

    def format_json(self, json_obj: str):
        formatted_json = json.dumps(json_obj, indent=2)
        return formatted_json


    def debug(self, content: str, **kwargs):
        if check_log():
            self.logger.debug(content, **kwargs)


    def info(self, content: str, **kwargs):
        if check_log():
            self.logger.info(content, **kwargs)


    def warning(self, content: str, **kwargs):
        if check_log():
            self.logger.warning(content, **kwargs)


    def error(self, content: str, **kwargs):
        if check_log():
            self.logger.error(content, **kwargs)


    def critical(self, content: str, **kwargs):
        if check_log():
            self.logger.critical(content, **kwargs)
