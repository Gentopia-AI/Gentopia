import json
from abc import ABC
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from gentopia.model.completion_model import BaseCompletion


class BasicOutput:
    """
        A class for handling output.
    """

    def __init__(self):
        """
        Initializes a new instance of the BaseOutput class.
        """
        self.status_stack: List[str] = []

    def stop(self):
        """
        Stops the output.
        """
        pass

    def update_status(self, output: str, **kwargs):
        """
        Updates the status.

        :param output: The output to update the status with.
        :type output: str
        :param kwargs: Additional keyword arguments to pass to the update_status method.
        :type kwargs: Any
        """
        self.status_stack.append(output)
        print(self.status_stack, flush=True)

    def thinking(self, name: str):
        """
        Shows that a name is thinking.

        :param name: The name of the entity that is thinking.
        :type name: str
        """
        self.status_stack.append(name)
        print(f"{name} is thinking...", flush=True)

    def done(self, _all=False):
        """
        Marks the output as done.

        :param _all: If True, marks all output as done. If False, marks only the last output as done. Defaults to False.
        :type _all: bool
        """
        if _all:
            self.status_stack = []

        else:
            if len(self.status_stack) > 0:
                self.status_stack.pop()
            if len(self.status_stack) > 0:
                print(f"{self.status_stack[-1]} is thinking...")

    def stream_print(self, item: str):
        """
        Prints an item to the output as a stream.

        :param item: The item to print.
        :type item: str
        """
        print(item, end="", flush=True)

    def json_print(self, item: Dict[str, Any]):
        """
        Prints an item to the output as a JSON object.

        :param item: The item to print.
        :type item: Dict[str, Any]
        """
        print(item, end="", flush=True)

    def panel_print(self, item: Any, title: str = "Output", stream: bool = False):
        """
        Prints an item to the output as a panel.

        :param item: The item to print.
        :type item: Any
        :param title: The title of the panel. Defaults to "Output".
        :type title: str
        :param stream: If True, prints the item as a stream. If False, prints the item as a panel. Defaults to False.
        :type stream: bool
        """
        print(title.center(80, "-"), flush=True)
        print(item, flush=True)

    def clear(self):
        """
        Clears the output.
        """
        pass

    def print(self, content: str, **kwargs):
        """
        Prints content to the output.

        :param content: The content to print.
        :type content: str
        :param kwargs: Additional keyword arguments to pass to the print method.
        :type kwargs: Any
        """
        print(content, **kwargs)

    def format_json(self, json_obj: str):
        """
        Formats a JSON object.

        :param json_obj: The JSON object to format.
        :type json_obj: str
        :return: The formatted JSON object.
        :rtype: str
        """
        formatted_json = json.dumps(json_obj, indent=2)
        return formatted_json
