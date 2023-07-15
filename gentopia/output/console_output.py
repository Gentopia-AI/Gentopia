import json
from abc import ABC
from typing import Union, Dict, Any, List, Optional

from rich.box import Box
from rich.console import Console
from pydantic import BaseModel
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax

from gentopia.model.completion_model import BaseCompletion
from gentopia.output.base_output import BaseOutput

HEAVY: Box = Box(
    """\
┏━┳┓
┃ ┃┃
┣━╋┫
┃ ┃┃
┣━╋┫
┣━╋┫
┃ ┃┃
┗━┻┛
"""
)


class ConsoleOutput(BaseOutput):
    """
        A class for displaying output on the console using the rich library.
    """
    def __init__(self):
        """
        Initializes a new instance of the ConsoleOutput class.
        """
        self.console: Console = Console()
        self.status_stack: List[str] = []
        self.status: Optional[Status] = None
        self.live: Optional[Live] = None
        self.cache: str = ""
        super().__init__()

    def stop(self):
        """
        Stops the Live status.
        """
        if self.status is not None:
            self.status.stop()

    def update_status(self, output: str, style: str = "[bold green]"):
        """
        Updates the status, push it into a stack.

        :param output: The output to update the status with.
        :type output: str
        :param style: The style to use for the output. Defaults to "[bold green]".
        :type style: str
        """
        self.status_stack.append(output)
        if self.status is not None:
            self.status.update(f"{style}{output}")
        else:
            self.status = self.console.status(f"{style}{output}")
        self.status.start()
        super().update_status(output)

    def thinking(self, name: str):
        """
        Shows that a name is thinking.

        :param name: The name of the entity that is thinking.
        :type name: str
        """
        self.status_stack.append(name)
        if self.status is not None:
            self.status.update(f"[bold green]{name} is thinking...")
        else:
            self.status = self.console.status(f"[bold green]{name} is thinking...")
        self.status.start()
        super().thinking(name)

    def done(self, _all=False):
        """
        Marks the status as done.

        :param _all: If True, marks all status as done. If False, marks only the last status as done. Defaults to False.
        :type _all: bool
        """
        if _all:
            self.status_stack = []
            self.status.stop()
        else:
            if len(self.status_stack) > 0:
                self.status_stack.pop()
            if len(self.status_stack) > 0:
                self.status.update(f"[bold green]{self.status_stack[-1]} is thinking...")
            else:
                self.status.stop()
        super().done(_all)

    def stream_print(self, item: str):
        """
        Prints an item to the output as a stream.

        :param item: The item to print.
        :type item: str
        """
        self.console.print(item, end="")

    def json_print(self, item: Dict[str, Any]):
        """
        Prints an item to the output as a JSON object.

        :param item: The item to print.
        :type item: Dict[str, Any]
        """
        self.console.print_json(data=item)
        super().json_print(item)

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
        if not stream:
            super().panel_print(item, title, stream)
        if not stream:
            self.console.print(Panel(item, title=title))
            return
        if self.live is None:
            # if self.status is not None:
            #     self.status.stop()
            self.cache = item
            self.live = Live(Panel(Markdown(self.cache), title=title, style="yellow", box=HEAVY), console=self.console,
                             refresh_per_second=12)
            self.live.start()
            return
        self.cache += item
        self.live.update(Panel(Markdown(self.cache), title=title, style="yellow", box=HEAVY))


    def clear(self):
        """
        Clears the Live status and print cache.
        """
        if self.live is not None:
            self.live.stop()
            self.live = None
        super().print(self.cache)
        self.cache = ""

    def print(self, content: str, **kwargs):
        """
       Prints content to the output.

       :param content: The content to print.
       :type content: str
       :param kwargs: Additional keyword arguments to pass to the print method.
       :type kwargs: Any
       """
        self.console.print(content, **kwargs)
        super().print(content, **kwargs)

    def format_json(self, json_obj: str):
        """
       Formats a JSON object.

       :param json_obj: The JSON object to format.
       :type json_obj: str
       :return: The formatted JSON object.
       :rtype: Syntax
       """
        formatted_json = json.dumps(json_obj, indent=2)
        return Syntax(formatted_json, "json", theme="monokai", line_numbers=True)
