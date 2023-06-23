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
    def __init__(self):
        self.console: Console = Console()
        self.status_stack: List[str] = []
        self.status: Optional[Status] = None
        self.live: Optional[Live] = None
        self.cache: str = ""

    def stop(self):
        if self.status is not None:
            self.status.stop()

    def update_status(self, output: str, style: str = "[bold green]"):
        self.status_stack.append(output)
        if self.status is not None:
            self.status.update(f"{style}{output}")
        else:
            self.status = self.console.status(f"{style}{output}")
        self.status.start()

    def thinking(self, name: str):
        self.status_stack.append(name)
        if self.status is not None:
            self.status.update(f"[bold green]{name} is thinking...")
        else:
            self.status = self.console.status(f"[bold green]{name} is thinking...")
        self.status.start()

    def done(self, _all=False):
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

    def stream_print(self, item: str):
        self.console.print(item, end="")

    def json_print(self, item: Dict[str, Any]):
        self.console.print_json(data=item)

    def panel_print(self, item: Any, title: str = "Output", stream: bool = False):
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
        if self.live is not None:
            self.live.stop()
            self.live = None
        self.cache = ""

    def print(self, content: str, **kwargs):
        self.console.print(content, **kwargs)

    def format_json(self, json_obj: str):
        formatted_json = json.dumps(json_obj, indent=2)
        return Syntax(formatted_json, "json", theme="monokai", line_numbers=True)
