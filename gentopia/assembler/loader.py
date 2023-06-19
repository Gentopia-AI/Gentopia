from pathlib import Path
from typing import Any, IO
from gentopia.prompt import *
from gentopia.tools.basetool import BaseTool
import yaml


class Loader(yaml.SafeLoader):
    def __init__(self, stream: IO[Any]) -> None:
        self._root = Path(stream.name).resolve().parent
        super(Loader, self).__init__(stream)
        self.add_constructor("!include", Loader.include)
        self.add_constructor("!prompt", Loader.prompt)
        self.add_constructor("!tool", Loader.tool)

    def include(self, node: yaml.Node) -> Any:
        filename = Path(self.construct_scalar(node))
        if not filename.is_absolute():
            filename = self._root / filename
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

    def prompt(self, node: yaml.Node) -> Any:
        prompt = self.construct_scalar(node)
        prompt_ins = eval(prompt)
        assert isinstance(prompt_ins, PromptTemplate)
        return prompt_ins

    def tool(self, node: yaml.Node) -> Any:
        tool = self.construct_scalar(node)
        tool_ins = eval(tool)
        assert isinstance(tool_ins, BaseTool)
        return tool_ins
