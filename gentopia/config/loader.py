from pathlib import Path
from typing import Any, IO

import yaml


class Loader(yaml.SafeLoader):
    def __init__(self, stream: IO[Any]) -> None:
        self._root = Path(stream.name).resolve().parent
        super(Loader, self).__init__(stream)
        self.add_constructor("!include", Loader.include)
        self.add_constructor("!prompt", Loader.prompt)

    def include(self, node: yaml.Node) -> Any:
        filename = Path(self.construct_scalar(node))
        if not filename.is_absolute():
            filename = self._root / filename
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

    def prompt(self, node: yaml.Node) -> Any:
        prompt = self.construct_scalar(node)
        return eval(prompt)
