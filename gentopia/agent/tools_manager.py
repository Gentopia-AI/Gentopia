from pathlib import Path

from gentopia.config.tools_config import ToolsConfig


class ToolsManager:
    def __init__(self, config):
        if isinstance(config, str) or isinstance(config, Path):
            config = ToolsConfig(file=config)
        elif isinstance(config, list):
            config = ToolsConfig(config=config)
        self.config = config
        self.tools = {}
        self._load_tools()

    def _load_tools(self):
        if self.config is None or self.config.config is None:
            return
        self.tools = self.config.get_tools()

    def generate_worker_prompt(self):
        prompt = "Tools can be one of the following:\n"
        for name in self.tools:
            worker = self.tools[name]
            prompt += f"{worker.name}[input]: {worker.description}\n"
        return prompt + "\n"

    def run(self, name, args):
        if name not in self.tools:
            return "No evidence found"
        tool = self.tools[name]
        return tool.run(args)

    @property
    def cost(self):
        return {name: self.tools[name].cost for name in self.tools}
