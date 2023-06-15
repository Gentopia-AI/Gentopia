from pathlib import Path

from gentopia.config.agent_config import AgentConfig


class PluginManager:
    def __init__(self, config):
        if isinstance(config, str) or isinstance(config, Path):
            config = AgentConfig(file=config)
        elif isinstance(config, list):
            config = AgentConfig(config=config)
        config.get_agent()
        self.config = config
        self.plugins = self.config.plugins

    def generate_worker_prompt(self):
        prompt = "Tools can be one of the following:\n"
        for name in self.plugins:
            worker = self.plugins[name]
            prompt += f"{worker.name}[input]: {worker.description}\n"
        return prompt + "\n"

    def run(self, name, *args, **kwargs):
        if name not in self.plugins:
            return "No evidence found"
        plugin = self.plugins[name]
        return plugin.run(*args, **kwargs)

    def __call__(self, plugin, *args, **kwargs):
        self.run(plugin, *args, **kwargs)

    # @property
    # def cost(self):
    #     return {name: self.plugins[name].cost for name in self.tools}
