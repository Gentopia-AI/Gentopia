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
