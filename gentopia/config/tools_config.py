from gentopia.agent import WORKER_REGISTRY

from gentopia.config.config import Config


class ToolsConfig:
    def __init__(self, file=None, config=None):
        if file is not None:
            self.config = Config.from_file(file)['tools']
        elif config is not None:
            self.config = config

    def get_tools(self):
        assert self.config is not None
        tools = dict()
        for i in self.config:
            if isinstance(i, str):
                tools[i] = WORKER_REGISTRY[i]()
            elif isinstance(i, dict):
                assert len(i) == 1
                name = list(i.keys())[0]
                tools[name] = WORKER_REGISTRY[name](**i[name])
        return tools
