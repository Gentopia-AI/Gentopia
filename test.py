from gentopia.config.agent_config import AgentConfig
from gentopia.config.config import Config


def print_tree(obj, indent=0):
    for attr in dir(obj):
        if not attr.startswith('_'):
            value = getattr(obj, attr)
            if not callable(value):
                if not isinstance(value, dict) and not isinstance(value, list):
                    print('|   ' * indent + '|--', f'{attr}: {value}')
                else:
                    if not value:
                        print('|   ' * indent + '|--', f'{attr}: {value}')
                    print('|   ' * indent + '|--', f'{attr}:')
                if hasattr(value, '__dict__'):
                    print_tree(value, indent + 1)
                elif isinstance(value, list):
                    for item in value:
                        print_tree(item, indent + 1)
                elif isinstance(value, dict):
                    for key, item in value.items():
                        print_tree(item, indent + 1)


if __name__ == '__main__':
    config = Config.load('config.yaml')
    print(config)
    # exit(0)
    agent = AgentConfig(file='config.yaml').get_agent()
    print(agent.run("1+1=?"))
    # print(agent.plugins)
