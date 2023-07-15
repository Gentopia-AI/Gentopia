from gentopia.model.param_model import BaseParamModel
from gentopia.prompt import fewshots

def get_default_client_param_model(model_name: str) -> BaseParamModel:
    """
    Get the default client parameter model.

    Args:
        model_name: The name of the model.

    Returns:
        The default client parameter model.
    """
    return None

def print_tree(obj, indent=0):
    """
    Print the tree structure of an object.

    Args:
        obj: The object to print the tree structure of.
        indent: The indentation level.

    Returns:
        None
    """
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
