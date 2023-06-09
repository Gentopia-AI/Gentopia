from pathlib import Path
from typing import Dict, Any, Union, AnyStr

import yaml

from .loader import Loader


class Config:
    def __init__(self, file: Union[Path, AnyStr] = None):
        pass

    @staticmethod
    def load(path: Union[Path, AnyStr]) -> Dict[AnyStr, Any]:
        # TODO: logging
        try:
            with open(path, "r") as f:
                return yaml.load(f, Loader=Loader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {path} not found")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(e)
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def from_file(path: Union[Path, AnyStr]) -> 'Config':
        config = Config.load(path)
        return config

    @staticmethod
    def from_dict(config: Dict[AnyStr, Any]) -> 'Config':
        return Config(**config)
