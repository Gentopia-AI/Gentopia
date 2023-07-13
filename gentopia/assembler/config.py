from pathlib import Path
from typing import Dict, Any, Union, AnyStr
import logging
import yaml

from .loader import Loader

# Setup the logging



class Config:
    """
        A class for loading and creating configuration dictionaries from files or dictionaries.

    """
    @staticmethod
    def load(path: Union[Path, AnyStr]) -> Dict[AnyStr, Any]:
        """
           Load a configuration dictionary from a YAML file.

           :param path: The path to the configuration file.
           :type path: Union[Path, AnyStr]
           :raises FileNotFoundError: If the file is not found.
           :raises yaml.YAMLError: If a YAML error occurred while loading the file.
           :raises Exception: If an unexpected error occurred.
           :return: A dictionary containing the configuration.
           :rtype: Dict[AnyStr, Any]
       """
        # Logging the start of the loading process
        logging.info(f"Starting to load configuration from {path}")
        try:
            with open(path, "r") as f:
                config = yaml.load(f, Loader=Loader)
                logging.info(f"Successfully loaded configuration from {path}")
                return config
        except FileNotFoundError:
            logging.error(f"Config file {path} not found")
            raise FileNotFoundError(f"Config file {path} not found")
        except yaml.YAMLError as e:
            logging.error(f"YAML error occurred while loading the configuration: {e}")
            raise yaml.YAMLError(e)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise Exception(e)

    @staticmethod
    def from_file(path: Union[Path, AnyStr]) -> Dict[AnyStr, Any]:
        """
            Create a configuration dictionary from a YAML file.

            :param path: The path to the configuration file.
            :type path: Union[Path, AnyStr]
            :return: A dictionary containing the configuration.
            :rtype: Dict[AnyStr, Any]
        """
        logging.info(f"Creating Config from file: {path}")
        config = Config.load(path)
        return config

    @staticmethod
    def from_dict(config: Dict[AnyStr, Any]) -> Dict[AnyStr, Any]:
        """
            Create a configuration dictionary from a Python dictionary.

            :param config: A dictionary containing configuration parameters.
            :type config: Dict[AnyStr, Any]
            :return: A dictionary containing the configuration.
            :rtype: Dict[AnyStr, Any]
        """
        logging.info(f"Creating Config from dictionary")
        return Config(**config)
