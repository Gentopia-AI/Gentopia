import json
import logging
from abc import ABC
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from gentopia.model.completion_model import BaseCompletion
from gentopia.output import check_log


class BaseOutput:
    """
        Base class for output handlers.

        Attributes:
        -----------
        logger : logging.Logger
            The logger object to log messages.

        Methods:
        --------
        stop():
            Stop the output.

        update_status(output: str, **kwargs):
            Update the status of the output.

        thinking(name: str):
            Log that a process is thinking.

        done(_all=False):
            Log that the process is done.

        stream_print(item: str):
            Not implemented.

        json_print(item: Dict[str, Any]):
            Log a JSON object.

        panel_print(item: Any, title: str = "Output", stream: bool = False):
            Log a panel output.

        clear():
            Not implemented.

        print(content: str, **kwargs):
            Log arbitrary content.

        format_json(json_obj: str):
            Format a JSON object.

        debug(content: str, **kwargs):
            Log a debug message.

        info(content: str, **kwargs):
            Log an informational message.

        warning(content: str, **kwargs):
            Log a warning message.

        error(content: str, **kwargs):
            Log an error message.

        critical(content: str, **kwargs):
            Log a critical message.
    """

    def __init__(self):
        """
        Initialize the BaseOutput object.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self.logger = logging

    def stop(self):
        """
        Stop the output.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        pass

    def update_status(self, output: str, **kwargs):
        """
        Update the status of the output.

        Parameters:
        -----------
        output : str
            The output message.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.info(output)


    def thinking(self, name: str):
        """
        Log that a process is thinking.

        Parameters:
        -----------
        name : str
            The name of the process.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.info(f"{name} is thinking...")


    def done(self, _all=False):
        """
        Log that the process is done.

        Parameters:
        -----------
        _all : bool, optional
            Not used, defaults to False.

        Returns:
        --------
        None
        """

        if check_log():
            self.logger.info("Done")


    def stream_print(self, item: str):
        """
        Stream print.

        Parameters:
        -----------
        item : str
            Not used.

        Returns:
        --------
        None
        """

        pass

    def json_print(self, item: Dict[str, Any]):
        """
        Log a JSON object.

        Parameters:
        -----------
        item : Dict[str, Any]
            The JSON object to log.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.info(json.dumps(item, indent=2))

    def panel_print(self, item: Any, title: str = "Output", stream: bool = False):
        """
        Log a panel output.

        Parameters:
        -----------
        item : Any
            The item to log.
        title : str, optional
            The title of the panel, defaults to "Output".
        stream : bool, optional
            Not used, defaults to False.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.info('-'*20)
            self.logger.info(item)
            self.logger.info('-' * 20)

    def clear(self):
        """
        Not implemented.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        pass

    def print(self, content: str, **kwargs):
        """
        Log arbitrary content.

        Parameters:
        -----------
        content : str
            The content to log.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.info(content)

    def format_json(self, json_obj: str):
        """
        Format a JSON object.

        Parameters:
        -----------
        json_obj : str
            The JSON object to format.

        Returns:
        --------
        str
            The formatted JSON object.
        """
        formatted_json = json.dumps(json_obj, indent=2)
        return formatted_json


    def debug(self, content: str, **kwargs):
        """
        Log a debug message.

        Parameters:
        -----------
        content : str
            The message to log.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.debug(content, **kwargs)


    def info(self, content: str, **kwargs):
        """
        Log an informational message.

        Parameters:
        -----------
        content : str
            The message to log.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.info(content, **kwargs)


    def warning(self, content: str, **kwargs):
        """
        Log a warning message.

        Parameters:
        -----------
        content : str
            The message to log.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.warning(content, **kwargs)


    def error(self, content: str, **kwargs):
        """
        Log an error message.

        Parameters:
        -----------
        content : str
            The message to log.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.error(content, **kwargs)


    def critical(self, content: str, **kwargs):
        """
        Log a critical message.

        Parameters:
        -----------
        content : str
            The message to log.

        Returns:
        --------
        None
        """
        if check_log():
            self.logger.critical(content, **kwargs)
