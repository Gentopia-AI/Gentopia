import logging
import os


def enable_log(path: str = "./agent.log", log_level: str= "info" ):
    if path is None:
        path = "./agent.log"
    os.environ["LOG_LEVEL"] = log_level
    os.environ["LOG_PATH"] = path
    logging.basicConfig(level=log_level.upper(), filename=path)



def check_log():
    return os.environ.get("LOG_PATH", None) is not None