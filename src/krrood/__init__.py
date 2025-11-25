import importlib.metadata
import logging

__version__ = importlib.metadata.version("krrood")


logger = logging.Logger("krrood")
logger.setLevel(logging.INFO)
