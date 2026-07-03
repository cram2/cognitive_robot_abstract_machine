import importlib.metadata
import logging

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

logger = logging.getLogger("krrood")
logger.setLevel(logging.INFO)
