import sys
import logging

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

format = "%(levelname)s:%(filename)s::%(lineno)s %(funcName)s %(message)s"
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
