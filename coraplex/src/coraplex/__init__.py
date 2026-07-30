import sys
import logging

from ._version import __version__

format = "%(levelname)s:%(filename)s::%(lineno)s %(funcName)s %(message)s"
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
