__version__ = "0.7.2"

import logging

logger = logging.Logger("rdr")
logger.setLevel(logging.INFO)


# Re-exports used by generated RDR Python files (``from krrood.ripple_down_rules import *``).
from .datastructures.dataclasses import CaseQuery  # noqa: E402, F401
from .rdr_decorators import RDRDecorator  # noqa: E402, F401
from .rdr import MultiClassRDR, SingleClassRDR, GeneralRDR  # noqa: E402, F401
