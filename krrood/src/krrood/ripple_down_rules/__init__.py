__version__ = "0.7.2"

import logging
import sys

logger = logging.Logger("rdr")
logger.setLevel(logging.INFO)

try:
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
except ImportError:
    app = None


# Trigger patch
try:
    from .predicates import *
    from .datastructures.tracked_object import TrackedObjectMixin
    from .datastructures.dataclasses import CaseQuery
    from .rdr_decorators import RDRDecorator
    from .rdr import MultiClassRDR, SingleClassRDR, GeneralRDR
    import ripple_down_rules_meta._apply_overrides
except ImportError:
    pass