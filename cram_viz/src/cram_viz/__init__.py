"""
Browser-based visualization for the CRAM architecture.

.. note:: Configuring logging is left to the entry points (:func:`cram_viz.server.main`
   and friends). Calling :func:`logging.basicConfig` here would install a root handler
   at import time, which silently turns their own ``basicConfig`` calls into no-ops.
"""

import logging

__version__ = "1.0.0"


def get_logger(name: str) -> logging.Logger:
    """
    A named logger that propagates to the root logger's handlers.

    Some libraries (for example ROS 2's ``launch`` package) call
    :func:`logging.setLoggerClass` at import time with a class whose
    constructor forces ``propagate = False``. Since :func:`logging.getLogger`
    reuses that class for every logger created afterwards, an unrelated import
    elsewhere in the process can silently stop ``cram_viz``'s own log records
    from ever reaching a handler. Setting ``propagate`` explicitly overrides
    whatever the ambient logger class did.

    :param name: the logger's name, typically a module's ``__name__``
    """
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger


logger = get_logger(__name__)
