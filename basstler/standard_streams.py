"""
The package's logging, which is also how its commands print what they output.

A record at information level is a command's output and goes to standard output; a
warning or an error goes to standard error. Either is written as its bare message,
because a command's caller reads that output as data.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TextIO


@dataclass(eq=False)
class StandardStreamHandler(logging.Handler):
    """
    Writes each record's bare message to standard output, or to standard error from a
    warning upward.

    The stream is looked up when a record is written rather than when the handler is
    made, so a record goes wherever ``sys.stdout`` or ``sys.stderr`` points at that
    moment.

    .. note:: Compared by identity, as every handler is, so that the logging module can
        tell two of them apart.
    """

    message_format: str = "%(message)s"
    """
    How a record is written: its message alone, with no level or logger name.
    """

    def __post_init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(self.message_format))

    @staticmethod
    def stream_for(record: logging.LogRecord) -> TextIO:
        """
        :param record: The record to write.
        :return: The standard stream a record of its level belongs on.
        """
        return sys.stderr if record.levelno >= logging.WARNING else sys.stdout

    def emit(self, record: logging.LogRecord) -> None:
        """
        Write *record*'s message, and a line break, to its standard stream.

        :param record: The record to write.
        """
        self.stream_for(record).write(f"{self.format(record)}\n")

    @staticmethod
    def import_name(module_name: str) -> str:
        """
        A module run with ``python -m`` is named ``__main__``, which no package logger
        is the parent of, so the name is read from the module's spec, which holds the
        name it is imported by however it was run.

        :param module_name: The ``__name__`` of a module that has been loaded.
        :return: The name the module is imported by.
        """
        return sys.modules[module_name].__spec__.name

    @classmethod
    def logger_for(cls, module_name: str) -> logging.Logger:
        """
        A module's logger, under a package logger that writes through this handler.

        The package logger is configured on the first request and left as it is on every
        later one.

        :param module_name: The ``__name__`` of a module inside the package.
        :return: That module's logger.
        """
        package_logger = logging.getLogger(__package__)
        if not any(isinstance(handler, cls) for handler in package_logger.handlers):
            package_logger.addHandler(cls())
            package_logger.setLevel(logging.INFO)
            package_logger.propagate = False
        return logging.getLogger(cls.import_name(module_name))
