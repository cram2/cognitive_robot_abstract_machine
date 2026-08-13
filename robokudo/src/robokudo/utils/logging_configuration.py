"""Logging configuration utilities for Robokudo.

This module provides functionality for configuring Python logging in a ROS environment.
It handles:

* Logger configuration from YAML files
* ROS-Python logging integration
* Custom log formatting
* Per-module log level control

.. note::
   ROS normally disables Python's default logging system. This module provides
   a workaround to enable proper Python logging alongside ROS logging.
"""

import os
import re
import sys
from logging import Formatter, LogRecord, StreamHandler, getLogger

import yaml


class DynamicCompactFormatter(Formatter):
    """Extended `logging.Formatter` that supports dynamic truncation of filenames."""

    def __init__(self, fmt: str = None, datefmt: str = None, style: str = "%"):
        """Initialize formatter with custom format string.

        :param fmt: Format string used to determine line truncation
        :param datefmt: Date format string passed to `super().__init__()`
        :param style: Format style passed to `super().__init__()`
        """
        super().__init__(fmt, datefmt, style)
        self._parse_field_width()

    def _parse_field_width(self) -> None:
        """Parse format string to find filename_line width."""
        # Match %(filename_line)-WIDTHs pattern
        match = re.search(r"%\(filename_line\)-(\d+)s", self._fmt)
        if match:
            self.max_file_line_width = int(match.group(1))
        else:
            self.max_file_line_width = 44  # Default

    def format(self, record: LogRecord) -> str:
        """Format the given log record according to the formatter's format string.

        :param record: Log record to format
        :return: Formatted log message as a string
        """
        filename = os.path.basename(record.filename)
        funcname = record.funcName

        # New format: filename@methodname:lineno
        file_line = f"{filename}/{funcname}:{record.lineno}"

        # Dynamic truncation using parsed width
        if len(file_line) > self.max_file_line_width:
            # Reserve space for @methodname:lineno + "..."
            lineno_part = f":{record.lineno}"
            method_part = f"/{funcname}"
            reserved_len = len(method_part) + len(lineno_part) + 3  # "..."

            max_filename_len = self.max_file_line_width - reserved_len
            if max_filename_len > 0:
                short_filename = filename[: max_filename_len - 3] + "..."
                file_line = f"{short_filename}/{funcname}:{record.lineno}"
            else:
                # Fallback: truncate method name too
                max_method_len = (
                    self.max_file_line_width - len(lineno_part) - 4
                )  # "...@"
                if max_method_len > 0:
                    short_method = funcname[: max_method_len - 3] + "..."
                    file_line = f".../{short_method}:{record.lineno}"
                else:
                    file_line = f"...:{record.lineno}"

        record.filename_line = file_line.ljust(self.max_file_line_width)
        return super().format(record)


def configure_logging(logging_config_file_name: str) -> None:
    """Configure Python logging system with ROS integration.

    Sets up Python logging with:

    * Console output to stdout
    * Custom formatter for detailed log messages
    * Per-module log levels from YAML config
    * ROS logging integration

    The YAML config file should map logger names to logging levels:

    .. code-block:: yaml

        logger_name1: INFO
        logger_name2: DEBUG
        ...

    :param logging_config_file_name: Path to YAML config file
    :raises FileNotFoundError: If config file not found
    :raises yaml.YAMLError: If config file has invalid format

    .. note::
       This is needed because ROS disables standard Python logging.
       See: https://github.com/ros/ros_comm/issues/1384
    """
    console = StreamHandler(stream=sys.stdout)
    # formatter = logging.Formatter(
    #     '%(name)-28s %(levelname)-8s %(asctime)-28s %(filename)-36s:%(lineno)-8s %('
    #     'funcName)-42s %(message)s')
    formatter = DynamicCompactFormatter(
        "%(name)-28s %(levelname)-8s %(asctime)-24s %(filename_line)-50s %(message)s"
    )
    console.setFormatter(formatter)
    getLogger().addHandler(console)
    with open(logging_config_file_name, "r") as yaml_file:
        logging_config = yaml.safe_load(yaml_file)
        for logger_name, level in logging_config.items():
            getLogger(logger_name).setLevel(level)
