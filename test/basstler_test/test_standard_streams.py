"""
Tests for standard_streams.py: the package's logging, which is also what its commands
print through.
"""

import logging
from enum import StrEnum

import basstler
import basstler.standard_streams
from basstler.standard_streams import StandardStreamHandler

MODULE_NAME = basstler.standard_streams.__name__
"""
A module inside the package, whose logger the package's handler writes for.
"""

NOTHING_WRITTEN = ""
"""
What a standard stream holds when no record was written to it.
"""


class LoggedMessage(StrEnum):
    """
    Messages the tests log, one per standard stream they should reach.
    """

    OUTPUT = "some output"
    """
    A command's output, logged at information level.
    """

    FAILURE = "some failure"
    """
    A command's failure, logged as an error.
    """

    @property
    def written_line(self) -> str:
        """
        The message as it appears on its stream: bare, ending in a line break.
        """
        return f"{self}\n"


# %% where a record goes


def test_an_information_record_is_printed_bare_on_standard_output(capsys):
    """
    A command's output is read by its caller, so it carries no level or logger prefix.
    """
    StandardStreamHandler.logger_for(MODULE_NAME).info(LoggedMessage.OUTPUT)
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (
        LoggedMessage.OUTPUT.written_line,
        NOTHING_WRITTEN,
    )


def test_an_error_record_is_printed_bare_on_standard_error(capsys):
    StandardStreamHandler.logger_for(MODULE_NAME).error(LoggedMessage.FAILURE)
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (
        NOTHING_WRITTEN,
        LoggedMessage.FAILURE.written_line,
    )


# %% configuring the package logger once


def test_asking_for_loggers_twice_attaches_one_handler():
    """
    Every module asks for its logger, so a second request must not print every record a
    second time.
    """
    StandardStreamHandler.logger_for(MODULE_NAME)
    StandardStreamHandler.logger_for(MODULE_NAME)
    package_handlers = logging.getLogger(basstler.__name__).handlers
    assert (
        len(
            [
                handler
                for handler in package_handlers
                if isinstance(handler, StandardStreamHandler)
            ]
        )
        == 1
    )
