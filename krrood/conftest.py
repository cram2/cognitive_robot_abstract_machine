"""
Pytest configuration for runs rooted in the :mod:`krrood` package directory.
"""

from __future__ import annotations

import logging

import pytest
from _pytest.doctest import DoctestItem
from typing_extensions import Any, Generator


# %% doctests under live logging


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, Any, None]:
    """
    Keep live logging silent while a doctest runs its examples.

    Printing a record live hands the standard output back to pytest's capture, so an
    example that logs loses the output its doctest compares. The record still appears in
    the doctest's captured log.
    """
    if not isinstance(item, DoctestItem):
        yield
        return
    live_logging_handler = item.config.pluginmanager.get_plugin(
        "logging-plugin"
    ).log_cli_handler
    level_outside_doctest = live_logging_handler.level
    live_logging_handler.setLevel(logging.CRITICAL + 1)
    yield
    live_logging_handler.setLevel(level_outside_doctest)
