"""
Tests that the plugin behind ``@pytest.mark.flaky`` is actually installed.

The mark is supplied by ``pytest-rerunfailures``, which has to be declared as a dependency
for it to do anything. Undeclared, pytest treats it as an unknown mark: a test marked flaky
is collected and run exactly once, so it fails the very run the mark was added to survive.
Nothing in the marked test's own file reveals that -- the mark is spelled identically either
way, and the only outward sign is an unknown-mark warning among the hundreds a run prints.

What the mark then *does* is the plugin's own contract and is not re-tested here.
"""

from __future__ import annotations

import pytest

RERUN_PLUGIN_NAME = "rerunfailures"
"""
Name ``pytest-rerunfailures`` registers itself under in pytest's plugin manager.
"""


def test_the_plugin_supplying_the_flaky_mark_is_installed(pytestconfig: pytest.Config):
    """
    ``pytest-rerunfailures`` is available to this run.

    A failure here means every ``@pytest.mark.flaky`` in the repository is inert and the
    tests wearing it are running unprotected.
    """
    assert pytestconfig.pluginmanager.hasplugin(RERUN_PLUGIN_NAME)
