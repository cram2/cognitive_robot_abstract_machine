"""
A live page showing SegMind's events as they are detected, served with flask.

Needs segmind's ``dashboard`` extra; loading it without flask raises
:class:`~segmind.exceptions.DashboardNeedsFlask`.
"""

from __future__ import annotations

import importlib.util

from segmind.exceptions import DashboardNeedsFlask, OptionalDependency

if importlib.util.find_spec(OptionalDependency.FLASK) is None:
    raise DashboardNeedsFlask()
