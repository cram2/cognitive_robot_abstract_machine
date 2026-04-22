"""Shared fixtures for the bridge test package."""

from __future__ import annotations

import pytest

from llmr.bridge.introspect import PycramIntrospector


@pytest.fixture
def introspector() -> PycramIntrospector:
    """Return a fresh :class:`PycramIntrospector` for each test."""
    return PycramIntrospector()
