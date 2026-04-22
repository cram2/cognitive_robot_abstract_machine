"""PyCRAM adapter boundary for execution and action discovery.

This is the module in llmr that imports or scans PyCRAM packages.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import pkgutil
from typing_extensions import Any, Dict, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class PycramContext(Protocol):
    """Structural match for a PyCRAM context object."""

    query_backend: Any


@runtime_checkable
class PycramPlanNode(Protocol):
    """Structural match for a PyCRAM plan node object."""

    def perform(self) -> None: ...


def execute_single(match: Any, context: Any) -> PycramPlanNode:
    """Wrap PyCRAM's single-action execution factory."""
    from pycram.plans.factories import execute_single as _fn

    return _fn(match, context)


def discover_action_classes() -> Dict[str, type]:
    """Return all concrete PyCRAM action classes rooted at ActionDescription.

    Loads every module under ``pycram.robot_plans.actions`` once so that
    Python registers all subclasses, then uses krrood's recursive_subclasses
    to collect them — no name-suffix heuristics or module filters needed.
    """
    from krrood.utils import recursive_subclasses

    try:
        _pkg = importlib.import_module("pycram.robot_plans.actions")
    except ImportError:
        return {}

    for _, modname, _ in pkgutil.walk_packages(
        _pkg.__path__, prefix=_pkg.__name__ + "."
    ):
        try:
            importlib.import_module(modname)
        except Exception as exc:
            logger.debug("discover_action_classes: skipping %s: %s", modname, exc)

    from pycram.robot_plans.actions.base import ActionDescription

    return {
        cls.__name__: cls
        for cls in recursive_subclasses(ActionDescription)
        if dataclasses.is_dataclass(cls) and not inspect.isabstract(cls)
    }
