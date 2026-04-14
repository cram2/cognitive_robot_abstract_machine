"""PyCRAM adapter boundary for execution and action discovery.

This is the module in llmr that imports or scans PyCRAM packages.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing_extensions import Any, Dict, Protocol, runtime_checkable


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


def discover_action_classes(
    package_root: str = "pycram.robot_plans.actions",
) -> Dict[str, type]:
    """Scan *package_root* and return concrete PyCRAM action classes."""
    try:
        actions_pkg = importlib.import_module(package_root)
    except ImportError:
        return {}

    result: Dict[str, type] = {}
    for _, module_name, _ in pkgutil.walk_packages(
        actions_pkg.__path__, prefix=actions_pkg.__name__ + "."
    ):
        try:
            mod = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    name.endswith("Action")
                    and not inspect.isabstract(obj)
                    and obj.__module__.startswith("pycram")
                ):
                    result[name] = obj
        except Exception:
            continue
    return result
