"""Tests for pycram_bridge.adapter — action discovery and protocols.

Coverage target: 70% (6 tests covering discover_action_classes and protocol protocols).
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from llmr.pycram_bridge.adapter import (
    PycramContext,
    PycramPlanNode,
    discover_action_classes,
)


class TestDiscoverActionClasses:
    """discover_action_classes() — action class discovery."""

    def test_returns_dict(self) -> None:
        """Returns a dict mapping class name → class object."""
        result = discover_action_classes()
        assert isinstance(result, dict)

    def test_returns_empty_when_pycram_not_installed(self, monkeypatch) -> None:
        """Returns empty dict when pycram is not installed."""
        # Monkeypatch importlib.import_module to raise ImportError for pycram
        import importlib
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if "pycram" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("importlib.import_module", mock_import)
        result = discover_action_classes(package_root="pycram.robot_plans.actions")
        assert result == {}

    def test_discovers_action_classes_with_action_suffix(self) -> None:
        """Only classes ending with 'Action' are included."""
        result = discover_action_classes()
        # All keys should end with 'Action'
        for name in result.keys():
            assert name.endswith("Action"), f"{name} does not end with 'Action'"

    def test_discovers_non_abstract_classes_only(self) -> None:
        """Only non-abstract classes are included."""
        result = discover_action_classes()
        # All discovered classes should be concrete (not abstract)
        import inspect
        for cls in result.values():
            assert not inspect.isabstract(cls), f"{cls} is abstract"

    def test_discovers_only_pycram_module_classes(self) -> None:
        """Only classes from pycram modules are included."""
        result = discover_action_classes()
        for cls in result.values():
            assert cls.__module__.startswith("pycram"), (
                f"{cls.__module__} does not start with 'pycram'"
            )


class TestPycramContextProtocol:
    """PycramContext protocol — structural type check."""

    def test_object_with_query_backend_satisfies_protocol(self) -> None:
        """Object with query_backend attribute satisfies PycramContext."""
        obj = SimpleNamespace(query_backend=None)
        assert isinstance(obj, PycramContext)

    def test_object_without_query_backend_does_not_satisfy(self) -> None:
        """Object without query_backend does not satisfy PycramContext."""
        obj = SimpleNamespace(other_field=None)
        assert not isinstance(obj, PycramContext)

    def test_real_context_would_satisfy_protocol(self) -> None:
        """Typical PyCRAM context structure satisfies protocol."""
        # Simulate a PyCRAM context: has query_backend attribute
        mock_context = SimpleNamespace(
            query_backend=None, world=None, other_fields={}
        )
        assert isinstance(mock_context, PycramContext)


class TestPycramPlanNodeProtocol:
    """PycramPlanNode protocol — structural type check."""

    def test_object_with_perform_satisfies_protocol(self) -> None:
        """Object with perform() method satisfies PycramPlanNode."""
        obj = SimpleNamespace(perform=lambda: None)
        assert isinstance(obj, PycramPlanNode)

    def test_object_without_perform_does_not_satisfy(self) -> None:
        """Object without perform() does not satisfy PycramPlanNode."""
        obj = SimpleNamespace(other_method=lambda: None)
        assert not isinstance(obj, PycramPlanNode)

    def test_class_with_perform_method_satisfies(self) -> None:
        """Class defining perform() method satisfies protocol."""

        class FakePlanNode:
            def perform(self) -> None:
                pass

        node = FakePlanNode()
        assert isinstance(node, PycramPlanNode)
