"""Tests for :mod:`llmr` public surface."""

from __future__ import annotations


class TestPackagePublicSurface:
    """``llmr.__all__`` exports its advertised public API."""

    def test_top_level_exports(self) -> None:
        import llmr

        expected = {
            "LLMBackend",
            "nl_plan",
            "nl_sequential",
            "resolve_match",
            "resolve_params",
            "LLMActionClassificationFailed",
            "LLMActionRegistryEmpty",
            "LLMProviderNotSupported",
            "LLMSlotFillingFailed",
            "LLMUnresolvedRequiredFields",
        }
        assert expected.issubset(set(llmr.__all__))
        for name in expected:
            assert hasattr(llmr, name), f"llmr missing advertised export {name}"


class TestPycramBridgeSurface:
    """:mod:`llmr.pycram_bridge` exposes the PyCRAM adapter surface."""

    def test_adapter_exports(self) -> None:
        from llmr.pycram_bridge import (
            PycramContext,
            PycramPlanNode,
            discover_action_classes,
            execute_single,
        )

        assert callable(discover_action_classes)
        assert callable(execute_single)
        assert PycramContext is not None
        assert PycramPlanNode is not None


class TestBridgeSurface:
    """:mod:`llmr.bridge` submodules are importable and expose the documented symbols."""

    def test_introspect_exports(self) -> None:
        from llmr.bridge.introspect import (
            ActionSchema,
            FieldKind,
            FieldSpec,
            PycramIntrospector,
        )

        assert FieldKind.ENTITY.name == "ENTITY"
        assert FieldSpec is not None
        assert ActionSchema is not None
        assert PycramIntrospector is not None

    def test_match_reader_exports(self) -> None:
        from llmr.bridge.match_reader import (
            MatchData,
            MatchSlot,
            finalize_match,
            read_match,
            required_match,
            unresolved_required_fields,
            write_slot_value,
        )

        assert MatchData is not None
        assert MatchSlot is not None
        for fn in (
            read_match,
            write_slot_value,
            finalize_match,
            required_match,
            unresolved_required_fields,
        ):
            assert callable(fn)
