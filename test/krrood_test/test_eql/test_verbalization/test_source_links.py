"""
Tests for source-link hyperlink support in EQL verbalization.

Coverage:
- SourceReference: frozen dataclass, cls-only and cls+attribute forms
- AutoAPIResolver: Sphinx AutoAPI URL structure; for_package() auto-detection
- Formatter.wrap_link: PlainFormatter (no-op), HTMLFormatter (<a>), ANSIFormatter (OSC 8)
- ANSIFormatter OSC 8 detection: enabled / disabled paths
- RoleFragment.source_reference: default None, explicit value
- FragmentRenderer._render_role: link injected when resolver + ref both present
- ParagraphRenderer / HierarchicalRenderer: <a> tags appear in HTML output
- VerbalizationPipeline: html() with link_resolver produces clickable class names
- Verbalizer: source_reference propagation for Variable, Attribute chain, bool Attribute
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Optional
from unittest.mock import patch

import pytest

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.formatter import (
    ANSIFormatter,
    HTMLFormatter,
    PlainFormatter,
)
from krrood.entity_query_language.verbalization.rendering.renderer import (
    HierarchicalRenderer,
    ParagraphRenderer,
)
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
    AutoAPIResolver,
)
from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer

# ── Test fixtures ──────────────────────────────────────────────────────────────


@dataclass
class _Sensor:
    level: int
    active: bool
    name: str


@dataclass
class _SensorChild(_Sensor):
    """Subclass to verify MRO walking (kept for potential future use)."""

    extra: str


class _ConstantResolver:
    """Stub resolver that always returns the same URL, for isolation."""

    def __init__(self, url: str = "http://example.com/source") -> None:
        self._url = url

    def resolve(self, ref: SourceReference) -> Optional[str]:
        return self._url


class _NoneResolver:
    """Stub resolver that always returns None (no link available)."""

    def resolve(self, ref: SourceReference) -> Optional[str]:
        return None


# ── SourceReference ─────────────────────────────────────────────────────────────────


def test_source_ref_cls_only():
    ref = SourceReference(owner_type=_Sensor)
    assert ref.owner_type is _Sensor
    assert ref.attribute is None


def test_source_ref_with_attribute():
    ref = SourceReference(owner_type=_Sensor, attribute="level")
    assert ref.owner_type is _Sensor
    assert ref.attribute == "level"


def test_source_ref_is_frozen():
    ref = SourceReference(owner_type=_Sensor)
    with pytest.raises((AttributeError, TypeError)):
        ref.owner_type = int  # type: ignore[misc]


def test_source_ref_equality():
    assert SourceReference(owner_type=_Sensor) == SourceReference(owner_type=_Sensor)
    assert SourceReference(owner_type=_Sensor, attribute="level") == SourceReference(
        owner_type=_Sensor, attribute="level"
    )
    assert SourceReference(owner_type=_Sensor) != SourceReference(
        owner_type=_Sensor, attribute="level"
    )


# ── AutoAPIResolver ────────────────────────────────────────────────────────────


def test_autoapi_resolver_class_url_structure():
    r = AutoAPIResolver(base_url="https://docs.example.com")
    url = r.resolve(SourceReference(owner_type=_Sensor))
    assert url is not None
    assert url.startswith("https://docs.example.com/autoapi/")
    assert "#" in url


def test_autoapi_resolver_class_anchor_contains_qualname():
    r = AutoAPIResolver(base_url="https://docs.example.com")
    url = r.resolve(SourceReference(owner_type=_Sensor))
    assert url is not None
    anchor = url.split("#")[1]
    assert "_Sensor" in anchor


def test_autoapi_resolver_attribute_anchor_contains_attr():
    r = AutoAPIResolver(base_url="https://docs.example.com")
    url = r.resolve(SourceReference(owner_type=_Sensor, attribute="level"))
    assert url is not None
    anchor = url.split("#")[1]
    assert "level" in anchor


def test_autoapi_resolver_strips_trailing_slash_from_base():
    r1 = AutoAPIResolver(base_url="https://docs.example.com/")
    r2 = AutoAPIResolver(base_url="https://docs.example.com")
    ref = SourceReference(owner_type=_Sensor)
    assert r1.resolve(ref) == r2.resolve(ref)


def test_autoapi_resolver_module_path_uses_slashes():
    r = AutoAPIResolver(base_url="https://docs.example.com")
    url = r.resolve(SourceReference(owner_type=_Sensor))
    assert url is not None
    path_part = url.split("/autoapi/")[1].split("/index.html")[0]
    assert "." not in path_part


def test_autoapi_resolver_no_warning_when_html_root_not_set(caplog):
    """Without html_root, resolve() never logs a warning."""
    import logging

    r = AutoAPIResolver(base_url="https://docs.example.com")
    with caplog.at_level(logging.WARNING):
        r.resolve(SourceReference(owner_type=_Sensor))
    assert not caplog.records


def test_autoapi_resolver_warns_when_local_page_missing(tmp_path):
    """When html_root is set and the AutoAPI page is absent, a WARNING is logged."""
    from unittest.mock import patch as _patch
    import krrood.entity_query_language.verbalization.rendering.source_link_resolver as _slr

    r = AutoAPIResolver(base_url="https://docs.example.com", html_root=tmp_path)
    with _patch.object(_slr._log, "warning") as mock_warn:
        url = r.resolve(SourceReference(owner_type=_Sensor))
    assert url is not None, "resolve() must still return the URL"
    mock_warn.assert_called_once()
    assert "_Sensor" in str(mock_warn.call_args)


def test_autoapi_resolver_no_warning_when_page_exists(tmp_path):
    """When the AutoAPI page exists on disk, no WARNING is logged."""
    from unittest.mock import patch as _patch
    import krrood.entity_query_language.verbalization.rendering.source_link_resolver as _slr

    module_path = _Sensor.__module__.replace(".", "/")
    page = tmp_path / "autoapi" / module_path / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("")
    r = AutoAPIResolver(base_url="https://docs.example.com", html_root=tmp_path)
    with _patch.object(_slr._log, "warning") as mock_warn:
        r.resolve(SourceReference(owner_type=_Sensor))
    mock_warn.assert_not_called()


# ── AutoAPIResolver.for_package ───────────────────────────────────────────────

# Detect at module load whether the krrood docs are built so dependent tests
# can be skipped cleanly in CI environments where Sphinx has not been run.
_krrood_resolver: Optional[AutoAPIResolver] = None
try:
    _krrood_resolver = AutoAPIResolver.for_package("krrood")
    _KRROOD_DOCS_AVAILABLE = True
except (FileNotFoundError, ImportError):
    _KRROOD_DOCS_AVAILABLE = False

_requires_krrood_docs = pytest.mark.skipif(
    not _KRROOD_DOCS_AVAILABLE,
    reason="krrood Sphinx docs not built — run: sphinx-build doc doc/_build/html",
)


@_requires_krrood_docs
def test_autoapi_resolver_for_package_returns_resolver():
    resolver = AutoAPIResolver.for_package("krrood")
    assert isinstance(resolver, AutoAPIResolver)


@_requires_krrood_docs
def test_autoapi_resolver_for_package_base_url_has_krrood_docs():
    resolver = AutoAPIResolver.for_package("krrood")
    assert "localhost" in resolver.base_url
    assert "krrood" in resolver.base_url
    assert "doc/_build/html" in resolver.base_url


@_requires_krrood_docs
def test_autoapi_resolver_for_package_custom_port():
    resolver = AutoAPIResolver.for_package("krrood", port=8080)
    assert "localhost:8080" in resolver.base_url


@_requires_krrood_docs
def test_autoapi_resolver_for_package_sets_html_root():
    resolver = AutoAPIResolver.for_package("krrood")
    assert resolver.html_root is not None
    assert resolver.html_root.is_dir()


def test_autoapi_resolver_for_package_unknown_package_raises():
    with pytest.raises(ImportError):
        AutoAPIResolver.for_package("_nonexistent_pkg_xyz")


def test_autoapi_resolver_for_package_missing_docs_raises(tmp_path):
    """for_package raises FileNotFoundError when doc/_build/html is absent."""
    import types
    import krrood.entity_query_language.verbalization.rendering.source_link_resolver as _slr
    from unittest.mock import patch as _patch

    pkg_src = tmp_path / "fakepkg" / "src" / "fakepkg"
    pkg_src.mkdir(parents=True)
    init = pkg_src / "__init__.py"
    init.write_text("")
    (tmp_path / "fakepkg" / "pyproject.toml").write_text("")

    fake_mod = types.ModuleType("fakepkg")
    fake_mod.__file__ = str(init)

    with _patch.object(_slr, "importlib") as mock_importlib:
        mock_importlib.import_module.return_value = fake_mod
        with pytest.raises(FileNotFoundError, match="_build"):
            AutoAPIResolver.for_package("fakepkg")


@_requires_krrood_docs
def test_autoapi_resolver_for_package_url_for_real_class():
    """URL for a real krrood class has the correct module-path structure."""
    from krrood.entity_query_language.query.query import Query

    url = _krrood_resolver.resolve(SourceReference(owner_type=Query))
    assert url is not None
    assert "krrood/entity_query_language/query/query/index.html" in url
    assert "#krrood.entity_query_language.query.query.Query" in url


@_requires_krrood_docs
def test_autoapi_resolver_for_package_html_file_exists():
    """The HTML page for a real krrood class exists on disk after docs build."""
    from krrood.entity_query_language.query.query import Query
    from pathlib import Path
    import importlib as _il, inspect as _ins

    pkg_file = Path(_ins.getfile(_il.import_module("krrood"))).resolve()
    for parent in pkg_file.parents:
        if (parent / "pyproject.toml").exists():
            module_path = Query.__module__.replace(".", "/")
            html_file = (
                parent
                / "doc"
                / "_build"
                / "html"
                / "autoapi"
                / module_path
                / "index.html"
            )
            assert html_file.exists(), f"Expected autoapi page at {html_file}"
            return
    pytest.fail("krrood package root not found")


@_requires_krrood_docs
def test_autoapi_resolver_for_package_end_to_end_html():
    """Full pipeline with for_package() produces <a> tags pointing at the local docs."""
    x = variable(EQLVerbalizer, [])
    text = VerbalizationPipeline.html(link_resolver=_krrood_resolver).verbalize(
        an(entity(x))
    )
    assert "localhost" in text
    assert "EQLVerbalizer" in text


# ── Formatter.wrap_link ────────────────────────────────────────────────────────


def test_plain_formatter_wrap_link_is_noop():
    f = PlainFormatter()
    assert f.wrap_link("Robot", "http://example.com") == "Robot"


def test_html_formatter_wrap_link_produces_anchor():
    f = HTMLFormatter()
    result = f.wrap_link("Robot", "http://example.com")
    assert (
        result
        == '<a target="_blank" rel="noopener" href="http://example.com">Robot</a>'
    )


def test_html_formatter_wrap_link_preserves_inner_markup():
    f = HTMLFormatter()
    colored = '<span style="color:cornflowerblue">Robot</span>'
    result = f.wrap_link(colored, "http://example.com")
    assert 'href="http://example.com"' in result
    assert "cornflowerblue" in result


def test_ansi_formatter_wrap_link_osc8_enabled():
    f = ANSIFormatter()
    object.__setattr__(
        f, "_hyperlinks_enabled", True
    )  # force-enable for test isolation
    result = f.wrap_link("Robot", "http://example.com")
    assert "\033]8;;http://example.com\033\\" in result
    assert "Robot" in result
    assert result.endswith("\033]8;;\033\\")


def test_ansi_formatter_wrap_link_osc8_disabled_returns_text():
    f = ANSIFormatter()
    object.__setattr__(f, "_hyperlinks_enabled", False)
    result = f.wrap_link("Robot", "http://example.com")
    assert result == "Robot"


# ── ANSIFormatter OSC 8 detection ─────────────────────────────────────────────


def test_ansi_formatter_detects_gnome_terminal_via_vte_version():
    with patch.dict("os.environ", {"VTE_VERSION": "6800"}, clear=False):
        f = ANSIFormatter()
        assert f._hyperlinks_enabled is True


def test_ansi_formatter_detects_vscode_terminal():
    with patch.dict("os.environ", {"TERM_PROGRAM": "vscode"}, clear=False):
        f = ANSIFormatter()
        assert f._hyperlinks_enabled is True


def test_ansi_formatter_detects_kitty():
    with patch.dict("os.environ", {"TERM": "xterm-kitty"}, clear=False):
        f = ANSIFormatter()
        assert f._hyperlinks_enabled is True


def test_ansi_formatter_unknown_terminal_disables_hyperlinks():
    env = {"VTE_VERSION": "", "TERM_PROGRAM": "unknown-term", "TERM": "xterm"}
    with patch.dict("os.environ", env, clear=False):
        f = ANSIFormatter()
        assert f._hyperlinks_enabled is False


# ── RoleFragment.source_reference ────────────────────────────────────────────────────


def test_role_fragment_source_ref_defaults_to_none():
    frag = RoleFragment(text="Robot", role=SemanticRole.VARIABLE)
    assert frag.source_reference is None


def test_role_fragment_accepts_source_ref():
    ref = SourceReference(owner_type=_Sensor)
    frag = RoleFragment(text="Sensor", role=SemanticRole.VARIABLE, source_reference=ref)
    assert frag.source_reference is ref


# ── FragmentRenderer._render_role ─────────────────────────────────────────────


def test_paragraph_renderer_injects_link_when_resolver_and_ref_present():
    resolver = _ConstantResolver("http://example.com")
    r = ParagraphRenderer(HTMLFormatter(), resolver)
    ref = SourceReference(owner_type=_Sensor)
    frag = RoleFragment(text="Sensor", role=SemanticRole.VARIABLE, source_reference=ref)
    result = r.render(frag)
    assert 'href="http://example.com"' in result
    assert "Sensor" in result


def test_paragraph_renderer_no_link_when_no_resolver():
    r = ParagraphRenderer(HTMLFormatter())
    ref = SourceReference(owner_type=_Sensor)
    frag = RoleFragment(text="Sensor", role=SemanticRole.VARIABLE, source_reference=ref)
    result = r.render(frag)
    assert "<a " not in result
    assert "Sensor" in result


def test_paragraph_renderer_no_link_when_resolver_returns_none():
    r = ParagraphRenderer(HTMLFormatter(), _NoneResolver())
    ref = SourceReference(owner_type=_Sensor)
    frag = RoleFragment(text="Sensor", role=SemanticRole.VARIABLE, source_reference=ref)
    result = r.render(frag)
    assert "<a " not in result


def test_paragraph_renderer_no_link_when_no_source_ref():
    resolver = _ConstantResolver("http://example.com")
    r = ParagraphRenderer(HTMLFormatter(), resolver)
    frag = RoleFragment(text="Sensor", role=SemanticRole.VARIABLE)
    result = r.render(frag)
    assert "<a " not in result


def test_hierarchical_renderer_injects_link():
    resolver = _ConstantResolver("http://example.com")
    r = HierarchicalRenderer(HTMLFormatter(), resolver)
    ref = SourceReference(owner_type=_Sensor)
    block = BlockFragment(
        header=RoleFragment(
            text="Sensor", role=SemanticRole.VARIABLE, source_reference=ref
        ),
        items=[WordFragment("some condition")],
    )
    result = r.render(block)
    assert 'href="http://example.com"' in result


# ── Pipeline: html() with link_resolver ───────────────────────────────────────


def test_pipeline_html_with_resolver_links_variable_name():
    r = variable(_Sensor, [])
    resolver = _ConstantResolver("http://example.com/sensor")
    text = VerbalizationPipeline.html(link_resolver=resolver).verbalize(an(entity(r)))
    assert 'href="http://example.com/sensor"' in text
    assert "_Sensor" in text


def test_pipeline_html_without_resolver_no_anchor_tags():
    r = variable(_Sensor, [])
    text = VerbalizationPipeline.html().verbalize(an(entity(r)))
    assert "<a " not in text


def test_pipeline_ansi_with_resolver_and_osc8_emits_escape():
    r = variable(_Sensor, [])
    resolver = _ConstantResolver("http://example.com/sensor")
    with patch.dict("os.environ", {"VTE_VERSION": "6800"}, clear=False):
        text = VerbalizationPipeline.ansi(link_resolver=resolver).verbalize(
            an(entity(r))
        )
    assert "\033]8;;" in text


def test_pipeline_ansi_with_resolver_no_osc8_logs_warning_and_no_osc8():
    import krrood.entity_query_language.verbalization.pipeline as pipeline_mod
    from unittest.mock import patch as _patch

    r = variable(_Sensor, [])
    resolver = _ConstantResolver("http://example.com/sensor")
    env = {"VTE_VERSION": "", "TERM_PROGRAM": "unknown", "TERM": "xterm"}
    with patch.dict("os.environ", env, clear=False):
        with _patch.object(pipeline_mod._log, "warning") as mock_warn:
            text = VerbalizationPipeline.ansi(link_resolver=resolver).verbalize(
                an(entity(r))
            )
    assert "\033]8;;" not in text
    mock_warn.assert_called_once()
    assert "OSC 8" in mock_warn.call_args[0][0]


# ── Verbalizer source_reference propagation ─────────────────────────────────────────


def _collect_source_refs(fragment: Fragment) -> list[SourceReference]:
    """Recursively collect all non-None SourceReference values from a fragment tree."""
    match fragment:
        case RoleFragment(source_reference=ref) if ref is not None:
            return [ref]
        case PhraseFragment(parts=parts):
            return [r for p in parts for r in _collect_source_refs(p)]
        case BlockFragment(header=header, items=items):
            result = _collect_source_refs(header) if header else []
            return result + [r for item in items for r in _collect_source_refs(item)]
        case _:
            return []


def test_variable_fragment_carries_source_ref_for_its_type():
    x = variable(_Sensor, [])
    frag = EQLVerbalizer().build(x)
    refs = _collect_source_refs(frag)
    assert any(r.owner_type is _Sensor and r.attribute is None for r in refs)


def test_attribute_fragment_carries_source_ref_with_attribute_name():
    x = variable(_Sensor, [])
    frag = EQLVerbalizer().build(x.level > 5)
    refs = _collect_source_refs(frag)
    assert any(r.owner_type is _Sensor and r.attribute == "level" for r in refs)


def test_bool_attribute_chain_carries_source_ref():
    x = variable(_Sensor, [])
    frag = EQLVerbalizer().build(x.active)
    refs = _collect_source_refs(frag)
    assert any(r.owner_type is _Sensor and r.attribute == "active" for r in refs)


def test_comparator_fragment_has_both_class_and_attr_refs():
    x = variable(_Sensor, [])
    frag = EQLVerbalizer().build(x.level > 0)
    refs = _collect_source_refs(frag)
    class_refs = [r for r in refs if r.owner_type is _Sensor and r.attribute is None]
    attr_refs = [r for r in refs if r.owner_type is _Sensor and r.attribute == "level"]
    assert class_refs, "Expected a SourceReference for the _Sensor class"
    assert attr_refs, "Expected a SourceReference for _Sensor.level"


# ── VerbalizationPipeline.display / display_fragment ──────────────────────────


def test_display_opens_browser_outside_jupyter():
    """Outside Jupyter, display() writes a temp HTML file and calls webbrowser.open."""
    from unittest.mock import patch as _patch
    import krrood.entity_query_language.verbalization.pipeline as pipeline_mod

    x = variable(_Sensor, [])
    pipeline = VerbalizationPipeline.html(link_resolver=_ConstantResolver())
    with _patch.object(pipeline_mod, "_is_ipython", return_value=False):
        with _patch("webbrowser.open") as mock_open:
            pipeline.display(an(entity(x)))

    mock_open.assert_called_once()
    opened_url = mock_open.call_args[0][0]
    assert opened_url.startswith("file://")
    assert opened_url.endswith(".html")


def test_display_writes_full_html_page_to_temp_file():
    """The temp file written by display() is a complete HTML document."""
    from unittest.mock import patch as _patch
    import krrood.entity_query_language.verbalization.pipeline as pipeline_mod

    x = variable(_Sensor, [])
    pipeline = VerbalizationPipeline.html()
    captured_path: list[str] = []

    def fake_open(url: str) -> None:
        captured_path.append(url.removeprefix("file://"))

    with _patch.object(pipeline_mod, "_is_ipython", return_value=False):
        with _patch("webbrowser.open", side_effect=fake_open):
            pipeline.display(an(entity(x)))

    assert captured_path, "webbrowser.open was not called"
    path = Path(captured_path[0])
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "_Sensor" in content
    path.unlink()


def test_display_in_jupyter_calls_ipython_display():
    """Inside a Jupyter session, display() calls IPython.display.HTML."""
    from unittest.mock import MagicMock, patch as _patch
    import krrood.entity_query_language.verbalization.pipeline as pipeline_mod

    x = variable(_Sensor, [])
    pipeline = VerbalizationPipeline.html(link_resolver=_ConstantResolver())
    mock_html_cls = MagicMock()
    mock_ipython_display = MagicMock()
    with _patch.object(pipeline_mod, "_is_ipython", return_value=True):
        with _patch.dict(
            "sys.modules",
            {
                "IPython.display": MagicMock(
                    display=mock_ipython_display, HTML=mock_html_cls
                )
            },
        ):
            pipeline.display(an(entity(x)))

    mock_html_cls.assert_called_once()
    html_arg = mock_html_cls.call_args[0][0]
    assert "_Sensor" in html_arg
    mock_ipython_display.assert_called_once()


def test_display_fragment_works_like_display():
    """display_fragment() accepts a pre-built fragment instead of an expression."""
    from unittest.mock import patch as _patch
    import krrood.entity_query_language.verbalization.pipeline as pipeline_mod

    x = variable(_Sensor, [])
    fragment = EQLVerbalizer().build(an(entity(x)))
    pipeline = VerbalizationPipeline.html()

    with _patch.object(pipeline_mod, "_is_ipython", return_value=False):
        with _patch("webbrowser.open") as mock_open:
            pipeline.display_fragment(fragment)

    mock_open.assert_called_once()


def test_display_html_page_has_dark_background_style():
    """The browser-fallback page includes the dark-background stylesheet."""
    from unittest.mock import patch as _patch
    import krrood.entity_query_language.verbalization.pipeline as pipeline_mod

    x = variable(_Sensor, [])
    pipeline = VerbalizationPipeline.html()
    captured_path: list[str] = []

    def fake_open(url: str) -> None:
        captured_path.append(url.removeprefix("file://"))

    with _patch.object(pipeline_mod, "_is_ipython", return_value=False):
        with _patch("webbrowser.open", side_effect=fake_open):
            pipeline.display(an(entity(x)))

    content = Path(captured_path[0]).read_text(encoding="utf-8")
    assert "background" in content
    Path(captured_path[0]).unlink()
