"""
Tests verifying the mock AutoAPI documentation solution.

The verbalization notebook demonstrates hyperlinks using `verbalization_domain.Robot`
and `verbalization_domain.Mission`.  Because these are not real krrood source classes
they will never appear in the Sphinx AutoAPI output.  Instead a static mock HTML page
(`krrood/doc/eql/user/api_mock/autoapi/verbalization_domain/index.html`) carries the
same anchor IDs that `AutoAPIResolver` would generate, and Sphinx copies it into the
build via `html_extra_path`.

Coverage:
- Mock HTML contains every anchor ID that AutoAPIResolver generates for Robot / Mission
- `_config.yml` lists `eql/user/api_mock` in `html_extra_path`
- AutoAPIResolver produces the correct GitHub Pages URL for verbalization_domain classes
- No shadow copy of verbalization_domain.py exists inside test_tmp/
- verbalization_domain.py is importable from the canonical doc/eql/user/ directory
"""

from __future__ import annotations

import dataclasses
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
    AutoAPIResolver,
)
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef

# ── Paths ──────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parents[4]
_DOC_EQL_USER = _PROJECT_ROOT / "krrood" / "doc" / "eql" / "user"
_MOCK_HTML = (
    _DOC_EQL_USER / "api_mock" / "autoapi" / "verbalization_domain" / "index.html"
)
_CONFIG_YML = _PROJECT_ROOT / "krrood" / "doc" / "_config.yml"
_TEST_TMP = _DOC_EQL_USER / "test_tmp"

_GITHUB_PAGES_BASE = "https://cram2.github.io/cognitive_robot_abstract_machine/krrood"


# ── Fixture: import verbalization_domain from the canonical location ───────────


@pytest.fixture(scope="module")
def verbalization_domain():
    """Import verbalization_domain from doc/eql/user/ without polluting sys.path."""
    canonical = str(_DOC_EQL_USER.resolve())
    added = canonical not in sys.path
    if added:
        sys.path.insert(0, canonical)
    try:
        import importlib

        mod = importlib.import_module("verbalization_domain")
        return mod
    finally:
        if added:
            sys.path.remove(canonical)
        # Keep the module in sys.modules so the fixture value stays valid,
        # but remove the transient path entry we added.


# ── Anchor ID extraction ───────────────────────────────────────────────────────


class _AnchorCollector(HTMLParser):
    """Collect all id= attribute values from an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)


def _collect_anchor_ids(html_path: Path) -> set[str]:
    parser = _AnchorCollector()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return parser.ids


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_mock_html_file_exists():
    assert _MOCK_HTML.exists(), (
        f"Mock AutoAPI HTML not found at {_MOCK_HTML}. "
        "It should be committed to the repo."
    )


def test_mock_html_robot_class_anchor(verbalization_domain):
    """Mock HTML contains the class-level anchor for Robot."""
    ids = _collect_anchor_ids(_MOCK_HTML)
    Robot = verbalization_domain.Robot
    resolver = AutoAPIResolver(base_url="http://localhost")
    url = resolver.resolve(SourceRef(owner_type=Robot))
    assert url is not None
    anchor = url.split("#")[1]
    assert (
        anchor in ids
    ), f"Anchor '{anchor}' missing from mock HTML. Found: {sorted(ids)}"


def test_mock_html_robot_name_attribute_anchor(verbalization_domain):
    ids = _collect_anchor_ids(_MOCK_HTML)
    Robot = verbalization_domain.Robot
    resolver = AutoAPIResolver(base_url="http://localhost")
    url = resolver.resolve(SourceRef(owner_type=Robot, attribute="name"))
    assert url is not None
    anchor = url.split("#")[1]
    assert anchor in ids, f"Anchor '{anchor}' missing from mock HTML."


def test_mock_html_robot_battery_attribute_anchor(verbalization_domain):
    ids = _collect_anchor_ids(_MOCK_HTML)
    Robot = verbalization_domain.Robot
    resolver = AutoAPIResolver(base_url="http://localhost")
    url = resolver.resolve(SourceRef(owner_type=Robot, attribute="battery"))
    assert url is not None
    anchor = url.split("#")[1]
    assert anchor in ids, f"Anchor '{anchor}' missing from mock HTML."


def test_mock_html_mission_class_anchor(verbalization_domain):
    ids = _collect_anchor_ids(_MOCK_HTML)
    Mission = verbalization_domain.Mission
    resolver = AutoAPIResolver(base_url="http://localhost")
    url = resolver.resolve(SourceRef(owner_type=Mission))
    assert url is not None
    anchor = url.split("#")[1]
    assert anchor in ids, f"Anchor '{anchor}' missing from mock HTML."


def test_mock_html_mission_assigned_to_attribute_anchor(verbalization_domain):
    ids = _collect_anchor_ids(_MOCK_HTML)
    Mission = verbalization_domain.Mission
    resolver = AutoAPIResolver(base_url="http://localhost")
    url = resolver.resolve(SourceRef(owner_type=Mission, attribute="assigned_to"))
    assert url is not None
    anchor = url.split("#")[1]
    assert anchor in ids, f"Anchor '{anchor}' missing from mock HTML."


def test_mock_html_mission_priority_attribute_anchor(verbalization_domain):
    ids = _collect_anchor_ids(_MOCK_HTML)
    Mission = verbalization_domain.Mission
    resolver = AutoAPIResolver(base_url="http://localhost")
    url = resolver.resolve(SourceRef(owner_type=Mission, attribute="priority"))
    assert url is not None
    anchor = url.split("#")[1]
    assert anchor in ids, f"Anchor '{anchor}' missing from mock HTML."


def test_mock_html_all_dataclass_fields_have_anchors(verbalization_domain):
    """Parametric check: every field of every domain class has an anchor in the mock HTML."""
    ids = _collect_anchor_ids(_MOCK_HTML)
    resolver = AutoAPIResolver(base_url="http://localhost")
    missing = []
    for cls_name in ("Robot", "Mission"):
        cls = getattr(verbalization_domain, cls_name)
        # class-level anchor
        url = resolver.resolve(SourceRef(owner_type=cls))
        anchor = url.split("#")[1]
        if anchor not in ids:
            missing.append(anchor)
        # field-level anchors
        for field in dataclasses.fields(cls):
            url = resolver.resolve(SourceRef(owner_type=cls, attribute=field.name))
            anchor = url.split("#")[1]
            if anchor not in ids:
                missing.append(anchor)
    assert not missing, f"Missing anchors in mock HTML: {missing}"


def test_config_yml_lists_api_mock_in_html_extra_path():
    """_config.yml must declare eql/user/api_mock in html_extra_path."""
    content = _CONFIG_YML.read_text(encoding="utf-8")
    assert "eql/user/api_mock" in content, (
        "_config.yml must list 'eql/user/api_mock' under html_extra_path "
        "so the mock HTML is copied into the Sphinx build."
    )


def test_github_pages_resolver_url_for_robot(verbalization_domain):
    """AutoAPIResolver with the GitHub Pages base URL produces the expected URL for Robot."""
    Robot = verbalization_domain.Robot
    resolver = AutoAPIResolver(base_url=_GITHUB_PAGES_BASE)
    url = resolver.resolve(SourceRef(owner_type=Robot))
    assert url is not None
    assert url.startswith(_GITHUB_PAGES_BASE)
    assert "autoapi/verbalization_domain/index.html" in url
    assert "#verbalization_domain.Robot" in url


def test_github_pages_resolver_url_for_robot_battery(verbalization_domain):
    Robot = verbalization_domain.Robot
    resolver = AutoAPIResolver(base_url=_GITHUB_PAGES_BASE)
    url = resolver.resolve(SourceRef(owner_type=Robot, attribute="battery"))
    assert url is not None
    assert url.endswith("#verbalization_domain.Robot.battery")


def test_github_pages_resolver_url_for_mission(verbalization_domain):
    Mission = verbalization_domain.Mission
    resolver = AutoAPIResolver(base_url=_GITHUB_PAGES_BASE)
    url = resolver.resolve(SourceRef(owner_type=Mission))
    assert url is not None
    assert "#verbalization_domain.Mission" in url


def test_no_shadow_copy_of_verbalization_domain_in_test_tmp():
    """test_tmp/ must not contain verbalization_domain.py — that would shadow the canonical copy."""
    shadow = _TEST_TMP / "verbalization_domain.py"
    assert not shadow.exists(), (
        f"Shadow copy found at {shadow}. "
        "Delete it so the notebook setup cell picks up the canonical verbalization_domain.py."
    )


def test_verbalization_domain_is_importable_from_canonical_path(verbalization_domain):
    """verbalization_domain can be imported and has Robot and Mission classes."""
    assert hasattr(verbalization_domain, "Robot")
    assert hasattr(verbalization_domain, "Mission")


def test_verbalization_domain_robot_is_dataclass(verbalization_domain):
    assert dataclasses.is_dataclass(verbalization_domain.Robot)


def test_verbalization_domain_mission_is_dataclass(verbalization_domain):
    assert dataclasses.is_dataclass(verbalization_domain.Mission)
