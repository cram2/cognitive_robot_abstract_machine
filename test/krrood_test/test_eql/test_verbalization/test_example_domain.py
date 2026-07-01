"""
Tests for the verbalization documentation's example domain.

The verbalization user guide queries over a small example domain (``Robot``, ``Mission``,
``BankTransaction``, …).  Those classes live in
:mod:`krrood.entity_query_language.verbalization.example_domain` — a *real* module under
``src`` — precisely so Sphinx AutoAPI documents them like any other module and the
:class:`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.AutoAPIResolver`
resolves the rendered hyperlinks to the generated AutoAPI pages (on GitHub Pages and in a
locally built docs tree) with no hand-maintained mock API.

Coverage:
- the example classes are importable dataclasses from the real module;
- ``AutoAPIResolver`` builds URLs into the module's *real* AutoAPI page (class + field anchors),
  for the GitHub Pages base URL;
- the retired mock apparatus is gone (no ``api_mock`` HTML, no ``verbalization_domain.py``,
  no ``html_extra_path`` entry).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from krrood.entity_query_language.verbalization import example_domain
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
    AutoAPIResolver,
)

_PROJECT_ROOT = Path(__file__).parents[4]
_DOC_EQL_USER = _PROJECT_ROOT / "krrood" / "doc" / "eql" / "user"
_CONFIG_YML = _PROJECT_ROOT / "krrood" / "doc" / "_config.yml"

_GITHUB_PAGES_BASE = "https://cram2.github.io/cognitive_robot_abstract_machine/krrood"
_MODULE = "krrood.entity_query_language.verbalization.example_domain"
_MODULE_PATH = _MODULE.replace(".", "/")

# Every example class the user guide uses, so a rename/removal trips this test. Mirrors the
# canonical grouped list in the ``example_domain`` module docstring — keep the two in sync.
_EXAMPLE_CLASSES = [
    "Robot",
    "Mission",
    "Task",
    "Worker",
    "AmountDetails",
    "BankTransaction",
    "Employee",
    "Location",
    "IsReachable",
    "Department",
    "StaffMember",
    "WorksIn",
    "Bird",
    "LoveBirds",
    "BirdView",
    "StrongLoveBird",
    "WeakLoveBird",
    "Handle",
    "Container",
    "FixedConnection",
    "PrismaticConnection",
    "Drawer",
    "Cabinet",
]


def test_all_example_classes_present_and_dataclasses():
    """Each documented example class exists in the real module and is a dataclass."""
    for name in _EXAMPLE_CLASSES:
        cls = getattr(example_domain, name, None)
        assert cls is not None, f"{name} missing from {_MODULE}"
        assert dataclasses.is_dataclass(cls), f"{name} is not a dataclass"


def test_classes_belong_to_the_real_module():
    """Their ``__module__`` is the real ``src`` module — what AutoAPI documents and the
    resolver keys URLs off (not ``__main__`` or a doc-local module)."""
    for name in _EXAMPLE_CLASSES:
        assert getattr(example_domain, name).__module__ == _MODULE


def test_resolver_builds_real_autoapi_url_for_class():
    """The GitHub Pages resolver points a class at the module's real AutoAPI page."""
    resolver = AutoAPIResolver(base_url=_GITHUB_PAGES_BASE)
    url = resolver.resolve(SourceReference(owner_type=example_domain.Robot))
    assert url is not None
    assert url.startswith(_GITHUB_PAGES_BASE)
    assert f"autoapi/{_MODULE_PATH}/index.html" in url
    assert url.endswith(f"#{_MODULE}.Robot")


def test_resolver_builds_real_autoapi_url_for_fields():
    """Field references resolve to the field anchor on the same real AutoAPI page."""
    resolver = AutoAPIResolver(base_url=_GITHUB_PAGES_BASE)
    for cls_name, field_name in [
        ("Robot", "battery"),
        ("Mission", "assigned_to"),
        ("BankTransaction", "amount_details"),
    ]:
        cls = getattr(example_domain, cls_name)
        url = resolver.resolve(SourceReference(owner_type=cls, attribute=field_name))
        assert url is not None
        assert f"autoapi/{_MODULE_PATH}/index.html" in url
        assert url.endswith(f"#{_MODULE}.{cls_name}.{field_name}")


def test_mock_apparatus_is_gone():
    """The retired mock API must not come back: no static HTML, no doc-local module."""
    assert not (_DOC_EQL_USER / "api_mock").exists()
    assert not (_DOC_EQL_USER / "verbalization_domain.py").exists()


def test_config_no_longer_copies_api_mock():
    """``_config.yml`` must not declare the api_mock ``html_extra_path`` any more."""
    content = _CONFIG_YML.read_text(encoding="utf-8")
    assert "api_mock" not in content


@pytest.mark.parametrize("cls_name", ["IsReachable", "WorksIn"])
def test_predicates_build_a_verbalization_fragment(cls_name):
    """The custom-predicate examples build their verbalization as a :class:`VerbalizationFragment` from the shared
    vocabulary, given the rendered fragment for each field."""
    cls = getattr(example_domain, cls_name)
    fields = {
        field.name: WordFragment(text=field.name) for field in dataclasses.fields(cls)
    }
    assert isinstance(cls._verbalization_fragment_(fields), VerbalizationFragment)
