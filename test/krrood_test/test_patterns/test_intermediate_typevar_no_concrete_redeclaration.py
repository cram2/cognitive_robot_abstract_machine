"""
Tests that when an intermediate ancestor narrows a TypeVar and the concrete class does
not further narrow it, the RoleFor for the concrete class does NOT redeclare the field.

Dataset: independent_typevar_takers.py
    RootHolder[TRoot]
        └─ NarrowedRootHolder[TSpecificRoot]   (narrows TRoot → TSpecificRoot)
               └─ ContentHolder[TContent]       (inherits root; adds content: TContent)
                      └─ MultiTaker[TContent2]  (narrows TContent → TContent2; root unchanged)

Expected mixin output:
    RoleForRootHolder           – root: TRoot
    RoleForNarrowedRootHolder   – root: TSpecificRoot  (genuine narrowing)
    RoleForContentHolder        – content: TContent; no root redeclaration
    RoleForMultiTaker           – content: TContent2; no root redeclaration
"""

import pytest

from krrood.patterns.role.role_transformer import RoleTransformer, TransformationMode
from test.krrood_test.dataset.role_and_ontology import independent_typevar_takers

TRANSFORMED = TransformationMode.TRANSFORMED.value


@pytest.fixture(scope="module")
def mixin_source():
    transformer = RoleTransformer(independent_typevar_takers, file_name_prefix=TRANSFORMED)
    _, src = transformer.transform()[independent_typevar_takers]
    return src


def _body_of_class(source: str, class_name: str) -> str:
    after = source.split(f"class {class_name}")[1]
    next_class = after.find("\nclass ")
    return after[:next_class] if next_class != -1 else after


def test_narrowed_root_holder_declares_root_with_specific_typevar(mixin_source):
    body = _body_of_class(mixin_source, "RoleForNarrowedRootHolder")
    assert "def root(self) -> TSpecificRoot" in body


def test_content_holder_does_not_redeclare_root(mixin_source):
    body = _body_of_class(mixin_source, "RoleForContentHolder")
    assert "def root" not in body


def test_multi_taker_does_not_redeclare_root(mixin_source):
    body = _body_of_class(mixin_source, "RoleForMultiTaker")
    assert "def root" not in body


def test_multi_taker_redeclares_content_with_narrowed_typevar(mixin_source):
    body = _body_of_class(mixin_source, "RoleForMultiTaker")
    assert "def content(self) -> TContent2" in body
