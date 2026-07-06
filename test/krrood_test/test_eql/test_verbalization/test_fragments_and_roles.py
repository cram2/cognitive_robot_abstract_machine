"""
VerbalizationFragment-structure tests.

These tests verify that verbalization produces properly typed VerbalizationFragment trees —
specifically that semantic roles (VARIABLE, ATTRIBUTE, KEYWORD, …) are preserved
inside constraint clauses, binding phrases, grouped-by, ordered-by, and set_of
output.  They complement the plain-text regression tests in test_verbalization.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing_extensions import List

import pytest

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import (
    an,
    a,
    entity,
    flat_variable,
    inference,
    variable,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
from ...dataset.semantic_world_like_classes import (
    Cabinet,
    Container,
    Drawer,
    FixedConnection,
    Handle,
    PrismaticConnection,
)
from ...dataset.department_and_employee import Department, Employee

# ── VerbalizationFragment-tree helpers ──────────────────────────────────────────────────────


def _collect_role_fragments(
    frag: VerbalizationFragment, role: SemanticRole
) -> list[RoleFragment]:
    """Recursively collect all RoleFragments with the given role."""
    found: list[RoleFragment] = []
    _walk(frag, role, found)
    return found


def _walk(
    frag: VerbalizationFragment, role: SemanticRole, accumulator: list[RoleFragment]
) -> None:
    if isinstance(frag, RoleFragment) and frag.role == role:
        accumulator.append(frag)
    elif isinstance(frag, PhraseFragment):
        for part in frag.parts:
            _walk(part, role, accumulator)
    elif isinstance(frag, BlockFragment):
        if frag.header:
            _walk(frag.header, role, accumulator)
        for item in frag.items:
            _walk(item, role, accumulator)


def _all_texts(frag: VerbalizationFragment) -> list[str]:
    """Collect all leaf text values from a fragment tree."""
    texts: list[str] = []
    _walk_texts(frag, texts)
    return texts


def _walk_texts(frag: VerbalizationFragment, accumulator: list[str]) -> None:
    if isinstance(frag, (WordFragment, RoleFragment)):
        accumulator.append(frag.text)
    elif isinstance(frag, PhraseFragment):
        for part in frag.parts:
            _walk_texts(part, accumulator)
    elif isinstance(frag, BlockFragment):
        if frag.header:
            _walk_texts(frag.header, accumulator)
        for item in frag.items:
            _walk_texts(item, accumulator)


# ── 1. Constraint fragments retain semantic roles ──────────────────────────────


def test_constraint_fragment_preserves_variable_role(doors_and_drawers_world):
    """'such that' clause must contain VARIABLE-role fragments, not plain WordFragments."""
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer = inference(Drawer)(container=fc.expression.parent)

    frag = EQLVerbalizer().build(drawer)

    var_frags = _collect_role_fragments(frag, SemanticRole.VARIABLE)
    var_texts = [f.text for f in var_frags]
    # PrismaticConnection and Handle appear in the such-that clause
    assert any(
        "PrismaticConnection" in t for t in var_texts
    ), f"Expected VARIABLE role for PrismaticConnection in {var_texts!r}"
    assert any(
        "Handle" in t for t in var_texts
    ), f"Expected VARIABLE role for Handle in {var_texts!r}"


def test_constraint_fragment_preserves_attribute_role(doors_and_drawers_world):
    """Attribute names in the 'such that' clause must be ATTRIBUTE-role fragments."""
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )

    frag = EQLVerbalizer().build(drawer)

    attr_frags = _collect_role_fragments(frag, SemanticRole.ATTRIBUTE)
    attr_texts = [f.text for f in attr_frags]
    assert (
        "parent" in attr_texts or "container" in attr_texts
    ), f"Expected ATTRIBUTE role for 'parent'/'container' in {attr_texts!r}"


# ── 2. Binding override resolves field-ref in constraints ──────────────────────


def test_binding_override_replaces_entity_ref_in_constraint(doors_and_drawers_world):
    """The 'such that' clause must use field-reference paths, not raw entity names."""
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )

    text = verbalize_expression(drawer)

    # The constraint should reference the field paths, not the raw chain
    assert (
        "the container of the Drawer" in text
    ), f"Expected field-ref substitution in: {text!r}"
    assert (
        "the handle of the Drawer" in text
    ), f"Expected field-ref substitution in: {text!r}"


# ── 3. WHERE keyword role ──────────────────────────────────────────────────────


def test_where_keyword_has_keyword_role():
    """'where' in an InstantiatedVariable binding must carry KEYWORD semantic role."""

    @dataclass
    class Box:
        item: Any

    @dataclass
    class Item:
        value: int

    item_var = variable(Item, [])
    box = inference(Box)(item=item_var)

    frag = EQLVerbalizer().build(box)

    kw_frags = _collect_role_fragments(frag, SemanticRole.KEYWORD)
    kw_texts = [f.text for f in kw_frags]
    assert "where" in kw_texts, f"Expected KEYWORD role for 'where', got: {kw_texts!r}"


def test_such_that_keyword_has_keyword_role(doors_and_drawers_world):
    """'such that' in the deferred-constraint clause must carry KEYWORD role."""
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer = inference(Drawer)(container=fc.expression.parent)

    frag = EQLVerbalizer().build(drawer)

    kw_frags = _collect_role_fragments(frag, SemanticRole.KEYWORD)
    kw_texts = [f.text for f in kw_frags]
    assert (
        "such that" in kw_texts
    ), f"Expected KEYWORD role for 'such that', got: {kw_texts!r}"


# ── 4. GroupedBy variables retain roles ───────────────────────────────────────


def test_grouped_by_vars_are_role_fragments(departments_and_employees_fixture):
    """Variables inside a grouped-by clause must be VARIABLE-role fragments."""
    _, _ = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    avg_salary = eql.average(emp.salary)
    query = a(
        so := eql.set_of(emp.department, avg_salary)
        .grouped_by(emp.department)
        .having(avg_salary > 30000)
    )

    frag = EQLVerbalizer().build(query)

    var_frags = _collect_role_fragments(frag, SemanticRole.VARIABLE)
    var_texts = [f.text for f in var_frags]
    assert any(
        "Employee" in t for t in var_texts
    ), f"Expected VARIABLE role for Employee in {var_texts!r}"


# ── 5. OrderedBy direction is a vocabulary fragment ───────────────────────────


def test_ordered_by_direction_fragment(handles_and_containers_world):
    """The direction word in ordered-by must come from the ordering-range vocabulary (not a bare string)."""
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = flat_variable(cabinet.drawers)
    query = an(
        entity(cabinet)
        .grouped_by(cabinet)
        .ordered_by(eql.count(drawer), descending=True)
    )

    frag = EQLVerbalizer().build(query)
    texts = _all_texts(frag)

    assert (
        "from highest to lowest" in texts
    ), f"direction not found in fragment texts: {texts!r}"
    # the direction must be a fragment leaf, not embedded in a larger string
    word_leaf_texts = []
    _collect_word_leaves(frag, word_leaf_texts)
    assert (
        "from highest to lowest" in word_leaf_texts
    ), f"direction should be a standalone leaf fragment, got: {word_leaf_texts!r}"


def _collect_word_leaves(frag: VerbalizationFragment, accumulator: list[str]) -> None:
    if isinstance(frag, (WordFragment, RoleFragment)):
        accumulator.append(frag.text)
    elif isinstance(frag, PhraseFragment):
        for part in frag.parts:
            _collect_word_leaves(part, accumulator)
    elif isinstance(frag, BlockFragment):
        if frag.header:
            _collect_word_leaves(frag.header, accumulator)
        for item in frag.items:
            _collect_word_leaves(item, accumulator)


# ── 6. set_of selected vars are role fragments ────────────────────────────────


def test_set_of_vars_are_role_fragments(departments_and_employees_fixture):
    """Selected variables in a set_of query must be VARIABLE-role fragments."""
    _, _ = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    dept = emp.department
    avg_salary = eql.average(emp.salary)
    query = a(eql.set_of(dept, avg_salary).grouped_by(dept))

    frag = EQLVerbalizer().build(query)

    var_frags = _collect_role_fragments(frag, SemanticRole.VARIABLE)
    var_texts = [f.text for f in var_frags]
    assert any(
        "Employee" in t for t in var_texts
    ), f"Expected VARIABLE role for Employee in {var_texts!r}"


# ── 7. WHERE clause in entity query retains roles when binding overrides exist ─


def test_where_clause_with_instantiated_var_preserves_roles(doors_and_drawers_world):
    """WHERE clause on an entity query whose selected var is InstantiatedVariable
    must preserve semantic roles (not be collapsed to a plain WordFragment)."""
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer_var = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )

    query = entity(drawer_var)
    frag = EQLVerbalizer().build(query)

    # The If/Then rule form is used; both VARIABLE and ATTRIBUTE roles must appear
    var_frags = _collect_role_fragments(frag, SemanticRole.VARIABLE)
    attr_frags = _collect_role_fragments(frag, SemanticRole.ATTRIBUTE)
    assert (
        var_frags
    ), "Expected at least one VARIABLE-role fragment in rule verbalization"
    assert (
        attr_frags
    ), "Expected at least one ATTRIBUTE-role fragment in rule verbalization"


# ── 8. Nested InstantiatedVariable — such-that count and field-ref ─────────────


def test_double_nested_constraint_field_refs(doors_and_drawers_world):
    """Wrapper(drawer=Drawer(...)) — the such-that clause uses field-ref paths."""

    @dataclass
    class Wrapper:
        drawer: Any

    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer_var = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )
    wrapper_var = inference(Wrapper)(drawer=drawer_var)

    text = verbalize_expression(wrapper_var)

    assert text.count("such that") == 1
    assert "the container of the Drawer" in text
    assert "the handle of the Drawer" in text


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def departments_and_employees_fixture():
    d1 = Department("HR")
    d2 = Department("Finance")
    e1 = Employee("John", d1, 10000)
    e2 = Employee("Anna", d2, 40000)
    return [d1, d2], [e1, e2]
