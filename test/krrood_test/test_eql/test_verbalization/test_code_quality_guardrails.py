"""
Ratchet guardrails for the verbalization package's adherence to the AGENTS.md code-
quality rules.

Each test counts a forbidden construct across the verbalization source tree and asserts the count
does not *exceed* a recorded baseline, so no new violation can slip in. The baselines are the debt
inventory taken at the start of the verbalization refactor
(:doc:`/eql/developer/verbalization_refactor_plan`); each is driven to zero in the phase noted on its
constant, at which point the baseline is tightened (the ratchet only moves down).

Counting is AST-based, so occurrences in comments or docstrings never inflate a count.
"""

from __future__ import annotations

import ast
from pathlib import Path

from typing_extensions import List, Set

import krrood.entity_query_language.verbalization as verbalization_package

#: Forbidden ``getattr`` calls remaining (AGENTS.md: "always access attributes via '.', never via
#: getattr"). These are defensive ``getattr(node, "_id_"/"_type_", default)`` accesses where the
#: node may not carry the attribute; converting them safely needs per-site isinstance/None guards
#: (the ``_id_`` base attribute is universal, but ``_type_`` is not, and several sites rely on a
#: ``None``/sentinel default), so the ratchet holds the line rather than forcing a risky bulk rewrite.
GETATTR_BASELINE = 37

#: Attribute-access ``try/except`` handlers remaining (AGENTS.md: "do not wrap attribute access in
#: try-except blocks"). Only ``AttributeError`` / ``KeyError`` handlers count; ``ImportError`` guards
#: for optional dependencies are legitimate. The single remaining one is ``recognition.references``,
#: whose fallback depends on which expression types expose ``_unique_variables_``.
ATTRIBUTE_EXCEPT_BASELINE = 1

#: Module-level mutable empty containers remaining (AGENTS.md: "avoid using global variables").
#: Zero — the unused ``morphology`` override registry (the only such state) was removed in Phase 7.
MODULE_MUTABLE_STATE_BASELINE = 0

_VERBALIZATION_ROOT = Path(verbalization_package.__file__).parent


def _verbalization_modules() -> List[Path]:
    """:return: Every Python source file in the verbalization package, sorted for stable reporting."""
    return sorted(_VERBALIZATION_ROOT.rglob("*.py"))


def _syntax_tree(path: Path) -> ast.Module:
    """:return: The parsed AST of *path*."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _getattr_call_sites(tree: ast.Module) -> int:
    """:return: The number of ``getattr(...)`` calls in *tree*."""
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
    )


def _attribute_except_handlers(tree: ast.Module) -> int:
    """:return: The number of ``except AttributeError``/``except KeyError`` handlers in *tree*."""
    forbidden = {"AttributeError", "KeyError"}
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        caught: Set[str] = set()
        if isinstance(node.type, ast.Name):
            caught = {node.type.id}
        elif isinstance(node.type, ast.Tuple):
            caught = {
                element.id
                for element in node.type.elts
                if isinstance(element, ast.Name)
            }
        if forbidden & caught:
            count += 1
    return count


def _module_level_mutable_containers(tree: ast.Module) -> int:
    """:return: The number of module-level names bound to an empty ``{}`` / ``[]`` / ``set()``."""
    count = 0
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Dict) and not value.keys:
            count += 1
        elif isinstance(value, (ast.List, ast.Set)) and not value.elts:
            count += 1
    return count


def _total(counter) -> int:
    return sum(counter(_syntax_tree(path)) for path in _verbalization_modules())


def test_no_new_getattr_calls() -> None:
    """
    No new ``getattr`` call may be added to the verbalization package.
    """
    assert _total(_getattr_call_sites) <= GETATTR_BASELINE


def test_no_new_attribute_access_try_except() -> None:
    """
    No new attribute-access ``try/except`` may be added to the verbalization package.
    """
    assert _total(_attribute_except_handlers) <= ATTRIBUTE_EXCEPT_BASELINE


def test_no_new_module_level_mutable_state() -> None:
    """
    No new module-level mutable container may be added to the verbalization package.
    """
    assert _total(_module_level_mutable_containers) <= MODULE_MUTABLE_STATE_BASELINE
