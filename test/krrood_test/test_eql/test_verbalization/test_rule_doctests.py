"""
Doctest harness for the whole verbalization package.

Every rule / form / assembler / planner docstring carries a concrete ``>>> verbalize_expression(...)``
example next to its template form, so the documented output is *executed* and cannot silently drift
from what the grammar actually produces.

The set of scanned modules is **auto-discovered** by walking the verbalization package: any module
whose docstrings contain ``>>>`` examples is found and run, so a doctest added anywhere in the
package is executed without editing this harness (the previous hand-maintained list silently dropped
doctests in modules it omitted — e.g. ``grammar.conditions.placement``).

The examples are kept to a single readable line by injecting a shared namespace — the EQL
factories, ``Not``, ``verbalize_expression`` and the example-domain classes — so a docstring need
only write ``verbalize_expression(variable(Task, []).completed)`` rather than re-import everything.
The example domain is the same module Sphinx AutoAPI documents, so the rendered examples also
hyperlink to real API pages.
"""

from __future__ import annotations

import datetime
import doctest
import importlib
import pkgutil

import pytest
from typing_extensions import List

import krrood.entity_query_language.factories as eql
import krrood.entity_query_language.verbalization as verbalization_package
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.verbalization import example_domain
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

# A shared namespace for all examples. It includes the EQL factories, so an example can write
# *"verbalize_expression(variable(Task, []).completed)"* rather than re-import everything.
# It also includes the example-domain classes, so the rendered examples also hyperlink to real API pages.
factories = [
    eql.variable,  # variables
    eql.a,
    eql.an,
    eql.the,  # quantifiers
    eql.entity,
    eql.set_of,  # query construction
    eql.and_,
    eql.or_,  # boolean logic
    eql.max,
    eql.min,
    eql.sum,
    eql.count,  # aggregations
    eql.contains,
    eql.in_,  # membership
    eql.for_all,
    eql.exists,  # quantified conditionals
    eql.not_,  # negation
    eql.inference,  # rule-tree inference queries
    eql.match,
    eql.match_variable,  # construction-pattern matches
    eql.underspecified,  # generative (underspecified) requests
]
_GLOBS = {factory.__name__: factory for factory in factories}
_GLOBS.update(
    verbalize_expression=verbalize_expression,
    Not=Not,
    Exists=Exists,
    ForAll=ForAll,
    datetime=datetime,
)
# The example-domain classes (defined in that module, not imported into it).
_GLOBS.update(
    {
        name: obj
        for name, obj in vars(example_domain).items()
        if isinstance(obj, type) and obj.__module__ == example_domain.__name__
    }
)


def _verbalization_modules_with_doctests() -> List[str]:
    """
    :return: The dotted names of every verbalization sub-module that carries ``>>>`` doctest
        examples, discovered by walking the package — so a doctest added anywhere is executed without
        editing this harness.
    """
    discovered: List[str] = []
    finder = doctest.DocTestFinder()
    for module_info in pkgutil.walk_packages(
        verbalization_package.__path__, verbalization_package.__name__ + "."
    ):
        module = importlib.import_module(module_info.name)
        if finder.find(module, module.__name__, extraglobs=_GLOBS):
            discovered.append(module_info.name)
    return sorted(discovered)


_MODULE_NAMES = _verbalization_modules_with_doctests()


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_rule_docstring_examples_execute(module_name):
    """Each rule/form docstring's ``>>>`` example produces exactly the documented output."""
    module = importlib.import_module(module_name)
    finder = doctest.DocTestFinder()
    failures: list[str] = []
    for test in finder.find(module, module.__name__, extraglobs=_GLOBS):
        # A fresh runner per docstring — DocTestRunner.run reports cumulative counts.
        runner = doctest.DocTestRunner(optionflags=doctest.FAIL_FAST)
        result = runner.run(test, clear_globs=False)
        if result.failed:
            failures.append(test.name)
    assert not failures, f"doctest failures in {module.__name__}: {failures}"


def test_doctest_discovery_includes_previously_uncovered_modules():
    """
    The auto-discovery must execute doctests the old hand-maintained list silently dropped — in
    particular ``grammar.conditions.placement`` — and must still cover the originally-curated modules.
    """
    previously_curated = {
        "grammar.terms.rules",
        "grammar.chain.rules",
        "grammar.chain.planner",
        "grammar.conditions.rules",
        "grammar.conditions.subject",
        "grammar.query.rules",
        "grammar.query.assembler",
        "grammar.query.planner",
        "grammar.clauses.assembler",
        "grammar.clauses.planner",
        "grammar.aggregation.rules",
        "grammar.aggregation.assembler",
    }
    discovered_suffixes = {name.split("verbalization.", 1)[1] for name in _MODULE_NAMES}
    missing = previously_curated - discovered_suffixes
    assert not missing, f"auto-discovery regressed on: {sorted(missing)}"
    assert "grammar.conditions.placement" in discovered_suffixes
