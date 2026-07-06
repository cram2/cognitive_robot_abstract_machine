from __future__ import annotations

import importlib
import pkgutil

from typing_extensions import List

from krrood.entity_query_language.verbalization import grammar as grammar_package
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
)
from krrood.patterns.specificity_ranking import concrete_subclasses


def _load_rule_modules() -> None:
    """
    Import every construct's ``rules`` module so its :class:`PhraseRule` subclasses are loaded (and
    therefore discoverable) before ``RULES`` is built. Discovered by walking the ``grammar`` package
    rather than a hand-maintained import list, so a new construct is registered simply by adding its
    ``grammar/<construct>/rules.py`` — no edit here.
    """
    for module_info in pkgutil.walk_packages(
        grammar_package.__path__, grammar_package.__name__ + "."
    ):
        if module_info.name.endswith(".rules"):
            importlib.import_module(module_info.name)


_load_rule_modules()

# Auto-discovered: one instance of every concrete ``PhraseRule`` subclass. Order is irrelevant —
# ``select`` decides specificity — and a new rule is registered simply by defining its class in one
# of the construct ``rules`` modules. Uses the same ``concrete_subclasses`` primitive the
# ``SpecificityRule`` families use to discover their alternatives.
RULES: List[PhraseRule] = [rule_cls() for rule_cls in concrete_subclasses(PhraseRule)]
