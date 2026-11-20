"""
The grammar a verbalization dispatches over.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from krrood.entity_query_language.verbalization import grammar as grammar_package
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
)
from krrood.patterns.specificity_ranking import concrete_subclasses

# %% loading the grammar package's rules


def _load_rule_modules() -> None:
    """
    Import every ``rules`` module of the grammar package, so a construct's
    :class:`PhraseRule` subclasses are loaded once its ``grammar/<construct>/rules.py``
    exists.
    """
    for module_info in pkgutil.walk_packages(
        grammar_package.__path__, grammar_package.__name__ + "."
    ):
        if module_info.name.endswith(".rules"):
            importlib.import_module(module_info.name)


_load_rule_modules()

# %% reading the grammar


@dataclass(kw_only=True)
class PhraseRuleRegistry:
    """
    The grammar rules a verbalization dispatches over.
    """

    @classmethod
    def loaded_rules(cls) -> list[PhraseRule]:
        """
        :return: One instance of every concrete :class:`PhraseRule` subclass loaded when
            this is called whose :attr:`PhraseRule.joins_grammar` is ``True``: those of the
            grammar package, and those of any other module, such as a krrood part's own
            constructs.
        """
        return [
            rule_class()
            for rule_class in concrete_subclasses(PhraseRule)
            if rule_class.joins_grammar
        ]
