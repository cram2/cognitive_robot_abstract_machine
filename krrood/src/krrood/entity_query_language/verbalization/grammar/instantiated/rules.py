from __future__ import annotations

from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.instantiated.assembler import (
    InstantiatedAssembler,
)
from krrood.entity_query_language.verbalization.grammar.instantiated.planner import (
    InstantiatedPlanner,
)
from krrood.entity_query_language.verbalization.rendering.realization import (
    realize_subtree,
)


class InstantiatedVariableRule(PhraseRule):
    """*"a TypeName where the field of the TypeName is … such that …"*."""

    construct = InstantiatedVariable
    name = "instantiated-variable"

    def build(self, node: InstantiatedVariable, context: RuleContext) -> Fragment:
        return InstantiatedAssembler(context).assemble(node)


class InstantiatedVerbalizableRule(PhraseRule):
    """An InstantiatedVariable whose type supplies a verbalization template string."""

    construct = InstantiatedVariable
    name = "instantiated-verbalizable"

    def when(self, node: InstantiatedVariable, context: RuleContext) -> bool:
        return InstantiatedPlanner.has_template(node)

    def build(self, node: InstantiatedVariable, context: RuleContext) -> Fragment:
        # An opaque format string: it consumes finalized child text, so it realizes its
        # children locally (morphology pass + flatten) rather than deferring to the global pass.
        template = node._type_._verbalization_template_()
        kwargs = {
            name: realize_subtree(context.child(child))
            for name, child in node._child_vars_.items()
        }
        return WordFragment(text=template.format(**kwargs))
