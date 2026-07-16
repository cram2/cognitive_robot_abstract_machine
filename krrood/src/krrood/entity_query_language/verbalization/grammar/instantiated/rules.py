from __future__ import annotations

from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.predicate import RenderedFields, Verbalizable
from krrood.entity_query_language.verbalization.exceptions import (
    NonFragmentPredicateError,
    PredicateFragmentRequiredError,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
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


class InstantiatedVariableRule(PhraseRule):
    """
    *"a TypeName where the field of the TypeName is … such that …"*.
    """

    construct = InstantiatedVariable

    def build(
        self, node: InstantiatedVariable, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: The instantiated variable's *"a TypeName, where the field of the TypeName is …"*
        noun phrase, built by the :class:`InstantiatedAssembler`.

        Its contribution is selecting the generic decomposed surface for a *non-predicate*
        constructed entity: with no verbalization fragment on ``Drawer``, this fallback rule fires
        and delegates to the assembler, which is why the result is the long *"a Drawer, where …"*
        form. A :class:`Verbalizable` predicate, by contrast, is *required* to supply a fragment —
        reaching this rule without one is an error, not a name-based fallback.

        >>> connection = variable(FixedConnection, [])
        >>> verbalize_expression(inference(Drawer)(container=connection.parent, handle=connection.child))
        'a Drawer, where the container of the Drawer is the parent of a FixedConnection, and the handle of the Drawer is the child of the FixedConnection'
        """
        type_ = node._type_
        if isinstance(type_, type) and issubclass(type_, Verbalizable):
            raise PredicateFragmentRequiredError(node=node)
        return InstantiatedAssembler(context).assemble(node)


class InstantiatedVerbalizableRule(PhraseRule):
    """
    An InstantiatedVariable whose type builds its own verbalization
    :class:`VerbalizationFragment`.
    """

    construct = InstantiatedVariable

    def when(self, node: InstantiatedVariable, context: RuleContext) -> bool:
        """:return: ``True`` when *node*'s type supplies a verbalization fragment, selecting this rule
        over the generic *"a TypeName, where …"* form.

        Its contribution is the guard that admits this rule: ``IsReachable`` supplies a fragment, so
        this rule wins and the example renders as *"a Robot is reachable"* instead of the generic
        decomposed phrase. :meth:`build` then assembles that fragment.

        >>> verbalize_expression(inference(IsReachable)(body=variable(Robot, [])))
        'a Robot is reachable'
        """
        return InstantiatedPlanner.has_fragment(node)

    def build(
        self, node: InstantiatedVariable, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: the type's verbalization fragment, built from its rendered field fragments
        (*"a Robot is reachable"*).

        The type composes the surface from the shared vocabulary, so the result is a structured
        fragment that flows through the remaining passes (coreference, determiner, morphology) — not
        an opaque string blob — which is why a wrapping ``Not`` can negate it inline.

        >>> verbalize_expression(inference(IsReachable)(body=variable(Robot, [])))
        'a Robot is reachable'
        """
        fields = RenderedFields(
            fragments={
                name: context.child(child) for name, child in node._child_vars_.items()
            },
            raw=node._child_vars_,
        )
        fragment = node._type_._verbalization_fragment_(fields)
        if not isinstance(fragment, VerbalizationFragment):
            raise NonFragmentPredicateError(
                predicate_type=node._type_, returned=fragment
            )
        return fragment
