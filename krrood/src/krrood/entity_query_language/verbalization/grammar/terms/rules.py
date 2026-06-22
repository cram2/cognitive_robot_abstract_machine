from __future__ import annotations

import enum
import itertools

from typing_extensions import Any, List, Optional

from krrood.entity_query_language.core.mapped_variable import FlatVariable
from krrood.entity_query_language.core.variable import (
    ExternallySetVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    NounPhrase,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_concrete_object_literal,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    FallbackNouns,
    Prepositions,
    SetMembership,
    Specificity,
)

#: Field names tried, in order, to identify a concrete object when its class declares no
#: ``_identifying_attributes_`` — the first one present names it (*"a specific Body with name
#: 'door'"*). Conventional identity fields, so the heuristic is deterministic, not a guess.
_CONVENTIONAL_ID_FIELDS = ("name", "id", "label", "key", "uuid")

#: Only scalar identifying values are shown — a nested object would re-open the very ``repr`` blow-up
#: that *"a specific <Type>"* exists to avoid.
_SCALAR_ID_TYPES = (str, int, float, bool, enum.Enum)

#: The most domain values a value-type variable lists before falling back to its type name — a
#: bounded peek keeps the rendering cheap and never enumerates a large (or entity) domain.
_MAX_DOMAIN_CHOICES = 6

#: The value types whose explicit domain is listed (*"one of 1, 2, or 3"*); an entity / ``Symbol``
#: type's domain is the inferred population, which must never be listed.
_PRIMITIVE_VALUE_TYPES = (int, float, str, bool)


class VariableRule(PhraseRule):
    """*"a/an Robot"* (first mention), *"the Robot"* (subsequent), or *"Robot N"* (numbered).

    >>> verbalize_expression(variable(Robot, []))
    'a Robot'
    """

    construct = Variable
    name = "variable"

    def build(self, node: Variable, context: RuleContext) -> Fragment:
        if context.as_value:
            choice = self._domain_choice(node, context)
            if choice is not None:
                return choice
        if context.number is Number.PLURAL:
            return self._plural(node, context)
        noun_form = context.refer.noun_for_parts(node)
        return NounPhrase(
            head=RoleFragment.for_variable(noun_form.label, node),
            definiteness=noun_form.definiteness,
            referent_id=node._id_,
        )

    @staticmethod
    def _domain_choice(node: Variable, context: RuleContext) -> Optional[Fragment]:
        """In value position, a domain-constrained value-type variable says its candidate set —
        *"one of OPTION_A, OPTION_B, or OPTION_C"* (an enum) / *"one of 1, or 2"* (a primitive), or
        just the value for a singleton domain. ``None`` (use the type-name noun) unless the type is
        an ``enum`` / primitive value type and its domain is a small, bounded set — an entity type's
        domain is the inferred population and is never listed.

        :param node: The variable in value position.
        :param context: The per-node context (for value lexicalisation).
        :return: The candidate-set fragment, or ``None`` to fall back to the noun form.
        """
        type_ = getattr(node, "_type_", None)
        is_enum = isinstance(type_, type) and issubclass(type_, enum.Enum)
        if not (is_enum or type_ in _PRIMITIVE_VALUE_TYPES):
            return None
        values = list(
            itertools.islice(
                node._re_enterable_domain_generator_, _MAX_DOMAIN_CHOICES + 1
            )
        )
        if not values or len(values) > _MAX_DOMAIN_CHOICES:
            return None
        fragments: List[Fragment] = [
            RoleFragment.for_literal(value) for value in values
        ]
        if len(fragments) == 1:
            return fragments[0]
        return PhraseFragment(
            parts=[
                SetMembership.ONE_OF.as_fragment(),
                oxford_comma(fragments, Conjunctions.OR.as_fragment()),
            ]
        )

    @staticmethod
    def _plural(node: Variable, context: RuleContext) -> Fragment:
        """Bare plural variable noun phrase (*"Robots"*); the determiner phase drops the article and
        the morphology pass inflects the head.

        A numbered label (*"Robot 2"*) is surface-final — kept singular and bare; a plain type
        name is a plural indefinite noun phrase (the concord table renders it bare-then-pluralised).
        """
        numbered = context.refer.numbered_label(node)
        return NounPhrase(
            head=RoleFragment.for_variable(numbered.text, node),
            number=Number.SINGULAR if numbered.is_numbered else Number.PLURAL,
            definiteness=(
                Definiteness.BARE if numbered.is_numbered else Definiteness.INDEFINITE
            ),
            referent_id=node._id_,
        )


class LiteralRule(PhraseRule):
    """A literal value (e.g. ``42``, ``"hello"``, ``True``), or *"a specific <Type>"* for a concrete
    object literal — we mean its identity, and its ``repr`` can be arbitrarily large.
    """

    construct = Literal
    name = "literal"

    def build(self, node: Literal, context: RuleContext) -> Fragment:
        if is_concrete_object_literal(node):
            return self._concrete_object(node, context)
        return RoleFragment.for_literal(node._value_)

    def _concrete_object(self, node: Literal, context: RuleContext) -> Fragment:
        """:return: *"a specific <Type>"* for a concrete object literal — identity, not its (possibly
        huge) repr — qualified by its identifying field(s) when any are known (*"a specific Body with
        name 'door'"*). The fields come from the class's ``_identifying_attributes_`` if it declares
        any, else the first present :data:`conventional identity field <_CONVENTIONAL_ID_FIELDS>`.
        """
        value = node._value_
        details = [
            PhraseFragment(
                parts=[
                    RoleFragment.for_attribute(None, name),
                    RoleFragment.for_literal(field_value),
                ]
            )
            for name, field_value in self._identifying_fields(value)
        ]
        modifiers: List[Fragment] = (
            [
                Prepositions.WITH.as_fragment(),
                oxford_comma(details, Conjunctions.AND.as_fragment()),
            ]
            if details
            else []
        )
        return NounPhrase(
            head=RoleFragment.for_variable(type(value).__name__, node),
            definiteness=Definiteness.INDEFINITE,
            pre_head=Specificity.SPECIFIC.as_fragment(),
            modifiers=modifiers,
        )

    @staticmethod
    def _identifying_fields(value: Any) -> List[tuple]:
        """:return: The ``(name, value)`` pairs that identify *value* for display — the class's
        declared ``_identifying_attributes_`` (a name or iterable of names) if present, else the first
        present conventional identity field — keeping only scalar values."""
        declared = getattr(type(value), "_identifying_attributes_", None)
        if callable(declared):
            try:
                declared = declared()
            except TypeError:
                declared = None
        if isinstance(declared, str):
            names: List[str] = [declared]
        elif declared:
            names = list(declared)
        else:
            names = [f for f in _CONVENTIONAL_ID_FIELDS if hasattr(value, f)][:1]
        return [
            (name, getattr(value, name))
            for name in names
            if hasattr(value, name)
            and isinstance(getattr(value, name), _SCALAR_ID_TYPES)
        ]


class ExternalVariableRule(PhraseRule):
    """*"a/an TypeName"* for an opaque externally-set variable (no coreference)."""

    construct = ExternallySetVariable
    name = "external-variable"

    def build(self, node: ExternallySetVariable, context: RuleContext) -> Fragment:
        type_name = (
            node._type_.__name__
            if getattr(node, "_type_", None)
            else FallbackNouns.VARIABLE.text
        )
        return NounPhrase(
            head=RoleFragment.for_type(getattr(node, "_type_", None), text=type_name)
        )


class FlatVariableRule(PhraseRule):
    """A transparent SetOf wrapper → unwrap to its child (forwarding the requested number)."""

    construct = FlatVariable
    name = "flat-variable"

    def build(self, node: FlatVariable, context: RuleContext) -> Fragment:
        return context.child(node._child_, number=context.number)
