from __future__ import annotations

import enum
import itertools
from dataclasses import fields, is_dataclass

from typing_extensions import Any, List, Optional

from krrood.entity_query_language.core.mapped_variable import FlatVariable
from krrood.entity_query_language.core.variable import (
    ExternallySetVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
    NounPhrase,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
)
from krrood.patterns.field_metadata import FieldMetadata, GrammarMetadata
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_concrete_object_literal,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    MAX_SET_MEMBERS,
    one_of,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    FallbackNouns,
    Prepositions,
    Specificity,
)

#: Field names tried, in order, to identify a concrete object whose class marks no field with
#: :attr:`GrammarMetadata.is_identifying_field` — the first one present names it (*"a specific
#: Body with name 'door'"*). Conventional identity fields, so the heuristic is deterministic.
_CONVENTIONAL_ID_FIELDS = ("name", "id", "label", "key", "uuid")

#: Only scalar identifying values are shown — a nested object would re-open the very ``repr`` blow-up
#: that *"a specific <Type>"* exists to avoid.
_SCALAR_ID_TYPES = (str, int, float, bool, enum.Enum)

#: The value types whose explicit domain is listed (*"one of 1, 2, or 3"*); an entity / ``Symbol``
#: type's domain is the inferred population, which must never be listed.
_PRIMITIVE_VALUE_TYPES = (int, float, str, bool)


class VariableRule(PhraseRule):
    """*"a/an Robot"* (first mention), *"the Robot"* (subsequent), or *"Robot N"* (numbered).

    >>> verbalize_expression(variable(Robot, []))
    'a Robot'
    """

    construct = Variable

    def build(self, node: Variable, context: RuleContext) -> VerbalizationFragment:
        """:return: The variable noun phrase (*"a Robot"* / *"the Robot"* / *"Robot N"*)."""
        if context.as_value:
            choice = self._domain_choice(node, context)
            if choice is not None:
                return choice
        if context.number is GrammaticalNumber.PLURAL:
            return self._plural(node, context)
        noun_form = context.refer.noun_for_parts(node)
        return NounPhrase(
            head=RoleFragment.for_variable(noun_form.label, node),
            definiteness=noun_form.definiteness,
            referent_id=node._id_,
        )

    @staticmethod
    def _domain_choice(
        node: Variable, context: RuleContext
    ) -> Optional[VerbalizationFragment]:
        """In value position, a domain-constrained value-type variable says its candidate set —
        *"one of OPTION_A, OPTION_B, or OPTION_C"* (an enum) / *"one of 1, or 2"* (a primitive), or
        just the value for a singleton domain. ``None`` (use the type-name noun) unless the type is
        an ``enum`` / primitive value type and its domain is a small, bounded set — an entity type's
        domain is the inferred population and is never listed.

        :param node: The variable in value position.
        :param context: The per-node context (for value lexicalisation).
        :return: The candidate-set fragment, or ``None`` to fall back to the noun form.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(entity(employee).where(employee.department == variable(str, ["Sales", "Eng"]))))
        "Find an Employee whose department is one of 'Sales' or 'Eng'"
        """
        type_ = node._type_
        is_enum = isinstance(type_, type) and issubclass(type_, enum.Enum)
        if not (is_enum or type_ in _PRIMITIVE_VALUE_TYPES):
            return None
        values = itertools.islice(
            node._re_enterable_domain_generator_, MAX_SET_MEMBERS + 1
        )
        return one_of([RoleFragment.for_literal(value) for value in values])

    @staticmethod
    def _plural(node: Variable, context: RuleContext) -> VerbalizationFragment:
        """Bare plural variable noun phrase (*"Robots"*); the determiner phase drops the article and
        the morphology pass inflects the head.

        A numbered label (*"Robot 2"*) is surface-final — kept singular and bare; a plain type
        name is a plural indefinite noun phrase (the concord table renders it bare-then-pluralised).

        >>> verbalize_expression(count(variable(Robot, [])))
        'the number of Robots'
        """
        numbered = context.refer.numbered_label(node)
        return NounPhrase(
            head=RoleFragment.for_variable(numbered.text, node),
            number=(
                GrammaticalNumber.SINGULAR
                if numbered.is_numbered
                else GrammaticalNumber.PLURAL
            ),
            definiteness=(
                Definiteness.BARE if numbered.is_numbered else Definiteness.INDEFINITE
            ),
            referent_id=node._id_,
        )


class LiteralRule(PhraseRule):
    """A literal value (e.g. ``42``, ``"hello"``, ``True``), or *"a specific <Type>"* for a concrete
    object literal — we mean its identity, and its ``repr`` can be arbitrarily large.

    >>> verbalize_expression(variable(Robot, []).battery == 42)
    'the battery of a Robot is 42'
    """

    construct = Literal

    def build(self, node: Literal, context: RuleContext) -> VerbalizationFragment:
        """:return: The literal value, or *"a specific <Type>"* for a concrete object literal.

        A bare class used as a value renders as a linked type reference, and a tuple of classes as a
        coordinated value (*"Robot and Task"*, each linked). A tuple is a *value* here — reading it
        as a membership set (*"one of …"*) is the consuming predicate's call (see
        :meth:`HasTypes._verbalization_fragment_`), not the literal's, so an equality
        ``x == (Robot, Task)`` is not mis-read as membership.
        """
        value = node._value_
        if is_concrete_object_literal(node):
            return self._concrete_object(node, context)
        if isinstance(value, type):
            return RoleFragment.for_type(value)
        type_members = self._type_members(value)
        if type_members is not None:
            return oxford_comma(
                [RoleFragment.for_type(member) for member in type_members],
                Conjunctions.AND.as_fragment(),
            )
        return RoleFragment.for_literal(value)

    @staticmethod
    def _type_members(value: Any) -> Optional[List[type]]:
        """:return: the classes of a non-empty tuple/list of types, or ``None`` when *value* is not
        such a sequence."""
        if (
            isinstance(value, (tuple, list))
            and value
            and all(isinstance(member, type) for member in value)
        ):
            return list(value)
        return None

    def _concrete_object(
        self, node: Literal, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: *"a specific <Type>"* for a concrete object literal — identity, not its (possibly
        huge) repr — qualified by its identifying field(s) when any are known (*"a specific Body with
        name 'door'"*). The fields are those marked with :attr:`GrammarMetadata.is_identifying_field`,
        else the first present :data:`conventional identity field <_CONVENTIONAL_ID_FIELDS>`.

        This method builds the whole *"a specific Robot with name 'R2D2'"* noun phrase — the
        identity head plus the qualifying detail :meth:`_identifying_fields` supplies:

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(a(entity(robot).where(robot == Robot("R2D2", 80, True))))
        "Find a Robot such that the Robot is a specific Robot with name 'R2D2'"
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
        modifiers: List[VerbalizationFragment] = (
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
        """:return: The ``(name, value)`` pairs that identify *value* for display — the attributes a
        :class:`HasIdentifyingAttributes` declares, else the first present conventional identity
        field — keeping only scalar values.

        Its contribution is just the identifying detail: it returns ``[("name", "R2D2")]`` here, which
        is what becomes the *"with name 'R2D2'"* qualifier (drop it and the phrase is the bare *"a
        specific Robot"*):

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(a(entity(robot).where(robot == Robot("R2D2", 80, True))))
        "Find a Robot such that the Robot is a specific Robot with name 'R2D2'"
        """
        names = LiteralRule._declared_identifying_names(value)
        return [
            (name, getattr(value, name))
            for name in names
            if hasattr(value, name)
            and isinstance(getattr(value, name), _SCALAR_ID_TYPES)
        ]

    @staticmethod
    def _declared_identifying_names(value: Any) -> List[str]:
        """:return: The identifying attribute names for *value* — the dataclass fields its class marks
        with :attr:`GrammarMetadata.is_identifying_field`, else the first present conventional
        identity field."""
        marked = LiteralRule._identifying_attribute_fields(value)
        return (
            marked
            or [name for name in _CONVENTIONAL_ID_FIELDS if hasattr(value, name)][:1]
        )

    @staticmethod
    def _identifying_attribute_fields(value: Any) -> List[str]:
        """:return: The names of *value*'s dataclass fields marked as identifying attributes, in
        declaration order (empty when *value* is not a dataclass or marks none)."""
        if not is_dataclass(value):
            return []
        return [
            data_field.name
            for data_field in fields(value)
            if (grammar := GrammarMetadata.of_field(value, data_field.name)) is not None
            and grammar.is_identifying_field
        ]


class ExternalVariableRule(PhraseRule):
    """*"a/an TypeName"* for an opaque externally-set variable (no coreference).

    >>> verbalize_expression(ExternallySetVariable(_type_=Robot))
    'a Robot'
    """

    construct = ExternallySetVariable

    def build(
        self, node: ExternallySetVariable, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: The indefinite type-name noun phrase for the externally-set variable."""
        type_name = FallbackNouns.VARIABLE.name_of(node)
        return NounPhrase(head=RoleFragment.for_type(node._type_, text=type_name))


class FlatVariableRule(PhraseRule):
    """A transparent SetOf wrapper → unwrap to its child (forwarding the requested number).

    >>> verbalize_expression(FlatVariable(_child_=variable(Worker, []).tasks))
    'the tasks of a Worker'
    """

    construct = FlatVariable

    def build(self, node: FlatVariable, context: RuleContext) -> VerbalizationFragment:
        """:return: The child's rendering, unwrapped from the transparent SetOf wrapper."""
        return context.child(node._child_, number=context.number)
