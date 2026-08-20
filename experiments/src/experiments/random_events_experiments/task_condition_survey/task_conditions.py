"""
This module states what a task's success condition says, in the Entity Query Language.

A condition is an EQL expression and nothing else: EQL's own conjunction, disjunction,
negation and quantifiers compose it, and what a task suite brings of its own -- the
relations it states of the objects a scene provides, and the values it measures over them
-- enters as subclasses of :class:`~krrood.entity_query_language.predicate.Predicate` and
:class:`~krrood.entity_query_language.predicate.SymbolicFunction`. So a survey supplies a
vocabulary to EQL rather than a formula language of its own.

Each relation and each measured value states either a condition over a finite domain or
one on a continuous quantity, since the two pose different questions for a
:mod:`random_events` representation: the discrete ones are always exactly representable
and the open question is what a formula over them costs, whereas the continuous ones are
exactly representable only when they bound a variable per axis.

Nothing here is specific to one task suite. Which names a suite states its conditions
under is supplied by the suite's own :class:`PredicateVocabulary`.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Any, ClassVar, Iterator, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import (
    InstantiatedVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.logical_quantifiers import (
    QuantifiedConditional,
)
from krrood.entity_query_language.predicate import (
    Predicate,
    RenderedFields,
    SymbolicFunction,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    FunctionVerbalizationTemplates,
    Noun,
    Verb,
    clause,
)
from krrood.exceptions import DataclassException

# %% conditions that are stated but never checked


@dataclass
class ConditionIsNotObservable(DataclassException):
    """
    Raised when a condition read from a suite's source is asked for its truth.

    A survey reads what a task states and never runs it, so the objects a condition is
    stated of are source text and no scene is at hand to check them against.
    """

    stated_under: str
    """
    The name the unobservable condition is stated under.
    """

    def error_message(self) -> str:
        return (
            f"{self.stated_under} was read from a task's source, which names its objects "
            "rather than providing them, so its truth is not observable."
        )

    def suggest_correction(self) -> str:
        return (
            "Ask what the condition states -- its propositions, its quantifications, its "
            "random event -- rather than whether it holds."
        )


# %% the objects a condition is stated about


@dataclass(frozen=True)
class SceneObject:
    """
    An object a scene provides, identified by the source text a task names it with.

    A survey reads a suite's source and never runs it, so the only handle on an object
    is how the task writes it. Conditions stated of differently written objects are
    therefore different conditions, which is what keeps one relation applied to two
    objects from counting as one.
    """

    named_as: str
    """
    The source text naming the object.
    """

    @classmethod
    def written_as(cls, source_text: str) -> Literal:
        """
        :param source_text: The objects as the task writes them.
        :return: The EQL literal standing for them.
        """
        return Literal(_value_=cls(named_as=source_text), _name__=source_text)

    @classmethod
    def provided_by_the_scene(cls) -> Variable:
        """
        The collection's contents are settled by the scene rather than by the task, so the
        variable is declared without a domain: what it ranges over is not in the source.

        :return: An EQL variable standing for any one object of a collection the scene
            provides, which is what a quantified condition is stated of.
        """
        return Variable(_type_=cls, _domain_=())

    @staticmethod
    def named_by(objects: Variable) -> str:
        """
        :param objects: The EQL variable a condition is stated of.
        :return: The objects it stands for, as the task writes them, which is the
            quantifier's own naming when the condition is stated of a quantified object.
        """
        if isinstance(objects, Literal):
            return objects._value_.named_as
        return objects._name_


# %% what kind of set a condition describes


class PredicateKind(enum.Enum):
    """
    What kind of condition a name is stated under, and therefore which question a
    :mod:`random_events` representation of it has to answer.
    """

    DISCRETE = "a condition over a finite domain of states or relations"
    """
    The condition holds or fails on a finite domain, so it maps onto a symbolic variable
    and is always exactly representable.
    """

    CONTINUOUS = "a continuous quantity compared against a threshold"
    """
    The condition bounds a real quantity, so it maps onto a continuous variable and is
    exactly representable only when the bound is per axis.
    """

    UNCLASSIFIED = "a name this survey does not recognise"
    """
    The survey does not know this name, so it makes no claim about representing what it
    states.
    """

    def relation_stated_of(
        self, stated_under: str, objects: Variable
    ) -> SymbolicExpression:
        """
        :param stated_under: The dotted name the task states the relation under.
        :param objects: The objects it is stated of.
        :return: The EQL condition asserting that the relation holds of them.
        """
        return {
            PredicateKind.DISCRETE: DiscreteRelation,
            PredicateKind.CONTINUOUS: ContinuousRelation,
            PredicateKind.UNCLASSIFIED: UnclassifiedRelation,
        }[self](stated_under=stated_under, objects=objects)

    def value_measured_over(
        self, stated_under: str, objects: Variable
    ) -> SymbolicExpression:
        """
        :param stated_under: The dotted name the task takes the measure under.
        :param objects: The objects it is measured over.
        :return: The EQL value the task compares, which becomes a condition once compared.
        """
        return {
            PredicateKind.DISCRETE: DiscreteValue,
            PredicateKind.CONTINUOUS: ContinuousValue,
            PredicateKind.UNCLASSIFIED: UnclassifiedValue,
        }[self](stated_under=stated_under, objects=objects)


# %% the relations a task states


@dataclass(eq=False)
class StatedRelation(Predicate, ABC):
    """
    A relation a task's success condition states of the objects a scene provides.

    A relation is what a condition applies without writing a bound of its own: a
    containment, a contact, or a helper hiding the threshold it thresholds. What the
    relation means is left to the suite; what is recorded is the name it is stated
    under, the objects it is stated of, and the kind of set it describes.
    """

    stated_under: str
    """
    The dotted name the task states the relation under.
    """

    objects: Variable
    """
    The objects the relation is stated of.
    """

    kind: ClassVar[PredicateKind]
    """
    The kind of set this relation describes.
    """

    def __call__(self) -> bool:
        raise ConditionIsNotObservable(stated_under=self.stated_under)

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :return: The clause *"<objects> satisfies <relation>"*, which is all a relation read
            from source says: the suite names the requirement and the objects meet it.
        """
        return clause(
            Noun(fields["objects"]), Verb("satisfy"), Noun(fields["stated_under"])
        )


@dataclass(eq=False)
class DiscreteRelation(StatedRelation):
    """
    A relation over a finite domain of states, such as a containment or a contact.
    """

    kind: ClassVar[PredicateKind] = PredicateKind.DISCRETE
    """
    A finite domain, so a symbolic variable represents it exactly.
    """


@dataclass(eq=False)
class ContinuousRelation(StatedRelation):
    """
    A relation defined by a threshold on a continuous quantity, which the task states by
    name rather than by writing the threshold itself.
    """

    kind: ClassVar[PredicateKind] = PredicateKind.CONTINUOUS
    """
    A bound on a real quantity, so how exactly it is represented depends on the region
    it describes.
    """


@dataclass(eq=False)
class UnclassifiedRelation(StatedRelation):
    """
    A relation whose kind the suite's vocabulary does not establish.
    """

    kind: ClassVar[PredicateKind] = PredicateKind.UNCLASSIFIED
    """
    Unestablished, so the survey makes no claim about representing it.
    """


# %% the values a task measures and compares


@dataclass(eq=False)
class MeasuredValue(SymbolicFunction, ABC):
    """
    A value a task's condition measures over the objects a scene provides, in order to
    compare it.

    A condition that writes its own bound states a comparison, so what it names is a
    value and the bound is EQL's comparison over it. The bound itself stays as the task
    writes it, which is a number when the task writes a literal and the source text of
    the bound otherwise.
    """

    stated_under: str
    """
    The dotted name the task takes the measure under.
    """

    objects: Variable
    """
    The objects the value is measured over.
    """

    kind: ClassVar[PredicateKind]
    """
    The kind of set a comparison of this value describes.
    """

    def __call__(self) -> Any:
        raise ConditionIsNotObservable(stated_under=self.stated_under)

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :return: The noun phrase naming the value, as *"the <kind> value of <measure> and
            <objects>"*.
        """
        return FunctionVerbalizationTemplates.possessive(
            cls, fields["stated_under"], fields["objects"]
        )


@dataclass(eq=False)
class DiscreteValue(MeasuredValue):
    """
    A value drawn from a finite domain, such as a mode or a state a task compares by
    name.
    """

    kind: ClassVar[PredicateKind] = PredicateKind.DISCRETE
    """
    A finite domain, so a symbolic variable represents a comparison of it exactly.
    """


@dataclass(eq=False)
class ContinuousValue(MeasuredValue):
    """
    A real quantity, such as a distance or a displacement a task bounds.
    """

    kind: ClassVar[PredicateKind] = PredicateKind.CONTINUOUS
    """
    A real quantity, so how exactly a bound on it is represented depends on the region
    the bound describes.
    """


@dataclass(eq=False)
class UnclassifiedValue(MeasuredValue):
    """
    A value whose kind the suite's vocabulary does not establish.
    """

    kind: ClassVar[PredicateKind] = PredicateKind.UNCLASSIFIED
    """
    Unestablished, so the survey makes no claim about representing it.
    """


# %% a condition the survey did not read


@dataclass(eq=False)
class UnreadCondition(Predicate):
    """
    A part of a success condition the survey did not read.

    Stated rather than dropped, so a survey reports its own coverage instead of
    presenting a partial reading as a complete one.
    """

    description: Variable
    """
    What was found in place of a condition the survey recognises.
    """

    @classmethod
    def found(cls, description: str) -> SymbolicExpression:
        """
        :param description: What was found in place of a condition the survey recognises.
        :return: The EQL condition standing for the part that was not read.
        """
        return cls(description=Literal(_value_=description, _name__=description))

    def __call__(self) -> bool:
        raise ConditionIsNotObservable(stated_under=self.description._value_)

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :return: The clause *"<description> stays unread"*.
        """
        return clause(Noun(fields["description"]), Verb("stay"), Noun.bare("unread"))


# %% a condition that holds always or never


class ConstantCondition(enum.Enum):
    """
    The outcome a bare ``return True`` or ``return False`` states, which decides nothing
    about a scene but still composes with what surrounds it.
    """

    ALWAYS = True
    """
    Holds whatever the scene is, the identity of conjunction.
    """

    NEVER = False
    """
    Holds for no scene, the identity of disjunction.
    """

    @property
    def expression(self) -> Literal:
        """
        :return: The EQL literal stating this outcome.
        """
        return Literal(_value_=self.value, _name__=self.name.lower())

    @classmethod
    def stated_by(cls, expression: SymbolicExpression) -> Optional[ConstantCondition]:
        """
        :param expression: An EQL condition.
        :return: The outcome it states outright, or ``None`` when it depends on a scene.
        """
        if not isinstance(expression, Literal) or not isinstance(
            expression._value_, bool
        ):
            return None
        return cls(expression._value_)


# %% the propositions a condition states


@dataclass(frozen=True)
class StatedProposition(ABC):
    """
    One indivisible statement a task's condition makes about a scene.

    Everything above it is EQL's own composition, so this is where the survey's
    questions -- what kind of set is described, over what objects, and which variable
    represents it -- are answered.
    """

    expression: SymbolicExpression
    """
    The EQL node stating it.
    """

    @classmethod
    def stated_by(cls, expression: SymbolicExpression) -> Optional[StatedProposition]:
        """
        :param expression: A node of an EQL condition.
        :return: The proposition it states, or ``None`` when it composes others rather than
            stating one of its own.
        """
        if isinstance(expression, Comparator) and MeasuredComparison.measure_of(
            expression
        ):
            return MeasuredComparison(expression=expression)
        if cls.applies(expression, StatedRelation):
            return AssertedRelation(expression=expression)
        return None

    @staticmethod
    def applies(
        expression: SymbolicExpression, applied: type[Predicate | SymbolicFunction]
    ) -> bool:
        """
        EQL represents a predicate or a symbolic function applied to variables as an
        :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable` naming the
        class rather than as an instance of it, so what a node applies is read off that
        name.

        :param expression: A node of an EQL condition.
        :param applied: The predicate or symbolic function the node may apply.
        :return: Whether the node applies it.
        """
        return (
            isinstance(expression, InstantiatedVariable)
            and isinstance(expression._type_, type)
            and issubclass(expression._type_, applied)
        )

    @property
    @abstractmethod
    def kind(self) -> PredicateKind:
        """
        :return: The kind of set this proposition describes.
        """

    @property
    @abstractmethod
    def stated_under(self) -> str:
        """
        :return: The dotted name the task states this proposition under.
        """

    @property
    @abstractmethod
    def stated_of(self) -> str:
        """
        :return: The objects it is stated of, as the task writes them.
        """

    @property
    @abstractmethod
    def variable_name(self) -> str:
        """
        :return: What this proposition constrains, which is what two occurrences have to
            agree on to be represented by one variable.
        """


@dataclass(frozen=True)
class AssertedRelation(StatedProposition):
    """
    A relation the condition states of named objects.
    """

    @property
    def kind(self) -> PredicateKind:
        return self.expression._type_.kind

    @property
    def stated_under(self) -> str:
        return self.expression._kwargs_["stated_under"]

    @property
    def stated_of(self) -> str:
        return SceneObject.named_by(self.expression._kwargs_["objects"])

    @property
    def variable_name(self) -> str:
        """
        :return: The relation together with the objects it is stated of, since one relation
            asserted of two objects states two conditions.
        """
        return f"{self.stated_under}{self.stated_of}"


@dataclass(frozen=True)
class MeasuredComparison(StatedProposition):
    """
    A value the condition measures over named objects and compares against a bound.
    """

    @staticmethod
    def measure_of(comparison: Comparator) -> Optional[InstantiatedVariable]:
        """
        :param comparison: A comparison a condition states.
        :return: The measured value it compares, or ``None`` when it compares none.
        """
        for operand in (comparison.left, comparison.right):
            if StatedProposition.applies(operand, MeasuredValue):
                return operand
        return None

    @property
    def measure(self) -> InstantiatedVariable:
        """
        :return: The measured value the comparison bounds.
        """
        return self.measure_of(self.expression)

    @property
    def bound(self) -> Any:
        """
        :return: What the value is compared against, as the task writes it.
        """
        compared_against = (
            self.expression.right
            if self.expression.left is self.measure
            else self.expression.left
        )
        return compared_against._value_

    @property
    def kind(self) -> PredicateKind:
        return self.measure._type_.kind

    @property
    def stated_under(self) -> str:
        return self.measure._kwargs_["stated_under"]

    @property
    def stated_of(self) -> str:
        return SceneObject.named_by(self.measure._kwargs_["objects"])

    @property
    def variable_name(self) -> str:
        """
        :return: The measured value together with the bound put on it, written in the order
            the comparison relates them, since bounding one value two ways describes two
            different sets and a containment does not say the same thing either way round.
        """
        measured = f"{self.stated_under}{self.stated_of}"
        if self.expression.left is self.measure:
            return f"{measured} {self.expression._name_} {self.bound}"
        return f"{self.bound} {self.expression._name_} {measured}"


# %% what a whole condition combines


class ConditionKind(enum.Enum):
    """
    What kinds of proposition a whole success condition combines.
    """

    DISCRETE = "states only conditions over finite domains"
    """
    Every proposition is discrete, so the whole condition maps onto symbolic variables.
    """

    CONTINUOUS = "states only conditions on continuous quantities"
    """
    Every proposition is continuous, so the whole condition maps onto continuous
    variables.
    """

    HYBRID = "states both discrete and continuous conditions"
    """
    The condition mixes both, so representing it needs a product space over variables of
    both kinds at once.
    """

    UNDETERMINED = "states no proposition this survey recognises"
    """
    No proposition was recognised, so the condition's kind is not established.
    """

    @classmethod
    def of(cls, kinds: set[PredicateKind]) -> ConditionKind:
        """
        :param kinds: Kinds of the propositions a condition states.
        :return: The kind of the condition as a whole.
        """
        has_discrete = PredicateKind.DISCRETE in kinds
        has_continuous = PredicateKind.CONTINUOUS in kinds
        if has_discrete and has_continuous:
            return cls.HYBRID
        if has_discrete:
            return cls.DISCRETE
        if has_continuous:
            return cls.CONTINUOUS
        return cls.UNDETERMINED


@dataclass(frozen=True)
class StatedTaskCondition:
    """
    The success condition one task states, as an EQL expression.

    Wrapping the expression is what gives a survey its questions in one place: EQL
    composes the condition, and this reads back the propositions, the quantifications
    and the coverage that composition holds.
    """

    expression: SymbolicExpression
    """
    The EQL condition the task states.
    """

    def nodes(self) -> Iterator[SymbolicExpression]:
        """
        :return: The condition's expression followed by every node nested in it, each once
            however many times the condition reaches it.
        """
        yield self.expression
        yield from self.expression._descendants_

    def propositions(self) -> list[StatedProposition]:
        """
        :return: Every proposition the condition states, each once however many times
            the condition states it.
        """
        stated = (StatedProposition.stated_by(node) for node in self.nodes())
        return [proposition for proposition in stated if proposition is not None]

    def quantifications(self) -> list[QuantifiedConditional]:
        """
        :return: Every quantification over the objects a scene provides that the
            condition states.
        """
        return [
            node for node in self.nodes() if isinstance(node, QuantifiedConditional)
        ]

    def unread_parts(self) -> list[str]:
        """
        :return: What was found in place of a condition the survey recognises, once per part
            it did not read.
        """
        return [
            node._kwargs_["description"]._value_
            for node in self.nodes()
            if StatedProposition.applies(node, UnreadCondition)
        ]

    def is_fully_read(self) -> bool:
        """
        :return: Whether the survey recovered the whole condition, with no part left unread.
        """
        return not self.unread_parts()

    @property
    def kind(self) -> ConditionKind:
        """
        :return: The kind of set the condition as a whole describes.
        """
        return ConditionKind.of(
            {proposition.kind for proposition in self.propositions()}
        )


@dataclass(frozen=True)
class PredicateNameRule:
    """
    Assigns a kind to every name carrying a given marker.

    A suite that names its success predicates by convention rather than registering them
    has a long tail of per-fixture checks, which is recognised by the marker a name
    carries instead of being enumerated one by one.
    """

    marker: str
    """
    Substring a name carries when this rule applies to it.
    """
    kind: PredicateKind
    """
    Kind assigned to names carrying the marker.
    """

    def applies_to(self, name: str) -> bool:
        """
        :param name: Dotted name to test.
        :return: Whether this rule assigns that name its kind.
        """
        return self.marker in name


@dataclass
class PredicateVocabulary:
    """
    Classifies the names a suite's success conditions are stated under.

    A suite states which conditions it applies and what they mean, so a suite supplies
    its own vocabulary: the frequent names outright, and the long tail by the naming
    convention the suite follows. Recognising the tail by convention is what keeps a
    name the survey has never seen from being silently dropped.
    """

    exact_kinds: dict[str, PredicateKind] = field(default_factory=dict)
    """
    Kind of each name the suite states outright, keyed by dotted name.
    """

    name_rules: tuple[PredicateNameRule, ...] = ()
    """
    Rules recognising the suite's remaining names by naming convention, consulted in
    order once no exact name matched.
    """

    def classify(self, name: str) -> PredicateKind:
        """
        :param name: Dotted name to classify.
        :return: The kind the name states, or :attr:`PredicateKind.UNCLASSIFIED` when
            neither an exact name nor a naming convention recognises it.
        """
        if name in self.exact_kinds:
            return self.exact_kinds[name]
        for rule in self.name_rules:
            if rule.applies_to(name):
                return rule.kind
        return PredicateKind.UNCLASSIFIED

    def classify_comparison(
        self, measure_name: str, compared_values: tuple[object, ...]
    ) -> PredicateKind:
        """
        What a comparison bounds is settled by the value it is compared against rather than
        by what the measure is called: a threshold is a number, whereas a state or a mode is
        a string or a truth value. The measure's name decides only when nothing is compared
        against a literal.

        :param measure_name: Dotted name of the value being compared.
        :param compared_values: Literal values the value is compared against.
        :return: The kind the comparison states.
        """
        for value in compared_values:
            if isinstance(value, (bool, str)):
                return PredicateKind.DISCRETE
            if isinstance(value, (int, float)):
                return PredicateKind.CONTINUOUS
        return self.classify(measure_name)
