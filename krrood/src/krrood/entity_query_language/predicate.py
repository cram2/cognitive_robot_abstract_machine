"""
Predicates and symbolic function utilities for the Entity Query Language.

This module defines predicate classes for boolean checks and a decorator to build symbolic expressions
from regular Python functions when variables are present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

from typing_extensions import (
    Callable,
    Iterator,
    Optional,
    Any,
    Type,
    Tuple,
    ClassVar,
    Mapping,
    Sized,
    TYPE_CHECKING,
    Dict,
    Union,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.fragments.base import (
        VerbalizationFragment,
    )

from krrood.entity_query_language.utils import T, merge_args_and_kwargs
from krrood.entity_query_language.core.variable import (
    Variable,
    InstantiatedVariable,
    Literal,
)
from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.core.base_expressions import Selectable
from krrood.entity_query_language.utils import camel_case_to_words
from krrood.patterns.code_parsing_utils import (
    get_accessed_attribute_name_in_return_statement_of_property,
)
from krrood.symbol_graph.symbol_graph import Symbol


def symbolic_function(
    function: Callable[..., T],
) -> Union[Callable[..., Variable[T]], T]:
    """
    Function decorator that constructs a symbolic expression representing the function call
     when inside a symbolic_rule context.

    When symbolic mode is active, calling the method returns a Call instance which is a SymbolicExpression bound to
    representing the method call that is not evaluated until the evaluate() method is called on the query/rule.

    :param function: The function to decorate.
    :return: The decorated function.
    """

    @wraps(function)
    def wrapper(*args, **kwargs) -> Optional[Any]:
        all_kwargs = merge_args_and_kwargs(function, args, kwargs)
        if _any_of_the_kwargs_is_a_variable(all_kwargs):
            return InstantiatedVariable(
                _type_=function,
                _kwargs_=all_kwargs,
            )
        return function(*args, **kwargs)

    return wrapper


def functional_form(symbolic_callable: Type[T]) -> Callable[..., Any]:
    """:return: a function that calls *symbolic_callable* -- the class-form counterpart of the
    :func:`symbolic_function` decorator.

    It returns a symbolic expression when any argument is a variable (so it composes in a query) and
    the directly computed value otherwise. Binding an existing function name to
    ``functional_form(TheClass)`` lets a migration to the class form keep that name's call behaviour
    unchanged -- the logic moves into the class's :meth:`__call__` and the name just constructs it.
    """

    def call(*args: Any, **kwargs: Any) -> Any:
        result = symbolic_callable(*args, **kwargs)
        return result() if isinstance(result, symbolic_callable) else result

    return call


@dataclass(frozen=True)
class VerbalizationField:
    """One predicate field as ``_verbalization_fragment_`` sees it.

    It carries both the field's already-rendered (and source-linked) :attr:`fragment` and the raw
    :attr:`value` bound to it (a :class:`Literal`'s value unwrapped). A part-of-speech element takes
    whichever it needs — :class:`Noun` uses the fragment, :class:`OneOf` uses the value — so the
    author just passes ``fields[name]`` and the right thing happens, never an explicit accessor.
    """

    fragment: VerbalizationFragment
    """The field's rendered, source-linked fragment — what :class:`Noun` uses."""

    value: Any
    """The raw Python value bound to the field (a literal's value) — what :class:`OneOf` enumerates."""

    def as_fragment(self) -> VerbalizationFragment:
        """:return: the field's rendered fragment, so a :class:`VerbalizationField` is a clause constituent like
        the part-of-speech elements — ``clause(field)`` and ``Noun(field)`` both work.
        """
        return self.fragment


@dataclass(frozen=True)
class RenderedFields(Mapping):
    """The arguments passed to :meth:`Verbalizable._verbalization_fragment_`.

    A mapping of *field name → :class:`VerbalizationField`*. Each ``fields["x"]`` carries both the rendered
    fragment and the raw value, so it can be passed straight to a part-of-speech element — ``Noun``
    takes the fragment, ``OneOf`` takes the value — without the author choosing between them.
    """

    fragments: "Mapping[str, VerbalizationFragment]"
    """The rendered fragment for each field, keyed by field name."""

    raw: "Mapping[str, SymbolicExpression]"
    """The raw child expression for each field, keyed by field name."""

    def __getitem__(self, field_name: str) -> VerbalizationField:
        raw = self.raw[field_name]
        value = raw._value_ if isinstance(raw, Literal) else raw
        return VerbalizationField(fragment=self.fragments[field_name], value=value)

    def __iter__(self) -> Iterator[str]:
        return iter(self.fragments)

    def __len__(self) -> int:
        return len(self.fragments)


@dataclass(eq=False)
class Verbalizable(ABC):
    """
    A mixin for classes that want to add custom verbalization, such that when a query that is using them is verbalized,
    the final output text is more correct or intuitivie.
    """

    @classmethod
    @abstractmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Structured verbalization for this predicate — a required clause (no string fallback).

        Build the clause from the typed part-of-speech vocabulary
        (:func:`~…vocabulary.parts_of_speech.clause` with ``Noun`` / ``Verb`` / ``Copula`` /
        ``Prepositions`` / ``Adjective``), composing the already-rendered field fragments in *fields*
        (keyed by field name). State only the **affirmative, present-tense** form: a ``Verb`` is given
        as its lemma, and the realisation passes inflect it (*"work"* → *"works"*) and agree its
        number. Returning a typed clause rather than a string keeps the predicate composable — a
        wrapping ``Not`` negates it automatically (a verb with do-support, *"does not love"*; a copula
        with suppletion, *"is not reachable"*) and coreference still reduces the operands.

        ..note:: Abstract, so every concrete predicate must supply a clause; a missing implementation
            fails at (concrete) instantiation rather than only when the predicate is verbalized.

        Example::

            @dataclass(eq=False)
            class Loves(Predicate):
                person_1: Person
                person_2: Person

                @classmethod
                def _verbalization_fragment_(cls, fields):
                    return clause(Noun(fields["person_1"]), Verb("love"), Noun(fields["person_2"]))

        :param fields: The rendered fragment for each predicate field, keyed by field name.
        :return: The predicate's verbalization fragment.
        """


@dataclass(eq=False)
class SymbolicCallable(Symbol, Verbalizable, ABC):
    """A user-defined, self-verbalizing symbolic operation.

    It is CALLED with arguments, is represented as an :class:`InstantiatedVariable` in a query when
    any argument is symbolic, and renders itself through its required
    :meth:`Verbalizable._verbalization_fragment_`. :class:`Predicate` (a boolean operation) and
    :class:`SymbolicFunction` (a value operation) are its two concrete kinds, so the
    symbolic-construction machinery lives here once rather than being duplicated in each.
    """

    _cache_instances_: ClassVar[bool] = False
    """Instances are not cached -- they do not persist."""

    def __new__(cls, *args, **kwargs):
        all_kwargs = merge_args_and_kwargs(
            cls.__init__, args, kwargs, ignore_first=True
        )
        if _any_of_the_kwargs_is_a_variable(all_kwargs):
            return InstantiatedVariable(
                _type_=cls,
                _kwargs_=all_kwargs,
            )
        return super().__new__(cls)

    @classmethod
    def _construct_normally_(cls, **kwargs) -> SymbolicCallable:
        """
        Construct a concrete instance directly, bypassing the symbolic ``__new__`` redirect.

        Normally, calling ``cls(**kwargs)`` when any kwarg is a :class:`Selectable` redirects
        construction to an :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`
        so the call can be represented as a symbolic expression in a query graph.  During evaluation,
        however, the bound values themselves may be :class:`Selectable` objects (e.g. in a
        meta-query that reasons about EQL nodes).  In that case the redirect is wrong — we want the
        real instance so its :meth:`__call__` can be evaluated immediately.

        :meth:`_construct_normally_` is the escape hatch for that situation.  It calls
        ``object.__new__`` directly (skipping ``__new__`` entirely) and then ``__init__``, so the
        caller always receives a fully initialised concrete instance regardless of what the kwargs
        contain.
        """
        instance = object.__new__(cls)
        instance.__init__(**kwargs)
        return instance

    @classmethod
    def _bound_value_(cls, **kwargs) -> Any:
        """:return: the value this operation contributes to a query result when its arguments have
        concrete values -- the constructed instance itself by default (a :class:`Predicate`'s truth is
        then read from that instance). A value operation overrides this to its COMPUTED value.

        ..note:: This is the class-form counterpart of calling a ``@symbolic_function``: for a function
            the query binds ``function(**values)``; for a callable class it binds this.
        """
        return cls._construct_normally_(**kwargs)

    @abstractmethod
    def __call__(self) -> Any:
        """
        Evaluate the operation for the supplied values.
        """


@dataclass(eq=False)
class Predicate(SymbolicCallable, ABC):
    """
    The super predicate class that represents a filtration operation or asserts a relation.
    """

    @abstractmethod
    def __call__(self) -> bool:
        """
        Evaluate the predicate for the supplied values.
        """

    def __bool__(self):
        """
        Bool casting a predicate evaluates it.
        """
        return bool(self.__call__())


@dataclass(eq=False)
class SymbolicFunction(SymbolicCallable, ABC):
    """A user-defined operation that computes a VALUE, with its own verbalization.

    Like :class:`Predicate` it is a self-verbalizing symbolic callable, but its :meth:`__call__`
    returns a value (not a truth value), so its :meth:`Verbalizable._verbalization_fragment_` names
    that value as a NOUN PHRASE rather than a clause. When the class name itself reads as the value
    (``Length`` → *"the length of a list"*), mix in :class:`NameVerbalized` instead of writing the
    fragment by hand.
    """

    @classmethod
    def _bound_value_(cls, **kwargs) -> Any:
        """:return: the COMPUTED value -- a value operation is constructed AND called, so a query binds
        what it computes (exactly as a ``@symbolic_function`` is called), not the instance.
        """
        return cls._construct_normally_(**kwargs)()

    @abstractmethod
    def __call__(self) -> Any:
        """
        Compute the value for the supplied arguments.
        """


@dataclass(eq=False)
class NameVerbalized(Verbalizable):
    """Opt-in mixin declaring that the class NAME is its verbalization surface.

    Mixing this in (``class Length(NameVerbalized, SymbolicFunction)``) is an explicit, greppable
    choice that the CamelCase / snake_case class name reads as the operation's surface, so no
    hand-written fragment is needed: a :class:`SymbolicFunction` reads as the value noun phrase
    *"the <name> of <arguments>"* (``Length`` → *"the length of a list"*) and a :class:`Predicate`
    as the name-based clause — copular for an ``Is…`` name (``IsReachable`` → *"<subject> is
    reachable"*), verb-first otherwise (``ConnectsTo`` → *"<subject> connects to <object>"*).

    Keeping :meth:`Verbalizable._verbalization_fragment_` abstract and making the name-based reading
    an explicit mixin (rather than a silently inherited default) means an author must always *decide*
    how a new operation is said — a class that decides nothing fails loudly instead of producing an
    unreviewed sentence.

    ..warning:: Mix it in only after checking the produced sentence — the surface snapshot test
        (``test_verbalization_surfaces.py``) prints it for every symbolic callable, so it shows up
        in the PR diff. When the name does not read as the surface, implement
        :meth:`Verbalizable._verbalization_fragment_` with a
        :func:`~krrood.entity_query_language.verbalization.vocabulary.parts_of_speech.clause`
        instead.
    """

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """:return: the name-based surface — the value noun phrase for a :class:`SymbolicFunction`,
        the name-based clause (first operand as subject, the rest trailing objects) for a
        :class:`Predicate`.
        """
        # Imported locally to avoid the core -> verbalization import cycle (as Triple does).
        from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
            predicate_clause,
            value_function_phrase,
        )

        if issubclass(cls, SymbolicFunction):
            return value_function_phrase(cls.__name__, *fields.values())
        subject, *objects = fields.values()
        return predicate_clause(cls.__name__, subject, *objects)


@dataclass(eq=False)
class Triple(Predicate):
    """
    A Triple is a type predicate that represents a relation between two entities.
    To know if your predicate is a Triple or not ask yourself can I say "subject" "predicate_name" "object" and it
    makes sense? if so then yes. Check the verbalization function below as a reference.
    """

    @property
    @abstractmethod
    def subject(self) -> Any:
        """
        The subject of the predicate.
        """

    @property
    @abstractmethod
    def object(self) -> Any:
        """
        The object of the predicate.
        """

    @classmethod
    def _verbalization_fragment_(
        cls, fields: Mapping[str, VerbalizationFragment]
    ) -> VerbalizationFragment:
        """
        Verbalization of a Triple is a subject - verb-phrase - object, where the verb phrase is read
        off the class name (``ConnectsTo`` → *"connects to"*). The leading word is a :class:`Verb`
        (its lemma), so a wrapping ``Not`` negates with do-support (*"does not connect to"*).
        """
        # Imported locally: the verbalization layer depends on the core predicate types, so a
        # module-level import here would close an import cycle.
        from krrood.entity_query_language.verbalization import morphology
        from krrood.entity_query_language.verbalization.fragments.base import (
            WordFragment,
        )
        from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
            clause,
            Noun,
            Verb,
        )

        words = camel_case_to_words(cls.__name__).split()
        subject_name = get_accessed_attribute_name_in_return_statement_of_property(
            cls.subject, cls
        )
        object_name = get_accessed_attribute_name_in_return_statement_of_property(
            cls.object, cls
        )
        particles = [WordFragment(text=word) for word in words[1:]]
        return clause(
            Noun(fields[subject_name]),
            Verb(morphology.verb_lemma(words[0])),
            *particles,
            Noun(fields[object_name]),
        )


@dataclass(eq=False)
class HasType(Triple):
    """
    Represents a predicate to check if a given variable is an instance of a specified type.

    This class is used to evaluate whether the domain value belongs to a given type by leveraging
    Python's built-in `isinstance` functionality. It provides methods to retrieve the domain and
    range values and perform direct checks.
    """

    variable: Any
    """
    The variable whose type is being checked.
    """
    types_: Type
    """
    The type or tuple of types against which the `variable` is validated.
    """

    def __call__(self) -> bool:
        return isinstance(self.variable, self.types_)

    @property
    def subject(self):
        return self.variable

    @property
    def object(self) -> Any:
        return self.types_

    @classmethod
    def _verbalization_fragment_(
        cls, fields: Mapping[str, VerbalizationFragment]
    ) -> VerbalizationFragment:
        # Imported locally to avoid the core → verbalization import cycle (see :class:`Triple`).
        from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
            Adjective,
            clause,
            Copula,
            Noun,
        )

        return clause(
            Noun(fields["variable"]),
            Copula(),
            Adjective("of type"),
            Noun(fields["types_"]),
        )


@dataclass(eq=False)
class HasTypes(HasType):
    """
    Represents a specialized data structure holding multiple types.

    This class is a data container designed to store and manage a tuple of
    types. It inherits from the `HasType` class and extends its functionality
    to handle multiple types efficiently. The primary goal of this class is to
    allow structured representation and access to a collection of type
    information with equality comparison explicitly disabled.
    """

    types_: Tuple[Type, ...]
    """
    A tuple containing Type objects that are associated with this instance.
    """

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """Say membership over the admissible types — *"<variable> is one of A, B, or C"*. The
        :class:`OneOf` element handles the bounded listing (linking, *"or"*, the count cap), so the
        types are read from the field's value (an ``isinstance`` over the tuple is membership, not the
        tuple value an equality would mean)."""
        # Imported locally to avoid the core → verbalization import cycle (see :class:`Triple`).
        from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
            clause,
            Copula,
            Noun,
            OneOf,
        )

        return clause(Noun(fields["variable"]), Copula(), OneOf(fields["types_"]))


@dataclass(eq=False)
class Length(NameVerbalized, SymbolicFunction):
    """The number of items in an iterable, as a value operation.

    Reads through the explicit :class:`NameVerbalized` surface (*"the length of <iterable>"*).
    """

    iterable: Sized
    """The iterable whose length is computed."""

    def __call__(self) -> int:
        return len(self.iterable)


length = functional_form(Length)
"""Backward-compatible functional form of :class:`Length` (keeps the ``length(iterable)`` call)."""


def _any_of_the_kwargs_is_a_variable(bindings: Dict[str, Any]) -> bool:
    """
    :param bindings: A kwarg like dict mapping strings to objects
    :return: ``True`` if any value in ``bindings`` is a :class:`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`, ``False`` otherwise.
    """
    return any(isinstance(binding, SymbolicExpression) for binding in bindings.values())


@dataclass(eq=False)
class Is(Predicate):
    """
    Predicate asserting that two operands refer to the same object in memory.
    """

    first_entity: Any
    """The first entity"""

    second_entity: Any
    """The second entity."""

    def __call__(self) -> bool:
        return self.first_entity is self.second_entity
