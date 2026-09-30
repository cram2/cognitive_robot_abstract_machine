"""
What a query box may name: the variables, types, classes and keywords one query's
namespace holds, and the members that follow a name's dot.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum

from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.entity_query_language.scope import eql_factory_namespace
from krrood.exceptions import DataclassException
from typing_extensions import Any, Dict, List, Optional

from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.workspace_classes import WorkspaceClassIndex


class VocabularyKind(Enum):
    """
    What one name a query may use stands for.
    """

    VARIABLE = "variable"
    """
    A ready-made variable ranging over one of the queried domains.
    """

    ENTITY_TYPE = "entity_type"
    """
    The type one ready-made variable ranges over.
    """

    CLASS = "class"
    """
    A class of the workspace, resolved when a query first names it.
    """

    FACTORY = "factory"
    """
    One of EQL's own keywords, such as ``the`` or ``entity``.
    """

    VALUE = "value"
    """
    A value put within a query's reach by hand, such as a domain's raw list.
    """

    FIELD = "field"
    """
    A dataclass field of the type before the dot.
    """

    PROPERTY = "property"
    """
    A property of the type before the dot.
    """

    METHOD = "method"
    """
    A method of the type before the dot.
    """


@dataclass
class UnknownVocabularyName(DataclassException):
    """
    Raised when the members of a name no query can use are asked for.
    """

    name: str
    """
    The name whose members were asked for.
    """

    def error_message(self) -> str:
        """Identify the type missing from the query vocabulary."""
        return f"No type named '{self.name}' is in reach of a query."

    def suggest_correction(self) -> str:
        """Direct the query author to the available vocabulary."""
        return "ask for the members of a type the vocabulary offers."


@dataclass(frozen=True)
class VocabularyEntry:
    """
    One name a query may use, as the query box shows it.
    """

    name: str
    """
    The name itself, as it is written in a query.
    """

    kind: VocabularyKind
    """
    What the name stands for.
    """

    detail: str = ""
    """
    One line about the name: a docstring summary, or the type a variable ranges over.
    """

    module: str = ""
    """
    Module the name is defined in, for a name that has one.
    """

    type_name: str = ""
    """
    Name of the type whose members follow this name's dot, when it has one.
    """

    further_modules: int = 0
    """
    How many further modules define a class of the same name, which this name does not
    resolve to.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the query box reads.
        """
        return {
            "name": self.name,
            "kind": self.kind.value,
            "detail": self.detail,
            "module": self.module,
            "type": self.type_name,
            "further_modules": self.further_modules,
        }


@dataclass
class QueryVocabulary:
    """
    Everything one query's namespace answers to, described for the query box.

    Built from the same domains, hand-placed names and class index the namespace is,
    so what the box offers is what a query can actually name.
    """

    domains: List[QueryDomain] = field(default_factory=list)
    """
    The ready-made variables a query may range over.
    """

    extra_names: Dict[str, Any] = field(default_factory=dict)
    """
    Further names placed within a query's reach by hand.
    """

    class_index: WorkspaceClassIndex = field(default_factory=WorkspaceClassIndex)
    """
    The workspace classes a query may name.
    """

    def entries(self) -> List[VocabularyEntry]:
        """
        Every name a query may use, the ones it was built for first.

        Ordered variables, entity types, hand-placed names, EQL keywords, then the
        workspace's classes alphabetically, and each name appears once.
        """
        offered: Dict[str, VocabularyEntry] = {}
        for entry in self._variables() + self._hand_placed() + self._factories():
            offered.setdefault(entry.name, entry)
        for entry in self._workspace_classes():
            offered.setdefault(entry.name, entry)
        return list(offered.values())

    def members_of(self, name: str) -> List[VocabularyEntry]:
        """
        The members that follow this name's dot: fields first, then properties, then
        methods, each named once even when a base class also declares it.

        :param name: A type's name, or a variable's, as a query writes it.
        :raises UnknownVocabularyName: When no type of that name is in reach.
        """
        owner = self._type_named(name)
        if owner is None:
            raise UnknownVocabularyName(name=name)
        members: Dict[str, VocabularyEntry] = {}
        offered = (
            self._fields_of(owner)
            + self._declared_members(owner, VocabularyKind.PROPERTY)
            + self._declared_members(owner, VocabularyKind.METHOD)
        )
        for member in offered:
            members.setdefault(member.name, member)
        return list(members.values())

    def to_payload(self) -> Dict[str, Any]:
        """
        The vocabulary as the query box is served it.
        """
        return {
            "ok": True,
            "entries": [entry.to_payload() for entry in self.entries()],
        }

    def members_payload(self, name: str) -> Dict[str, Any]:
        """
        One type's members as the query box is served them.

        :param name: A type's name, or a variable's, as a query writes it.
        :raises UnknownVocabularyName: When no type of that name is in reach.
        """
        return {
            "ok": True,
            "name": name,
            "members": [member.to_payload() for member in self.members_of(name)],
        }

    # %% where each kind of name comes from

    def _variables(self) -> List[VocabularyEntry]:
        """
        The ready-made variables and the types they range over.
        """
        offered = []
        for domain in self.domains:
            entity_type = domain.entity_type
            offered.append(
                VocabularyEntry(
                    name=domain.name,
                    kind=VocabularyKind.VARIABLE,
                    detail=f"ranges over {entity_type.__name__}",
                    module=entity_type.__module__,
                    type_name=entity_type.__name__,
                )
            )
        for domain in self.domains:
            offered.append(self._type_entry(domain.entity_type))
        return offered

    def _hand_placed(self) -> List[VocabularyEntry]:
        """
        The names placed within a query's reach beside its domains.
        """
        return [
            (
                self._type_entry(value)
                if isinstance(value, type)
                else VocabularyEntry(
                    name=name,
                    kind=VocabularyKind.VALUE,
                    detail=self._summary_of(value) or type(value).__name__,
                )
            )
            for name, value in self.extra_names.items()
        ]

    def _factories(self) -> List[VocabularyEntry]:
        """
        EQL's own keywords, as the language puts them in every namespace.
        """
        return [
            VocabularyEntry(
                name=name,
                kind=VocabularyKind.FACTORY,
                detail=self._summary_of(value),
            )
            for name, value in sorted(eql_factory_namespace().items())
        ]

    def _workspace_classes(self) -> List[VocabularyEntry]:
        """
        Every class of the workspace, named by the module a bare name resolves to.
        """
        offered = []
        for name in self.class_index.names():
            candidates = self.class_index.candidates(name)
            offered.append(
                VocabularyEntry(
                    name=name,
                    kind=VocabularyKind.CLASS,
                    detail=candidates[0].docstring_summary,
                    module=candidates[0].module,
                    type_name=name,
                    further_modules=len(candidates) - 1,
                )
            )
        return offered

    def _type_entry(self, entity_type: type) -> VocabularyEntry:
        """
        One type as the query box offers it.

        :param entity_type: The type to describe.
        """
        return VocabularyEntry(
            name=entity_type.__name__,
            kind=VocabularyKind.ENTITY_TYPE,
            detail=self._summary_of(entity_type),
            module=entity_type.__module__,
            type_name=entity_type.__name__,
        )

    # %% resolving a name to the type whose members follow its dot

    def _type_named(self, name: str) -> Optional[type]:
        """
        The type a name stands for, or whose instances a variable ranges over.

        :param name: A type's name, or a variable's.
        """
        for domain in self.domains:
            if name == domain.name or name == domain.entity_type.__name__:
                return domain.entity_type
        placed = self.extra_names.get(name)
        if isinstance(placed, type):
            return placed
        resolved = self.class_index.resolve(name)
        return resolved if isinstance(resolved, type) else None

    def _fields_of(self, owner: type) -> List[VocabularyEntry]:
        """
        The dataclass fields of a type, described by what they hold.

        A field with no default is declared as an annotation only, so the classes
        themselves cannot be read for it -- krrood's own introspector is what knows
        which attributes a dataclass really offers.

        :param owner: The type whose fields are wanted.
        """
        return [
            VocabularyEntry(
                name=discovered.public_name,
                kind=VocabularyKind.FIELD,
                detail=str(discovered.field.type),
                module=owner.__module__,
                type_name=owner.__name__,
            )
            for discovered in DataclassOnlyIntrospector().discover(owner)
        ]

    def _declared_members(
        self, owner: type, kind: VocabularyKind
    ) -> List[VocabularyEntry]:
        """
        One kind of member declared on a type or inherited by it, the type's own first.

        Read off the classes themselves rather than through attribute access, so
        describing a type never runs one of its properties.

        :param owner: The type whose members are wanted.
        :param kind: Which kind of member to collect.
        """
        collected = []
        for declaring_type in inspect.getmro(owner):
            for name, attribute in vars(declaring_type).items():
                if name.startswith("_") or self._kind_of(attribute) is not kind:
                    continue
                collected.append(
                    VocabularyEntry(
                        name=name,
                        kind=kind,
                        detail=self._summary_of(attribute),
                        module=declaring_type.__module__,
                        type_name=declaring_type.__name__,
                    )
                )
        return collected

    @staticmethod
    def _kind_of(attribute: Any) -> Optional[VocabularyKind]:
        """
        Which kind of member an attribute of a class is, or None for neither.

        :param attribute: The attribute as the declaring class holds it.
        """
        if isinstance(attribute, property):
            return VocabularyKind.PROPERTY
        if inspect.isroutine(attribute) or isinstance(
            attribute, (classmethod, staticmethod)
        ):
            return VocabularyKind.METHOD
        return None

    @staticmethod
    def _summary_of(value: Any) -> str:
        """
        The first line of something's docstring, or ``''``.

        :param value: Whatever carries the docstring.
        """
        documentation = inspect.getdoc(value) or ""
        return documentation.strip().splitlines()[0] if documentation.strip() else ""
