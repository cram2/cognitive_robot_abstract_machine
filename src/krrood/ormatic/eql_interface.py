from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List, Any, Optional
import operator

import sqlalchemy.inspection
from sqlalchemy import and_, or_, select, Select, func, literal, not_ as sa_not
from sqlalchemy.orm import Session

from ..entity_query_language.symbolic import (
    SymbolicExpression,
    Attribute,
    Comparator,
    AND,
    OR,
    An,
    The,
    Variable,
    Literal,
)

from .dao import get_dao_class


class EQLTranslationError(Exception):
    """Raised when an EQL expression cannot be translated into SQLAlchemy."""


class UnsupportedQueryTypeError(EQLTranslationError):
    """Raised when an unsupported query type is encountered."""


class UnsupportedOperatorError(EQLTranslationError):
    """Raised when an unsupported operator is encountered."""


class UnsupportedQuantifierError(EQLTranslationError):
    """Raised when an unsupported quantifier is encountered."""


class AttributeResolutionError(EQLTranslationError):
    """Raised when an attribute cannot be resolved."""


class MissingDAOError(EQLTranslationError):
    """Raised when a DAO class cannot be found for a type."""


class DomainExtractionError(EQLTranslationError):
    """Raised when a value cannot be extracted from a domain."""


@dataclass
class VariableTypeExtractor:
    """Extracts underlying Variable and its python type from a leaf-like node."""

    def extract(self, node: Any) -> tuple[Optional[Variable], Optional[type]]:
        """
        Extract variable and type from a node.

        :param node: The node to extract from
        :return: Tuple of (variable, type)
        """
        if isinstance(node, Variable):
            return node, node._type_

        if hasattr(node, "_var_"):
            var = node._var_
            if isinstance(var, Variable):
                return var, var._type_

        node_type = node._type_
        return None, node_type


@dataclass
class AttributeChainResolver:
    """Resolves attribute chains for EQL Attribute expressions."""

    def extract_leaf_variable(self, attribute: Attribute) -> Any:
        """
        Extract the leaf variable from an attribute chain.

        :param attribute: The attribute to extract from
        :return: The leaf variable or node
        """
        extractor = VariableTypeExtractor()
        node = attribute
        while isinstance(node, Attribute):
            node = node._child_
        var, _ = extractor.extract(node)
        return var or node

    def extract_base_dao(self, attribute: Attribute) -> Optional[type]:
        """
        Extract the base DAO class from an attribute chain.

        :param attribute: The attribute to extract from
        :return: The DAO class or None
        """
        extractor = VariableTypeExtractor()
        node = attribute
        while isinstance(node, Attribute):
            node = node._child_
        _, node_type = extractor.extract(node)
        return get_dao_class(node_type) if node_type is not None else None


@dataclass
class RelationshipResolver:
    """Resolves relationships and foreign keys for DAO classes."""

    def resolve_relationship_and_foreign_key(
        self, dao_class: type, attribute_name: str
    ) -> tuple[Any, Any]:
        """
        Resolve the relationship and foreign key column for a DAO attribute.

        :param dao_class: The DAO class
        :param attribute_name: The attribute name
        :return: Tuple of (relationship, foreign_key_column)
        """
        mapper = sqlalchemy.inspection.inspect(dao_class)
        relationship = self._find_relationship(mapper, attribute_name)

        if relationship is None:
            return None, None

        local_column = next(iter(relationship.local_columns))
        foreign_key_column = getattr(dao_class, local_column.key)
        return relationship, foreign_key_column

    def _find_relationship(self, mapper: Any, attribute_name: str) -> Any:
        """
        Find a relationship by name in a mapper or aliased mapper.

        :param mapper: The SQLAlchemy mapper or alias inspection
        :param attribute_name: The attribute name to find
        :return: The relationship or None
        """
        # Support both Mapper and AliasedInsp from sqlalchemy.inspection.inspect()
        relationships = None
        if hasattr(mapper, "relationships"):
            relationships = mapper.relationships
        elif hasattr(mapper, "mapper") and hasattr(mapper.mapper, "relationships"):
            relationships = mapper.mapper.relationships
        else:
            return None

        relationship = relationships.get(attribute_name)
        if relationship is not None:
            return relationship

        for rel in relationships:
            if rel.key == attribute_name:
                return rel

        return None


@dataclass
class OperatorMapper:
    """Maps EQL operators to SQLAlchemy expressions."""

    def map_comparison_operator(self, operation: Any, left: Any, right: Any) -> Any:
        """
        Map a comparison operator to a SQLAlchemy expression.

        :param operation: The operator
        :param left: Left operand
        :param right: Right operand
        :return: SQLAlchemy expression
        """
        operator_name = operation.__name__

        if operation is operator.eq or operator_name == "eq":
            return left == right
        if operation is operator.gt or operator_name == "gt":
            return left > right
        if operation is operator.lt or operator_name == "lt":
            return left < right
        if operation is operator.ge or operator_name == "ge":
            return left >= right
        if operation is operator.le or operator_name == "le":
            return left <= right
        if operation is operator.ne or operator_name == "ne":
            return left != right

        raise UnsupportedOperatorError(f"Unknown operator: {operation}")

    def map_contains_operator(self, operation: Any, left: Any, right: Any) -> Any:
        """
        Map a contains operator to a SQLAlchemy expression.

        :param operation: The operator
        :param left: Left operand
        :param right: Right operand
        :return: SQLAlchemy expression
        """
        operator_name = operation.__name__
        is_negated = operator_name == "not_contains"

        if isinstance(left, (list, tuple, set)):
            expression = right.in_(left)
        elif isinstance(right, (list, tuple, set)):
            expression = left.in_(right)
        elif isinstance(left, str) and not isinstance(right, str):
            expression = func.instr(literal(left), right) > 0
        elif not isinstance(left, str) and isinstance(right, str):
            if hasattr(left, "contains"):
                expression = left.contains(right)
            else:
                expression = left.like("%" + right + "%")
        elif isinstance(left, str) and isinstance(right, str):
            expression = literal(right in left)
        else:
            expression = func.instr(left, right) > 0

        return sa_not(expression) if is_negated else expression


@dataclass
class DomainValueExtractor:
    """Extracts values from EQL Variable/Literal domains."""

    session: Session

    def extract_from_literal(self, literal_node: Literal) -> Any:
        """
        Extract values from a Literal node.

        :param literal_node: The Literal node
        :return: The extracted value(s)
        """

        if not hasattr(literal_node, "_domain_"):
            return (
                literal_node.value if hasattr(literal_node, "value") else literal_node
            )

        values = [hashed_value.value for hashed_value in literal_node._domain_]

        if len(values) > 1:
            return values
        if len(values) == 1:
            single_value = values[0]
            if isinstance(single_value, (list, tuple, set)):
                return single_value
            return single_value

        return literal_node.value if hasattr(literal_node, "value") else literal_node

    def extract_from_variable(self, variable: Variable) -> Any:
        """
        Extract a value from a Variable domain.

        :param variable: The Variable
        :return: The extracted value
        """
        if not hasattr(variable, "_domain_"):
            return variable.value if hasattr(variable, "value") else variable

        try:
            sample = next(iter(variable._domain_)).value
        except (StopIteration, AttributeError):
            return variable.value if hasattr(variable, "value") else variable

        if isinstance(variable, Literal):
            return sample

        dao_class = get_dao_class(type(sample))
        if dao_class is None:
            return sample

        if isinstance(sample, dao_class):
            return sample.id if hasattr(sample, "id") else sample

        return self._resolve_dao_instance(sample, dao_class)

    def _resolve_dao_instance(self, sample: Any, dao_class: type) -> Any:
        """
        Resolve a DAO instance from a sample entity.

        :param sample: The sample entity
        :param dao_class: The DAO class
        :return: The DAO id or the sample itself
        """
        filters = {}
        if hasattr(sample, "id_"):
            filters["id_"] = sample.id_
        elif hasattr(sample, "name"):
            filters["name"] = sample.name

        if filters:
            dao_instance = self.session.query(dao_class).filter_by(**filters).first()
            if dao_instance is not None:
                return dao_instance.id if hasattr(dao_instance, "id") else dao_instance

        return sample


@dataclass
class JoinManager:
    """Manages JOIN operations for the EQL translator.

    Tracks both which relationship paths have been joined and the SQLAlchemy
    alias used for each path so that downstream column references can bind to
    the correct FROM element without triggering implicit joins.
    """

    aliases_by_path: dict[tuple[type, str], Any] = field(default_factory=dict)
    joined_tables: set[type] = field(default_factory=set)

    def add_path_join(self, dao_class: type, attribute_name: str, alias: Any) -> None:
        """
        Register a path-based JOIN and its alias.

        :param dao_class: The DAO class
        :param attribute_name: The attribute name
        :param alias: The SQLAlchemy aliased entity used for the join
        """
        self.aliases_by_path[(dao_class, attribute_name)] = alias

    def is_path_joined(self, dao_class: type, attribute_name: str) -> bool:
        """
        Check if a path has already been joined.

        :param dao_class: The DAO class
        :param attribute_name: The attribute name
        :return: True if already joined
        """
        return (dao_class, attribute_name) in self.aliases_by_path

    def get_alias_for_path(self, dao_class: type, attribute_name: str) -> Any:
        """
        Get the alias associated with a previously joined path.
        """
        return self.aliases_by_path.get((dao_class, attribute_name))

    def add_table_join(self, dao_class: type) -> None:
        """
        Register a table-level JOIN.

        :param dao_class: The DAO class
        """
        self.joined_tables.add(dao_class)

    def is_table_joined(self, dao_class: type) -> bool:
        """
        Check if a table has already been joined.

        :param dao_class: The DAO class
        :return: True if already joined
        """
        return dao_class in self.joined_tables


@dataclass
class EQLTranslator:
    """
    Translate an EQL query into an SQLAlchemy query.

    This assumes the query has a structure like:
    - quantifier (an/the)
        - select like (entity, setof)
            - Root Condition
                - child 1
                - child 2
                - ...

    """

    eql_query: SymbolicExpression
    session: Session

    sql_query: Optional[Select] = None
    join_manager: JoinManager = field(default_factory=JoinManager)

    @property
    def quantifier(self) -> SymbolicExpression:
        """Get the quantifier from the query."""
        return self.eql_query

    @property
    def select_like(self) -> Any:
        """Get the select-like expression from the query."""
        return self.eql_query._child_

    @property
    def root_condition(self) -> SymbolicExpression:
        """Get the root condition from the query."""
        return self.eql_query._child_._child_

    def translate(self) -> None:
        """Translate the EQL query to SQL."""
        dao_class = get_dao_class(self.select_like.selected_variable._type_)
        if dao_class is None:
            raise MissingDAOError(
                f"No DAO class found for {self.select_like.selected_variable._type_}"
            )

        self.sql_query = select(dao_class)
        conditions = self.translate_query(self.root_condition)

        if conditions is not None:
            self.sql_query = self.sql_query.where(conditions)

    def evaluate(self) -> List[Any]:
        """
        Evaluate the translated SQL query.

        :return: Query results
        """
        bound_query = self.session.scalars(self.sql_query)

        if isinstance(self.quantifier, The):
            return bound_query.one()

        elif isinstance(self.quantifier, An):
            return bound_query.all()

        raise UnsupportedQuantifierError(f"Unknown quantifier: {type(self.quantifier)}")

    def __iter__(self):
        """Iterate over evaluation results."""
        yield from self.evaluate()

    def translate_query(self, query: SymbolicExpression) -> Optional[Any]:
        """
        Translate an EQL query expression to SQL.

        :param query: The EQL query expression
        :return: SQLAlchemy expression or None
        """
        if isinstance(query, AND):
            return self.translate_and(query)
        if isinstance(query, OR):
            return self.translate_or(query)
        if isinstance(query, Comparator):
            return self.translate_comparator(query)
        if isinstance(query, Attribute):
            return self.translate_attribute(query)

        raise UnsupportedQueryTypeError(f"Unknown query type: {type(query)}")

    def translate_and(self, query: AND) -> Optional[Any]:
        """
        Translate an eql.AND query into an sql.AND.

        :param query: EQL query
        :return: SQL expression or None if all parts are handled via JOINs.
        """
        parts = self._collect_logical_parts(query)
        return self._combine_logical_parts(parts, and_)

    def translate_or(self, query: OR) -> Optional[Any]:
        """
        Translate an eql.OR query into an sql.OR.

        :param query: EQL query
        :return: SQL expression or None if all parts are handled via JOINs.
        """
        parts = self._collect_logical_parts(query)
        return self._combine_logical_parts(parts, or_)

    def _collect_logical_parts(self, query: Any) -> List[Any]:
        """
        Collect parts from a binary logical expression.

        :param query: The logical expression (AND/OR)
        :return: List of translated parts
        """
        parts = []

        if hasattr(query, "left") and hasattr(query, "right"):
            left_part = self.translate_query(query.left)
            right_part = self.translate_query(query.right)
            if left_part is not None:
                parts.append(left_part)
            if right_part is not None:
                parts.append(right_part)
        else:
            children = query._children_ if hasattr(query, "_children_") else []
            for child in children:
                part = self.translate_query(child)
                if part is not None:
                    parts.append(part)

        return parts

    def _combine_logical_parts(self, parts: List[Any], combiner: Any) -> Optional[Any]:
        """
        Combine logical parts using a combiner function.

        :param parts: List of parts to combine
        :param combiner: The combining function (and_ or or_)
        :return: Combined expression or None
        """
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return combiner(*parts)

    def translate_comparator(self, query: Comparator) -> Optional[Any]:
        """
        Translate an eql.Comparator query into a SQLAlchemy expression.

        :param query: The comparator query
        :return: SQLAlchemy expression or None if handled via JOIN
        """
        if self._is_attribute_equality_join(query):
            join_result = self._handle_attribute_equality_join(query)
            if join_result is not None:
                return None

        left = self._translate_comparator_operand(query.left)
        right = self._translate_comparator_operand(query.right)

        operation = query.operation
        operator_name = operation.__name__

        if operation is operator.contains or operator_name in (
            "contains",
            "not_contains",
            "in_",
        ):
            return self._handle_contains_operator(query, left, right, operator_name)

        mapper = OperatorMapper()
        return mapper.map_comparison_operator(operation, left, right)

    def _is_attribute_equality_join(self, query: Comparator) -> bool:
        """
        Check if a comparator represents an attribute equality join.

        :param query: The comparator query
        :return: True if it's an attribute equality join
        """
        operation_name = query.operation.__name__
        is_equality = query.operation is operator.eq or operation_name == "eq"
        both_attributes = isinstance(query.left, Attribute) and isinstance(
            query.right, Attribute
        )
        return is_equality and both_attributes

    def _handle_attribute_equality_join(self, query: Comparator) -> Optional[bool]:
        """
        Handle an attribute equality join.

        :param query: The comparator query
        :return: True if JOIN was performed, None otherwise
        """
        resolver = AttributeChainResolver()

        left_leaf = resolver.extract_leaf_variable(query.left)
        right_leaf = resolver.extract_leaf_variable(query.right)
        left_dao = resolver.extract_base_dao(query.left)
        right_dao = resolver.extract_base_dao(query.right)

        if left_leaf is right_leaf or left_dao is None or right_dao is None:
            return None

        rel_resolver = RelationshipResolver()
        left_attribute_name = query.left._attr_name_
        right_attribute_name = query.right._attr_name_

        left_rel, left_fk = rel_resolver.resolve_relationship_and_foreign_key(
            left_dao, left_attribute_name
        )
        right_rel, right_fk = rel_resolver.resolve_relationship_and_foreign_key(
            right_dao, right_attribute_name
        )

        if left_rel is None or right_rel is None:
            return None

        anchor_dao = get_dao_class(self.select_like.selected_variable._type_)
        if anchor_dao is None:
            raise MissingDAOError("Selected variable has no DAO class")

        if left_dao is anchor_dao:
            target_dao, target_fk, anchor_fk = right_dao, right_fk, left_fk
        else:
            target_dao, target_fk, anchor_fk = left_dao, left_fk, right_fk

        if not self.join_manager.is_table_joined(target_dao):
            onclause = target_fk == anchor_fk
            self.sql_query = self.sql_query.join(target_dao, onclause=onclause)
            self.join_manager.add_table_join(target_dao)

        return True

    def _translate_comparator_operand(self, operand: Any) -> Any:
        """
        Translate a comparator operand to SQL.

        :param operand: The operand
        :return: Translated SQL value or expression
        """
        if isinstance(operand, Attribute):
            return self.translate_attribute(operand)

        if isinstance(operand, Literal):
            extractor = DomainValueExtractor(self.session)
            return extractor.extract_from_literal(operand)

        if isinstance(operand, Variable):
            extractor = DomainValueExtractor(self.session)
            return extractor.extract_from_variable(operand)

        return operand

    def _handle_contains_operator(
        self, query: Comparator, left: Any, right: Any, operator_name: str
    ) -> Any:
        """
        Handle contains/in operators with special cases.

        :param query: The comparator query
        :param left: Left operand (translated)
        :param right: Right operand (translated)
        :param operator_name: Name of the operator
        :return: SQLAlchemy expression
        """
        is_negated = operator_name == "not_contains"

        if (
            operator_name in ("contains", "in_")
            and isinstance(query.left, Literal)
            and isinstance(query.right, Attribute)
        ):
            extractor = DomainValueExtractor(self.session)
            values = extractor.extract_from_literal(query.left)

            if not isinstance(values, list):
                values = [values]

            if len(values) == 1 and isinstance(values[0], (list, tuple)):
                values = values[0]

            if len(values) != 1 or (values and not isinstance(values[0], str)):
                column = self.translate_attribute(query.right)
                expression = column.in_(values)
                return sa_not(expression) if is_negated else expression

        mapper = OperatorMapper()
        return mapper.map_contains_operator(query.operation, left, right)

    def translate_attribute(self, query: Attribute) -> Any:
        """
        Translate an eql.Attribute query into a SQLAlchemy column.

        :param query: The attribute query
        :return: SQLAlchemy column expression
        """
        attribute_names = self._collect_attribute_chain(query)
        base_class = self._extract_base_class(query)

        if base_class is None:
            raise AttributeResolutionError(
                "Attribute chain leaf does not have a class."
            )

        current_dao = get_dao_class(base_class)
        if current_dao is None:
            raise MissingDAOError(f"No DAO class found for {base_class}.")

        return self._walk_attribute_chain(current_dao, attribute_names)

    def _collect_attribute_chain(self, query: Attribute) -> List[str]:
        """
        Collect attribute names from the chain.

        :param query: The attribute query
        :return: List of attribute names (reversed, from base to leaf)
        """
        names = []
        node = query
        while isinstance(node, Attribute):
            names.append(node._attr_name_)
            node = node._child_
        return list(reversed(names))

    def _extract_base_class(self, query: Attribute) -> Optional[type]:
        """
        Extract the base class from an attribute chain.

        :param query: The attribute query
        :return: The base class or None
        """
        node = query
        while isinstance(node, Attribute):
            node = node._child_

        base_class = node._type_
        if base_class is None:
            if hasattr(node, "_var_"):
                var = node._var_
                if var is not None:
                    base_class = var._type_ if hasattr(var, "_type_") else None

        return base_class

    def _walk_attribute_chain(self, current_dao: type, names: List[str]) -> Any:
        """
        Walk the attribute chain and return the final column.

        :param current_dao: The starting DAO class
        :param names: List of attribute names to walk
        :return: SQLAlchemy column expression
        """
        rel_resolver = RelationshipResolver()

        for index, name in enumerate(names):
            mapper = sqlalchemy.inspection.inspect(current_dao)
            relationship = rel_resolver._find_relationship(mapper, name)

            if relationship is not None:
                if index == len(names) - 1:
                    local_column = next(iter(relationship.local_columns))
                    return getattr(current_dao, local_column.key)

                alias = self._apply_relationship_join(current_dao, name, relationship)
                current_dao = alias or relationship.entity.class_
                continue

            if index != len(names) - 1:
                raise AttributeResolutionError(
                    f"Attribute '{name}' on {current_dao.__name__} is not a relationship but chain continues."
                )

            if not hasattr(current_dao, name):
                raise AttributeResolutionError(
                    f"Column '{name}' not found on {current_dao.__name__}."
                )

            return getattr(current_dao, name)

        raise AttributeResolutionError("Attribute chain processing error.")

    def _apply_relationship_join(
        self, dao_class: type, attribute_name: str, relationship: Any
    ) -> Any:
        """
        Apply a JOIN for a relationship if not already joined.

        This uses an explicit SQLAlchemy alias for the relationship target and
        joins using the relationship attribute itself, allowing SQLAlchemy to
        derive the ON clause while still binding subsequent column references
        to the correct FROM element. This mirrors the default Core/ORM behavior
        for join(<entity>) that the tests compare against, but avoids implicit
        joins by returning the alias for downstream attribute resolution.

        :param dao_class: The DAO class where the relationship is defined
        :param attribute_name: The relationship attribute name on dao_class
        :param relationship: The SQLAlchemy relationship object
        """
        if self.join_manager.is_path_joined(dao_class, attribute_name):
            # Return the existing alias so downstream uses the same FROM element
            return self.join_manager.get_alias_for_path(dao_class, attribute_name)

        # Resolve target DAO class and create a dedicated alias for this path
        target_dao = relationship.entity.class_
        from sqlalchemy.orm import aliased

        aliased_target = aliased(target_dao, flat=True)

        # Relationship attribute on the source class, e.g., PoseDAO.position
        relationship_attr = getattr(dao_class, attribute_name)

        # Perform the join using the relationship attribute so SQLAlchemy
        # determines the ON clause, while we control aliasing of the right side
        self.sql_query = self.sql_query.join(aliased_target, relationship_attr)

        # Record both the logical path and the table as joined to avoid duplicates
        self.join_manager.add_path_join(dao_class, attribute_name, aliased_target)
        # Track underlying table class as joined; alias class type differs but table is the same
        self.join_manager.add_table_join(target_dao)

        return aliased_target


def eql_to_sql(query: SymbolicExpression, session: Session) -> EQLTranslator:
    """
    Translate an EQL query to SQL.

    :param query: The EQL query
    :param session: The SQLAlchemy session
    :return: The translator instance
    """
    result = EQLTranslator(query, session)
    result.translate()
    return result
