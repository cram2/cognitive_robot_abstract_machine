from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import Dict, FrozenSet, List, Optional, Tuple, Union

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    flatten_operands,
)
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.core.expression_structure import chain_root
from krrood.entity_query_language.verbalization.grammar.framework.planner import Planner
from krrood.entity_query_language.query.aggregation_structure import (
    unwrap_result_quantifiers,
)
from krrood.entity_query_language.verbalization.vocabulary.english import FallbackNouns


class AggregationStatus(Enum):
    """
    How a consequent binding or antecedent relates to the GROUP BY clause.
    """

    GROUP_KEY = auto()
    """
    This expression is one of the ``grouped_by`` key variables.
    """

    AGGREGATED = auto()
    """
    Present in the query but not a group key — rendered in plural form.
    """

    NONE = auto()
    """
    No grouping context in this query.
    """

@dataclass
class AntecedentInformation:
    """
    Descriptor for one antecedent variable in the IF clause.
    """

    root: Union[Variable, Entity]
    """
    The underlying Variable/Entity (unwrapped from any ResultQuantifier).
    """

    variable: Optional[Variable]
    """
    The antecedent's restriction subject — the variable its conditions attach
    to (``root`` for a Variable root, the selected variable for an Entity
    root).
    """

    type_name: str
    """
    Human-readable Python type name of *root* (e.g. ``"Robot"``).
    """

    aggregation_status: AggregationStatus
    """
    Whether this antecedent is a group key, aggregated, or neither.
    """

    conditions: List[SymbolicExpression] = field(default_factory=list)
    """
    The raw WHERE conditions attributable to this antecedent; their surface
    form/slot is the condition-form registry's concern at render time, not the
    plan's.
    """

@dataclass
class ConsequentBinding:
    """
    Descriptor for one field binding in the THEN clause.
    """

    field_name: str
    """
    Python attribute name on the consequent type (e.g. ``"tasks"``).
    """

    value_expression: SymbolicExpression
    """
    EQL expression providing the value for *field_name*.
    """

    is_plural_field: bool
    """
    ``True`` when *field_name* is already plural.
    """

    aggregation_status: AggregationStatus
    """
    Whether the value is a group key, aggregated, or neither.
    """


@dataclass
class RuleStructure:
    """
    Complete decomposition of an inference-rule Entity query (the plan).
    """

    primary_antecedents: List[AntecedentInformation]
    """
    Antecedents with at least one condition — items in the IF block.
    """

    secondary_antecedents: List[AntecedentInformation]
    """
    Antecedents with no conditions — only registered for coreference.
    """

    consequent_type: str
    """
    Python type name of the inferred variable (e.g. ``"Drawer"``).
    """

    consequent_bindings: List[ConsequentBinding]
    """
    Ordered field bindings for the THEN clause.
    """

    unmatched_conditions: List[SymbolicExpression]
    """
    Outer WHERE conditions not attributable to any antecedent.
    """

    group_key_ids: FrozenSet[uuid.UUID]
    """
    ``_id_`` values of the GROUP BY key variables.
    """


@dataclass
class InferencePlanner(Planner[Entity, RuleStructure]):
    """
    Decompose an inference-rule query (an entity whose selected variable is an
    instantiated variable) into a ``RuleStructure`` (the IF/THEN
    decomposition).

    Reference: :cite:t:`reiter2000building` — content/structure determination (microplanning).
    """

    @staticmethod
    def can_handle(entity: Entity) -> bool:
        """
        :param entity: Candidate query.
        :return: ``True`` when *entity*'s selected variable is an instantiated variable.

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> InferencePlanner.can_handle(entity(drawer))
        True
        >>> InferencePlanner.can_handle(entity(variable(Robot, [])))
        False
        """
        entity.build()
        return isinstance(entity.selected_variable, InstantiatedVariable)

    def plan(self) -> RuleStructure:
        """:return: The IF/THEN decomposition: antecedents, consequent bindings, and grouping.

        Its contribution is assembling the whole :class:`RuleStructure`; the shown ``'Drawer'`` is the
        ``consequent_type`` field it fills from the inferred (THEN) variable.

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> InferencePlanner(entity(drawer)).plan().consequent_type
        'Drawer'
        """
        self.node.build()
        group_key_ids = self._group_key_ids()
        antecedents, unmatched = self._plan_antecedents(group_key_ids)
        return RuleStructure(
            primary_antecedents=[
                antecedent for antecedent in antecedents if antecedent.conditions
            ],
            secondary_antecedents=[
                antecedent for antecedent in antecedents if not antecedent.conditions
            ],
            consequent_type=self._consequent_type(),
            consequent_bindings=self._plan_consequent(group_key_ids),
            unmatched_conditions=unmatched,
            group_key_ids=group_key_ids,
        )

    # %% shared analysis helpers

    @property
    def _inferred(self) -> InstantiatedVariable:
        """:return: The instantiated variable selected by the entity (after build).

        >>> connection = variable(FixedConnection, [])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=connection.parent, handle=connection.child)))
        >>> _ = planner.node.build()
        >>> planner._inferred._type_.__name__
        'Drawer'
        """
        return self.node.selected_variable

    def _consequent_type(self) -> str:
        """:return: The Python type name of the inferred (THEN) variable (*"Drawer"*).

        >>> connection = variable(FixedConnection, [])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=connection.parent, handle=connection.child)))
        >>> _ = planner.node.build()
        >>> planner._consequent_type()
        'Drawer'
        """
        inferred = self._inferred
        return getattr(inferred._type_, "__name__", str(inferred._type_))

    def _group_key_ids(self) -> FrozenSet[uuid.UUID]:
        """:return: The ``_id_`` set of the GROUP BY key variables — empty when the query is
        ungrouped.

        >>> connection = variable(FixedConnection, [])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=connection.parent, handle=connection.child)))
        >>> _ = planner.node.build()
        >>> len(planner._group_key_ids())
        0
        """
        grouped = self.node._grouped_by_expression_
        if grouped is not None and grouped.variables_to_group_by:
            return frozenset(
                variable._id_ for variable in grouped.variables_to_group_by
            )
        return frozenset()

    @staticmethod
    def _aggregation_status(
        node_id: uuid.UUID, group_key_ids: FrozenSet[uuid.UUID]
    ) -> AggregationStatus:
        """:return: GROUP_KEY if a group key, else AGGREGATED when grouping is present, else NONE.

        >>> import uuid
        >>> key = uuid.uuid4()
        >>> InferencePlanner._aggregation_status(key, frozenset({key})).name
        'GROUP_KEY'
        >>> InferencePlanner._aggregation_status(uuid.uuid4(), frozenset()).name
        'NONE'
        """
        if node_id in group_key_ids:
            return AggregationStatus.GROUP_KEY
        return AggregationStatus.AGGREGATED if group_key_ids else AggregationStatus.NONE

    # %% consequent (THEN bindings)

    def _plan_consequent(
        self, group_key_ids: FrozenSet[uuid.UUID]
    ) -> List[ConsequentBinding]:
        """:return: One :class:`ConsequentBinding` per THEN-clause field, in construction order.

        >>> connection = variable(FixedConnection, [])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=connection.parent, handle=connection.child)))
        >>> _ = planner.node.build()
        >>> [binding.field_name for binding in planner._plan_consequent(frozenset())]
        ['container', 'handle']
        """
        return [
            ConsequentBinding(
                field_name=field_name,
                value_expression=child,
                is_plural_field=morphology.is_plural(field_name),
                aggregation_status=self._aggregation_status(child._id_, group_key_ids),
            )
            for field_name, child in self._inferred._child_vars_.items()
        ]

    # %% antecedents (IF roots + their conditions)

    def _plan_antecedents(
        self, group_key_ids: FrozenSet[uuid.UUID]
    ) -> Tuple[List[AntecedentInformation], List[SymbolicExpression]]:
        """Discover antecedent roots, then attribute outer-WHERE conditions to them.

        :return: The antecedents (their ``conditions`` mutated in place) and the conditions that
            matched no antecedent.

        >>> handle = variable(Handle, [])
        >>> prismatic = variable(PrismaticConnection, [])
        >>> fixed = a(FixedConnection)(parent=prismatic.child, child=handle).from_([])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=fixed.expression.parent, handle=fixed.expression.child)))
        >>> _ = planner.node.build()
        >>> antecedents, unmatched = planner._plan_antecedents(frozenset())
        >>> (len(antecedents), len(unmatched))
        (1, 0)
        """
        antecedents = self._discover_antecedents(group_key_ids)
        unmatched = self._attribute_conditions(antecedents, self._outer_conditions())
        return antecedents, unmatched

    def _discover_antecedents(
        self, group_key_ids: FrozenSet[uuid.UUID]
    ) -> List[AntecedentInformation]:
        """:return: One :class:`AntecedentInformation` per distinct root reached from the consequent
        bindings (the IF-clause subjects).

        >>> handle = variable(Handle, [])
        >>> prismatic = variable(PrismaticConnection, [])
        >>> fixed = a(FixedConnection)(parent=prismatic.child, child=handle).from_([])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=fixed.expression.parent, handle=fixed.expression.child)))
        >>> _ = planner.node.build()
        >>> [antecedent.type_name for antecedent in planner._discover_antecedents(frozenset())]
        ['FixedConnection']
        """
        antecedents_by_root_id: Dict[uuid.UUID, AntecedentInformation] = {}
        for child in self._inferred._child_vars_.values():
            root = self._find_root(child)
            if root is None or root._id_ in antecedents_by_root_id:
                continue
            type_name, own_conditions = self._extract_root_info(root)
            antecedents_by_root_id[root._id_] = AntecedentInformation(
                root=root,
                variable=self._root_variable(root),
                type_name=type_name,
                aggregation_status=self._aggregation_status(root._id_, group_key_ids),
                conditions=own_conditions,
            )
        return list(antecedents_by_root_id.values())

    @staticmethod
    def _root_variable(root: Union[Variable, Entity]) -> Optional[Variable]:
        """:return: The restriction subject of an antecedent root — the selected variable for an
        Entity root, the root itself for a Variable root.

        >>> robot = variable(Robot, [])
        >>> InferencePlanner._root_variable(robot) is robot
        True
        """
        if isinstance(root, Entity):
            root.build()
            selected = root.selected_variable
            return selected if isinstance(selected, Variable) else None
        return root if isinstance(root, Variable) else None

    def _outer_conditions(self) -> List[SymbolicExpression]:
        """:return: The flattened outer-WHERE conditions of the query — empty when it has no
        ``.where(...)``.

        >>> connection = variable(FixedConnection, [])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=connection.parent, handle=connection.child)))
        >>> _ = planner.node.build()
        >>> len(planner._outer_conditions())
        0
        """
        where = self.node._where_expression_
        return flatten_operands(where.condition, AND) if where is not None else []

    def _attribute_conditions(
        self,
        antecedents: List[AntecedentInformation],
        extra_conditions: List[SymbolicExpression],
    ) -> List[SymbolicExpression]:
        """Distribute outer-WHERE conditions to owning antecedents (in place).

        :return: The conditions that matched no antecedent.

        >>> connection = variable(FixedConnection, [])
        >>> planner = InferencePlanner(entity(inference(Drawer)(container=connection.parent, handle=connection.child)))
        >>> _ = planner.node.build()
        >>> antecedents = planner._discover_antecedents(frozenset())
        >>> len(planner._attribute_conditions(antecedents, planner._outer_conditions()))
        0
        """
        id_to_antecedent = {
            self._antecedent_variable_id(antecedent): antecedent
            for antecedent in antecedents
        }
        unmatched: List[SymbolicExpression] = []
        for condition in extra_conditions:
            owner_id = self._condition_left_owner_id(condition)
            if owner_id is not None and owner_id in id_to_antecedent:
                id_to_antecedent[owner_id].conditions.append(condition)
            else:
                unmatched.append(condition)
        return unmatched

    def _antecedent_variable_id(
        self, antecedent: AntecedentInformation
    ) -> Optional[uuid.UUID]:
        """:return: The stable ``_id_`` of the underlying variable for an antecedent."""
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            selected_variable = root.selected_variable
            return selected_variable._id_ if selected_variable is not None else None
        return root._id_

    def _condition_left_owner_id(
        self, condition: SymbolicExpression
    ) -> Optional[uuid.UUID]:
        """:return: The ``_id_`` of the root variable on the left-hand side of an equality condition, else
        ``None``.

        >>> planner = InferencePlanner(entity(variable(Robot, [])))
        >>> planner._condition_left_owner_id(variable(Robot, []).battery > 50) is None
        True
        """
        if (
            not isinstance(condition, Comparator)
            or condition.operation is not operator.eq
        ):
            return None
        current = unwrap_result_quantifiers(chain_root(condition.left))
        return current._id_

    def _find_root(
        self, expression: SymbolicExpression
    ) -> Optional[Union[Variable, Entity]]:
        """:return: The :class:`Variable`/:class:`Entity` at the root of *expression*'s chain, else
        ``None``.

        >>> robot = variable(Robot, [])
        >>> planner = InferencePlanner(entity(robot))
        >>> planner._find_root(robot.battery) is robot
        True
        """
        current = unwrap_result_quantifiers(chain_root(expression))
        if isinstance(current, (Variable, Entity)):
            return current
        return None

    def _extract_root_info(
        self, root: Union[Variable, Entity]
    ) -> Tuple[str, List[SymbolicExpression]]:
        """:return: ``(type_name, own_conditions)`` for a root variable or entity.

        >>> planner = InferencePlanner(entity(variable(Robot, [])))
        >>> type_name, conditions = planner._extract_root_info(variable(Robot, []))
        >>> (type_name, len(conditions))
        ('Robot', 0)
        """
        if isinstance(root, Entity):
            root.build()
            selected = root.selected_variable
            type_name = FallbackNouns.ENTITY.name_of(selected)
            conditions: List[SymbolicExpression] = []
            if root._where_expression_ is not None:
                conditions = flatten_operands(root._where_expression_.condition, AND)
            return type_name, conditions
        if isinstance(root, Variable):
            return FallbackNouns.VARIABLE.name_of(root), []
        return FallbackNouns.ENTITY.text, []
