from __future__ import annotations

import ast
import linecache
import textwrap
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing_extensions import Any, Callable, List, Optional, Type, TYPE_CHECKING

from ordered_set import OrderedSet

from krrood.entity_query_language._monitoring import monitored
from krrood.entity_query_language._stack import CallStack, StackFrame

# Resolved once at import time: krrood/src/krrood/ — used as the path-based
# fallback in _is_krrood_internal_frame for frames where module_name is absent.
_KRROOD_SRC_ROOT: Path = Path(__file__).parents[2].resolve()
from krrood.entity_query_language.core.base_expressions import Selectable, Bindings
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    FlatVariable,
)
from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.factories import (
    and_,
    contains,
    entity,
    exists,
    flat_variable,
    is_class,
    issubclass_,
    node_children,
    node_descendants,
    node_id,
    node_type,
    variable_from,
    concatenation,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.symbol_graph.symbol_graph import Symbol

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        OperationResult,
        SymbolicExpression,
    )
    from krrood.entity_query_language.query.query import Entity, Query
    from uuid import UUID


def _build_type_existence_condition(
    node_variable: SymbolicExpression, type_: Type
) -> SymbolicExpression:
    """
    Build an exists-condition that checks whether *node_variable* (or one of its
    descendants) has a ``_type_`` that is a subclass of *type_*.

    :param node_variable: The EQL variable node to test.
    :param type_: The type to check for subclass membership.
    :return: An :func:`~krrood.entity_query_language.factories.exists` expression encoding the check.
    """
    node_type_variable = node_type(node_variable)
    return exists(
        node_variable,
        and_(
            HasType(node_variable, Selectable),
            node_type_variable != None,
            is_class(node_type_variable),
            issubclass_(node_type_variable, type_),
        ),
    )


def _is_krrood_internal_frame(frame: StackFrame) -> bool:
    """
    Check whether *frame* belongs to the krrood package internals.

    Uses ``frame.module_name`` as the primary signal.  Falls back to a
    :mod:`pathlib`-based path check (relative to :data:`_KRROOD_SRC_ROOT`) for frames
    where ``module_name`` is absent — e.g. notebook cells or frames captured inside
    ``eval()``.

    :param frame: The stack frame to test.
    :return:``True`` when the frame originates from within the krrood package.
    """
    if frame.module_name and (
        frame.module_name == "krrood" or frame.module_name.startswith("krrood.")
    ):
        return True
    if frame.filename:
        try:
            Path(frame.filename).resolve().relative_to(_KRROOD_SRC_ROOT)
            return True
        except ValueError:
            pass
    return False


def _get_query_source(frame: StackFrame) -> Optional[str]:
    """
    Extract the full source statement that contains ``frame.lineno``.

    :param frame: The stack frame whose source to extract.
    :return: The full source statement at ``frame.lineno``, or ``frame.code_snippet`` as
        a fallback.
    """
    if not frame or not frame.filename:
        return None
    lines = linecache.getlines(frame.filename)
    if not lines:
        return frame.code_snippet

    # Find the candidate statement that contains the line number.
    source = "".join(lines)
    tree = ast.parse(source)
    candidates: list = list(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates.extend(node.body)

    # Among candidates, find the one that contains the line number and has the smallest span.
    best = None
    for node in candidates:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if (start is None) or (end is None) or not (start <= frame.lineno <= end):
            continue
        if (best is None) or (end - start) < (best.end_lineno - best.lineno):
            best = node

    if best is not None:
        stmt_lines = lines[best.lineno - 1 : best.end_lineno]
        return textwrap.dedent("".join(stmt_lines)).rstrip()

    # Fallback to the entire source snippet if no candidate statement was found.
    return frame.code_snippet


def _select_source_frame(
    filtered_frames: List[StackFrame],
) -> Optional[StackFrame]:
    """
    Select the best user-facing frame from *filtered_frames*.

    Prefers frames from real source files over synthetic ones (``<string>``, ``<frozen …>``),
    and excludes krrood-internal frames.  Falls back to the innermost non-krrood frame when
    all real-file frames are internal, and finally to the very first frame if nothing else
    qualifies.

    :param filtered_frames: Frames already filtered by :meth:`~krrood.entity_query_language._stack.CallStack.filter`.
    :return: The selected :class:`~krrood.entity_query_language._stack.StackFrame`, or ``None``
        if *filtered_frames* is empty.
    """
    real_user_frames = [
        frame
        for frame in filtered_frames
        if not frame.filename.startswith("<") and not _is_krrood_internal_frame(frame)
    ]
    if real_user_frames:
        return real_user_frames[0]
    fallback = [
        frame for frame in filtered_frames if not _is_krrood_internal_frame(frame)
    ]
    if fallback:
        return fallback[0]
    return filtered_frames[0] if filtered_frames else None


def _format_source_context(source_frame: StackFrame) -> str:
    """
    Render the ``(function, file:line)`` context string for *source_frame*.

    :param source_frame: The frame to describe.
    :return: A parenthesised string of the form ``(func_name, basename:lineno)``.
    """
    basename = Path(source_frame.filename).name
    return f"({source_frame.function_name}, {basename}:{source_frame.lineno})"


def _format_indented_source(source_frame: Optional[StackFrame]) -> str:
    """
    Return the source statement at *source_frame*, indented by two spaces per line.

    :param source_frame: The frame whose source to render, or ``None``.
    :return: The indented source string, or ``" (unavailable)"`` when no source is
        found.
    """
    query_source = _get_query_source(source_frame) if source_frame else None
    if query_source:
        return "\n".join(f"  {line}" for line in query_source.splitlines())
    return "  (unavailable)"


def _format_stack_trace(stack: CallStack, focus_package: Optional[str]) -> str:
    """
    Render up to ten frames of *stack* as a formatted call-stack string.

    :param stack: The call stack to render.
    :param focus_package: When given, only frames whose module name contains this string
        are included; ``None`` includes all frames.
    :return: A multi-line string with one frame per entry.
    """
    display_stack = stack.filter(package=focus_package)
    formatted = []
    for frame in display_stack:
        formatted.append(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.function_name}\n'
            f"    {frame.code_snippet if frame.code_snippet else '???'}\n"
        )
    return "".join(formatted[:10])


@dataclass
class ConditionAndBindings:
    """
    Represents a condition and its associated bindings in the inference process.
    """

    condition: SymbolicExpression
    """
    The condition expression.
    """

    bindings: Bindings
    """
    A dictionary mapping UUIDs of condition children to their corresponding bindings.
    """

    def __repr__(self):
        if isinstance(self.condition, Comparator):
            return f"({self.condition.left} {self.condition} {self.condition.right})"
        else:
            return f"{self.condition} ({','.join(str(child) for child in self.condition._children_)})"


@dataclass
class InferenceExplanation(Symbol):
    """
    Explanation of how an instance was created through inference.

    Inherits from :class:`~krrood.symbol_graph.symbol_graph.Symbol` so that every
    explanation is a first-class entity in the SymbolGraph and therefore queryable
    via EQL like any other domain object.

    Lifecycle is tied to the inferred instance: the instance stores a strong reference
    to this object via its ``_inference_explanation_`` attribute (see
    :func:`register_inference`), while this object stores only a *weak* reference back
    to the instance.  This means the explanation is part of the same reference cluster
    as the instance and is collected together with it — no global registry required.
    """

    query_node: SymbolicExpression
    """
    The query node that was used to create the instance.
    """

    stack: CallStack
    """
    The call stack at the point of creation, as a
    :class:`~krrood.entity_query_language._stack.CallStack`.
    """

    query_root: Query | InstantiatedVariable
    """
    The root of the query that was used to create the instance.
    """

    satisfied_condition_ids: Optional[OrderedSet[UUID]] = None
    """
    An ordered set of UUIDs of condition expressions that were satisfied (truth value = True)
    during the evaluation that produced this instance. None if no condition information is available.
    """
    operation_result: Optional[OperationResult] = None
    """
    The full :class:`OperationResult` from the evaluation iteration that produced this
    instance.

    Contains bindings, all_bindings, is_false, operand, previous_operation_result, and
    satisfied_condition_ids. None if no result information is available.
    """

    # Internal weak reference to the inferred instance.  Not part of the public
    # constructor — populated by __post_init__.
    _instance_ref: Optional[weakref.ref] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def instance(self) -> Any:
        """
        The inferred instance, or ``None`` if it has been garbage-collected.
        """
        if self._instance_ref is None:
            return None
        return self._instance_ref()

    def get_satisfied_conditions_as_string(self) -> str:
        """
        Render all satisfied conditions as a single string, with each condition
        separated by ' AND '.

        :return: A string containing all satisfied conditions joined by ``\\nAND ``, or
            an empty string when no conditions were satisfied.
        """
        return "\nAND ".join(
            str(c) for c in self.get_satisfied_conditions_and_their_bindings()
        )

    def get_satisfied_conditions_and_their_bindings(self) -> List[ConditionAndBindings]:
        """
        Retrieve the list of satisfied non-logical condition expressions along with
        their bindings.

        :return: A list of :class:`ConditionAndBindings` objects, each pairing a satisfied
            condition expression (excluding :class:`~krrood.entity_query_language.operators.core_logical_operators.LogicalOperator`
            wrappers) with the full variable bindings from the evaluation. An empty list is
            returned when no satisfaction data is available.
        """
        if (
            self.operation_result is None
            or not self.operation_result.satisfied_condition_ids
        ):
            return []

        satisfied_conditions = []
        for condition_id in self.operation_result.satisfied_condition_ids:
            condition_expr = self.query_root._get_expression_by_id_(condition_id)
            if isinstance(condition_expr, (LogicalOperator,)):
                continue
            satisfied_conditions.append(
                ConditionAndBindings(condition_expr, self.operation_result.all_bindings)
            )
        return satisfied_conditions

    def condition_graph(self) -> Optional[QueryGraph]:
        """
        Build a QueryGraph of the full query tree with satisfaction data overlaid.

        Each ``QueryNode`` carries an ``is_satisfied`` flag grounded directly on the
        satisfied condition IDs.  Unsatisfied condition subtrees are also marked as
        *faded* for visualization purposes.

        :return: A :class:`QueryGraph` instance, or None if no conditions exist or no
            satisfaction data is available.
        """
        if self.query_root is None or not self.satisfied_condition_ids:
            return None
        return QueryGraph(
            self.query_root,
            satisfied_condition_ids=self.satisfied_condition_ids,
        )

    def as_string(
        self, focus_package: Optional[str | ModuleType] = None, show_trace: bool = False
    ) -> str:
        """
        Convert an InferenceExplanation into a human-readable string.

        :param focus_package: Optional package name; only applies when
            ``show_trace=True``.
        :param show_trace: When ``True``, append the call stack recorded at query-
            definition time.
        :return: A formatted string explaining the inference.
        """
        source_frame = _select_source_frame(self.stack.filter().frames)
        source_context = _format_source_context(source_frame) if source_frame else ""
        indented_source = _format_indented_source(source_frame)

        conditions = self.get_satisfied_conditions_and_their_bindings()
        conds_str = (
            "\n  AND ".join(str(c) for c in conditions) if conditions else "(none)"
        )

        result = (
            f"Instance: {self.instance}\n"
            f"Produced by inference variable: {self.query_node}\n"
            f"Query source {source_context}:\n{indented_source}\n"
            f"Satisfied conditions:\n  {conds_str}"
        )

        if show_trace:
            if isinstance(focus_package, ModuleType):
                focus_package = focus_package.__name__
            result += f"\nCall stack at definition:\n{_format_stack_trace(self.stack, focus_package)}"

        return result

    # ------------------------------------------------------------------
    # Stack query methods
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """
        Number of frames in the captured call stack.
        """
        return len(self.stack)

    def is_triggered_from_method(self) -> bool:
        """
        Check whether any frame in the call stack is inside a class method or
        classmethod.

        :return:``True`` if at least one frame has a non-``None`` ``class_object``,
            ``False`` otherwise.
        """
        return self.stack.is_from_method()

    def triggering_classes(self) -> List[type]:
        """
        Retrieve the distinct class objects that appear in the call stack, in order of
        first occurrence (innermost first).

        Useful for answering "from which class was this inference triggered?"

        :return: A list of class objects appearing in the stack, deduplicated and in
            innermost-first order.
        """
        return self.stack.classes()

    def triggering_functions(self) -> List[Callable]:
        """
        Retrieve the distinct function objects that appear in the call stack, in order
        of first occurrence (innermost first).

        Nested functions defined inside other functions may not be resolvable and will
        be absent from this list.

        :return: A list of callable objects appearing in the stack, deduplicated and in
            innermost-first order.
        """
        return self.stack.functions()

    def root_frame_in(self, package: str) -> Optional[StackFrame]:
        """
        Find the outermost :class:`~krrood.entity_query_language._stack.StackFrame`
        whose ``module_name`` contains *package*.

        This identifies the highest-level entry point into *package* that triggered the
        inference, which is useful for understanding where inside your own library the
        query was constructed.

        :param package: Substring matched against ``StackFrame.module_name``.
        :return: The outermost matching frame, or ``None`` if no frame matches.
        """
        return self.stack.root_frame_in(package)

    def get_satisfied_condition_expressions_for_the_instance(
        self,
    ) -> Entity[SymbolicExpression]:
        """
        :return: An entity containing condition expressions that were satisfied during the inference of the instance.
        """
        explanation = self.create_explanation_variable()
        node = self.create_query_node_variable(explanation)
        return (
            entity(node)
            .where(
                explanation.satisfied_condition_ids != None,
                contains(explanation.satisfied_condition_ids, node_id(node)),
            )
            .distinct()
        )

    def get_values_of_variable_nodes_of_given_type(
        self, type_: Type
    ) -> Entity[SymbolicExpression]:
        """
        :param type_: The type of the variable nodes to retrieve.
        :return: An entity containing variable nodes of the specified type that participated in the inference of the instance.
        """
        explanation = self.create_explanation_variable()
        node = self.get_variable_nodes_of_given_type(
            type_, self.create_query_node_variable(explanation)
        )
        operation_result = explanation.operation_result
        node_identifier = node_id(node)
        return (
            entity(operation_result.all_bindings[node_identifier])
            .where(contains(operation_result.all_bindings, node_identifier))
            .distinct()
        )

    def get_variable_nodes_of_given_type(
        self, type_: Type, node_variable: Optional[SymbolicExpression] = None
    ) -> Entity[SymbolicExpression] | SymbolicExpression:
        """
        :return: An entity containing instances that participated in the inference of this instance.
        """
        if node_variable is None:
            node_variable = self.create_query_node_variable(
                self.create_explanation_variable()
            )
        return (
            entity(node_variable)
            .where(
                HasType(node_variable, Selectable),
                (node_type_ := node_type(node_variable)) != None,
                is_class(node_type_),
                issubclass_(node_type_, type_),
            )
            .distinct(node_variable)
        )

    def get_satisfied_comparator_conditions(self) -> Entity[SymbolicExpression]:
        """
        :return: An entity containing condition expressions that are comparators and were satisfied during the
         inference of the instance.
        """
        condition_node = self.get_satisfied_condition_expressions_for_the_instance()
        return (
            entity(condition_node).where(HasType(condition_node, Comparator)).distinct()
        )

    def get_satisfied_comparator_conditions_between_attributes(
        self,
    ) -> Entity[SymbolicExpression]:
        """
        :return: An entity containing condition expressions that are comparators and were satisfied during the
         inference of the instance and have Attribute nodes as both their left and right children.
        """
        condition_node = self.get_satisfied_condition_expressions_for_the_instance()
        return (
            entity(condition_node)
            .where(
                HasType(condition_node, Comparator),
                HasType(condition_node.left, Attribute),
                HasType(condition_node.right, Attribute),
            )
            .distinct()
        )

    def get_conditions_that_relate_the_variables_of_type(
        self, type_: Type
    ) -> Entity[SymbolicExpression]:
        """
        :return: An entity containing condition expressions that relate the participating instances in the inference of this instance.
        """
        condition_node = self.get_satisfied_condition_expressions_for_the_instance()
        child1 = flat_variable(node_children(condition_node))
        child2 = flat_variable(node_children(condition_node))

        # Use exists(node, conditions) directly — avoids the _expression_/An-quantifier
        # problem that arises when passing a built entity to exists().  When .build() is
        # called on an entity, _expression_ is set to an An quantifier; _update_children_
        # then passes that An to Exists.left, and An._evaluate__ ignores sources entirely.
        child1_with_descendants = concatenation(
            child1, flat_variable(node_descendants(child1))
        )
        child2_with_descendants = concatenation(
            child2, flat_variable(node_descendants(child2))
        )

        return (
            entity(condition_node)
            .where(
                HasType(condition_node, (Comparator, InstantiatedVariable)),
                node_id(child1) != node_id(child2),
                _build_type_existence_condition(child1_with_descendants, type_),
                _build_type_existence_condition(child2_with_descendants, type_),
            )
            .distinct()
        )

    def get_conditions_that_relate_variables_of_types(
        self, type_a: Type, type_b: Type
    ) -> Entity[SymbolicExpression]:
        """
        Generalisation of :meth:`get_conditions_that_relate_the_variables_of_type` for
        two potentially different types.  Returns satisfied condition expressions that
        have at least one descendant variable node whose ``_type_`` is a subclass of
        *type_a* and at least one (different) descendant variable node whose ``_type_``
        is a subclass of *type_b*.

        When ``type_a == type_b`` the semantics reduce to
        :meth:`get_conditions_that_relate_the_variables_of_type`.

        :param type_a: First participant type.
        :param type_b: Second participant type.
        :return: An entity containing the matching condition expressions.
        """
        condition_node = self.get_satisfied_condition_expressions_for_the_instance()
        desc_a = flat_variable(node_descendants(condition_node))
        desc_b = flat_variable(node_descendants(condition_node))

        return (
            entity(condition_node)
            .where(
                HasType(condition_node, (Comparator, InstantiatedVariable)),
                node_id(desc_a) != node_id(desc_b),
                _build_type_existence_condition(desc_a, type_a),
                _build_type_existence_condition(desc_b, type_b),
            )
            .distinct(node_id(condition_node))
        )

    @staticmethod
    def create_query_node_variable(
        explanation_variable: Selectable[InferenceExplanation] | InferenceExplanation,
    ) -> FlatVariable[SymbolicExpression] | SymbolicExpression:
        """
        Build a flat variable ranging over all descendant nodes of the explanation's
        query root.

        :param explanation_variable: A :class:`~krrood.entity_query_language.core.base_expressions.Selectable`
            wrapping an :class:`InferenceExplanation`, or an explanation instance used as a domain source.
        :return: A :class:`~krrood.entity_query_language.core.mapped_variable.FlatVariable` iterating
            over all descendant symbolic expressions of the query root.
        """
        return flat_variable(node_descendants(explanation_variable.query_root))

    def create_explanation_variable(
        self,
    ) -> Selectable[InferenceExplanation] | InferenceExplanation:
        """
        :return: A variable representing the explanation in the inference process.
        """
        return variable_from(self)


def register_inference(
    instance: Any,
    variable_node: SymbolicExpression,
    result: Optional[OperationResult] = None,
) -> None:
    """
    Register an instance created via inference by attaching an
    :class:`InferenceExplanation` directly to the instance.

    Only :class:`~krrood.symbol_graph.symbol_graph.Symbol` instances are supported.
    Non-Symbol values (plain ints, strings, frozen third-party objects) are silently
    ignored so that callers need no special-casing.

    The explanation is stored directly as the ``_inference_explanation_`` field on
    the instance (declared in :class:`~krrood.symbol_graph.symbol_graph.Symbol`),
    keeping the explanation's lifecycle identical to the instance's lifecycle — no
    separate global registry is needed.

    :param instance: The instance to record.
    :param variable_node: The variable node that produced the instance.
    :param result: The OperationResult from the evaluation, carrying satisfied condition IDs.
    """
    if not isinstance(instance, Symbol):
        return
    if not monitored.is_monitored(type(variable_node)):
        return

    satisfied_ids = result.satisfied_condition_ids if result else None
    explanation = InferenceExplanation(
        query_node=variable_node,
        stack=monitored.get_stack(variable_node) or CallStack([]),
        query_root=variable_node._root_,
        satisfied_condition_ids=satisfied_ids,
        operation_result=result,
    )
    try:
        explanation._instance_ref = weakref.ref(instance)
    except TypeError:
        explanation._instance_ref = lambda: instance  # type: ignore[assignment]
    instance._inference_explanation_ = explanation


def explain_inference(instance: Any) -> Optional[InferenceExplanation]:
    """
    Retrieve the explanation of how the given instance was created through inference.

    Returns ``None`` for non-:class:`~krrood.symbol_graph.symbol_graph.Symbol` values
    or for Symbol instances that were not produced by an inference variable.

    :param instance: The instance to explain.
    :return: An :class:`InferenceExplanation` if the instance was inferred, otherwise ``None``.
    """
    if not isinstance(instance, Symbol):
        return None
    return instance._inference_explanation_
