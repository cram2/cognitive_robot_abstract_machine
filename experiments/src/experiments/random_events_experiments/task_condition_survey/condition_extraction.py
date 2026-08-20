"""
This module recovers the EQL condition a task states, from the ordinary Python the
task's success condition is written in.

Suites such as RoboCasa do not state their query-like objects in EQL, so the condition
has to be recovered from the syntax tree instead of read off. What is here covers the
suites the experiments survey, and is not a general translation from Python to EQL.

A task's success condition is a formula over relations, not a single value: the value a
scene happens to hold after a reset is a sample drawn from that condition, not the
condition itself. Reading the condition instead of observing it is entirely static, so a
whole task suite is covered rather than a curated subset, and the coverage reported is a
property of the suite rather than of the tasks someone picked.

What is recovered is an expression of the Entity Query Language: a suite's control flow
-- named intermediates, guard clauses returning early, loops deciding on the first
object -- is read back into EQL's conjunction, disjunction, negation and quantifiers
over the vocabulary of :mod:`experiments.random_events_experiments.task_condition_survey.task_conditions`.
Whatever is not recognised becomes an :class:`~...task_conditions.UnreadCondition`
rather than a guess.

Nothing here is specific to one task suite. Which method states a suite's condition, how
that method returns its outcome, and what its relations are called are supplied by the
suite's own reader.
"""

from __future__ import annotations

import ast
import importlib.util
import operator
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, ClassVar, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    and_,
    contains,
    exists,
    for_all,
    in_,
    not_,
    or_,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.logical_quantifiers import (
    Exists,
    ForAll,
)
from krrood.exceptions import DataclassException

from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConstantCondition,
    PredicateVocabulary,
    SceneObject,
    StatedTaskCondition,
    UnreadCondition,
)

# %% where a condition method states its outcome


class StatedOutcome(ABC):
    """
    Where in a return statement a condition method states its outcome.

    A suite either returns the outcome itself or returns it among other measurements,
    and which of the two decides what a return statement is read as.
    """

    @abstractmethod
    def stated_by(self, returned: ast.expr) -> Optional[ast.expr]:
        """
        :param returned: The expression a condition method returns.
        :return: The part of it stating the outcome, or ``None`` when it states none.
        """


@dataclass(frozen=True)
class ReturnedValue(StatedOutcome):
    """
    The outcome is the returned expression itself.
    """

    def stated_by(self, returned: ast.expr) -> Optional[ast.expr]:
        return returned


@dataclass(frozen=True)
class ReturnedEntry(StatedOutcome):
    """
    The outcome is one named entry of a returned mapping, which also carries
    measurements the condition does not depend on.
    """

    key: str
    """
    Name of the entry stating the outcome.
    """

    def stated_by(self, returned: ast.expr) -> Optional[ast.expr]:
        if isinstance(returned, ast.Dict):
            for name, value in zip(returned.keys, returned.values):
                if isinstance(name, ast.Constant) and name.value == self.key:
                    return value
            return None
        if isinstance(returned, ast.Call) and self.constructs_a_mapping(returned):
            for keyword in returned.keywords:
                if keyword.arg == self.key:
                    return keyword.value
        return None

    @staticmethod
    def constructs_a_mapping(call: ast.Call) -> bool:
        """
        :param call: A call whose result is returned.
        :return: Whether it builds a mapping, rather than merely carrying a keyword of the
            same name as the entry stating the outcome.
        """
        return isinstance(call.func, ast.Name) and call.func.id == "dict"


# %% quantifying a condition over the objects a scene provides


@dataclass(frozen=True)
class QuantifiedCondition:
    """
    A condition together with the object of a scene it is quantified over.

    The collection's size is not known statically, which is precisely why the object is
    kept apart from the condition stated of it: quantifying is what turns one stated
    condition into as many conjuncts or disjuncts as the scene holds objects.
    """

    objects: Variable
    """
    The EQL variable standing for each object of the collection.
    """

    body: SymbolicExpression
    """
    The condition stated of that object.
    """

    def holds_for_every_object(self) -> ForAll:
        """
        :return: The condition requiring every object of the collection to satisfy the body.
        """
        return for_all(self.objects, self.body)

    def holds_for_some_object(self) -> Exists:
        """
        :return: The condition requiring at least one object of the collection to satisfy
            the body.
        """
        return exists(self.objects, self.body)


# %% reading a success condition


@dataclass
class ConditionExtractor:
    """
    Recovers the EQL condition a task's condition method states, from its syntax tree.

    Suites state these conditions as ordinary Python: named intermediates, guard clauses
    that return early, and loops that return on the first object to decide the outcome.
    This reads that control flow back into the EQL condition it computes, and records
    whatever it does not recognise as an
    :class:`~experiments.random_events_experiments.task_condition_survey.task_conditions.UnreadCondition` rather
    than guessing.
    """

    vocabulary: PredicateVocabulary
    """
    Classifies the names encountered while reading.
    """

    outcome: StatedOutcome = field(default_factory=ReturnedValue)
    """
    Where in a return statement the method states its outcome.
    """

    bindings: dict[str, SymbolicExpression] = field(default_factory=dict)
    """
    Conditions bound to local names by assignments seen so far.
    """

    name_aliases: dict[str, str] = field(default_factory=dict)
    """
    Dotted name each local name stands for, so a value read back through a local is
    still named after where it came from.
    """

    quantified_objects: dict[str, Variable] = field(default_factory=dict)
    """
    The EQL variable each local name of a quantified object stands for, so a condition
    stated of a loop's object is stated of the variable the quantifier binds.
    """

    comparisons: ClassVar[dict[type, Callable[[Any, Any], bool]]] = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
    }
    """
    The comparisons a task can bound a measured value with, and the operation each
    states.

    ..note:: Membership is left out because it is not a bound on a value but a containment
        between the value and a collection, which EQL states either way round.
    """

    reversed_comparisons: ClassVar[dict[Callable[[Any, Any], bool], Callable]] = {
        operator.lt: operator.gt,
        operator.le: operator.ge,
        operator.gt: operator.lt,
        operator.ge: operator.le,
        operator.eq: operator.eq,
        operator.ne: operator.ne,
        operator.is_: operator.is_,
        operator.is_not: operator.is_not,
    }
    """
    The operation stating the same bound once the measured value moves to the left, so a
    threshold written before its value bounds it the same way as one written after it.
    """

    combining_calls: ClassVar[dict[str, Callable[..., SymbolicExpression]]] = {
        "logical_and": and_,
        "logical_or": or_,
    }
    """
    Array-library functions that combine outcomes, and the condition each one states.
    """

    truth_value_casts: ClassVar[frozenset[str]] = frozenset({"bool"})
    """
    Methods that cast an outcome to a truth value without changing where it holds.
    """

    def extract(self, function: ast.FunctionDef) -> StatedTaskCondition:
        """
        :param function: The condition method to read.
        :return: The condition the method states its outcome as.
        """
        self.bindings = {}
        self.name_aliases = {}
        self.quantified_objects = {}
        return StatedTaskCondition(expression=self.condition_of_body(function.body))

    def relation_of(self, name: str, objects: Variable) -> SymbolicExpression:
        """
        :param name: Name the task states the relation under.
        :param objects: The objects it is stated of.
        :return: The relation, classified by the vocabulary.
        """
        return self.vocabulary.classify(name).relation_stated_of(name, objects)

    def condition_of_body(self, statements: list[ast.stmt]) -> SymbolicExpression:
        """
        :param statements: The statements of a function or block, in order.
        :return: The condition under which running the block returns True.
        """
        if not statements:
            return ConstantCondition.NEVER.expression

        statement, rest = statements[0], statements[1:]

        if isinstance(statement, ast.Return):
            return self.condition_of_return(statement)
        if isinstance(statement, ast.Assign):
            self.bind(statement)
            return self.condition_of_body(rest)
        if isinstance(statement, ast.If):
            return self.condition_of_if(statement, rest)
        if isinstance(statement, ast.For):
            return self.condition_of_for(statement, rest)
        return self.condition_of_body(rest)

    def condition_of_return(self, statement: ast.Return) -> SymbolicExpression:
        """
        :param statement: A return statement.
        :return: The condition the returned expression states.
        """
        if statement.value is None:
            return ConstantCondition.NEVER.expression
        stated = self.outcome.stated_by(statement.value)
        if stated is not None:
            return self.condition_of_expression(stated)
        if (
            isinstance(statement.value, ast.Name)
            and statement.value.id in self.bindings
        ):
            return self.bindings[statement.value.id]
        return UnreadCondition.found(
            f"return stating no outcome: {ast.unparse(statement.value)}"
        )

    def bind(self, statement: ast.Assign) -> None:
        """
        Binds the condition an assignment's value states to the name it assigns.

        Only single-name targets are bound, since those are the named intermediates the
        conditions actually read back.

        :param statement: An assignment whose value may be used as a condition later.
        """
        if len(statement.targets) != 1:
            return
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            return
        if isinstance(
            statement.value, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)
        ):
            self.name_aliases[target.id] = self.dotted_name(statement.value)
        self.bindings[target.id] = self.condition_of_assigned(statement.value)

    def condition_of_assigned(self, value: ast.expr) -> SymbolicExpression:
        """
        Reads an assigned value as the condition the name it is bound to states.

        Binding the outcome rather than the mapping is what lets a conditional that names
        the outcome differently in each branch state both: bindings are merged per branch,
        so anything read back through a name has to be a condition and not the syntax it
        came from.

        :param value: The value an assignment binds to a name.
        :return: The condition that name states, which is the outcome the value carries
            when the value is a mapping stating one.
        """
        stated = self.outcome.stated_by(value)
        if stated is not None and stated is not value:
            return self.condition_of_expression(stated)
        return self.condition_of_expression(value)

    def condition_of_if(
        self, statement: ast.If, rest: list[ast.stmt]
    ) -> SymbolicExpression:
        """
        :param statement: A conditional statement.
        :param rest: The statements following it.
        :return: The condition under which the conditional and what follows return True.
        """
        test = self.condition_of_expression(statement.test)

        if not self.decides_outcome(statement.body) and not self.decides_outcome(
            statement.orelse
        ):
            return self.condition_after_branching(statement, test, rest)

        consequent = self.condition_of_body(statement.body)

        if statement.orelse:
            alternative = self.condition_of_body(statement.orelse)
            return or_(and_(test, consequent), and_(not_(test), alternative))

        continuation = self.condition_of_body(rest)
        stated = ConstantCondition.stated_by(consequent)
        if stated is ConstantCondition.NEVER:
            return and_(not_(test), continuation)
        if stated is ConstantCondition.ALWAYS:
            return or_(test, continuation)
        return or_(and_(test, consequent), continuation)

    @staticmethod
    def decides_outcome(statements: list[ast.stmt]) -> bool:
        """
        Tells whether a branch settles the outcome or only prepares for what follows it.

        A branch may return from inside a nested conditional or loop, so what matters is
        whether it returns at all and not whether it returns immediately.

        :param statements: The statements of a branch.
        :return: Whether the branch settles the condition's outcome by returning, rather
            than only naming intermediates for what follows to use.
        """
        return any(
            isinstance(node, ast.Return)
            for statement in statements
            for node in ast.walk(statement)
        )

    def condition_after_branching(
        self, statement: ast.If, test: SymbolicExpression, rest: list[ast.stmt]
    ) -> SymbolicExpression:
        """
        Reads what follows a conditional that only names intermediates.

        A conditional that assigns rather than returns leaves the condition to be stated
        by what comes after it, so its branches contribute the names they bind and not
        the outcome.

        :param statement: A conditional whose branches only name intermediates.
        :param test: The condition the conditional branches on.
        :param rest: The statements following it.
        :return: The condition stated by what follows, with every name the branches gave
            a different meaning bound to both meanings, each guarded by the branch that
            gave it.
        """
        outer_bindings = dict(self.bindings)

        self.condition_of_body(statement.body)
        consequent_bindings = self.bindings

        self.bindings = dict(outer_bindings)
        self.condition_of_body(statement.orelse)
        alternative_bindings = self.bindings

        merged = dict(outer_bindings)
        for name in set(consequent_bindings) | set(alternative_bindings):
            unchanged = outer_bindings.get(name, ConstantCondition.NEVER.expression)
            merged[name] = or_(
                and_(test, consequent_bindings.get(name, unchanged)),
                and_(not_(test), alternative_bindings.get(name, unchanged)),
            )
        self.bindings = merged
        return self.condition_of_body(rest)

    def condition_of_for(
        self, statement: ast.For, rest: list[ast.stmt]
    ) -> SymbolicExpression:
        """
        :param statement: A loop, which decides the outcome when it returns inside its body.
        :param rest: The statements following the loop.
        :return: The quantified condition the loop states, combined with what follows.
        """
        returns = [
            node
            for each in statement.body
            for node in ast.walk(each)
            if isinstance(node, ast.Return)
        ]

        if not returns:
            self.record_bindings_of(statement.body)
            return self.condition_of_body(rest)

        objects = self.quantify_over(statement.target)

        if all(self.states_failure(each) for each in returns):
            held = self.condition_of_fallthrough(statement.body)
            return and_(
                QuantifiedCondition(
                    objects=objects, body=held
                ).holds_for_every_object(),
                self.condition_of_body(rest),
            )

        satisfied = QuantifiedCondition(
            objects=objects, body=self.condition_of_body(statement.body)
        ).holds_for_some_object()
        continuation = self.condition_of_body(rest)

        if ConstantCondition.stated_by(continuation) is ConstantCondition.NEVER:
            return satisfied
        return or_(satisfied, continuation)

    def quantify_over(self, target: ast.expr) -> Variable:
        """
        :param target: The name a loop or a comprehension gives each object it ranges over.
        :return: The EQL variable standing for that object, which every condition stated of
            the name is then stated of.
        """
        objects = SceneObject.provided_by_the_scene()
        if isinstance(target, ast.Name):
            self.quantified_objects[target.id] = objects
        return objects

    def record_bindings_of(self, statements: list[ast.stmt]) -> None:
        """
        Reads a block for the names it binds alone.

        A loop that never returns states no condition of its own, but the names it binds
        are still there for what follows it to read, so the block is read for its
        bindings and the condition it states is deliberately discarded.

        :param statements: The statements of a block that decides no outcome.
        """
        self.condition_of_body(statements)

    @staticmethod
    def states_failure(statement: ast.Return) -> bool:
        """
        :param statement: A return statement inside a loop.
        :return: Whether it states that the condition fails.
        """
        return (
            isinstance(statement.value, ast.Constant) and statement.value.value is False
        )

    def condition_of_fallthrough(
        self, statements: list[ast.stmt]
    ) -> SymbolicExpression:
        """
        :param statements: The statements of a loop body that returns only on failure.
        :return: The condition under which the body runs to its end without returning,
            which is what has to hold of every object for such a loop to be passed.
        """
        if not statements:
            return ConstantCondition.ALWAYS.expression

        statement, rest = statements[0], statements[1:]

        if isinstance(statement, ast.Return):
            return ConstantCondition.NEVER.expression
        if isinstance(statement, ast.Assign):
            self.bind(statement)
            return self.condition_of_fallthrough(rest)
        if isinstance(statement, ast.If) and self.decides_outcome(statement.body):
            return and_(
                not_(self.condition_of_expression(statement.test)),
                self.condition_of_fallthrough(rest),
            )
        return self.condition_of_fallthrough(rest)

    def condition_of_expression(self, expression: ast.expr) -> SymbolicExpression:
        """
        :param expression: An expression evaluated as a condition.
        :return: The condition it states.
        """
        if isinstance(expression, ast.BoolOp):
            operands = [
                self.condition_of_expression(operand) for operand in expression.values
            ]
            if isinstance(expression.op, ast.And):
                return and_(*operands)
            return or_(*operands)

        if isinstance(expression, ast.BinOp):
            return self.condition_of_combined_outcomes(expression)

        if isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op, (ast.Not, ast.Invert)
        ):
            return not_(self.condition_of_expression(expression.operand))

        if isinstance(expression, ast.IfExp):
            return or_(
                and_(
                    self.condition_of_expression(expression.test),
                    self.condition_of_expression(expression.body),
                ),
                and_(
                    not_(self.condition_of_expression(expression.test)),
                    self.condition_of_expression(expression.orelse),
                ),
            )

        if isinstance(expression, ast.Constant) and isinstance(expression.value, bool):
            return ConstantCondition(expression.value).expression

        if isinstance(expression, ast.Name):
            if expression.id in self.bindings:
                return self.bindings[expression.id]
            return UnreadCondition.found(f"unbound name {expression.id!r}")

        if isinstance(expression, ast.Compare):
            return self.condition_of_comparison(expression)

        if isinstance(expression, ast.Call):
            return self.condition_of_call(expression)

        if isinstance(expression, ast.Subscript):
            return self.relation_of(
                self.subscripted_name(expression), SceneObject.written_as("")
            )

        return UnreadCondition.found(f"unread expression {type(expression).__name__}")

    def condition_of_combined_outcomes(
        self, expression: ast.BinOp
    ) -> SymbolicExpression:
        """
        Reads outcomes combined element-wise as the condition they state together.

        Conditions evaluated over arrays of environments combine their outcomes
        element-wise rather than with the boolean keywords: ``&`` and ``*`` both hold
        exactly where all of their operands hold, and ``|`` wherever any of them does.

        ..warning:: This reads ``*`` as a conjunction, which holds only because a suite
            writing it means it of outcomes. Multiplying two quantities is arithmetic and
            not a condition, and nothing here can tell the two apart.

        :param expression: Two outcomes combined by an operator.
        :return: The condition their combination states.
        """
        operands = (
            self.condition_of_expression(expression.left),
            self.condition_of_expression(expression.right),
        )
        if isinstance(expression.op, (ast.BitAnd, ast.Mult)):
            return and_(*operands)
        if isinstance(expression.op, ast.BitOr):
            return or_(*operands)
        return UnreadCondition.found(
            f"outcomes combined by {type(expression.op).__name__}"
        )

    def condition_of_comparison(self, expression: ast.Compare) -> SymbolicExpression:
        """
        Reads a comparison as the bounds it states of the value it measures.

        A comparison against a threshold measures a value and bounds it, and the kind of
        set the bound describes follows from what the value is compared against.

        :param expression: A comparison evaluated as a condition.
        :return: The bounds it puts on the value being compared, required together.
        """
        polarity = self.truth_comparison_polarity(expression)
        if polarity is not None:
            stated = self.condition_of_expression(expression.left)
            return stated if polarity else not_(stated)

        quantity = self.compared_quantity(expression)
        name = self.dotted_name(quantity)
        compared_values = self.literals_compared_against(expression, quantity)
        value = self.vocabulary.classify_comparison(
            name, compared_values
        ).value_measured_over(name, self.measured_over(quantity))
        return and_(*self.bounds_on(value, expression, quantity))

    def bounds_on(
        self, value: SymbolicExpression, expression: ast.Compare, quantity: ast.expr
    ) -> list[SymbolicExpression]:
        """
        States every bound a comparison chain puts on the value it measures.

        A value bounded from both sides is bounded twice, and the two bounds describe
        different sets, so they are stated as two comparisons rather than collapsed into
        one.

        :param value: The measured value the comparison is about.
        :param expression: A comparison evaluated as a condition.
        :param quantity: The operand the comparison is about.
        :return: One bound per comparison the chain states of the value, each written
            with the value on the left so that a threshold written before its value
            states the same bound as one written after it.
        """
        bounds = []
        operands = (expression.left, *expression.comparators)
        for index, comparison in enumerate(expression.ops):
            left, right = operands[index], operands[index + 1]
            if left is not quantity and right is not quantity:
                continue
            value_written_first = left is quantity
            bounds.append(
                self.bound_stated_by(
                    comparison,
                    value,
                    self.bound_written_as(right if value_written_first else left),
                    value_written_first,
                )
            )
        return bounds

    def bound_stated_by(
        self,
        comparison: ast.cmpop,
        value: SymbolicExpression,
        compared_against: Any,
        value_written_first: bool,
    ) -> SymbolicExpression:
        """
        States the bound one comparison of a chain puts on the value it measures.

        An ordering, an equality or an identity is stated of the value first whichever
        side the task writes it on, since the same bound is stated either way. A
        containment is not symmetric, so which side the value is written on decides
        whether the value is held by what it is compared against or holds it.

        :param comparison: How the task compares the value it measures.
        :param value: The measured value.
        :param compared_against: What the task compares it against, as written.
        :param value_written_first: Whether the task writes the value before what it is
            compared against.
        :return: The condition the comparison states, which is unread when the survey
            does not recognise the comparison.
        """
        if isinstance(comparison, (ast.In, ast.NotIn)):
            held = (
                in_(value, compared_against)
                if value_written_first
                else contains(value, compared_against)
            )
            return not_(held) if isinstance(comparison, ast.NotIn) else held

        operation = self.comparisons.get(type(comparison))
        if operation is None:
            return UnreadCondition.found(
                f"value bounded by {type(comparison).__name__}"
            )
        if not value_written_first:
            operation = self.reversed_comparisons[operation]
        return Comparator(value, compared_against, operation)

    @staticmethod
    def bound_written_as(operand: ast.expr) -> Any:
        """
        :param operand: What a measured value is compared against.
        :return: The bound as the task writes it, which is the number or state itself when
            the task writes a literal and the source text of the bound otherwise.
        """
        if isinstance(operand, ast.Constant):
            return operand.value
        return ast.unparse(operand)

    @staticmethod
    def literals_compared_against(
        expression: ast.Compare, quantity: ast.expr
    ) -> tuple[object, ...]:
        """
        Collects what a comparison chain writes its quantity against as literals.

        Every operand of the chain except the quantity itself is something the quantity
        is compared against, whichever side of the comparison it is written on, so a
        threshold written before its quantity states the same bound as one written after
        it.

        :param expression: A comparison evaluated as a condition.
        :param quantity: The operand the comparison is about.
        :return: The literal values the quantity is compared against.
        """
        return tuple(
            operand.value
            for operand in (expression.left, *expression.comparators)
            if operand is not quantity and isinstance(operand, ast.Constant)
        )

    def measured_over(self, quantity: ast.expr) -> Variable:
        """
        Names what a bounded value was measured over.

        The same measure taken over a whole position and over a slice of one describes
        regions of different dimension, so what it was applied to is part of what the
        comparison states and not only of how it was written.

        :param quantity: The value a comparison bounds.
        :return: The objects the value was measured over, or the empty naming when its
            own name already says so.
        """
        if isinstance(quantity, ast.Call):
            return self.objects_of(quantity)
        return SceneObject.written_as("")

    def condition_of_call(self, expression: ast.Call) -> SymbolicExpression:
        """
        :param expression: A call evaluated as a condition.
        :return: The condition it states, which is a quantification when the call is
            ``all`` or ``any``, a combination when it combines outcomes element-wise, and a
            relation otherwise.
        """
        name = self.dotted_name(expression.func)

        if (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr in self.truth_value_casts
            and not expression.args
        ):
            return self.condition_of_expression(expression.func.value)

        combination = self.combining_calls.get(name.rsplit(".", 1)[-1])
        if combination is not None and len(expression.args) > 1:
            return combination(
                *(
                    self.condition_of_expression(argument)
                    for argument in expression.args
                )
            )

        if name.rsplit(".", 1)[-1] == "logical_not" and expression.args:
            return not_(self.condition_of_expression(expression.args[0]))

        if name in ("all", "any") and expression.args:
            quantified = self.quantified_comprehension(expression.args[0])
            if name == "all":
                return quantified.holds_for_every_object()
            return quantified.holds_for_some_object()

        return self.relation_of(name, self.objects_of(expression))

    def objects_of(self, expression: ast.Call) -> Variable:
        """
        :param expression: A call stating a relation or measuring a value.
        :return: The objects it is stated of, which is the variable a quantifier binds when
            the call is stated of a quantified object, and the objects as written
            otherwise. The scene every relation is passed carries nothing and is left out.
        """
        quantified = self.quantified_object_in(expression.args)
        if quantified is not None:
            return quantified
        arguments = [
            ast.unparse(argument)
            for argument in expression.args
            if not (isinstance(argument, ast.Name) and argument.id == "self")
        ]
        arguments += [
            f"{keyword.arg}={ast.unparse(keyword.value)}"
            for keyword in expression.keywords
        ]
        return SceneObject.written_as(f"({', '.join(arguments)})")

    def quantified_object_in(self, arguments: list[ast.expr]) -> Optional[Variable]:
        """
        Finds the quantified object a call is applied to.

        A quantified object is reached through whatever the task writes around it, so
        the arguments are searched rather than only matched.

        :param arguments: The arguments a call is applied to.
        :return: The quantified object they name, or ``None`` when they name none.
        """
        for argument in arguments:
            for node in ast.walk(argument):
                if isinstance(node, ast.Name) and node.id in self.quantified_objects:
                    return self.quantified_objects[node.id]
        return None

    def quantified_comprehension(self, expression: ast.expr) -> QuantifiedCondition:
        """
        :param expression: The argument of a quantifying call.
        :return: Its element condition together with the object it is quantified over.
        """
        if (
            not isinstance(expression, (ast.ListComp, ast.GeneratorExp, ast.SetComp))
            or not expression.generators
        ):
            return QuantifiedCondition(
                objects=SceneObject.provided_by_the_scene(),
                body=self.condition_of_expression(expression),
            )
        objects = self.quantify_over(expression.generators[0].target)
        return QuantifiedCondition(
            objects=objects, body=self.condition_of_expression(expression.elt)
        )

    @staticmethod
    def truth_comparison_polarity(expression: ast.Compare) -> Optional[bool]:
        """
        Tells whether a comparison against a truth value asserts or denies its left
        side.

        Conditions are often written by comparing a check against ``True`` or ``False``
        rather than by using it directly. Such a comparison states the check itself, or
        its negation, and reading it as an opaque relation would silently drop that
        distinction.

        :param expression: A comparison evaluated as a condition.
        :return: Whether the comparison asserts that its left side holds, whether it
            asserts that it fails, or ``None`` when it is not a comparison against a
            truth value at all.
        """
        if len(expression.ops) != 1 or len(expression.comparators) != 1:
            return None
        if not isinstance(expression.ops[0], (ast.Eq, ast.NotEq)):
            return None
        comparator = expression.comparators[0]
        if not isinstance(comparator, ast.Constant) or not isinstance(
            comparator.value, bool
        ):
            return None
        return comparator.value is isinstance(expression.ops[0], ast.Eq)

    @staticmethod
    def compared_quantity(expression: ast.Compare) -> ast.expr:
        """
        Picks out the operand a comparison is about.

        ..note:: A quantity bounded from both sides is the only operand of the chain that
            is compared against rather than compared with, so it is the subject whether or
            not its bounds are written as literals.

        :param expression: A comparison evaluated as a condition.
        :return: The operand naming what is being compared, which is the middle one when
            the quantity is bounded from both sides, and the right-hand one when the
            comparison is written with its threshold first.
        """
        if len(expression.ops) > 1:
            return expression.comparators[0]
        if not isinstance(expression.left, ast.Constant):
            return expression.left
        return expression.comparators[0]

    def subscripted_name(self, expression: ast.Subscript) -> str:
        """
        Names the entry a subscript reads out of a state collection.

        A fixture's state is read one key at a time, and each key stands for its own
        condition, so the key belongs in the relation's name rather than being collapsed
        into the collection it was read from.

        :param expression: A subscript reading one entry of a state collection.
        :return: The name of the entry read, keeping the key when it is a literal.
        """
        base = self.dotted_name(expression.value)
        if isinstance(expression.slice, ast.Constant):
            return f"{base}[{expression.slice.value!r}]"
        return base

    def dotted_name(self, expression: ast.expr) -> str:
        """
        :param expression: The expression naming a relation or a measure.
        :return: The dotted name it spells, or a placeholder when it is not a plain name.
        """
        if isinstance(expression, ast.Name):
            return self.name_aliases.get(expression.id, expression.id)
        if isinstance(expression, ast.Attribute):
            return f"{self.dotted_name(expression.value)}.{expression.attr}"
        if isinstance(expression, ast.Call):
            return self.dotted_name(expression.func)
        if isinstance(expression, ast.Subscript):
            return self.dotted_name(expression.value)
        if isinstance(expression, ast.BinOp):
            return self.dotted_name(expression.left)
        return type(expression).__name__


# %% reading a task suite


@dataclass
class SuiteSourceNotFound(DataclassException):
    """
    Raised when a suite's task definitions cannot be located, since a survey of a suite
    that is not there would report an empty suite rather than a missing one.
    """

    package_name: str
    """
    Name of the package whose task definitions were looked for.
    """

    def error_message(self) -> str:
        return (
            f"{self.package_name} is not installed, so its task definitions cannot be "
            "read."
        )

    def suggest_correction(self) -> str:
        return f"Install {self.package_name} and run the survey again."


@dataclass
class TaskConditionReader(ABC):
    """
    Reads the success condition every task of one suite states.

    A suite decides where its task definitions live, what it calls its relations, and
    how a task states its condition. Everything else -- walking the suite, and reporting
    one condition per task class -- is shared.
    """

    task_root: pathlib.Path
    """
    Directory holding the task definitions to read.
    """

    vocabulary: PredicateVocabulary
    """
    Classifies the names this suite's conditions are stated under.
    """

    def conditions(self) -> dict[str, StatedTaskCondition]:
        """
        :return: The condition each task class states, keyed by class name.
        """
        conditions: dict[str, StatedTaskCondition] = {}
        for path in sorted(self.task_root.rglob("*.py")):
            conditions.update(self.conditions_of_module(path))
        return conditions

    def conditions_of_module(
        self, path: pathlib.Path
    ) -> dict[str, StatedTaskCondition]:
        """
        :param path: A module possibly defining task classes.
        :return: The condition each task class in it states, keyed by class name.
        """
        tree = ast.parse(path.read_text())
        conditions = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            condition = self.condition_of_class(node)
            if condition is not None:
                conditions[node.name] = condition
        return conditions

    @abstractmethod
    def condition_of_class(self, task: ast.ClassDef) -> Optional[StatedTaskCondition]:
        """
        :param task: A class that may define a task.
        :return: The condition it states, or ``None`` when it states none.
        """

    @staticmethod
    def installed_source_directory(
        package_name: str, tasks_within: str
    ) -> pathlib.Path:
        """
        Locates the task definitions of an installed suite.

        A suite is located rather than imported, since the survey reads its source and
        would otherwise need the simulator the suite drives to be installed too.

        :param package_name: Name of the installed package holding the suite.
        :param tasks_within: Path of the task definitions inside that package.
        :return: Directory the suite's task definitions live in.
        :raises SuiteSourceNotFound: When the package is not installed.
        """
        specification = importlib.util.find_spec(package_name)
        if specification is None or specification.origin is None:
            raise SuiteSourceNotFound(package_name=package_name)
        return pathlib.Path(specification.origin).parent / tasks_within


@dataclass
class ComputedConditionReader(TaskConditionReader):
    """
    Reads the conditions of a suite whose tasks compute and return their outcome.

    Such a suite states its condition as ordinary Python in one named method, so reading
    it means following that method's control flow, and a suite is added by naming the
    method and saying where in a return statement the outcome sits.
    """

    @property
    @abstractmethod
    def condition_method_name(self) -> str:
        """
        :return: Name of the method a task of this suite states its success condition in.
        """

    @property
    @abstractmethod
    def outcome(self) -> StatedOutcome:
        """
        :return: Where in a return statement that method states its outcome.
        """

    def condition_of_class(self, task: ast.ClassDef) -> Optional[StatedTaskCondition]:
        """
        :param task: A class that may define a task.
        :return: The condition its condition method states, or ``None`` when it defines no
            such method.
        """
        for member in task.body:
            if (
                isinstance(member, ast.FunctionDef)
                and member.name == self.condition_method_name
            ):
                return self.condition_of_method(member)
        return None

    def condition_of_method(self, method: ast.FunctionDef) -> StatedTaskCondition:
        """
        :param method: The method a task states its condition in.
        :return: The condition it states.
        """
        return ConditionExtractor(
            vocabulary=self.vocabulary, outcome=self.outcome
        ).extract(method)
