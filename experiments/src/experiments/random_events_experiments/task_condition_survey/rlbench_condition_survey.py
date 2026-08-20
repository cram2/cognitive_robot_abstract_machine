"""
This module reads the success condition every RLBench task states.

RLBench does not compute its conditions: a task *registers* them, as a list of condition
objects each standing for one requirement, and the task succeeds when all of them are
met. So the condition is already a composed structure in the source, and reading it
needs no control flow to be followed -- which is why RLBench is the suite where what the
survey recovers is least in doubt.

What each registered condition states is read from that condition class rather than from
the task: a class bounding a quantity the task passes it states a continuous condition,
while one testing a relation states a condition over a finite domain.

The suite is read from source and never imported, so neither RLBench nor the simulator
it drives has to be installed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing_extensions import Callable, ClassVar

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import and_, not_, or_

from experiments.random_events_experiments.task_condition_survey.condition_extraction import (
    TaskConditionReader,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    PredicateKind,
    PredicateNameRule,
    PredicateVocabulary,
    SceneObject,
    StatedTaskCondition,
    UnreadCondition,
)

# %% what RLBench calls its predicates


@dataclass
class RLBenchPredicateVocabulary(PredicateVocabulary):
    """
    Classifies the conditions RLBench's tasks register.

    Every requirement is one of a small set of condition classes, so these are
    recognised by name outright rather than by convention. A class the task passes a
    threshold to bounds a continuous quantity; one testing whether an object is held,
    detected or absent states a condition over a finite domain.
    """

    exact_kinds: dict[str, PredicateKind] = field(
        default_factory=lambda: {
            "JointCondition": PredicateKind.CONTINUOUS,
            "FollowCondition": PredicateKind.CONTINUOUS,
            "DetectedCondition": PredicateKind.DISCRETE,
            "DetectedSeveralCondition": PredicateKind.DISCRETE,
            "GraspedCondition": PredicateKind.DISCRETE,
            "NothingGrasped": PredicateKind.DISCRETE,
            "EmptyCondition": PredicateKind.DISCRETE,
            "ColorCondition": PredicateKind.DISCRETE,
        }
    )
    """
    Kind each of RLBench's condition classes states.

    ..note:: ``DetectedCondition`` asks a proximity sensor whether it detects an object.
        The volume that sensor covers is set in the scene rather than by the task, so the
        task states a relation and not a bound, and it is read as discrete for the same
        reason a contact check is.
    """

    name_rules: tuple[PredicateNameRule, ...] = ()
    """
    No naming convention is consulted, since the suite's conditions are a closed set.
    """


# %% reading the suite


@dataclass
class RLBenchConditionSurvey(TaskConditionReader):
    """
    Reads every RLBench task's success condition from the conditions it registers.
    """

    vocabulary: PredicateVocabulary = field(default_factory=RLBenchPredicateVocabulary)
    """
    Classifies the conditions RLBench's tasks register.
    """

    registration_name: ClassVar[str] = "register_success_conditions"
    """
    Name of the call a task states its success condition through.
    """

    composing_conditions: ClassVar[dict[str, Callable[..., SymbolicExpression]]] = {
        "ConditionSet": and_,
        "OrConditions": or_,
    }
    """
    Condition classes that compose other conditions, and the condition each one states.
    """

    def condition_of_class(self, task: ast.ClassDef) -> StatedTaskCondition | None:
        """
        A task registers its conditions while setting itself up, or once its variation is
        known, so the registration is looked for anywhere in the class rather than in one
        named method.

        ..note:: A task registering its conditions per variation states a condition per
            variation. The first registration read stands for the task, so a suite of
            variations is counted once.

        :param task: A class that may define a task.
        :return: The conjunction of the conditions it registers, since a task succeeds only
            when every registered condition is met, or ``None`` when it registers none.
        """
        if not self.states_a_condition(task):
            return None
        registered = self.registered_conditions(task)
        if registered is None:
            return StatedTaskCondition(
                expression=UnreadCondition.found(
                    f"no {self.registration_name} call stating a list"
                )
            )
        return StatedTaskCondition(expression=self.condition_of_all(registered))

    def states_a_condition(self, task: ast.ClassDef) -> bool:
        """
        :param task: A class that may define a task.
        :return: Whether it registers success conditions at all.
        """
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == self.registration_name
            for node in ast.walk(task)
        )

    def registered_conditions(self, task: ast.ClassDef) -> list[ast.expr] | None:
        """
        A task either registers a list outright or names it first, and naming it states the
        same conditions.

        :param task: The class defining the task.
        :return: The conditions it registers, or ``None`` when it registers none the survey
            can read.
        """
        lists_by_name = {
            target.id: statement.value
            for statement in ast.walk(task)
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name) and isinstance(statement.value, ast.List)
        }
        for node in ast.walk(task):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == self.registration_name
                and node.args
            ):
                continue
            registered = node.args[0]
            if isinstance(registered, ast.Name):
                registered = lists_by_name.get(registered.id, registered)
            if isinstance(registered, ast.List):
                return list(registered.elts)
        return None

    def condition_of_all(self, conditions: list[ast.expr]) -> SymbolicExpression:
        """
        :param conditions: The conditions a task requires together.
        :return: The condition requiring all of them.
        """
        return and_(*(self.condition_of_registered(each) for each in conditions))

    def condition_of_registered(self, condition: ast.expr) -> SymbolicExpression:
        """
        :param condition: One registered condition.
        :return: The condition it states, which is a composition when it composes others
            and a relation otherwise.
        """
        if not isinstance(condition, ast.Call):
            return UnreadCondition.found(f"registered {type(condition).__name__}")

        name = self.condition_class_name(condition.func)
        composition = self.composing_conditions.get(name)
        if composition is not None:
            return self.condition_of_composition(condition, composition)

        stated = self.vocabulary.classify(name).relation_stated_of(
            name, SceneObject.written_as(self.objects_stated_of(condition))
        )
        if self.is_negated(condition):
            return not_(stated)
        return stated

    def condition_of_composition(
        self, condition: ast.Call, composition: Callable[..., SymbolicExpression]
    ) -> SymbolicExpression:
        """
        ..note:: A set of conditions may also require that they be met in a given order.
            Ordering is a constraint on the sequence of states rather than on any one state,
            so it is outside what this survey represents and is not recorded: such a set is
            read as requiring its members, which is what its members state.

        :param condition: A condition composing others.
        :param composition: The condition its composition states.
        :return: That composition over the conditions it composes.
        """
        if not condition.args or not isinstance(condition.args[0], ast.List):
            return UnreadCondition.found(
                f"{self.condition_class_name(condition.func)} over an unread collection"
            )
        return composition(
            *(self.condition_of_registered(each) for each in condition.args[0].elts)
        )

    @staticmethod
    def condition_class_name(function: ast.expr) -> str:
        """
        :param function: The expression naming the condition class.
        :return: The class's own name, without the module it was reached through.
        """
        if isinstance(function, ast.Attribute):
            return function.attr
        if isinstance(function, ast.Name):
            return function.id
        return type(function).__name__

    @staticmethod
    def is_negated(condition: ast.Call) -> bool:
        """
        :param condition: A registered condition.
        :return: Whether it requires its own test to fail.
        """
        return any(
            keyword.arg == "negated"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in condition.keywords
        )

    @staticmethod
    def objects_stated_of(condition: ast.Call) -> str:
        """
        ..note:: Every argument counts here, unlike a computed condition where the scene
            is passed to each predicate and carries nothing: a registered condition is
            constructed from the objects it is about alone.

        :param condition: A registered condition.
        :return: The objects it is stated of, as written, so that one condition class
            applied to two objects states two conditions rather than one.
        """
        return f"({', '.join(ast.unparse(argument) for argument in condition.args)})"

    @classmethod
    def for_installed_rlbench(cls) -> RLBenchConditionSurvey:
        """
        :return: A survey reading the task definitions of the installed RLBench.
        :raises SuiteSourceNotFound: When RLBench is not installed.
        """
        return cls(task_root=cls.installed_source_directory("rlbench", "tasks"))
