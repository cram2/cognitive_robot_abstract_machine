"""
Reading the success condition an RLBench task states.

RLBench states a condition by registering the requirements it is composed of, rather
than by computing it, so each test here pins how one registered composition is read -- a
list of requirements, a negated requirement, a set that requires all of its members, a
set that accepts any of them.
"""

import pathlib
import textwrap

from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not

from experiments.random_events_experiments.task_condition_survey.rlbench_condition_survey import (
    RLBenchConditionSurvey,
    RLBenchPredicateVocabulary,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConditionKind,
    PredicateKind,
    StatedTaskCondition,
)


def read_condition(body: str, task_root: pathlib.Path) -> StatedTaskCondition:
    """
    :param body: The body of a task's setup method, as written in the suite.
    :param task_root: Directory to write the task into and read it back from.
    :return: The condition the task states.
    """
    (task_root / "task.py").write_text(textwrap.dedent("""
            class SomeTask(Task):
                def init_task(self):
            """) + textwrap.indent(textwrap.dedent(body), " " * 8))
    [condition] = RLBenchConditionSurvey(task_root=task_root).conditions().values()
    return condition


def names_stated_by(condition: StatedTaskCondition) -> list[str]:
    """
    :param condition: A condition read from a task.
    :return: The names its propositions are stated under.
    """
    return [proposition.stated_under for proposition in condition.propositions()]


# %% how the suite states a condition


def test_registered_requirements_are_all_required(tmp_path: pathlib.Path):
    """
    A task registers the requirements it succeeds on, and succeeds only once every one
    of them is met, so registering several states their conjunction.
    """
    condition = read_condition(
        """self.register_success_conditions([ DetectedCondition(self.rubbish,
        success_sensor), NothingGrasped(self.robot.gripper), ])""",
        tmp_path,
    )

    assert isinstance(condition.expression, AND)
    assert sorted(names_stated_by(condition)) == [
        "DetectedCondition",
        "NothingGrasped",
    ]


def test_requirements_named_before_being_registered_are_read_the_same_way(
    tmp_path: pathlib.Path,
):
    """
    A task that names its requirements before registering them states the same condition
    as one registering them directly.
    """
    condition = read_condition(
        """
        conditions = [
            DetectedCondition(self.block, success_sensor),
            JointCondition(self.joint, 0.1),
        ]
        self.register_success_conditions(conditions)
        """,
        tmp_path,
    )

    assert isinstance(condition.expression, AND)
    assert len(condition.propositions()) == 2


def test_negated_requirement_states_a_negation(tmp_path: pathlib.Path):
    """
    A requirement registered as negated is met exactly when its own test fails, so it
    states the negation of that test rather than the test.
    """
    condition = read_condition(
        """self.register_success_conditions([ DetectedCondition(self.block,
        success_sensor, negated=True), ])""",
        tmp_path,
    )

    assert isinstance(condition.expression, Not)


def test_set_of_requirements_requires_all_of_them(tmp_path: pathlib.Path):
    """
    A set of requirements registered as one condition is met once all of its members
    are, so it states a conjunction nested inside the task's own.
    """
    condition = read_condition(
        """self.register_success_conditions([ ConditionSet([ DetectedCondition(self.block,
        success_sensor), JointCondition(self.joint, 0.1), ], order_matters=True), ])""",
        tmp_path,
    )

    assert isinstance(condition.expression, AND)
    assert len(condition.propositions()) == 2


def test_alternative_requirements_state_a_disjunction(tmp_path: pathlib.Path):
    """
    Requirements registered as alternatives are met once any of them is, so they state a
    disjunction and cost one term per alternative.
    """
    condition = read_condition(
        """self.register_success_conditions([ OrConditions([ DetectedCondition(self.block,
        left_sensor), DetectedCondition(self.block, right_sensor), ]), ])""",
        tmp_path,
    )

    assert isinstance(condition.expression, OR)


def test_task_registering_nothing_readable_is_recorded_as_unread(
    tmp_path: pathlib.Path,
):
    """
    A task whose requirements the survey cannot read states something unknown, so it is
    recorded as unread rather than as stating no condition.
    """
    condition = read_condition(
        """self.register_success_conditions(self.conditions_for_variation)""",
        tmp_path,
    )

    assert not condition.is_fully_read()


# %% what those requirements state


def test_requirement_the_task_passes_a_threshold_to_is_continuous():
    """
    A requirement the task gives a bound to constrains a quantity, so it states a
    continuous condition.
    """
    assert (
        RLBenchPredicateVocabulary().classify("JointCondition")
        is PredicateKind.CONTINUOUS
    )


def test_requirement_testing_a_relation_states_a_finite_domain():
    """
    A requirement asking whether an object is held or detected tests a relation the task
    puts no bound on, so it states a condition over a finite domain.
    """
    vocabulary = RLBenchPredicateVocabulary()

    assert vocabulary.classify("DetectedCondition") is PredicateKind.DISCRETE
    assert vocabulary.classify("NothingGrasped") is PredicateKind.DISCRETE


def test_condition_combining_a_bound_and_a_relation_is_hybrid(tmp_path: pathlib.Path):
    """
    A task requiring both a moved joint and a detected object states a condition of both
    kinds at once, which is the kind of set the survey counts.
    """
    condition = read_condition(
        """self.register_success_conditions([ JointCondition(self.joint, 0.1),
        DetectedCondition(self.block, success_sensor), ])""",
        tmp_path,
    )

    assert condition.kind is ConditionKind.HYBRID


def test_one_requirement_applied_to_two_objects_states_two_conditions(
    tmp_path: pathlib.Path,
):
    """
    The same requirement stated of two different objects constrains two things, so the
    two are not collapsed into one condition.
    """
    condition = read_condition(
        """self.register_success_conditions([ DetectedCondition(self.first,
        success_sensor), DetectedCondition(self.second, success_sensor), ])""",
        tmp_path,
    )

    constrained = {
        proposition.variable_name for proposition in condition.propositions()
    }
    assert len(constrained) == 2
