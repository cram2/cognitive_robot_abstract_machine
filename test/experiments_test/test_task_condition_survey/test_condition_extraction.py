"""
Recovering the EQL condition a task states from the ordinary Python its success condition
is written in.

Each test exercises one way a success condition is phrased -- a named intermediate, a guard
clause returning early, a loop deciding on the first object, a quantifier over a
comprehension -- rather than one task that happens to be phrased that way, so the survey's
coverage is pinned to the patterns and not to a particular task suite.
"""

import ast
import pathlib
import textwrap

from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll

from experiments.random_events_experiments.task_condition_survey.condition_extraction import (
    ConditionExtractor,
    ReturnedEntry,
    ReturnedValue,
    StatedOutcome,
)
from experiments.random_events_experiments.task_condition_survey.robocasa_condition_survey import (
    RoboCasaConditionSurvey,
    RoboCasaPredicateVocabulary,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    PredicateKind,
    StatedTaskCondition,
)

CHECK_SUCCESS = "def _check_success(self):"
"""
The header every condition method written out by these tests carries, so a test states only
the body whose phrasing it is about.
"""

EVALUATE = "def evaluate(self):"
"""
The header of a condition method reporting its outcome alongside other measurements.
"""


def extracted_condition(
    body: str, header: str = CHECK_SUCCESS, outcome: StatedOutcome | None = None
) -> StatedTaskCondition:
    """
    :param body: The body of a condition method, as written in a task.
    :param header: The condition method's signature.
    :param outcome: How the method states its outcome, defaulting to returning it directly.
    :return: The condition it states.
    """
    source = header + "\n" + textwrap.indent(textwrap.dedent(body).strip(), " " * 4)
    function = ast.parse(source).body[0]
    return ConditionExtractor(
        vocabulary=RoboCasaPredicateVocabulary(), outcome=outcome or ReturnedValue()
    ).extract(function)


def names_stated_by(condition: StatedTaskCondition) -> set[str]:
    """
    :param condition: A condition read from a task.
    :return: The names its propositions are stated under.
    """
    return {proposition.stated_under for proposition in condition.propositions()}


# %% phrasings of a condition


def test_named_intermediates_are_resolved_into_the_returned_condition():
    """
    A condition that names its parts before combining them states the same condition as one
    written inline.
    """
    condition = extracted_condition("""
        on_counter = OU.check_obj_fixture_contact(self, "tray", self.counter)
        gripper_far = OU.gripper_obj_far(self, "tray")
        return on_counter and gripper_far
        """)

    assert isinstance(condition.expression, AND)
    assert condition.is_fully_read()
    assert names_stated_by(condition) == {
        "OU.check_obj_fixture_contact",
        "OU.gripper_obj_far",
    }


def test_guard_clause_returning_false_states_a_conjunction():
    """
    A condition that bails out early on a failing check requires that check, so the check is
    a conjunct of the condition rather than a separate branch.
    """
    condition = extracted_condition("""
        if not OU.gripper_obj_far(self, "lid"):
            return False
        return self.blender.is_closed()
        """)

    assert isinstance(condition.expression, AND)
    assert condition.is_fully_read()
    assert len(condition.propositions()) == 2


def test_loop_deciding_on_the_first_object_states_an_existential_quantification():
    """
    A loop returning True as soon as one object satisfies its body holds exactly when some
    object does, so it states an existential quantification.
    """
    condition = extracted_condition("""
        for fixture in self.fixtures.values():
            if self.check_contact(self.lid, fixture):
                return True
        return False
        """)

    assert [type(each) for each in condition.quantifications()] == [Exists]


def test_condition_stated_of_a_quantified_object_is_stated_of_the_bound_variable():
    """
    A loop's object is what the quantifier binds, so the conditions the loop states are
    stated of that variable rather than of a name standing for nothing.
    """
    condition = extracted_condition("""
        for utensil in self.metals:
            if not OU.obj_inside_of(self, utensil, self.drawer):
                return False
        return True
        """)

    [quantification] = condition.quantifications()
    [proposition] = condition.propositions()
    assert proposition.expression._kwargs_["objects"] is quantification.left


def test_quantity_compared_through_named_intermediates_keeps_the_measure_it_came_from():
    """
    A quantity is named after the measure it was taken under however many named
    intermediates it passes through before being compared, so a condition that names its
    steps states the same proposition as one written inline.
    """
    condition = extracted_condition("""
        for site in self.stove.burner_sites:
            burner_position = self.position_of(site)
            distance = np.linalg.norm(burner_position - self.pan_position)
            if distance < 0.15:
                return True
        return False
        """)

    [proposition] = condition.propositions()
    assert proposition.stated_under == "np.linalg.norm"


def test_loop_rejecting_on_the_first_failure_states_a_universal_quantification():
    """
    A loop returning False as soon as one object fails requires every object to pass, so it
    states a universal quantification and not an existential one.
    """
    condition = extracted_condition("""
        for utensil in self.metals:
            if not OU.obj_inside_of(self, utensil, self.drawer):
                return False
        return OU.gripper_obj_far(self, "drawer")
        """)

    assert [type(each) for each in condition.quantifications()] == [ForAll]


def test_loop_that_only_names_intermediates_states_no_quantification():
    """
    A loop that never returns does not decide the outcome, so it quantifies nothing and what
    follows it still states the condition.
    """
    condition = extracted_condition("""
        count = 0
        for utensil in self.utensils:
            count = count + 1
        return OU.gripper_obj_far(self, "drawer")
        """)

    assert condition.quantifications() == []
    assert names_stated_by(condition) == {"OU.gripper_obj_far"}


def test_name_bound_inside_a_loop_is_still_read_after_it():
    """
    A loop that decides no outcome still binds the names it assigns, so a condition stated
    after the loop in terms of one of them is recovered rather than left unread.
    """
    condition = extracted_condition("""
        for utensil in self.metals:
            in_drawer = OU.obj_inside_of(self, utensil, self.drawer)
        return in_drawer
        """)

    assert condition.is_fully_read()
    assert names_stated_by(condition) == {"OU.obj_inside_of"}


def test_all_over_a_comprehension_states_a_universal_quantification():
    """
    A condition requiring every object of a collection to satisfy a relation states a
    universal quantification, whose expansion depends on how many objects the scene holds.
    """
    condition = extracted_condition("""
        return all(OU.check_obj_in_receptacle(self, name, "bowl") for name in self.objects)
        """)

    assert isinstance(condition.expression, ForAll)
    assert names_stated_by(condition) == {"OU.check_obj_in_receptacle"}


def test_check_compared_against_false_states_its_negation():
    """
    A check written as a comparison against ``False`` requires that the check fails, so it
    states the check's negation and not the check itself.

    Reading it as an opaque relation would make a condition that requires a check to fail of
    one object and to hold of another appear to require the same thing twice.
    """
    condition = extracted_condition("""
        removed = OU.obj_inside_of(self, "food", self.sink) == False
        placed = OU.obj_inside_of(self, "dish", self.sink) == True
        return removed and placed
        """)

    assert isinstance(condition.expression, AND)
    assert isinstance(condition.expression.left, Not)
    assert names_stated_by(condition) == {"OU.obj_inside_of"}
    assert {proposition.stated_of for proposition in condition.propositions()} == {
        "('food', self.sink)",
        "('dish', self.sink)",
    }


def test_check_compared_against_true_with_inequality_states_its_negation():
    """
    Requiring a check to differ from ``True`` requires it to fail, the same as negating it.
    """
    condition = extracted_condition("""
        return OU.gripper_obj_far(self, "cup") != True
        """)

    assert isinstance(condition.expression, Not)


def test_branch_that_only_names_an_intermediate_does_not_end_the_condition():
    """
    A conditional that assigns rather than returns leaves the condition to be stated by what
    follows it, so everything after the branch still counts.
    """
    condition = extracted_condition("""
        if OU.obj_inside_of(self, "bread", self.bowl):
            in_bowl = True
        else:
            in_bowl = self.check_contact(self.bread, self.other_bread)
        on_counter = OU.check_obj_fixture_contact(self, "bowl", self.counter)
        gripper_far = OU.gripper_obj_far(self, "bowl")
        return in_bowl and on_counter and gripper_far
        """)

    assert {
        "OU.check_obj_fixture_contact",
        "OU.gripper_obj_far",
        "self.check_contact",
    } <= names_stated_by(condition)


def test_unread_phrasing_is_recorded_rather_than_dropped():
    """
    A condition resting on a value the survey never saw bound is marked as unread, so a
    partial reading is never reported as a complete one.
    """
    condition = extracted_condition("""
        return decided_elsewhere
        """)

    assert not condition.is_fully_read()
    assert condition.unread_parts() == ["unbound name 'decided_elsewhere'"]


def test_quantity_initialised_to_a_literal_keeps_its_own_name():
    """
    A counter or accumulator started at a literal is still named after itself, rather than
    after the literal it was started from.
    """
    condition = extracted_condition("""
        contact_count = 0
        return contact_count > 2
        """)

    assert names_stated_by(condition) == {"contact_count"}


def test_threshold_written_before_the_quantity_states_the_same_bound():
    """
    A comparison states the same condition whichever side its threshold is written on, so
    the value is bounded either way and neither the kind nor the bound may follow the
    spelling.
    """
    written_last = extracted_condition("""
        return self.tray_offset < 0.15
        """)
    written_first = extracted_condition("""
        return 0.15 > self.tray_offset
        """)

    [bounded] = written_last.propositions()
    [bounded_the_other_way] = written_first.propositions()
    assert bounded.kind is PredicateKind.CONTINUOUS
    assert bounded.variable_name == bounded_the_other_way.variable_name


def test_quantity_bounded_from_both_sides_is_named_after_the_quantity():
    """
    A quantity written between a lower and an upper bound is what the comparison is about,
    so the propositions are named after it rather than after either bound.
    """
    condition = extracted_condition("""
        LOWER = self.stove.STOVE_LOW_MIN
        UPPER = self.stove.STOVE_HIGH_MIN
        return LOWER <= np.abs(knob_value) <= UPPER
        """)

    assert names_stated_by(condition) == {"np.abs"}


def test_quantity_bounded_from_both_sides_states_a_bound_each_way():
    """
    Bounding a quantity between two numbers states a continuous condition twice over, since
    a lower and an upper bound describe different sets and requiring both is their
    conjunction.
    """
    condition = extracted_condition("""
        return 0.35 <= np.abs(knob_value) <= 0.85
        """)

    assert [proposition.kind for proposition in condition.propositions()] == [
        PredicateKind.CONTINUOUS,
        PredicateKind.CONTINUOUS,
    ]
    assert [proposition.bound for proposition in condition.propositions()] == [
        0.35,
        0.85,
    ]


def test_membership_states_a_containment_the_way_round_it_is_written():
    """
    A containment does not say the same thing either way round, so which side of it the task
    writes the value on has to reach what the condition constrains.
    """
    held = extracted_condition("""
        return self.stove.knob in self.lit_burners
        """)
    holding = extracted_condition("""
        return "kettle" in self.stove.contents
        """)

    [inside] = held.propositions()
    [around] = holding.propositions()
    assert held.is_fully_read() and holding.is_fully_read()
    assert inside.variable_name == "self.lit_burners contains self.stove.knob"
    assert around.variable_name == "self.stove.contents contains kettle"


def test_refused_membership_states_the_negation_of_the_containment():
    """
    Requiring a value to be absent requires the containment to fail, so it states that
    containment's negation rather than a containment of its own.
    """
    condition = extracted_condition("""
        return self.pan not in self.stove.burners
        """)

    assert isinstance(condition.expression, Not)
    assert condition.is_fully_read()


def test_identity_against_a_reference_is_read_as_a_comparison():
    """
    A task comparing a value against a reference states a condition of that value, so it is
    read rather than left unread for not being an ordering.
    """
    condition = extracted_condition("""
        return self.gripper.held_object is not None
        """)

    [proposition] = condition.propositions()
    assert condition.is_fully_read()
    assert proposition.variable_name == "self.gripper.held_object is_not None"


def test_state_entry_keeps_the_key_it_was_read_under():
    """
    Each entry of a fixture's state stands for its own condition, so reading two entries of
    one state collection states two distinct propositions.
    """
    condition = extracted_condition("""
        state = self.blender.get_state()
        return state["lid_on_blender"] and state["turned_on"]
        """)

    assert names_stated_by(condition) == {
        "self.blender.get_state['lid_on_blender']",
        "self.blender.get_state['turned_on']",
    }


# %% conditions written over arrays rather than over truth values


def test_bitwise_conjunction_states_a_conjunction():
    """
    A condition combining element-wise results uses the bitwise operators rather than the
    boolean keywords, and states the same conjunction.
    """
    condition = extracted_condition(
        """
        is_placed = torch.linalg.norm(goal.p - cube.p, axis=1) <= 0.05
        is_static = self.agent.is_static(0.2)
        return is_placed & is_static
        """,
        header=EVALUATE,
    )

    assert isinstance(condition.expression, AND)
    assert len(condition.propositions()) == 2


def test_bitwise_disjunction_states_a_disjunction():
    """
    Accepting either of two element-wise results states a disjunction, however it is
    spelled.
    """
    condition = extracted_condition(
        """
        is_left = torch.linalg.norm(left.p - cube.p, axis=1) <= 0.05
        is_right = torch.linalg.norm(right.p - cube.p, axis=1) <= 0.05
        return is_left | is_right
        """,
        header=EVALUATE,
    )

    assert isinstance(condition.expression, OR)


def test_multiplying_outcomes_states_a_conjunction():
    """
    Multiplying element-wise results holds exactly where all of them hold, so it states a
    conjunction just as combining them with ``and`` would.
    """
    condition = extracted_condition(
        """
        is_on_top = torch.abs(offset) <= 0.005
        is_static = self.agent.is_static(0.2)
        return is_on_top * is_static
        """,
        header=EVALUATE,
    )

    assert isinstance(condition.expression, AND)


def test_bitwise_inversion_states_a_negation():
    """
    Inverting an element-wise result requires it to fail, so it states a negation just as
    ``not`` would.
    """
    condition = extracted_condition(
        """
        is_grasped = self.agent.is_grasping(self.cube)
        return ~is_grasped
        """,
        header=EVALUATE,
    )

    assert isinstance(condition.expression, Not)


def test_logical_and_of_two_results_states_a_conjunction():
    """
    A condition combining results through the array library's own function states the same
    conjunction as the operator would.
    """
    condition = extracted_condition(
        """
        is_placed = torch.linalg.norm(goal.p - cube.p, axis=1) <= 0.05
        is_static = self.agent.is_static(0.2)
        return torch.logical_and(is_placed, is_static)
        """,
        header=EVALUATE,
    )

    assert isinstance(condition.expression, AND)
    assert len(condition.propositions()) == 2


# %% conditions returned alongside other measurements


def test_condition_returned_as_one_entry_of_a_mapping_is_read_from_that_entry():
    """
    A task reporting its outcome alongside diagnostic measurements states its condition in
    one named entry, so the condition is read from that entry and not from the mapping.
    """
    condition = extracted_condition(
        """
        is_placed = torch.linalg.norm(goal.p - cube.p, axis=1) <= 0.05
        is_grasped = self.agent.is_grasping(self.cube)
        return {"success": is_placed, "is_grasped": is_grasped}
        """,
        header=EVALUATE,
        outcome=ReturnedEntry(key="success"),
    )

    assert names_stated_by(condition) == {"torch.linalg.norm"}


def test_condition_returned_as_a_keyword_of_a_mapping_is_read_the_same_way():
    """
    A mapping built by calling ``dict`` states its entries as keywords, and names the
    outcome no differently than a mapping written out does.
    """
    condition = extracted_condition(
        """
        is_static = self.agent.is_static(0.2)
        return dict(success=is_static, elapsed=self.elapsed_steps)
        """,
        header=EVALUATE,
        outcome=ReturnedEntry(key="success"),
    )

    assert names_stated_by(condition) == {"self.agent.is_static"}


def test_mapping_named_before_being_returned_is_read_the_same_way():
    """
    A task that names its mapping before returning it states its outcome no differently than
    one returning the mapping directly.
    """
    condition = extracted_condition(
        """
        is_placed = torch.linalg.norm(goal.p - cube.p, axis=1) <= 0.05
        info = {"success": is_placed}
        return info
        """,
        header=EVALUATE,
        outcome=ReturnedEntry(key="success"),
    )

    assert condition.is_fully_read()
    assert names_stated_by(condition) == {"torch.linalg.norm"}


def test_outcome_cast_to_a_truth_value_states_the_outcome_itself():
    """
    Casting an outcome to a truth value does not change where it holds, so the condition is
    the outcome rather than the cast.
    """
    condition = extracted_condition(
        """
        success = self.agent.is_grasping(self.cube)
        return {"success": success.bool()}
        """,
        header=EVALUATE,
        outcome=ReturnedEntry(key="success"),
    )

    assert names_stated_by(condition) == {"self.agent.is_grasping"}


def test_mapping_assigned_in_a_branch_is_read_from_the_branch_that_assigned_it():
    """
    A conditional that names the outcome differently in each branch states both, each guarded
    by the branch that gave it, so reading only the last assignment would report one branch's
    condition as the whole condition.
    """
    condition = extracted_condition(
        """
        if self.uses_lid:
            info = {"success": OU.check_obj_in_receptacle(self, "cup", "bowl")}
        else:
            info = {"success": OU.gripper_obj_far(self, "cup")}
        return info
        """,
        header=EVALUATE,
        outcome=ReturnedEntry(key="success"),
    )

    assert {"OU.check_obj_in_receptacle", "OU.gripper_obj_far"} <= names_stated_by(
        condition
    )


def test_call_that_is_not_a_mapping_states_no_outcome():
    """
    A call carrying a keyword of the outcome's name is not thereby a mapping, so a task
    delegating to a helper states nothing the survey can read rather than the helper's
    argument.
    """
    condition = extracted_condition(
        """
        return super()._evaluate(success=True)
        """,
        header=EVALUATE,
        outcome=ReturnedEntry(key="success"),
    )

    assert not condition.is_fully_read()


def test_mapping_without_the_outcome_entry_is_recorded_as_unread():
    """
    A mapping that never names the outcome states nothing the survey can read, so it is
    recorded as unread rather than treated as stating no condition.
    """
    condition = extracted_condition(
        """
        return {"elapsed": self.elapsed_steps}
        """,
        header=EVALUATE,
        outcome=ReturnedEntry(key="success"),
    )

    assert not condition.is_fully_read()


# %% surveying a task suite


def test_survey_reads_every_class_stating_a_success_condition(tmp_path: pathlib.Path):
    """
    The survey covers each task class defining a success condition, and reports one condition
    per class rather than per module.
    """
    module = tmp_path / "tasks.py"
    module.write_text(
        "class PlaceCup:\n"
        "    def _check_success(self):\n"
        '        return OU.check_obj_in_receptacle(self, "cup", "bowl")\n'
        "\n"
        "class MoveAway:\n"
        "    def _check_success(self):\n"
        '        return OU.gripper_obj_far(self, "cup")\n'
        "\n"
        "class NotATask:\n"
        "    def reset(self):\n"
        "        return None\n"
    )

    conditions = RoboCasaConditionSurvey(task_root=tmp_path).conditions()

    assert sorted(conditions) == ["MoveAway", "PlaceCup"]
    assert all(condition.is_fully_read() for condition in conditions.values())
