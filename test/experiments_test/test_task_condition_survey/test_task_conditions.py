"""
Stating a task's success condition in EQL, and reading back what it states.

The conditions here are built directly rather than parsed, so each test pins one
property of the vocabulary -- what a relation constrains, what a comparison constrains,
when a condition is first order -- independently of how any task suite happens to phrase
it.
"""

from krrood.entity_query_language.factories import and_, exists, for_all, not_, or_

from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConditionKind,
    ConstantCondition,
    ContinuousRelation,
    ContinuousValue,
    DiscreteRelation,
    PredicateKind,
    PredicateNameRule,
    PredicateVocabulary,
    SceneObject,
    StatedTaskCondition,
    UnreadCondition,
)


def relation(
    name: str,
    kind: PredicateKind = PredicateKind.DISCRETE,
    stated_of: str = "",
):
    """
    :param name: Name the relation is stated under.
    :param kind: What kind of set it describes.
    :param stated_of: The objects it is stated of, as a task would write them.
    :return: The EQL condition stating that relation.
    """
    return kind.relation_stated_of(name, SceneObject.written_as(stated_of))


def measured_value(
    name: str,
    kind: PredicateKind = PredicateKind.CONTINUOUS,
    stated_of: str = "",
):
    """
    :param name: Name the measure is taken under.
    :param kind: What kind of set a comparison of it describes.
    :param stated_of: The objects it is measured over, as a task would write them.
    :return: The EQL value a task compares.
    """
    return kind.value_measured_over(name, SceneObject.written_as(stated_of))


def condition(expression) -> StatedTaskCondition:
    """
    :param expression: An EQL condition.
    :return: That condition as one task's stated condition.
    """
    return StatedTaskCondition(expression=expression)


# %% what a vocabulary builds


def test_kind_decides_which_relation_a_name_is_stated_under():
    """
    The kind of set a name describes is what the condition is built from, so a discrete
    and a continuous requirement are different EQL conditions rather than one carrying a
    label.
    """
    discrete = relation("OU.door_is_open")
    continuous = relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS)

    assert discrete._type_ is DiscreteRelation
    assert continuous._type_ is ContinuousRelation


def test_relation_records_the_objects_it_is_stated_of():
    """
    One relation stated of two objects states two conditions, so the objects belong to
    what a relation constrains and not only to how it was written.
    """
    [first] = condition(
        relation("OU.obj_inside_of", stated_of="('food', sink)")
    ).propositions()
    [second] = condition(
        relation("OU.obj_inside_of", stated_of="('dish', sink)")
    ).propositions()

    assert first.stated_of == "('food', sink)"
    assert first.variable_name != second.variable_name


def test_comparison_constrains_the_bound_as_well_as_the_value():
    """
    Bounding one measured value two ways describes two different sets, so the bound
    belongs to what a comparison constrains.
    """
    value = measured_value("np.abs", stated_of="(knob_value)")
    both_bounds = condition(and_(value >= 0.35, value <= 0.85))

    lower, upper = both_bounds.propositions()
    assert lower.stated_under == upper.stated_under == "np.abs"
    assert lower.variable_name == "np.abs(knob_value) >= 0.35"
    assert upper.variable_name == "np.abs(knob_value) <= 0.85"


def test_comparison_takes_its_kind_from_the_value_it_compares():
    """
    What a comparison describes follows from the value being compared, so the kind is
    read off the measure rather than off the comparison EQL happens to build.
    """
    [proposition] = condition(measured_value("self.tray_offset") < 0.15).propositions()

    assert proposition.kind is PredicateKind.CONTINUOUS
    assert proposition.expression.left._type_ is ContinuousValue


# %% what a whole condition combines


def test_condition_mixing_both_kinds_of_proposition_is_hybrid():
    """
    A condition requiring a state and a distance at once needs variables of both kinds,
    which is the kind of set a survey counts.
    """
    mixed = condition(
        and_(
            relation("OU.door_is_open"),
            relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS),
        )
    )

    assert mixed.kind is ConditionKind.HYBRID


def test_condition_stating_one_kind_of_proposition_takes_that_kind():
    """
    A condition whose propositions all describe the same kind of set describes a set of
    that kind, however deeply the condition is nested.
    """
    discrete = condition(
        or_(relation("OU.door_is_open"), relation("OU.drawer_is_open"))
    )
    continuous = condition(
        for_all(
            SceneObject.provided_by_the_scene(),
            relation("OU.gripper_obj_far", kind=PredicateKind.CONTINUOUS),
        )
    )

    assert discrete.kind is ConditionKind.DISCRETE
    assert continuous.kind is ConditionKind.CONTINUOUS


def test_condition_of_unrecognised_names_has_no_established_kind():
    """
    A condition stating only names the survey does not recognise states nothing it can
    classify, so its kind is left undetermined rather than guessed.
    """
    unknown = condition(relation("some.unknown.check", kind=PredicateKind.UNCLASSIFIED))

    assert unknown.kind is ConditionKind.UNDETERMINED


def test_negated_and_nested_propositions_are_all_read():
    """
    Every proposition counts however it is composed, since a survey reports what a whole
    condition states and not only what its outermost operator does.
    """
    nested = condition(
        or_(
            and_(relation("first"), not_(relation("second"))),
            exists(SceneObject.provided_by_the_scene(), relation("third")),
        )
    )

    assert {proposition.stated_under for proposition in nested.propositions()} == {
        "first",
        "second",
        "third",
    }


# %% conditions that quantify over the objects a scene provides


def test_quantifying_over_a_collection_makes_a_condition_first_order():
    """
    A quantified condition states something about the objects a scene provides, which is
    what a survey counts separately from the conditions a fixed set of variables covers.
    """
    universal = condition(
        for_all(SceneObject.provided_by_the_scene(), relation("OU.obj_inside_of"))
    )
    existential = condition(
        exists(SceneObject.provided_by_the_scene(), relation("self.check_contact"))
    )

    assert len(universal.quantifications()) == 1
    assert len(existential.quantifications()) == 1
    assert condition(relation("OU.door_is_open")).quantifications() == []


def test_quantification_buried_under_other_operators_still_counts():
    """
    Quantifying a part of a condition makes the whole condition first order, so a
    quantification under a negation counts as one just as a bare one does.
    """
    buried = condition(
        and_(
            relation("first"),
            not_(exists(SceneObject.provided_by_the_scene(), relation("second"))),
        )
    )

    assert len(buried.quantifications()) == 1


# %% conditions the survey did not read


def test_unread_part_is_reported_rather_than_dropped():
    """
    A condition resting on something the survey never read is marked as unread, so a
    partial reading is never reported as a complete one.
    """
    partial = condition(
        and_(relation("first"), UnreadCondition.found("unbound name 'decided'"))
    )

    assert not partial.is_fully_read()
    assert partial.unread_parts() == ["unbound name 'decided'"]


def test_condition_without_an_unread_part_is_fully_read():
    """
    A condition built only from recognised propositions leaves nothing unread, so the
    survey reports it as covered.
    """
    assert condition(relation("OU.door_is_open")).is_fully_read()


# %% conditions that hold always or never


def test_constant_condition_is_recovered_from_the_condition_it_states():
    """
    A bare outcome still has to compose with what surrounds it, so it is stated as an
    EQL condition and read back as the outcome it states.
    """
    assert (
        ConstantCondition.stated_by(ConstantCondition.ALWAYS.expression)
        is ConstantCondition.ALWAYS
    )
    assert (
        ConstantCondition.stated_by(ConstantCondition.NEVER.expression)
        is ConstantCondition.NEVER
    )


def test_a_relation_states_no_constant_outcome():
    """
    A condition about a scene is not an outcome of its own, so reading one as a constant
    would report a requirement as though it were already settled.
    """
    assert ConstantCondition.stated_by(relation("OU.door_is_open")) is None


# %% classifying the names a suite states its conditions under


def test_comparison_kind_follows_the_value_compared_against():
    """
    A quantity compared against a number bounds a continuous variable, while one
    compared against a string or a truth value selects among finitely many states.
    """
    vocabulary = PredicateVocabulary()

    assert (
        vocabulary.classify_comparison("distance", (0.15,)) is PredicateKind.CONTINUOUS
    )
    assert vocabulary.classify_comparison("mode", ("open",)) is PredicateKind.DISCRETE
    assert vocabulary.classify_comparison("latched", (True,)) is PredicateKind.DISCRETE


def test_naming_convention_recognises_the_names_stated_outright():
    """
    A suite's long tail of per-fixture checks is recognised by the marker a name
    carries, so a name the survey has never seen is still classified rather than
    dropped.
    """
    vocabulary = PredicateVocabulary(
        exact_kinds={"OU.gripper_obj_far": PredicateKind.CONTINUOUS},
        name_rules=(PredicateNameRule("check_", PredicateKind.DISCRETE),),
    )

    assert vocabulary.classify("OU.gripper_obj_far") is PredicateKind.CONTINUOUS
    assert vocabulary.classify("OU.check_anything") is PredicateKind.DISCRETE


def test_unrecognised_name_is_reported_rather_than_assumed():
    """
    A name matching neither a known name nor a naming convention is left unclassified,
    so the survey never claims to represent what it does not recognise.
    """
    assert (
        PredicateVocabulary().classify("self.some_new_helper")
        is PredicateKind.UNCLASSIFIED
    )
