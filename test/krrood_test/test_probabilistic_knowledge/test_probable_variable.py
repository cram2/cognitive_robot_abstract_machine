import numpy as np

from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.query.match import (
    ProbableVariable,
    AttributeMatch,
    AbstractMatchExpression,
    construct_graph_and_get_root,
)
from krrood.probabilistic_knowledge.parameterizer import (
    DataAccessObjectParameterizer,
    copy_partial_object,
)
from krrood.rustworkx_utils import RWXNode
from random_events.interval import singleton, open, closed, closed_open
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous

from krrood.entity_query_language.factories import (
    variable,
    entity,
    and_,
    or_,
    match_variable,
    match,
    Entity,
    probable_variable,
    probable,
)
from krrood.probabilistic_knowledge.probable_variable import (
    QueryToRandomEventTranslator,
    is_disjunctive_normal_form,
    MatchToInstanceTranslator,
)

from ..dataset.example_classes import Pose, Position, Orientation
from ..dataset.ormatic_interface import *  # type: ignore


def test_parameterizer_with_where():
    pose_variable = variable(Pose, None)

    q = entity(pose_variable).where(
        pose_variable.position.y > 0.0,
        pose_variable.position.x == 0.0,
        pose_variable.position.y < 10.0,
        pose_variable.position.z >= -1.0,
        pose_variable.position.z <= 1.0,
        pose_variable.orientation.x != 1.0,
    )
    t = QueryToRandomEventTranslator(q)
    r = t.translate()

    result_by_hand = SimpleEvent(
        {
            Continuous("Pose.orientation.x"): ~singleton(1.0),
            Continuous("Pose.position.y"): open(0.0, 10),
            Continuous("Pose.position.z"): closed(-1.0, 1.0),
            Continuous("Pose.position.x"): singleton(0.0),
        }
    )

    assert result_by_hand.as_composite_set() == r


def test_dnf_checking():
    pose_variable = variable(Pose, None)

    q1 = entity(pose_variable).where(
        and_(
            or_(
                pose_variable.position.y > 0,
                pose_variable.position.x == 0,
            ),
            or_(
                pose_variable.position.z >= -1,
                pose_variable.position.x == 0,
            ),
        )
    )

    assert not is_disjunctive_normal_form(q1)

    q2 = entity(pose_variable).where(
        or_(
            pose_variable.position.x == 0,
            and_(
                pose_variable.position.z >= -1,
                pose_variable.position.z <= 1,
                pose_variable.position.y < 10,
            ),
            and_(pose_variable.orientation.z > 0),
        )
    )
    assert is_disjunctive_normal_form(q2)

    t = QueryToRandomEventTranslator(q2)
    translated = t.translate()

    variables = [
        Continuous("Pose.position.x"),
        Continuous("Pose.position.y"),
        Continuous("Pose.position.z"),
        Continuous("Pose.orientation.z"),
    ]
    [p_x, p_y, p_z, o_z] = variables

    e1 = SimpleEvent(
        {
            p_x: singleton(0.0),
        }
    )
    e1.fill_missing_variables(variables)
    e2 = SimpleEvent(
        {
            p_z: closed(-1.0, 1.0),
            p_y: closed_open(-np.inf, 10.0),
        }
    )
    e2.fill_missing_variables(variables)
    e3 = SimpleEvent({o_z: open(0.0, np.inf)})
    e3.fill_missing_variables(variables)

    result_by_hand = Event(e1, e2, e3)

    assert (result_by_hand - translated).is_empty()
    assert (translated - result_by_hand).is_empty()


def test_query_writing_with_match_and_copy():
    var: ProbableVariable = probable_variable(Pose)(
        position=probable(Position)(x=0.1, y=..., z=...), orientation=None
    )

    translator = MatchToInstanceTranslator(var)
    obj = translator.translate()
    assert obj.position.x == 0.1
    assert obj.position.y == ...
    assert obj.position.z == ...
    assert obj.orientation is None

    copied = copy_partial_object(obj)
    assert obj is not copied
    assert obj == copied
