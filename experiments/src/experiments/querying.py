"""
Queries about a robot's past execution behaviour, expressed in the KRROOD Entity Query
Language.

The plan mirrors the structure of the bullet-world demo
(coraplex/demos/coraplex_bullet_world_demo/demo.py): a PR2 parks its arms, raises its
torso, then transports three objects (milk, bowl, spoon) to a table.

Each :class:`BehaviourQuery` pairs a natural-language question with the EQL object that
answers it. :func:`run_experiment` evaluates all queries against a completed plan execution
and returns an :class:`~experiments.experiment_definitions.ExperimentsTable` suitable for
scientific reporting.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.querying

.. note::
    This module deliberately does not use ``from __future__ import annotations``: the
    :class:`~experiments.experiment_definitions.ExperimentResult` introspector requires
    dataclass field annotations to be real classes rather than strings.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import coraplex as _coraplex_pkg
import coraplex.orm.ormatic_interface  # type: ignore  # noqa: F401
import krrood.entity_query_language.factories as eql
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    TaskStatus,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.orm.ormatic_interface import Base, PlanMappingDAO  # type: ignore
from coraplex.plans.factories import sequential, try_in_order, code
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, PlanNode
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.testing import setup_world
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.ormatic.utils import create_engine, drop_database
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Drawer,
    Handle,
    Spoon,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)

_CORAPLEX_RESOURCES = Path(_coraplex_pkg.__file__).parent.parent.parent / "resources"
_DATABASE_PATH = Path(__file__).parent / "querying.db"


@dataclass
class BehaviourQuery:
    """
    A natural-language question paired with the EQL query that answers it.
    """

    question: str
    """
    The natural-language question posed to the robot.
    """

    query: Any
    """
    The EQL expression whose evaluation answers the question.
    """

    def evaluate(self) -> Any:
        """
        Evaluate the EQL query and return its raw result.

        :return: Whatever the EQL engine yields for this query.
        """
        return self.query.evaluate()

    def __repr__(self) -> str:
        return f"BehaviourQuery({self.question!r})"


@dataclass
class BehaviourQueryResult(ExperimentResult):
    """
    One row of the behaviour-query experiment table.

    Each row evaluates one query both via in-memory EQL and via SQL so the two
    approaches can be compared side-by-side.  Untranslatable SQL queries report ``-1`` /
    ``-1.0`` sentinel values.
    """

    question: str
    """
    The natural-language question posed to the robot.
    """

    eql_number_of_results: int
    """
    Number of results from in-memory EQL evaluation, or -1 on error.
    """

    eql_duration_ms: float
    """
    Wall-clock time in milliseconds for in-memory EQL evaluation.
    """

    sql_translation_duration_ms: float
    """
    Wall-clock time in milliseconds for EQL-to-SQL translation, or -1.0 on failure.
    """

    sql_number_of_results: int
    """
    Number of results returned by the SQL query, or -1 on error.
    """

    sql_execution_duration_ms: float
    """
    Wall-clock time in milliseconds for SQL execution against the database, or
    -1.0 on failure.
    """


def build_plan() -> Plan:
    """
    Set up the bullet-world scene, execute the plan in simulation, and return the
    completed :class:`~coraplex.plans.plan.Plan`.

    The scene and action sequence mirror
    ``coraplex/demos/coraplex_bullet_world_demo/demo.py`` exactly: the PR2 parks
    its arms, raises its torso, then transports milk, bowl, and spoon to the
    dining table.

    :return: The fully executed plan, ready for EQL queries.
    """
    world = setup_world()

    spoon = STLParser(str(_CORAPLEX_RESOURCES / "objects" / "spoon.stl")).parse()
    bowl = STLParser(str(_CORAPLEX_RESOURCES / "objects" / "bowl.stl")).parse()

    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1, reference_frame=world.root
            ),
        )
        connection = FixedConnection(
            parent=world.get_body_by_name("cabinet10_drawer_top"),
            child=spoon.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.05, -0.05, 0
            ),
        )
        world.merge_world(spoon, connection)

    try:
        import rclpy

        rclpy.init()
        from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
            VizMarkerPublisher,
        )

        ros_node = rclpy.create_node("viz_marker")
        VizMarkerPublisher(_world=world, node=ros_node).with_tf_publisher()
    except ImportError:
        ros_node = None

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=ros_node)

    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()
        world.add_semantic_annotations(
            [
                Bowl(root=world.get_body_by_name("bowl.stl")),
                Spoon(root=world.get_body_by_name("spoon.stl")),
            ]
        )
        world.add_semantic_annotation_recursively(
            Drawer(
                root=world.get_body_by_name("cabinet10_drawer_top"),
                handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
            )
        )

    context.evaluate_conditions = False

    def _failing_step():
        raise PlanFailure()

    root = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            try_in_order(
                [
                    code(_failing_step),
                    TransportAction(
                        world.get_body_by_name("milk.stl"),
                        Pose.from_xyz_rpy(
                            4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root
                        ),
                        Arms.LEFT,
                    ),
                ],
                context=context,
            ),
            TransportAction(
                world.get_body_by_name("bowl.stl"),
                Pose.from_xyz_rpy(5.0, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
                Arms.LEFT,
            ),
            TransportAction(
                world.get_body_by_name("spoon.stl"),
                Pose.from_xyz_rpy(5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
                Arms.LEFT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.TOP,
                    pr2.left_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan = root.plan
    with simulated_robot:
        plan.perform()

    return plan


def _q_what_did_you_do(plan: Plan) -> BehaviourQuery:
    n = eql.variable(ActionNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="What did you just do?",
        query=eql.an(eql.entity(n).where(n.status == TaskStatus.SUCCEEDED)).ordered_by(
            n.start_time,
            descending=False,
        ),
    )


def _q_walk_through_in_order(plan: Plan) -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Walk me through what you did in order.",
        query=eql.an(eql.entity(n).where(n.status == TaskStatus.SUCCEEDED)).ordered_by(
            n.start_time
        ),
    )


def _q_total_duration(plan: Plan) -> BehaviourQuery:
    n = eql.variable(ActionNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="How long did the whole task take?",
        query=eql.set_of(
            min_start := eql.min(n.start_time),
            max_end := eql.max(n.end_time),
        ).where(
            n.start_time != None, n.end_time != None
        ),  # noqa: E711
    )


def _q_duration_per_step(plan: Plan) -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="How long did each step take?",
        query=eql.an(eql.entity(n).where(n.end_time != None)).ordered_by(  # noqa: E711
            n.start_time
        ),
    )


def _q_did_anything_go_wrong(plan: Plan) -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Did anything go wrong?",
        query=eql.an(eql.entity(n).where(n.status == TaskStatus.FAILED)),
    )


def _q_why_did_you_fail(plan: Plan) -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Why did you fail at that step?",
        query=eql.an(eql.entity(n.reason).where(n.status == TaskStatus.FAILED)),
    )


def _q_how_many_retries(plan: Plan) -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="How many times did you retry before giving up?",
        query=(eql.set_of(c := eql.count_all()).where(n.status == TaskStatus.FAILED)),
    )


def _q_which_fallback(plan: Plan) -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    s = eql.variable(PlanNode, domain=n.left_siblings)
    return BehaviourQuery(
        question="Which fallback did you end up using?",
        query=eql.an(
            eql.entity(n).where(
                n.status == TaskStatus.SUCCEEDED,
                eql.exists(s, s.status == TaskStatus.FAILED),
            )
        ),
    )


def _q_longest_step(plan: Plan) -> BehaviourQuery:
    n = eql.variable(ActionNode, domain=plan.plan_graph.nodes())

    return BehaviourQuery(
        question="Which step took the longest?",
        query=eql.an(eql.entity(n).where(n.end_time != None))  # noqa: E711
        .ordered_by(n.end_time, descending=True)
        .limit(1),
    )


def _q_status_breakdown(plan: Plan) -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Were all subtasks successful, or did some fail?",
        query=(
            eql.set_of(status := n.status, c := eql.count(n))
            .grouped_by(status)
            .ordered_by(c, descending=True)
        ),
    )


def _q_world_state_at_start(plan: Plan) -> BehaviourQuery:
    n = eql.variable(Plan, domain=[plan])
    return BehaviourQuery(
        question="What was the state of the world when you started the task?",
        query=eql.an(eql.entity(n.initial_world.state)),
    )


def _q_world_state_at_end(plan: Plan) -> BehaviourQuery:
    n = eql.variable(Plan, domain=[plan])
    return BehaviourQuery(
        question="What was the state of the world when you finished?",
        query=eql.an(eql.entity(n.context.world.state)),
    )


def build_queries(plan: Plan) -> List[BehaviourQuery]:
    """
    Construct all behaviour queries for a completed plan execution.

    :param plan: The plan whose execution history the queries will inspect.
    :return: All behaviour queries, in presentation order.
    """
    return [
        _q_what_did_you_do(plan),
        _q_walk_through_in_order(plan),
        _q_total_duration(plan),
        _q_duration_per_step(plan),
        _q_did_anything_go_wrong(plan),
        _q_why_did_you_fail(plan),
        _q_how_many_retries(plan),
        _q_which_fallback(plan),
        _q_longest_step(plan),
        _q_status_breakdown(plan),
        _q_world_modifications(plan),
        _q_world_state_at_start(plan),
        _q_world_state_at_end(plan),
    ]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _count_results(raw: Any) -> int:
    """
    Count the number of results returned by an EQL evaluation.

    :param raw: The raw value returned by ``BehaviourQuery.evaluate()``.
    :return: Number of items for iterable results, 1 for a single value, 0 for ``None``.
    """
    if raw is None:
        return 0
    if hasattr(raw, "__iter__"):
        return len(list(raw))
    return 1


def run_experiment(plan: Plan, session: Session) -> ExperimentsTable:
    """
    Evaluate all behaviour queries both via in-memory EQL and via SQL, collecting
    timings and result counts for each approach in a single row per query.

    EQL or SQL failures are recorded as ``-1`` / ``-1.0`` sentinels so a single failing
    query does not abort the experiment.

    :param plan: The fully executed plan to query.
    :param session: An open SQLAlchemy session connected to the persisted plan database.
    :return: A table with one :class:`BehaviourQueryResult` row per query.
    """
    rows: List[BehaviourQueryResult] = []
    for query in build_queries(plan):
        # EQL evaluation
        t0 = time.perf_counter()
        try:
            eql_count = _count_results(query.evaluate())
        except Exception:
            eql_count = -1
        eql_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # SQL translation
        t0 = time.perf_counter()
        try:
            translator = eql_to_sql(query.query, session)
            sql_translation_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        except Exception:
            rows.append(
                BehaviourQueryResult(
                    question=query.question,
                    eql_number_of_results=eql_count,
                    eql_duration_ms=eql_ms,
                    sql_translation_duration_ms=-1.0,
                    sql_execution_duration_ms=-1.0,
                    sql_number_of_results=-1,
                )
            )
            continue

        # SQL execution
        t0 = time.perf_counter()
        try:
            sql_count = _count_results(translator.evaluate())
        except Exception:
            sql_count = -1
        sql_execution_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        rows.append(
            BehaviourQueryResult(
                question=query.question,
                eql_number_of_results=eql_count,
                eql_duration_ms=eql_ms,
                sql_translation_duration_ms=sql_translation_ms,
                sql_execution_duration_ms=sql_execution_ms,
                sql_number_of_results=sql_count,
            )
        )
    return ExperimentsTable(rows)


# ---------------------------------------------------------------------------
# Database serialization
# ---------------------------------------------------------------------------


def persist_plan(plan: Plan) -> tuple[Session, Engine]:
    """
    Serialise *plan* to a SQLite database at :data:`_DATABASE_PATH` via ORMatic.

    Any pre-existing database is dropped first so each run starts from a clean slate.
    Returns the open session and engine so the caller can run SQL queries against the
    same database and close them when done.

    :param plan: The fully executed plan to persist.
    :return: Tuple of ``(session, engine)`` pointing at the populated database.
    """
    engine = create_engine(f"sqlite:///{_DATABASE_PATH}")
    drop_database(engine)
    Base.metadata.create_all(bind=engine)

    dao = to_dao(plan)

    session = sessionmaker(engine)()
    session.add(dao)
    session.commit()
    print(f"Plan persisted to {_DATABASE_PATH} with database_id={dao.database_id}")
    return session, engine


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Run the bullet-world plan, persist it to a database, evaluate all behaviour queries
    both via EQL and via SQL, and print the combined result table.
    """
    plan = build_plan()
    session, engine = persist_plan(plan)
    try:
        table = run_experiment(plan, session)
    finally:
        session.close()
        engine.dispose()

    print(TypstRenderer(table).render_table())


if __name__ == "__main__":
    main()
