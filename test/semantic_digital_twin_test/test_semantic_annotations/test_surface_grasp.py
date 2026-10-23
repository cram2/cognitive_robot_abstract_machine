"""
Grasps described on an object's surface, and statements of where an object may be
grasped.
"""

import math

import numpy as np
import pytest
import trimesh

from krrood.parametrization.exceptions import UnboundedParameterError
from krrood.parametrization.model_registries import (
    RelationalCircuitRegistry,
    UniformPriorRegistry,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a, and_, or_
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import SurfaceGraspNotOnSurfaceError
from semantic_digital_twin.orm.ormatic_interface import SurfaceGraspDAO  # noqa: F401
from semantic_digital_twin.grasping.surface_grasp import (
    GraspResult,
    SurfaceGrasp,
    any_surface_grasp,
    draw_surface_grasps,
    lifting_surface_grasp,
)
from semantic_digital_twin.grasping.grasp_strategies import GraspFromAnySide
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from ._statements import allowed_intervals, statement_by
from semantic_digital_twin.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Twist,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures

CARTON_EXTENTS = (0.06, 0.08, 0.2)
"""
Width, depth and height of the carton the grasps are placed on.
"""


@pytest.fixture
def carton() -> Milk:
    """
    A box-shaped carton standing on its base, its middle at the origin.
    """
    body = Body(name=PrefixedName("carton", prefix="surface_grasp"))
    body.collision = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=trimesh.creation.box(extents=CARTON_EXTENTS),
                origin=HomogeneousTransformationMatrix(reference_frame=body),
            )
        ],
        reference_frame=body,
    )
    annotation = Milk(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


def grasp_frame(carton: Milk, surface_grasp: SurfaceGrasp) -> np.ndarray:
    """
    :return: The grasp frame of ``surface_grasp`` in the carton's own frame.
    """
    return surface_grasp.grasp_candidate(carton).grasp_pose.to_np()


# %% grasps on the surface


def test_an_untilted_unrolled_grasp_comes_from_above_and_closes_along_the_line_of_sight(
    carton,
):
    frame = grasp_frame(
        carton, SurfaceGrasp(yaw=0.0, height=0.9, depth=0.0, pitch=0.0, roll=0.0)
    )

    approach, closing = frame[:3, 0], frame[:3, 1]
    assert approach == pytest.approx([0.0, 0.0, -1.0])
    assert closing == pytest.approx([1.0, 0.0, 0.0])


def test_a_grasp_without_depth_takes_the_first_point_of_the_surface(carton):
    frame = grasp_frame(
        carton, SurfaceGrasp(yaw=0.0, height=0.5, depth=0.0, pitch=0.0, roll=0.0)
    )

    assert frame[:3, 3] == pytest.approx([CARTON_EXTENTS[0] / 2, 0.0, 0.0])


def test_the_depth_runs_along_the_line_of_sight(carton):
    """
    The line of sight, not the surface normal, sets where the gripper closes, so a
    handle seen from its end is grasped along its length.
    """
    yaw = 0.3
    depth = 0.01
    surface = grasp_frame(
        carton,
        SurfaceGrasp(yaw=yaw, height=0.5, depth=0.0, pitch=0.0, roll=0.0),
    )
    deeper = grasp_frame(
        carton,
        SurfaceGrasp(yaw=yaw, height=0.5, depth=depth, pitch=0.0, roll=0.0),
    )

    assert deeper[:3, 3] - surface[:3, 3] == pytest.approx(
        -depth * np.array([math.cos(yaw), math.sin(yaw), 0.0])
    )


def test_a_rolled_grasp_closes_along_a_turned_direction(carton):
    unrolled = grasp_frame(
        carton, SurfaceGrasp(yaw=2.0, height=0.8, depth=0.0, pitch=0.3, roll=0.0)
    )
    rolled = grasp_frame(
        carton,
        SurfaceGrasp(yaw=2.0, height=0.8, depth=0.0, pitch=0.3, roll=math.pi / 2),
    )

    assert rolled[:3, 0] == pytest.approx(unrolled[:3, 0])
    assert float(np.dot(rolled[:3, 1], unrolled[:3, 1])) == pytest.approx(0.0, abs=1e-9)


def test_a_point_above_the_object_is_not_on_its_surface(carton):
    above = SurfaceGrasp(yaw=0.0, height=1.5, depth=0.0, pitch=0.0, roll=0.0)

    assert not above.reaches_surface_of(carton)
    with pytest.raises(SurfaceGraspNotOnSurfaceError):
        above.grasp_candidate(carton)


# %% statements of where to grasp


def test_drawn_grasps_lie_where_the_statement_allows():
    sides = (0.0, math.pi)
    tolerance = 0.2
    grasp = any_surface_grasp()
    grasp.where(
        grasp.height >= 0.4,
        grasp.height < 0.6,
        grasp.depth >= 0.0,
        grasp.depth < 0.01,
        or_(
            *(
                and_(grasp.yaw >= side - tolerance, grasp.yaw < side + tolerance)
                for side in sides
            )
        ),
        grasp.pitch >= 0.0,
        grasp.pitch < 1.0,
        grasp.roll >= -0.5,
        grasp.roll < 0.5,
    )
    number_of_grasps = 60

    grasps = draw_surface_grasps(grasp, number_of_grasps)

    assert len(grasps) == number_of_grasps
    nearest_side = [
        min(sides, key=lambda side: abs(grasp.yaw - side)) for grasp in grasps
    ]
    for grasp, side in zip(grasps, nearest_side):
        assert abs(grasp.yaw - side) < tolerance
        assert 0.4 <= grasp.height < 0.6
        assert grasp.result is None
    assert set(nearest_side) == set(sides)


def test_drawing_no_grasps_draws_none(carton):
    assert draw_surface_grasps(carton.surface_grasp_statement(), 0) == []


def test_every_parameter_of_a_surface_grasp_needs_bounds():
    statement = a(SurfaceGrasp)(yaw=..., height=..., depth=..., pitch=..., roll=...)
    statement.where(statement.yaw < 1.0)

    with pytest.raises(UnboundedParameterError):
        list(
            statement.evaluate(
                backend=ProbabilisticBackend(
                    UniformPriorRegistry(), number_of_samples=1
                )
            )
        )


def test_a_carton_is_grasped_up_to_half_its_narrower_side_deep(carton):
    allowed = allowed_intervals(carton.surface_grasp_statement())

    assert allowed["depth"].simple_sets[-1].upper == pytest.approx(
        min(CARTON_EXTENTS[:2]) / 2
    )


def test_a_grasp_from_any_side_lies_within_the_bounds_asked(carton):
    strategy = GraspFromAnySide(
        minimal_height=0.3,
        maximal_height=0.7,
        minimal_depth=0.1,
        maximal_depth=0.4,
        maximal_pitch=0.3,
        maximal_roll=0.2,
    )
    allowed = allowed_intervals(statement_by(strategy, carton))
    narrower_extent = min(CARTON_EXTENTS[:2])

    assert allowed["height"].simple_sets[0].lower == pytest.approx(
        strategy.minimal_height
    )
    assert allowed["height"].simple_sets[-1].upper == pytest.approx(
        strategy.maximal_height
    )
    assert allowed["depth"].simple_sets[0].lower == pytest.approx(
        strategy.minimal_depth * narrower_extent
    )
    assert allowed["depth"].simple_sets[-1].upper == pytest.approx(
        strategy.maximal_depth * narrower_extent
    )
    assert allowed["pitch"].simple_sets[-1].upper == pytest.approx(
        strategy.maximal_pitch
    )
    assert allowed["roll"].simple_sets[0].lower == pytest.approx(-strategy.maximal_roll)
    assert allowed["roll"].simple_sets[-1].upper == pytest.approx(strategy.maximal_roll)


# %% asking a learned model for lifting grasps

LIFTING_HEIGHT = 0.5
"""
In the synthetic trials, a grasp lifts the carton exactly when it lies above this
fraction of the carton's height.
"""


def small_twist(rng: np.random.Generator, rise: float) -> Twist:
    """
    :param rise: How far the twist moves upwards, in meters.
    :return: A twist that moves ``rise`` upwards while turning by a few degrees about a
        random axis.
    """
    rotation_vector = rng.normal(0.0, 0.05, 3)
    return Twist.from_transformation(
        HomogeneousTransformationMatrix.from_xyz_axis_angle(
            z=rise,
            axis=Vector3(*rotation_vector),
            angle=float(np.linalg.norm(rotation_vector)),
        )
    )


@pytest.fixture
def learned_carton_model(carton) -> RelationalCircuitRegistry:
    """
    :return: A model learned from synthetic trials on the carton, in which the grasps
        above :data:`LIFTING_HEIGHT` lift it and the others do not.
    """
    grasps = draw_surface_grasps(carton.surface_grasp_statement(), 300)
    rng = np.random.default_rng(0)
    for grasp in grasps:
        raised = grasp.height > LIFTING_HEIGHT
        grasp.result = GraspResult(
            object_raised=raised,
            motion_completed=True,
            object_displacement=small_twist(rng, 0.2 if raised else 0.0),
            slip=small_twist(rng, 0.0 if raised else -0.2),
        )
    return RelationalCircuitRegistry(
        RelationalProbabilisticCircuit(
            learning_method=JointProbabilityTree(min_samples_per_leaf=20)
        ).fit(grasps, SurfaceGrasp)
    )


def test_a_learned_model_answers_with_lifting_grasps(carton, learned_carton_model):
    grasps = draw_surface_grasps(
        carton.surface_grasp_statement(lifting_surface_grasp), 50, learned_carton_model
    )

    assert len(grasps) == 50
    for grasp in grasps:
        assert grasp.height > LIFTING_HEIGHT
        assert grasp.result.lifted
        assert isinstance(grasp.result.slip.translation, Vector3)
        assert isinstance(grasp.result.slip.rotation, AxisAngle)


def test_a_uniform_prior_cannot_answer_for_lifting_grasps(carton):
    with pytest.raises(UnboundedParameterError):
        draw_surface_grasps(carton.surface_grasp_statement(lifting_surface_grasp), 1)


def test_an_annotation_given_a_model_draws_its_grasps_from_it(
    carton, learned_carton_model
):
    grasps = carton.grasp_candidates(learned_carton_model, lifting_surface_grasp)
    lowest, highest = carton.grasp_surface().bounds
    lifting_height = lowest[2] + LIFTING_HEIGHT * (highest[2] - lowest[2])

    assert grasps
    for grasp in grasps:
        assert grasp.grasp_pose.to_np()[2, 3] > lifting_height


def test_an_annotation_without_a_model_keeps_its_default_grasps(carton):
    for grasp in carton.grasp_candidates():
        assert grasp.grasp_pose.to_np()[:3, 3] == pytest.approx([0.0, 0.0, 0.0])
