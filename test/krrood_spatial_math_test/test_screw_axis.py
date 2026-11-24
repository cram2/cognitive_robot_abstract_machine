"""
Tests of :class:`krrood_spatial_math.spatial_types.ScrewAxis`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.linalg import expm

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from krrood.symbolic_math.exceptions import HasFreeVariablesError
from krrood_spatial_math.cartesian_frame_3d import ValueInCartesianFrame3D
from krrood_spatial_math.spatial_types import ScrewAxis, Twist, Vector3

from .dataset.frames import KnownFrames
from .reference_implementations import rigid_motion_rate_matrix

# %% the parts of a screw axis


def test_a_screw_axis_has_six_entries():
    """
    Three entries for the rotation and three for the translation per unit of position.
    """
    screw_axis = ScrewAxis()

    assert screw_axis.shape == (6, 1)


def test_a_screw_axis_without_parts_neither_turns_nor_moves():
    """
    Left out, both parts default to zero.
    """
    screw_axis = ScrewAxis()

    np.testing.assert_allclose(
        screw_axis.rotation_per_unit_position.to_np()[:3], [0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        screw_axis.translation_per_unit_position.to_np()[:3], [0.0, 0.0, 0.0]
    )


def test_a_screw_axis_returns_the_parts_it_was_given():
    """
    The rotation and translation per unit of position read back as they were given.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, 1.0),
        translation_per_unit_position=Vector3(0.0, -0.5, 0.0),
    )

    np.testing.assert_allclose(
        screw_axis.rotation_per_unit_position.to_np()[:3], [0.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(
        screw_axis.translation_per_unit_position.to_np()[:3], [0.0, -0.5, 0.0]
    )


# %% the frame a screw axis is expressed in


def test_a_screw_axis_is_a_value_given_in_a_cartesian_frame():
    """
    A screw axis is expressed in a reference frame.
    """
    assert issubclass(ScrewAxis, ValueInCartesianFrame3D)


def test_the_parts_of_a_screw_axis_are_expressed_in_its_frame(table_frame):
    """
    Both parts and the direction carry the screw axis's reference frame.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, 1.0),
        translation_per_unit_position=Vector3(0.1, 0.0, 0.0),
        reference_frame=table_frame,
    )

    assert screw_axis.rotation_per_unit_position.reference_frame is table_frame
    assert screw_axis.translation_per_unit_position.reference_frame is table_frame
    assert screw_axis.unit_direction.reference_frame is table_frame


def test_a_deserialized_screw_axis_refers_to_the_readers_frame(table_frame):
    """
    A JSON round trip keeps the values and resolves the frame on the reading side.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(1.0, 0.0, 0.0),
        translation_per_unit_position=Vector3(0.1, 0.2, 0.3),
        reference_frame=table_frame,
    )
    known_frames = KnownFrames()
    known_frames.add(key=table_frame.id, tracked_object=table_frame)

    screw_axis_copy = from_json(
        json.loads(json.dumps(to_json(screw_axis))), **known_frames.create_kwargs()
    )

    assert type(screw_axis_copy) is ScrewAxis
    assert screw_axis_copy.reference_frame is table_frame
    np.testing.assert_allclose(screw_axis_copy.to_np(), screw_axis.to_np())


# %% the direction of the axis


def test_the_direction_of_a_turning_screw_axis_is_that_of_its_rotation():
    """
    A screw axis that turns the body points along its rotation, whatever the
    translation.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, 2.0),
        translation_per_unit_position=Vector3(0.3, 0.0, 0.0),
    )

    np.testing.assert_allclose(screw_axis.unit_direction.to_np()[:3], [0.0, 0.0, 1.0])


def test_the_direction_of_a_sliding_screw_axis_is_that_of_its_translation():
    """
    A screw axis that does not turn the body points along its translation.
    """
    screw_axis = ScrewAxis(translation_per_unit_position=Vector3(0.0, 3.0, 4.0))

    np.testing.assert_allclose(screw_axis.unit_direction.to_np()[:3], [0.0, 0.6, 0.8])


def test_the_direction_of_a_zero_screw_axis_is_the_zero_vector():
    """
    A screw axis that neither turns nor moves the body has no direction.
    """
    np.testing.assert_allclose(ScrewAxis().unit_direction.to_np()[:3], [0.0, 0.0, 0.0])


# %% normalization


@pytest.mark.parametrize(
    "rotation, translation",
    [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.5)),
        ((0.0, 0.0, 1.0), (0.0, -0.4, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.6, 0.8)),
    ],
    ids=["revolute", "helical", "off-origin revolute", "prismatic"],
)
def test_a_screw_axis_with_a_unit_rotation_or_only_a_unit_translation_is_normalized(
    rotation, translation
):
    """
    Whatever the translation, a unit rotation normalizes a screw axis; one that does not
    turn is normalized by a unit translation.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(*rotation),
        translation_per_unit_position=Vector3(*translation),
    )

    assert screw_axis.is_normalized


@pytest.mark.parametrize(
    "rotation, translation",
    [
        ((0.0, 0.0, 2.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.5), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ],
    ids=["fast rotation", "slow rotation with unit translation", "long slide", "zero"],
)
def test_a_screw_axis_without_a_unit_rotation_or_only_a_unit_translation_is_not_normalized(
    rotation, translation
):
    """
    A unit translation does not normalize a screw axis that turns, and a screw axis that
    neither turns nor moves is never normalized.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(*rotation),
        translation_per_unit_position=Vector3(*translation),
    )

    assert not screw_axis.is_normalized


@pytest.mark.parametrize(
    "rotation, translation, passes_through_origin",
    [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.5), True),
        ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), True),
        ((0.0, 0.0, 1.0), (0.0, -0.4, 0.0), False),
    ],
    ids=["helical through the origin", "prismatic", "revolute off the origin"],
)
def test_a_screw_axis_passes_through_the_origin_if_its_translation_is_parallel_to_its_rotation(
    rotation, translation, passes_through_origin
):
    """
    A translation across the rotation is that of the origin circling an axis elsewhere;
    a screw axis that does not turn has no line to miss the origin.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(*rotation),
        translation_per_unit_position=Vector3(*translation),
    )

    assert screw_axis.passes_through_origin is passes_through_origin


def test_checking_whether_a_symbolic_screw_axis_passes_through_the_origin_raises():
    """
    Whether a screw axis with free variables passes through the origin depends on their
    values, so the check refuses to answer.
    """
    angle_per_unit_position = sm.FloatVariable(name="angle_per_unit_position")
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, angle_per_unit_position)
    )

    with pytest.raises(HasFreeVariablesError):
        screw_axis.passes_through_origin


def test_checking_whether_a_symbolic_screw_axis_is_normalized_raises():
    """
    Whether a screw axis with free variables is normalized depends on their values, so
    the check refuses to answer.
    """
    angle_per_unit_position = sm.FloatVariable(name="angle_per_unit_position")
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, angle_per_unit_position)
    )

    with pytest.raises(HasFreeVariablesError):
        screw_axis.is_normalized


# %% whether the body turns


@pytest.mark.parametrize(
    "rotation, tolerance, turns_body",
    [
        ((0.0, 0.0, 1.0), 0.0, True),
        ((0.0, 0.0, 0.0), 0.0, False),
        ((0.0, 0.0, 1e-12), 0.0, True),
        ((0.0, 0.0, 1e-12), 1e-9, False),
        ((0.0, 0.0, 1.0), 1e-9, True),
    ],
    ids=[
        "unit rotation",
        "no rotation",
        "rounding error, exactly",
        "rounding error, within the tolerance",
        "unit rotation, beyond the tolerance",
    ],
)
def test_a_screw_axis_and_its_twist_turn_the_body_if_the_rotation_exceeds_the_tolerance(
    rotation, tolerance, turns_body
):
    """
    A screw axis and a twist turn the body when their rotational part is longer than the
    tolerance, which absorbs rounding errors.
    """
    screw_axis = ScrewAxis(rotation_per_unit_position=Vector3(*rotation))
    twist = screw_axis.to_twist(position_velocity=2)

    for rate in (screw_axis, twist):
        assert rate.turns_body(tolerance=tolerance) is turns_body


def test_checking_whether_a_symbolic_screw_axis_turns_the_body_raises():
    """
    Whether a screw axis with free variables turns the body depends on their values, so
    the check refuses to answer.
    """
    angle_per_unit_position = sm.FloatVariable(name="angle_per_unit_position")
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, angle_per_unit_position)
    )

    with pytest.raises(HasFreeVariablesError):
        screw_axis.turns_body()


# %% the pitch and the line of the axis


@pytest.mark.parametrize(
    "rotation, translation, translation_per_radian",
    [
        ((0.0, 0.0, 2.0), (0.4, 0.0, 0.5), 0.25),
        ((0.0, 0.0, 1.0), (0.0, -0.4, 0.0), 0.0),
        ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), np.inf),
    ],
    ids=["helical off the origin", "revolute off the origin", "prismatic"],
)
def test_the_translation_per_radian_is_the_part_of_the_translation_along_the_rotation(
    rotation, translation, translation_per_radian
):
    """
    Only the translation along the axis advances the body; the translation across it is
    the origin circling the axis.

    A screw axis that does not turn advances without end.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(*rotation),
        translation_per_unit_position=Vector3(*translation),
    )

    assert float(screw_axis.translation_per_radian) == pytest.approx(
        translation_per_radian
    )


def test_the_axis_point_closest_to_the_origin_is_where_the_hinge_sits(table_frame):
    """
    A body turning about a hinge at (0.4, 0, 0) moves its point at the origin by
    -(rotation × hinge point), and the axis point recovers the hinge, in the screw
    axis's frame.
    """
    rotation = np.array([0.0, 0.0, 2.0])
    hinge_point = np.array([0.4, 0.0, 0.0])
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(*rotation),
        translation_per_unit_position=Vector3(
            *(-np.cross(rotation, hinge_point) + 0.3 * rotation)
        ),
        reference_frame=table_frame,
    )

    axis_point = screw_axis.axis_point_closest_to_origin

    np.testing.assert_allclose(axis_point[:3], hinge_point, atol=1e-12)
    assert axis_point.reference_frame == table_frame


def test_the_axis_point_of_a_screw_axis_that_does_not_turn_is_the_origin():
    """
    A screw axis that only slides has no line, and takes the origin as its point.
    """
    screw_axis = ScrewAxis(translation_per_unit_position=Vector3(0.3, 0.0, 0.0))

    np.testing.assert_allclose(screw_axis.axis_point_closest_to_origin[:3], 0)


# %% the twist at a given speed of the position


def test_the_twist_of_a_screw_axis_scales_both_parts_by_the_position_velocity(
    table_frame,
):
    """
    Moving the position at a given speed gives the twist the screw axis scaled by that
    speed, in the same frame.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, 1.0),
        translation_per_unit_position=Vector3(0.0, -0.5, 0.1),
        reference_frame=table_frame,
    )

    twist = screw_axis.to_twist(position_velocity=2.0)

    assert type(twist) is Twist
    assert twist.reference_frame is table_frame
    np.testing.assert_allclose(twist.to_np(), 2.0 * screw_axis.to_np())


# %% the displacement after moving along the axis


def test_a_screw_axis_through_the_origin_turns_the_body_in_place():
    """
    A revolute axis through the frame's origin rotates the body about its origin and
    does not move it.
    """
    screw_axis = ScrewAxis(rotation_per_unit_position=Vector3(0.0, 0.0, 1.0))

    displacement = screw_axis.to_displacement(position=0.8)

    np.testing.assert_allclose(displacement.translation.to_np()[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        displacement.rotation_vector.to_np()[:3], [0.0, 0.0, 0.8]
    )


def test_a_screw_axis_that_does_not_turn_slides_the_body():
    """
    Without rotation, the body moves along the translation scaled by the position.
    """
    screw_axis = ScrewAxis(translation_per_unit_position=Vector3(0.0, 0.6, 0.8))

    displacement = screw_axis.to_displacement(position=2.0)

    np.testing.assert_allclose(displacement.translation.to_np()[:3], [0.0, 1.2, 1.6])
    np.testing.assert_allclose(
        displacement.rotation_vector.to_np()[:3], [0.0, 0.0, 0.0]
    )


@pytest.mark.parametrize(
    "position, expected_translation",
    [(np.pi / 2, [0.5, -0.5, 0.0]), (np.pi, [1.0, 0.0, 0.0])],
)
def test_a_hinge_off_the_origin_swings_the_body_on_a_circle(
    position: float, expected_translation: list[float]
):
    """
    A hinge through (0.5, 0, 0) moves the body's origin on a circle around it, which no
    displacement linear in the position could do.
    """
    hinge = Vector3(0.5, 0.0, 0.0)
    rotation = Vector3(0.0, 0.0, 1.0)
    screw_axis = ScrewAxis(
        rotation_per_unit_position=rotation,
        translation_per_unit_position=-rotation.cross(hinge),
    )

    displacement = screw_axis.to_displacement(position=position)

    np.testing.assert_allclose(
        displacement.translation.to_np()[:3], expected_translation, atol=1e-12
    )


@pytest.mark.parametrize("position", [1e-9, 1e-4, 0.04, 0.042, 0.045, 1.0, 3.0])
def test_the_displacement_agrees_with_the_matrix_exponential(position: float):
    """
    For a helical axis off the origin, the displacement equals the exponential of the
    screw axis's matrix times the position, on both sides of the switch to the series.
    """
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.6, 0.8),
        translation_per_unit_position=Vector3(0.3, -0.2, 0.1),
    )
    expected = expm(
        rigid_motion_rate_matrix(
            rotation=screw_axis.rotation_per_unit_position.to_np()[:3],
            translation=screw_axis.translation_per_unit_position.to_np()[:3],
        )
        * position
    )

    displacement = screw_axis.to_displacement(position=position)

    np.testing.assert_allclose(
        displacement.translation.to_np()[:3], expected[:3, 3], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        displacement.rotation_vector.to_np()[:3],
        screw_axis.rotation_per_unit_position.to_np()[:3] * position,
    )


@pytest.mark.parametrize("scale", [0.5, 2.0, 3.0])
def test_a_scaled_screw_axis_moves_the_body_as_the_axis_does_over_the_scaled_position(
    scale: float,
):
    """
    A screw axis need not have a unit rotation: scaling it by a factor moves the body as
    the original axis does over that factor times the position.
    """
    rotation = Vector3(0.0, 0.6, 0.8)
    translation = Vector3(0.3, -0.2, 0.1)
    screw_axis = ScrewAxis(
        rotation_per_unit_position=rotation, translation_per_unit_position=translation
    )
    scaled_screw_axis = ScrewAxis(
        rotation_per_unit_position=rotation * scale,
        translation_per_unit_position=translation * scale,
    )

    displacement = scaled_screw_axis.to_displacement(position=0.7)

    np.testing.assert_allclose(
        displacement.to_np(),
        screw_axis.to_displacement(position=0.7 * scale).to_np(),
        atol=1e-12,
    )


def test_the_displacement_of_a_symbolic_screw_axis_is_finite_where_it_stops_turning():
    """
    Evaluated where its rotation is zero, the displacement of a symbolic screw axis and
    its derivative by the rotation are finite, so a branch not taken leaks no NaN.
    """
    angle_per_unit_position = sm.FloatVariable(name="angle_per_unit_position")
    screw_axis = ScrewAxis(
        rotation_per_unit_position=Vector3(0.0, 0.0, angle_per_unit_position),
        translation_per_unit_position=Vector3(0.3, 0.0, 0.0),
    )
    translation = screw_axis.to_displacement(position=2.0).translation

    value = translation.substitute(
        old_variables=[angle_per_unit_position], new_variables=[0.0]
    ).to_np()
    derivative = (
        translation[:3]
        .jacobian(variables=[angle_per_unit_position])
        .substitute(old_variables=[angle_per_unit_position], new_variables=[0.0])
        .to_np()
    )

    np.testing.assert_allclose(value[:3], [0.6, 0.0, 0.0])
    assert np.isfinite(derivative).all()
