import math

import numpy as np
import pytest

from semantic_digital_twin.world_description.geometry import Polygon2D

# %% measuring a polygon


def test_a_rectangles_area_and_centroid():
    rectangle = Polygon2D(np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]))

    assert rectangle.area == pytest.approx(2.0)
    assert rectangle.centroid.to_np() == pytest.approx([1.0, 0.5])


def test_a_triangle_balances_at_a_third_of_its_height():
    triangle = Polygon2D(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))

    assert triangle.area == pytest.approx(0.5)
    assert triangle.centroid.to_np() == pytest.approx([1 / 3, 1 / 3])


def test_the_area_does_not_depend_on_the_winding():
    corners = np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])

    assert Polygon2D(corners[::-1]).area == Polygon2D(corners).area


def test_a_closing_corner_that_repeats_the_first_is_dropped():
    corners = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])

    closed = Polygon2D(np.vstack([corners, corners[:1]]))

    assert closed.vertices == pytest.approx(corners)


def test_the_bounding_box_holds_every_corner():
    triangle = Polygon2D(np.array([(-1.0, 0.0), (3.0, 0.5), (0.0, 2.0)]))

    box = triangle.bounding_box

    assert (box.min_x, box.min_y, box.max_x, box.max_y) == (-1.0, 0.0, 3.0, 2.0)


def test_centering_moves_the_centroid_onto_the_origin():
    triangle = Polygon2D(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))

    assert triangle.centered().centroid.to_np() == pytest.approx([0.0, 0.0])


# %% building a polygon


def test_a_rectangle_is_as_wide_and_long_as_asked():
    box = Polygon2D.rectangle(0.02, 0.04).bounding_box

    assert (box.depth, box.width) == pytest.approx((0.02, 0.04))


def test_a_circle_has_the_asked_diameter_and_corner_count():
    circle = Polygon2D.circle(0.028, corner_count=12)

    assert len(circle.vertices) == 12
    assert np.linalg.norm(circle.vertices, axis=1) == pytest.approx(0.014)


def test_an_equilateral_triangle_has_three_equal_sides_about_its_centroid():
    triangle = Polygon2D.equilateral_triangle(0.037)

    sides = np.linalg.norm(triangle.vertices - np.roll(triangle.vertices, 1, 0), axis=1)
    assert sides == pytest.approx(0.037)
    assert triangle.centroid.to_np() == pytest.approx([0.0, 0.0], abs=1e-12)


# %% using a polygon


def test_turning_a_quarter_turn_moves_x_onto_y():
    point = Polygon2D(np.array([(1.0, 0.0), (0.0, 0.0), (0.0, -1.0)]))

    turned = point.turned(math.pi / 2)

    assert turned.vertices[0] == pytest.approx([0.0, 1.0])


def test_points_along_include_every_corner_at_the_asked_spacing():
    square = Polygon2D.rectangle(1.0, 1.0)

    points = square.points_along(0.25)

    assert len(points) == 16
    for corner in square.vertices:
        assert np.isclose(points, corner).all(axis=1).any()


def test_extruding_gives_a_watertight_solid_of_the_enclosed_area_times_thickness():
    triangle = Polygon2D.equilateral_triangle(0.037)

    solid = triangle.extrude(0.01)

    assert solid.is_watertight
    assert solid.volume == pytest.approx(triangle.area * 0.01)
    assert solid.bounds[:, 2] == pytest.approx([-0.005, 0.005])


# %% how far a polygon turns before it looks the same


def test_a_square_looks_the_same_after_a_quarter_turn():
    square = Polygon2D.rectangle(width=0.03, length=0.03)

    assert square.rotational_symmetry_angle() == pytest.approx(math.pi / 2)


def test_a_rectangle_looks_the_same_only_after_a_half_turn():
    rectangle = Polygon2D.rectangle(width=0.02, length=0.04)

    assert rectangle.rotational_symmetry_angle() == pytest.approx(math.pi)


def test_an_equilateral_triangle_looks_the_same_after_a_third_of_a_turn():
    triangle = Polygon2D.equilateral_triangle(side=0.037)

    assert triangle.rotational_symmetry_angle() == pytest.approx(2 * math.pi / 3)


def test_a_circle_looks_the_same_after_turning_from_one_corner_to_the_next():
    circle = Polygon2D.circle(diameter=0.028, corner_count=12)

    assert circle.rotational_symmetry_angle() == pytest.approx(2 * math.pi / 12)


def test_a_polygon_without_symmetry_looks_the_same_only_after_a_full_turn():
    triangle = Polygon2D(np.array([(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)]))

    assert triangle.rotational_symmetry_angle() == pytest.approx(2 * math.pi)


def test_a_polygon_turns_about_its_own_centroid_wherever_it_lies():
    square = Polygon2D.rectangle(width=0.03, length=0.03)
    moved = Polygon2D(square.vertices + np.array([5.0, -3.0]))

    assert moved.rotational_symmetry_angle() == square.rotational_symmetry_angle()


def test_corners_further_apart_than_the_tolerance_break_the_symmetry():
    corners = Polygon2D.rectangle(width=2.0, length=2.0).vertices
    corners[0] += 1e-6
    nudged = Polygon2D(corners)

    assert nudged.rotational_symmetry_angle() == pytest.approx(2 * math.pi)


def test_corners_within_the_tolerance_keep_the_symmetry():
    corners = Polygon2D.rectangle(width=2.0, length=2.0).vertices
    corners[0] += 1e-6
    nudged = Polygon2D(corners, corner_match_tolerance=1e-5)

    assert nudged.rotational_symmetry_angle() == pytest.approx(math.pi / 2)


# %% how far points lie from a polygon's edges


def test_a_corner_lies_on_the_boundary():
    square = Polygon2D.rectangle(width=2.0, length=2.0)

    assert square.distance_to_boundary(square.vertices) == pytest.approx(
        np.zeros(len(square.vertices))
    )


def test_a_point_is_as_far_from_the_boundary_as_from_its_nearest_edge():
    square = Polygon2D.rectangle(width=2.0, length=2.0)
    points = np.array([(0.0, 0.0), (3.0, 0.0), (2.0, 2.0)])

    assert square.distance_to_boundary(points) == pytest.approx(
        [1.0, 2.0, math.sqrt(2.0)]
    )


# %% turning one polygon onto another


def test_a_polygon_already_lined_up_with_another_needs_no_turn():
    rectangle = Polygon2D.rectangle(width=0.02, length=0.04)

    assert rectangle.turn_onto(rectangle) == pytest.approx(0.0)


def test_a_turned_triangle_is_turned_back_onto_its_original():
    triangle = Polygon2D.equilateral_triangle(side=0.037)
    turned = triangle.turned(math.radians(30))

    turn = turned.turn_onto(triangle)

    assert turn % (2 * math.pi / 3) == pytest.approx(
        math.radians(90), abs=turned.alignment_angle_step
    )


def test_a_smaller_polygon_is_lined_up_with_a_larger_one_of_the_same_shape():
    small = Polygon2D.rectangle(width=0.02, length=0.04)
    large_and_turned = Polygon2D.rectangle(width=0.042, length=0.022)

    turn = small.turn_onto(large_and_turned)

    assert turn % math.pi == pytest.approx(math.pi / 2, abs=small.alignment_angle_step)


def test_a_polygon_is_turned_onto_another_in_whole_steps_of_its_alignment_angle():
    step = math.radians(25.0)
    rectangle = Polygon2D.rectangle(width=0.02, length=0.04)
    coarse = Polygon2D(rectangle.vertices, alignment_angle_step=step)

    turn = coarse.turn_onto(rectangle.turned(math.radians(60)))

    assert turn / step == pytest.approx(round(turn / step))
    assert turn == pytest.approx(math.radians(50))


# %% a polygon's settings carried by the polygons made from it


def test_a_turned_polygon_keeps_its_alignment_angle_and_tolerance():
    step = math.radians(25.0)
    rectangle = Polygon2D.rectangle(width=0.02, length=0.04)
    tuned = Polygon2D(
        rectangle.vertices, alignment_angle_step=step, corner_match_tolerance=1e-5
    )

    turned = tuned.turned(math.pi / 3)

    assert (turned.alignment_angle_step, turned.corner_match_tolerance) == (
        tuned.alignment_angle_step,
        tuned.corner_match_tolerance,
    )


def test_a_centered_polygon_keeps_its_alignment_angle_and_tolerance():
    rectangle = Polygon2D.rectangle(width=0.02, length=0.04)
    tuned = Polygon2D(
        rectangle.vertices + 1.0,
        alignment_angle_step=math.radians(25.0),
        corner_match_tolerance=1e-5,
    )

    centered = tuned.centered()

    assert (centered.alignment_angle_step, centered.corner_match_tolerance) == (
        tuned.alignment_angle_step,
        tuned.corner_match_tolerance,
    )
