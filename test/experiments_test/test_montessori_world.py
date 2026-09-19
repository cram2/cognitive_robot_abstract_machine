import numpy as np
import pytest
from trimesh import Trimesh

from experiments.montessori.pieces import KNOWN_PIECE_BY_CATEGORY
from experiments.montessori.semantics import (
    MontessoriShape,
    MontessoriShapeCategory,
    ShapeSortingBoard,
    ShapeSortingHole,
)
from experiments.montessori.world import (
    BOARD_SCALE,
    FLOOR_Z,
    SHAPE_FOOTPRINT_CLEARANCE_SCALE,
    TABLE_POSITION,
    TABLE_SCALE,
    TABLE_SHAPE_ROW_X,
    MontessoriWorld,
    robot_installed,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor, Table
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Actuator

from .dataset.synthetic_fixed_arm_robot import SyntheticFixedArmRobot


def _sorted_xy(mesh: Trimesh) -> np.ndarray:
    """
    A mesh's vertices, projected to (x, y) and sorted into a canonical order, so two
    meshes built from the same footprint can be compared regardless of internal vertex
    ordering.
    """
    xy = np.round(mesh.vertices[:, :2], 6)
    return xy[np.lexsort((xy[:, 1], xy[:, 0]))]


def test_montessori_world_creates_one_board_with_holes_and_drawers():
    montessori = MontessoriWorld()

    [board] = montessori.world.get_semantic_annotations_by_type(ShapeSortingBoard)
    assert board is montessori.board
    assert len(board.apertures) == 6
    assert len(board.drawers) == 3
    assert all(drawer.handle is not None for drawer in board.drawers)


def test_montessori_world_cuts_a_watertight_board_mesh():
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    # The board's *visual* geometry is the single, precisely cut mesh; its *collision*
    # geometry is a hand-built grid of boxes (see
    # test_montessori_world_board_collision_leaves_every_holes_bounding_box_open) that
    # approximates each hole by its bounding box rather than preserving its exact cut
    # volume.
    board_mesh = montessori.board.root.visual.combined_mesh

    assert board_mesh.is_watertight
    # cutting 6 holes must strictly shrink the solid board's volume
    uncut_volume = BOARD_SCALE.x * BOARD_SCALE.y * BOARD_SCALE.z
    assert 0 < board_mesh.volume < uncut_volume


_BOUNDS_TOLERANCE = 1e-6
"""
Slack, in meters, subtracted from a hole's bounding box before checking it against a
collision box: the hole's bounds and the collision grid's exclusion cell for that same
hole are computed via two different (equivalent, but not bit-identical) arithmetic
paths, which leaves edges that are meant to exactly touch off by a sub-micron amount;
without this tolerance, that is indistinguishable from a genuine overlap.
"""


def _xy_bounds_overlap(a: tuple, b: tuple) -> bool:
    a_min_x, a_max_x, a_min_y, a_max_y = a
    b_min_x, b_max_x, b_min_y, b_max_y = b
    x_overlaps = a_min_x < b_max_x and b_min_x < a_max_x
    y_overlaps = a_min_y < b_max_y and b_min_y < a_max_y
    return x_overlaps and y_overlaps


def _collision_box_world_bounds(shapes, position) -> list:
    """
    World-frame (x, y) bounds of every :class:`Box` in ``shapes``, a body's collision
    shapes positioned at ``position``.

    Collision boxes are built axis-aligned with no rotation (see
    ``world.py::_tile_footprint_avoiding_holes``), so comparing their (x, y) bounds
    directly, without trimesh's ray-cast-based ``contains``, is both correct and immune
    to the numerical instability ray casting has against many mutually-touching box
    faces.
    """
    bounds = []
    for shape in shapes:
        origin = shape.origin.to_position()
        half_x, half_y = shape.scale.x / 2, shape.scale.y / 2
        bounds.append(
            (
                float(position.x) + float(origin.x) - half_x,
                float(position.x) + float(origin.x) + half_x,
                float(position.y) + float(origin.y) - half_y,
                float(position.y) + float(origin.y) + half_y,
            )
        )
    return bounds


def _hole_world_bounds(hole: ShapeSortingHole) -> tuple:
    """A hole's true (x, y) bounding box, in the world frame, with tolerance applied."""
    position = hole.global_transform.to_position()
    local_bounds = hole.root.area.combined_mesh.bounds
    return (
        float(position.x) + local_bounds[0][0] + _BOUNDS_TOLERANCE,
        float(position.x) + local_bounds[1][0] - _BOUNDS_TOLERANCE,
        float(position.y) + local_bounds[0][1] + _BOUNDS_TOLERANCE,
        float(position.y) + local_bounds[1][1] - _BOUNDS_TOLERANCE,
    )


def test_montessori_world_board_collision_leaves_every_holes_bounding_box_open():
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    board_position = montessori.board.global_transform.to_position()
    collision_box_bounds = _collision_box_world_bounds(
        montessori.board.root.collision.shapes, board_position
    )
    assert collision_box_bounds

    holes = montessori.world.get_semantic_annotations_by_type(ShapeSortingHole)
    assert holes
    for hole in holes:
        overlapping_boxes = [
            box_bounds
            for box_bounds in collision_box_bounds
            if _xy_bounds_overlap(box_bounds, _hole_world_bounds(hole))
        ]
        error = f"{hole.name} has collision material in its opening"
        assert not overlapping_boxes, error

    # sanity check the assertion above is not vacuous: solid material away from every
    # hole (a board corner) must still be covered by some collision box
    board_corner = (
        float(board_position.x) + BOARD_SCALE.x / 2 - 0.005,
        float(board_position.x) + BOARD_SCALE.x / 2 - 0.004,
        float(board_position.y) + BOARD_SCALE.y / 2 - 0.005,
        float(board_position.y) + BOARD_SCALE.y / 2 - 0.004,
    )
    assert any(
        _xy_bounds_overlap(box_bounds, board_corner)
        for box_bounds in collision_box_bounds
    )


def test_montessori_world_drawer_collision_leaves_overlapping_holes_open():
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    # Every hole cuts through the board's full thickness, and a drawer can sit
    # anywhere within that thickness directly behind a hole (see
    # world.py::_drawer_collision_boxes); a shape that cleanly passes the board's own
    # collision must not then be stopped by a drawer immediately behind it.
    holes = montessori.world.get_semantic_annotations_by_type(ShapeSortingHole)
    assert holes
    hole_bounds = [_hole_world_bounds(hole) for hole in holes]

    assert montessori.board.drawers
    for drawer in montessori.board.drawers:
        drawer_position = drawer.global_transform.to_position()
        collision_box_bounds = _collision_box_world_bounds(
            drawer.root.collision.shapes, drawer_position
        )
        assert collision_box_bounds

        for bounds in hole_bounds:
            overlapping_boxes = [
                box_bounds
                for box_bounds in collision_box_bounds
                if _xy_bounds_overlap(box_bounds, bounds)
            ]
            error = f"{drawer.name} has collision material overlapping a hole above it"
            assert not overlapping_boxes, error


def test_montessori_world_creates_one_shape_per_hole_category_plus_the_sphere():
    montessori = MontessoriWorld()

    holes = montessori.world.get_semantic_annotations_by_type(ShapeSortingHole)
    hole_categories = {hole.shape_category for hole in holes}
    shape_categories = [
        shape.shape_category
        for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape)
    ]

    # one shape per hole (some categories repeat, e.g. the two circular holes), plus
    # the sphere, which has no matching hole
    assert set(shape_categories) == hole_categories | {MontessoriShapeCategory.SPHERE}
    assert len(shape_categories) == len(holes) + 1


def test_montessori_world_pairs_each_circular_shape_with_its_own_sized_hole():
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    cylinder_shapes = [
        shape
        for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape)
        if shape.shape_category == MontessoriShapeCategory.CYLINDER
    ]

    # the board has two circular holes of different sizes; each cylinder shape must be
    # sized after its own hole's footprint (not a single shared, fixed size), and must
    # resolve back to that same hole rather than to the other, differently-sized one
    cross_section_sizes = {shape.cross_section_size for shape in cylinder_shapes}
    assert len(cylinder_shapes) == 2
    assert len(cross_section_sizes) == 2

    for shape in cylinder_shapes:
        hole = montessori.board.hole_for(shape)
        assert hole.name.name == shape.name.name.removesuffix("_shape")
        assert shape.cross_section_size <= hole.cross_section_size


@pytest.mark.parametrize(
    "category",
    [
        MontessoriShapeCategory.TRIANGULAR_PRISM,
        MontessoriShapeCategory.RECTANGULAR_PRISM,
    ],
)
def test_orientation_sensitive_shape_matches_its_holes_footprint_orientation(category):
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    [shape] = [
        shape
        for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape)
        if shape.shape_category == category
    ]
    hole = montessori.board.hole_for(shape)

    # a shape whose fit through its hole depends on rotation (unlike e.g. a cube or a
    # cylinder) must be a scaled, same-orientation copy of that hole's true footprint,
    # not an independently authored shape that merely shares its category: scaling the
    # shape's cross-section back up must recover the hole's own footprint, not some
    # rotation of it.
    shape_xy = _sorted_xy(shape.root.collision.combined_mesh)
    hole_xy = _sorted_xy(hole.root.area.combined_mesh)

    assert shape_xy.shape == hole_xy.shape
    np.testing.assert_allclose(
        shape_xy / SHAPE_FOOTPRINT_CLEARANCE_SCALE, hole_xy, atol=1e-4
    )


def test_montessori_world_places_loose_shapes_resting_on_the_table():
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    shapes = montessori.world.get_semantic_annotations_by_type(MontessoriShape)
    table_top_z = float(TABLE_POSITION.z) + TABLE_SCALE.z / 2
    table_min_y = float(TABLE_POSITION.y) - TABLE_SCALE.y / 2
    table_max_y = float(TABLE_POSITION.y) + TABLE_SCALE.y / 2

    y_positions = []
    for shape in shapes:
        position = shape.global_transform.to_position()
        lowest_local_z = shape.root.collision.combined_mesh.bounds[0][2]

        # the shape's lowest point rests exactly on the table's surface
        assert float(position.z) + lowest_local_z == pytest.approx(table_top_z)
        assert float(position.x) == TABLE_SHAPE_ROW_X
        assert table_min_y <= float(position.y) <= table_max_y
        y_positions.append(float(position.y))

    # every shape sits at its own spot along the row
    assert len(set(y_positions)) == len(shapes)


def test_montessori_world_welds_loose_shapes_by_default():
    montessori = MontessoriWorld()

    shapes = montessori.world.get_semantic_annotations_by_type(MontessoriShape)
    assert shapes
    for shape in shapes:
        assert isinstance(shape.root.parent_connection, FixedConnection)


def test_montessori_world_can_make_loose_shapes_movable():
    montessori = MontessoriWorld(shapes_are_movable=True)

    shapes = montessori.world.get_semantic_annotations_by_type(MontessoriShape)
    assert shapes
    for shape in shapes:
        assert isinstance(shape.root.parent_connection, Connection6DoF)


def test_movable_loose_shapes_still_rest_on_the_table():
    montessori = MontessoriWorld(shapes_are_movable=True)
    montessori.world.update_forward_kinematics()

    table_top_z = float(TABLE_POSITION.z) + TABLE_SCALE.z / 2
    for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape):
        position = shape.global_transform.to_position()
        lowest_local_z = shape.root.collision.combined_mesh.bounds[0][2]
        assert float(position.z) + lowest_local_z == pytest.approx(table_top_z)
        assert float(position.x) == pytest.approx(TABLE_SHAPE_ROW_X)


def test_montessori_world_creates_one_table_and_one_floor():
    montessori = MontessoriWorld()

    [table] = montessori.world.get_semantic_annotations_by_type(Table)
    assert table is not None
    [floor] = montessori.world.get_semantic_annotations_by_type(Floor)
    assert floor is not None


def test_montessori_world_table_stands_on_the_floor():
    montessori = MontessoriWorld()
    montessori.world.update_forward_kinematics()

    [table] = montessori.world.get_semantic_annotations_by_type(Table)
    position = table.global_transform.to_position()
    lowest_local_z = table.root.collision.combined_mesh.bounds[0][2]

    assert float(position.z) + lowest_local_z == pytest.approx(FLOOR_Z)


def test_montessori_world_has_no_robot_until_spawned():
    montessori = MontessoriWorld()

    assert montessori.robot is None


def test_montessori_world_spawn_robot_adds_the_robot_to_the_world():
    if not robot_installed(HSRB):
        pytest.skip("hsr_description is not installed")

    montessori = MontessoriWorld()
    body_count_before_spawn = len(montessori.world.bodies)

    robot = montessori.spawn_robot(HSRB)

    assert robot is montessori.robot
    assert len(montessori.world.bodies) > body_count_before_spawn


MOUNT_POSITION = Point3(0.25, 0.0, 0.5)
MOUNT_YAW = 3.14159265


def _parsed_synthetic_fixed_arm_robot():
    return URDFParser.from_file(SyntheticFixedArmRobot.get_ros_file_path()).parse()


def test_mount_stationary_robot_attaches_a_fixed_base_robot_to_the_world():
    montessori = MontessoriWorld()
    body_count_before_mount = len(montessori.world.bodies)

    robot = montessori.mount_stationary_robot(
        SyntheticFixedArmRobot, _parsed_synthetic_fixed_arm_robot(), MOUNT_POSITION
    )

    assert isinstance(robot, SyntheticFixedArmRobot)
    assert robot is montessori.robot
    assert len(montessori.world.bodies) > body_count_before_mount
    assert isinstance(robot.root.parent_connection, FixedConnection)


def test_mount_stationary_robot_places_the_robot_at_the_given_position_and_yaw():
    montessori = MontessoriWorld()

    robot = montessori.mount_stationary_robot(
        SyntheticFixedArmRobot,
        _parsed_synthetic_fixed_arm_robot(),
        MOUNT_POSITION,
        mount_yaw=MOUNT_YAW,
    )
    montessori.world.update_forward_kinematics()

    position = robot.root.global_transform.to_position()
    assert float(position.x) == pytest.approx(float(MOUNT_POSITION.x))
    assert float(position.y) == pytest.approx(float(MOUNT_POSITION.y))
    assert float(position.z) == pytest.approx(float(MOUNT_POSITION.z))
    yaw = robot.root.global_transform.to_rotation_matrix().to_rpy()[2]
    assert float(yaw) == pytest.approx(MOUNT_YAW)


def test_mount_stationary_robot_drops_actuators_from_the_parsed_world():
    montessori = MontessoriWorld()
    robot_world = _parsed_synthetic_fixed_arm_robot()
    dof = robot_world.get_connection_by_name("arm_joint").raw_dof
    actuator = Actuator()
    actuator.add_dof(dof=dof)
    with robot_world.modify_world():
        robot_world.add_actuator(actuator=actuator)

    montessori.mount_stationary_robot(
        SyntheticFixedArmRobot, robot_world, MOUNT_POSITION
    )

    assert actuator not in montessori.world.actuators


def test_add_robot_stand_spawns_a_table_at_the_mount_height():
    montessori = MontessoriWorld()
    body_count_before = len(montessori.world.bodies)

    stand = montessori.add_robot_stand(MOUNT_POSITION)
    montessori.world.update_forward_kinematics()

    assert isinstance(stand, Table)
    assert len(montessori.world.bodies) > body_count_before
    position = stand.global_transform.to_position()
    highest_local_z = stand.root.collision.combined_mesh.bounds[1][2]
    assert float(position.x) == pytest.approx(float(MOUNT_POSITION.x))
    assert float(position.y) == pytest.approx(float(MOUNT_POSITION.y))
    assert float(position.z) + highest_local_z == pytest.approx(float(MOUNT_POSITION.z))


def test_a_loose_shape_wears_the_colour_measured_off_the_real_piece():
    montessori = MontessoriWorld()

    coloured = {
        shape.shape_category: shape.root.visual[0].color
        for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape)
        if shape.shape_category in KNOWN_PIECE_BY_CATEGORY
    }

    assert coloured == {
        category: piece.color
        for category, piece in KNOWN_PIECE_BY_CATEGORY.items()
        if category in coloured
    }
    assert set(coloured) == set(KNOWN_PIECE_BY_CATEGORY)
