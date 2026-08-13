"""
The geometry contract of the warehouse storage demo.

The demo itself needs the full planning stack and takes half a minute, so it is not run
here. What is checked instead is everything that can go silently wrong between runs: a
measurement moved in :mod:`warehouse_layout` without regenerating the URDF, or a crate
pose that no longer lands on the surface it is supposed to rest on.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

DEMO_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_warehouse_storage_demo"
)
"""
The demo whose layout is under test.
"""


@pytest.fixture(scope="module", autouse=True)
def demo_modules_importable():
    """
    Put the demo directory on the import path.

    A demo directory is not a package, so its modules are only importable by name once
    it is on the path, which is also how the onboarder runs them.
    """
    sys.path.insert(0, str(DEMO_DIRECTORY))
    yield
    sys.path.remove(str(DEMO_DIRECTORY))
    for module_name in ("demo", "stow_tasks", "warehouse_layout"):
        sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def urdf_links(demo_modules_importable) -> dict[str, ElementTree.Element]:
    """
    :return: Every link of the committed warehouse URDF, keyed by name.
    """
    root = ElementTree.parse(DEMO_DIRECTORY / "storage_warehouse.urdf").getroot()
    return {link.get("name"): link for link in root.findall("link")}


def box_of(link: ElementTree.Element, tag: str) -> tuple[list[float], list[float]]:
    """
    :param link: The link to read.
    :param tag: Either ``visual`` or ``collision``.
    :return: The size and the origin of the link's box geometry.
    """
    element = link.find(tag)
    size = [float(value) for value in element.find("geometry/box").get("size").split()]
    origin = [float(value) for value in element.find("origin").get("xyz").split()]
    return size, origin


# %% the measurements the converter turns into collision geometry


class TestWarehouseFittings:
    """
    The boxes :mod:`warehouse_layout` derives for the surfaces the robot works on.
    """

    def test_target_shelf_top_is_the_shelf_height(self):
        import warehouse_layout

        shelf = next(
            fitting
            for fitting in warehouse_layout.FITTINGS
            if fitting.link_name == "target_shelf"
        )
        assert shelf.center[2] + shelf.size[2] / 2 == warehouse_layout.RACK_SHELF_HEIGHT

    def test_target_shelf_covers_the_clear_stretch(self):
        import warehouse_layout

        shelf = next(
            fitting
            for fitting in warehouse_layout.FITTINGS
            if fitting.link_name == "target_shelf"
        )
        assert (
            shelf.center[0] - shelf.size[0] / 2,
            shelf.center[0] + shelf.size[0] / 2,
        ) == warehouse_layout.TARGET_SHELF_X
        assert (
            shelf.center[1] - shelf.size[1] / 2,
            shelf.center[1] + shelf.size[1] / 2,
        ) == warehouse_layout.TARGET_SHELF_CLEAR_Y

    def test_pallet_load_top_is_the_height_crates_are_picked_from(self):
        import warehouse_layout

        load = next(
            fitting
            for fitting in warehouse_layout.FITTINGS
            if fitting.link_name == "incoming_pallet_load"
        )
        assert load.center[2] + load.size[2] / 2 == warehouse_layout.PALLET_LOAD_HEIGHT

    def test_pallet_load_rests_on_the_pallet_base(self):
        import warehouse_layout

        base, load = (
            next(
                fitting
                for fitting in warehouse_layout.FITTINGS
                if fitting.link_name == link_name
            )
            for link_name in ("incoming_pallet_base", "incoming_pallet_load")
        )
        assert base.center[2] + base.size[2] / 2 == pytest.approx(
            load.center[2] - load.size[2] / 2
        )
        assert base.center[2] + base.size[2] / 2 == warehouse_layout.PALLET_BASE_HEIGHT


# %% the URDF the converter wrote from those measurements


class TestGeneratedUrdfMatchesTheLayout:
    """
    The committed URDF against the measurements it was generated from.
    """

    def test_every_fitting_became_a_collision_box(self, urdf_links):
        import warehouse_layout

        for fitting in warehouse_layout.FITTINGS:
            size, origin = box_of(urdf_links[fitting.link_name], "collision")
            assert size == pytest.approx(list(fitting.size))
            assert origin == pytest.approx(list(fitting.center))

    def test_only_the_fittings_carry_collision(self, urdf_links):
        import warehouse_layout

        with_collision = {
            name
            for name, link in urdf_links.items()
            if link.find("collision") is not None
        }
        assert with_collision == {
            fitting.link_name for fitting in warehouse_layout.FITTINGS
        }

    def test_a_fitting_without_a_material_is_not_drawn(self, urdf_links):
        assert urdf_links["target_shelf"].find("visual") is None

    def test_a_fitting_with_a_material_is_drawn_as_the_same_box(self, urdf_links):
        assert box_of(urdf_links["incoming_pallet_load"], "visual") == box_of(
            urdf_links["incoming_pallet_load"], "collision"
        )


# %% the poses the demo derives from the layout


class TestStowTaskPoses:
    """
    Where the crates start, where they end up, and where the robot stands for each.
    """

    def test_crates_start_resting_on_the_pallet_load(self):
        import stow_tasks
        import warehouse_layout

        expected = warehouse_layout.PALLET_LOAD_HEIGHT + stow_tasks.CRATE_SCALE.z / 2
        for task in stow_tasks.STOW_TASKS:
            assert float(task.pick_pose.z) == pytest.approx(expected)

    def test_crates_end_up_resting_on_the_shelf(self):
        import stow_tasks
        import warehouse_layout

        expected = warehouse_layout.RACK_SHELF_HEIGHT + stow_tasks.CRATE_SCALE.z / 2
        for task in stow_tasks.STOW_TASKS:
            assert float(task.shelf_pose.z) == pytest.approx(expected)

    def test_crates_fit_on_the_pallet(self):
        import stow_tasks
        import warehouse_layout

        lower = (
            warehouse_layout.PALLET_CENTER[1] - warehouse_layout.PALLET_FOOTPRINT[1] / 2
        )
        upper = (
            warehouse_layout.PALLET_CENTER[1] + warehouse_layout.PALLET_FOOTPRINT[1] / 2
        )
        for task in stow_tasks.STOW_TASKS:
            assert float(task.pick_pose.y) - stow_tasks.CRATE_SCALE.y / 2 >= lower
            assert float(task.pick_pose.y) + stow_tasks.CRATE_SCALE.y / 2 <= upper

    def test_stowed_crates_fit_within_the_clear_stretch_of_shelf(self):
        import stow_tasks
        import warehouse_layout

        lower, upper = warehouse_layout.TARGET_SHELF_CLEAR_Y
        for task in stow_tasks.STOW_TASKS:
            assert float(task.shelf_pose.y) - stow_tasks.CRATE_SCALE.y / 2 >= lower
            assert float(task.shelf_pose.y) + stow_tasks.CRATE_SCALE.y / 2 <= upper

    def test_stowed_crates_do_not_overhang_the_shelf_edge(self):
        import stow_tasks
        import warehouse_layout

        for task in stow_tasks.STOW_TASKS:
            assert (
                float(task.shelf_pose.x) - stow_tasks.CRATE_SCALE.x / 2
                >= warehouse_layout.TARGET_SHELF_X[0]
            )

    def test_the_robot_only_ever_stands_in_the_free_aisle(self):
        import stow_tasks
        import warehouse_layout

        lower_x, upper_x = warehouse_layout.WORKING_AISLE_X
        lower_y, upper_y = warehouse_layout.WORKING_AISLE_Y
        standing_poses = [stow_tasks.ROBOT_START_POSE] + [
            pose
            for task in stow_tasks.STOW_TASKS
            for pose in (task.pallet_standing_pose, task.shelf_standing_pose)
        ]
        for pose in standing_poses:
            assert lower_x <= float(pose.x) <= upper_x
            assert lower_y <= float(pose.y) <= upper_y

    def test_the_robot_turns_around_between_pallet_and_shelf(self):
        import stow_tasks

        for task in stow_tasks.STOW_TASKS:
            assert abs(task.turn_towards_shelf) == pytest.approx(3.141592653589793)

    def test_each_crate_gets_its_own_colour(self):
        import stow_tasks

        colours = [task.color for task in stow_tasks.STOW_TASKS]
        assert len(colours) == len(stow_tasks.CRATE_COLORS)
        assert len({(colour.R, colour.G, colour.B) for colour in colours}) == len(
            colours
        )


class TestEvenlySpaced:
    """
    Spreading a number of positions over a stretch.
    """

    def test_positions_are_centred_in_equal_slots(self, demo_modules_importable):
        import stow_tasks

        assert stow_tasks.evenly_spaced((0.0, 3.0), 3) == [0.5, 1.5, 2.5]

    def test_a_single_position_lands_in_the_middle(self, demo_modules_importable):
        import stow_tasks

        assert stow_tasks.evenly_spaced((2.0, 4.0), 1) == [3.0]


# %% the colour conversion the converter applies


class TestDisplayColorConversion:
    """
    Turning the source model's linear ``Kd`` values into URDF display colours.
    """

    def test_black_and_white_are_unchanged(self):
        pytest.importorskip("open3d")
        import generate_warehouse_model

        assert generate_warehouse_model.to_display_color((0.0, 0.0, 0.0)) == (
            0.0,
            0.0,
            0.0,
        )
        assert generate_warehouse_model.to_display_color((1.0, 1.0, 1.0)) == (
            1.0,
            1.0,
            1.0,
        )

    def test_a_dark_linear_value_becomes_a_visibly_lighter_display_value(self):
        pytest.importorskip("open3d")
        import generate_warehouse_model

        # The warehouse floor: a Kd of 0.021 is a dark grey, not the near-black it
        # would be if the value were written out unconverted.
        red, green, blue = generate_warehouse_model.to_display_color(
            (0.0211, 0.0211, 0.0211)
        )
        assert red == green == blue
        assert 0.14 < red < 0.17
