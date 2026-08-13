"""
The geometry contract of the wind farm service demo.

The demo itself needs the full planning stack and takes half a minute, so it is not run
here. What is checked instead is that the constants describing the turbine really match
the model they were read from, and that every pose the demo derives lands on a surface
that exists and inside the space the robot is allowed to stand in.
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
    / "coraplex_wind_farm_service_demo"
)
"""
The demo whose layout is under test.
"""


@pytest.fixture(scope="module", autouse=True)
def demo_modules_importable():
    """
    Put the demo directory on the import path.

    A demo directory is not a package, so its modules are only importable by name once it
    is on the path, which is also how the onboarder runs them.
    """
    sys.path.insert(0, str(DEMO_DIRECTORY))
    yield
    sys.path.remove(str(DEMO_DIRECTORY))
    for module_name in ("demo", "service_tasks", "service_layout"):
        sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def service_turbine(demo_modules_importable) -> ElementTree.Element:
    """
    :return: The body of the turbine the demo services, from the committed model.
    """
    import service_layout

    root = ElementTree.parse(DEMO_DIRECTORY / "wind_farm.xml").getroot()
    return next(
        body
        for body in root.find("worldbody").findall("body")
        if body.get("name") == "%s_base" % service_layout.SERVICE_TURBINE
    )


def geom_named(body: ElementTree.Element, suffix: str) -> ElementTree.Element:
    """
    :param body: The body to search, including its descendants.
    :param suffix: The tail of the geom's name.
    :return: The geom whose name ends in that suffix.
    """
    return next(geom for geom in body.iter("geom") if geom.get("name").endswith(suffix))


def numbers(element: ElementTree.Element, attribute: str) -> list[float]:
    """
    :param element: The element to read.
    :param attribute: Name of a whitespace-separated numeric attribute.
    :return: The attribute's values.
    """
    return [float(value) for value in element.get(attribute).split()]


# %% the constants that were read out of the model


class TestLayoutMatchesTheModel:
    """
    The turbine measurements in :mod:`service_layout` against the committed wind farm.
    """

    def test_pad_surface_is_the_top_of_the_tower_base(self, service_turbine):
        import service_layout

        base = geom_named(service_turbine, "tower_base")
        body_z = numbers(service_turbine, "pos")[2]
        assert body_z + numbers(base, "size")[2] == service_layout.PAD_SURFACE_HEIGHT

    def test_pad_half_extent_matches_the_tower_base(self, service_turbine):
        import service_layout

        assert numbers(geom_named(service_turbine, "tower_base"), "size")[:2] == [
            service_layout.PAD_HALF_EXTENT,
            service_layout.PAD_HALF_EXTENT,
        ]

    def test_tower_radius_matches_the_tower_geom(self, service_turbine):
        import service_layout

        assert numbers(geom_named(service_turbine, "tower"), "size")[
            0
        ] == pytest.approx(service_layout.TOWER_RADIUS)

    def test_the_commanded_joints_exist_on_that_turbine(self, service_turbine):
        import service_layout

        joint_names = {joint.get("name") for joint in service_turbine.iter("joint")}
        assert service_layout.YAW_CONNECTION in joint_names
        assert service_layout.ROTOR_CONNECTION in joint_names


# %% the equipment the demo stands on the pad


class TestServiceSurfaces:
    """
    The boxes :mod:`service_layout` derives for the trailer and the bench.
    """

    def test_every_surface_rests_on_the_pad(self):
        import service_layout

        for surface in service_layout.SERVICE_SURFACES:
            assert surface.box_center[2] - surface.scale.z / 2 == pytest.approx(
                service_layout.PAD_SURFACE_HEIGHT
            )

    def test_every_surface_reaches_its_working_height(self):
        import service_layout

        for surface in service_layout.SERVICE_SURFACES:
            assert surface.box_center[2] + surface.scale.z / 2 == pytest.approx(
                surface.top_height
            )

    def test_the_working_edge_faces_the_lane_between_the_surfaces(self):
        import service_layout

        trailer, bench = (
            service_layout.DELIVERY_TRAILER,
            service_layout.SERVICE_BENCH,
        )
        assert bench.working_edge_x < trailer.working_edge_x
        assert bench.center[0] < bench.working_edge_x
        assert trailer.working_edge_x < trailer.center[0]

    def test_an_inset_moves_a_part_away_from_the_working_edge(self):
        import service_layout

        for surface in service_layout.SERVICE_SURFACES:
            towards_center = abs(surface.inset_x(0.2) - surface.center[0])
            assert towards_center < abs(surface.working_edge_x - surface.center[0])

    def test_no_surface_stands_in_the_tower_or_off_the_pad(self):
        import service_layout

        for surface in service_layout.SERVICE_SURFACES:
            nearest_x = abs(surface.center[0]) - surface.footprint[0] / 2
            assert nearest_x > service_layout.TOWER_RADIUS
            assert abs(surface.center[0]) + surface.footprint[0] / 2 < (
                service_layout.PAD_HALF_EXTENT
            )


# %% the poses the demo derives from the layout


class TestServiceTransferPoses:
    """
    Where the parts start, where they end up, and where the robot stands for each.
    """

    def test_parts_start_resting_on_the_trailer(self):
        import service_layout
        import service_tasks

        expected = (
            service_layout.DELIVERY_TRAILER.top_height + service_tasks.PART_SCALE.z / 2
        )
        for transfer in service_tasks.SERVICE_TRANSFERS:
            assert float(transfer.trailer_pose.z) == pytest.approx(expected)

    def test_parts_end_up_resting_on_the_bench(self):
        import service_layout
        import service_tasks

        expected = (
            service_layout.SERVICE_BENCH.top_height + service_tasks.PART_SCALE.z / 2
        )
        for transfer in service_tasks.SERVICE_TRANSFERS:
            assert float(transfer.bench_pose.z) == pytest.approx(expected)

    def test_parts_fit_on_the_surface_they_rest_on(self):
        import service_layout
        import service_tasks

        half_depth = service_tasks.PART_SCALE.y / 2
        for transfer in service_tasks.SERVICE_TRANSFERS:
            for pose, surface in (
                (transfer.trailer_pose, service_layout.DELIVERY_TRAILER),
                (transfer.bench_pose, service_layout.SERVICE_BENCH),
            ):
                lower, upper = surface.extent_y
                assert float(pose.y) - half_depth >= lower
                assert float(pose.y) + half_depth <= upper

    def test_the_robot_only_ever_stands_in_the_lane_between_the_surfaces(self):
        import service_layout
        import service_tasks

        lower, upper = service_layout.WORKING_LANE_X
        standing_poses = [
            pose
            for transfer in service_tasks.SERVICE_TRANSFERS
            for pose in (transfer.trailer_standing_pose, transfer.bench_standing_pose)
        ]
        for pose in standing_poses:
            assert lower < float(pose.x) < upper

    def test_the_robot_approaches_along_that_lane(self):
        import service_layout
        import service_tasks

        lower, upper = service_layout.WORKING_LANE_X
        for pose in (service_tasks.ROBOT_START_POSE, *service_tasks.APPROACH_WAYPOINTS):
            assert lower < float(pose.x) < upper

    def test_the_robot_stands_on_the_pad(self):
        import service_layout
        import service_tasks

        assert service_tasks.STANDING_HEIGHT == pytest.approx(
            service_layout.PAD_SURFACE_HEIGHT + service_tasks.PELVIS_HEIGHT_ABOVE_FLOOR
        )
        for transfer in service_tasks.SERVICE_TRANSFERS:
            assert float(transfer.trailer_standing_pose.z) == pytest.approx(
                service_tasks.STANDING_HEIGHT
            )

    def test_the_robot_turns_around_between_trailer_and_bench(self):
        import service_tasks

        for transfer in service_tasks.SERVICE_TRANSFERS:
            assert abs(transfer.turn_towards_bench) == pytest.approx(3.141592653589793)

    def test_each_part_gets_its_own_colour(self):
        import service_tasks

        colours = [transfer.color for transfer in service_tasks.SERVICE_TRANSFERS]
        assert len(colours) == len(service_tasks.PART_COLORS)
        assert len({(colour.R, colour.G, colour.B) for colour in colours}) == len(
            colours
        )


# %% shutting the turbine down over the approach


class TestTurbineShutdown:
    """
    The angles the turbine is commanded to while the robot walks up.
    """

    def test_the_turbine_starts_in_its_running_position(self):
        import service_tasks

        assert service_tasks.turbine_state(0.0) == {"yaw": 0.0, "rotor": 0.0}

    def test_the_turbine_ends_parked_for_service(self):
        import service_layout
        import service_tasks

        assert service_tasks.turbine_state(1.0) == {
            "yaw": service_layout.SERVICE_YAW,
            "rotor": service_layout.SERVICE_ROTOR,
        }

    def test_the_shutdown_only_ever_advances_over_the_approach(self):
        import service_tasks

        legs = len(service_tasks.APPROACH_WAYPOINTS)
        angles = [service_tasks.turbine_state(leg / legs) for leg in range(0, legs + 1)]
        for earlier, later in zip(angles, angles[1:]):
            assert later["yaw"] > earlier["yaw"]
            assert later["rotor"] > earlier["rotor"]

    def test_the_last_approach_leg_finishes_the_shutdown(self):
        import service_tasks

        legs = len(service_tasks.APPROACH_WAYPOINTS)
        assert service_tasks.turbine_state(legs / legs) == service_tasks.turbine_state(
            1.0
        )
