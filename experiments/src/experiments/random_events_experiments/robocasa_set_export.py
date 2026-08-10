"""
This module exports RoboCasa's own ground truth for what a scene or a task admits, for a
curated set of tasks.

RoboCasa states this admissible information in whatever form its own implementation
happens to use: a placement region as sampler parameters (a reference pose plus local
ranges), a gripper exclusion zone as a body position and a threshold, a fixture's open
or closed state as a normalized position on a MuJoCo joint, a device's state as an entry
in a plain dictionary. Reading a candidate representation's fidelity directly off those
forms would tie the ground truth to whatever RoboCasa happens to store internally.

The two Cartesian tracks -- placement regions and gripper exclusion zones -- are
therefore attached to a :class:`~semantic_digital_twin.world.World` as
:class:`~semantic_digital_twin.world_description.world_entity.Region`\\ s, in exactly the
shape model (:class:`~semantic_digital_twin.world_description.geometry.Box`,
:class:`~semantic_digital_twin.world_description.geometry.Sphere`) the rest of that world
already uses, via
:mod:`semantic_digital_twin.adapters.robocasa_dataset.region_extraction`. The remaining
two tracks -- joint openness and discrete device state -- are not shapes at all, so they
are read directly into plain intervals and values instead.

For placement regions, this module additionally measures how faithfully an axis-aligned
:mod:`random_events` representation captures that ground truth: a region's oriented
rectangle is compared, by intersection over union, against the smallest axis-aligned box
that contains it -- the only shape :class:`~random_events.product_algebra.Event` can
express. Gripper exclusion zones are spheres, which no union of axis-aligned boxes can
represent tightly, and approximating those is deliberately left for a later step. Joint
openness and discrete device state are not shapes at all, so they are read directly into
plain intervals and values instead, with no approximation to measure.
"""

from __future__ import annotations

import enum
import inspect
import json
import pathlib
import time
from dataclasses import dataclass
from typing_extensions import Any

import numpy as np

from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)

with robocasa_version_assertions_relaxed():
    import robocasa  # noqa: F401  # registers the kitchen tasks with robosuite

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from random_events.interval import Bound, SimpleInterval
from random_events.product_algebra import Event, SimpleEvent
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
)
from semantic_digital_twin.adapters.robocasa_dataset.region_extraction import (
    GripperExclusionZoneReader,
    PlacementSamplerRegion,
    PlacementSamplerRegionReader,
)
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_types import Point3

ALL_LAYOUTS_AND_STYLES = -3
"""
Layout and style selector that lets a task be instantiated in any kitchen it supports.

Passed straight through to robosuite beneath :class:`RoboCasaDatasetLoader`'s
``LayoutType``/``StyleType`` typing, which is how RoboCasa itself spells "pick any
layout/style this task supports."
"""

SURVEYED_TASKS = [
    "TurnOnBlender",
    "TurnOnToaster",
    "TurnOnMicrowave",
    "AddMarshmallow",
    "RetrieveIceTray",
    "PlaceIceInCup",
]
"""
Tasks the placement region export covers, spanning appliances, counters and the stools
and dining counters whose frames RoboCasa treats as a special case.
"""

JOINT_STATE_TASKS = ["OpenCabinet", "OpenMicrowave"]
"""
Tasks the joint openness export covers, spanning a multi door fixture and a single door
one.
"""

TASK_FIXTURE_ATTRIBUTE = {
    "TurnOnBlender": "blender",
    "TurnOnToaster": "toaster",
    "TurnOnMicrowave": "microwave",
}
"""
Tasks the discrete device state export covers, mapped to the attribute their built
environment exposes the relevant fixture under.
"""

GRIPPER_FAR_ZONES = {
    "AddMarshmallow": [("mug", 0.25)],
    "PlaceIceInCup": [("ice_cube1", 0.15), ("ice_cube2", 0.15)],
    "RetrieveIceTray": [("ice_cube_tray", 0.25)],
}
"""
Tasks the forbidden zone export covers, mapped to the ``(body_name, threshold)`` pairs
their own ``_check_success`` passes to ``gripper_obj_far``.

Read directly from each task's success condition rather than discovered generically,
since nothing in a built scene names which bodies a success condition happens to keep
the gripper away from.
"""


@dataclass
class AttachedPlacementRegion(ExperimentResult):
    """
    One placement region attached to a task's world as a Region, with the summary
    statistics used to sanity check the attachment.
    """

    task: str
    """
    Task the region belongs to.
    """

    object_name: str
    """
    Name the scene gave the object the region was resolved for.
    """

    width: float
    """
    Extent of the region along the sampler's own x axis.
    """

    depth: float
    """
    Extent of the region along the sampler's own y axis.
    """

    world_x: float
    """
    World x coordinate the region was attached at.
    """

    world_y: float
    """
    World y coordinate the region was attached at.
    """

    world_z: float
    """
    World z coordinate the region was attached at.
    """


@dataclass
class PlacementRegionAttacher:
    """
    Reads the placement regions a set of RoboCasa tasks resolve and attaches each as a
    Region to that task's own world.
    """

    task_names: list[str]
    """
    Tasks to instantiate.
    """

    def run(self) -> list[AttachedPlacementRegion]:
        """
        :return: One entry per attached region, across every task.
        """
        loader = RoboCasaDatasetLoader()
        records = []
        for task_name in self.task_names:
            records.extend(self.run_task(loader, task_name))
        return records

    def run_task(
        self, loader: RoboCasaDatasetLoader, task_name: str
    ) -> list[AttachedPlacementRegion]:
        """
        :param loader: Loader used to build the task's environment and parse its world.
        :param task_name: Task to instantiate.
        :return: One entry per region the task resolved.
        """
        environment = loader.build_task_environment(
            task_name,
            layout_id=ALL_LAYOUTS_AND_STYLES,
            style_id=ALL_LAYOUTS_AND_STYLES,
        )
        task = loader.load_task_from_environment(environment)
        region_data = PlacementSamplerRegionReader().read_all(
            environment, name_prefix=task_name
        )
        records = [self.attach(task.world, data) for data in region_data]
        environment.close()
        return records

    @staticmethod
    def attach(world, region_data: PlacementSamplerRegion) -> AttachedPlacementRegion:
        """
        :param world: The task's world to attach the region to.
        :param region_data: The region to attach.
        :return: The attached region's summary statistics.
        """
        region_data.attach_to(world)
        world_position = region_data.world_T_sampler.to_position()
        return AttachedPlacementRegion(
            task=region_data.name.prefix,
            object_name=region_data.name.name,
            width=region_data.width,
            depth=region_data.depth,
            world_x=float(world_position.x),
            world_y=float(world_position.y),
            world_z=float(world_position.z),
        )


@dataclass
class PlacementRegionFidelity(ExperimentResult):
    """
    How closely a placement region's axis-aligned random_events representation matches
    the oriented rectangle RoboCasa actually resolved, measured by intersection over
    union.
    """

    task: str
    """
    Task the region belongs to.
    """

    object_name: str
    """
    Name the scene gave the object the region was resolved for.
    """

    dimensions: int
    """
    Number of variables the random_events representation spans.
    """

    simple_set_count: int
    """
    Number of simple sets composing the random_events representation.
    """

    intersection_over_union: float
    """
    Volume of the region's oriented footprint divided by the volume of its axis-aligned
    random_events representation.

    The oriented footprint is always fully contained in its own axis-aligned bounding
    box, so the intersection is the footprint's own volume and the union is the bounding
    box's volume.
    """

    duration: float
    """
    Time, in seconds, spent building the representation and computing the measure.
    """


@dataclass
class PlacementRegionFidelityMeasurer:
    """
    Measures how faithfully an axis-aligned random_events representation captures the
    oriented rectangle a RoboCasa placement sampler resolved, via intersection over
    union.
    """

    task_names: list[str]
    """
    Tasks to instantiate.
    """

    def run(self) -> list[PlacementRegionFidelity]:
        """
        :return: One entry per placement region the tasks resolved.
        """
        loader = RoboCasaDatasetLoader()
        records = []
        for task_name in self.task_names:
            records.extend(self.run_task(loader, task_name))
        return records

    def run_task(
        self, loader: RoboCasaDatasetLoader, task_name: str
    ) -> list[PlacementRegionFidelity]:
        """
        :param loader: Loader used to build the task's environment.
        :param task_name: Task to instantiate.
        :return: One entry per placement region the task resolved.
        """
        environment = loader.build_task_environment(
            task_name,
            layout_id=ALL_LAYOUTS_AND_STYLES,
            style_id=ALL_LAYOUTS_AND_STYLES,
        )
        region_data = PlacementSamplerRegionReader().read_all(
            environment, name_prefix=task_name
        )
        records = [self.measure(data) for data in region_data]
        environment.close()
        return records

    @staticmethod
    def measure(region: PlacementSamplerRegion) -> PlacementRegionFidelity:
        """
        :param region: The placement region to measure the fidelity of.
        :return: The region's random_events representation size and its intersection
            over union with the region's own oriented footprint.
        """
        start = time.perf_counter()

        local_corners = [
            Point3(
                region.local_center.x + x_sign * region.width / 2.0,
                region.local_center.y + y_sign * region.depth / 2.0,
                region.local_center.z + z_sign * region.thickness / 2.0,
            )
            for x_sign in (-1.0, 1.0)
            for y_sign in (-1.0, 1.0)
            for z_sign in (-1.0, 1.0)
        ]
        world_corners = [region.world_T_sampler.dot(corner) for corner in local_corners]
        x_values = [float(corner.x) for corner in world_corners]
        y_values = [float(corner.y) for corner in world_corners]
        z_values = [float(corner.z) for corner in world_corners]

        simple_event = SimpleEvent.from_data(
            {
                SpatialVariables.x.value: SimpleInterval.from_data(
                    min(x_values), max(x_values), Bound.CLOSED, Bound.CLOSED
                ),
                SpatialVariables.y.value: SimpleInterval.from_data(
                    min(y_values), max(y_values), Bound.CLOSED, Bound.CLOSED
                ),
                SpatialVariables.z.value: SimpleInterval.from_data(
                    min(z_values), max(z_values), Bound.CLOSED, Bound.CLOSED
                ),
            }
        )
        event = Event.from_simple_sets(simple_event)

        footprint_volume = region.width * region.depth * region.thickness
        bounding_box_volume = (
            (max(x_values) - min(x_values))
            * (max(y_values) - min(y_values))
            * (max(z_values) - min(z_values))
        )
        duration = time.perf_counter() - start

        return PlacementRegionFidelity(
            task=region.name.prefix,
            object_name=region.name.name,
            dimensions=len(simple_event.variables),
            simple_set_count=len(event.simple_sets),
            intersection_over_union=footprint_volume / bounding_box_volume,
            duration=duration,
        )


@dataclass
class AttachedForbiddenZone(ExperimentResult):
    """
    One gripper exclusion zone attached to a task's world as a Region, with the summary
    statistics used to sanity check the attachment.
    """

    task: str
    """
    Task the zone belongs to.
    """

    body_name: str
    """
    Name of the body the gripper must stay away from.
    """

    radius: float
    """
    Distance from the centre the gripper must stay beyond.
    """

    world_x: float
    """
    World x coordinate the zone was attached at.
    """

    world_y: float
    """
    World y coordinate the zone was attached at.
    """

    world_z: float
    """
    World z coordinate the zone was attached at.
    """


@dataclass
class ForbiddenZoneAttacher:
    """
    Reads the gripper exclusion zones a set of RoboCasa tasks' success conditions
    resolve and attaches each as a Region to that task's own world.
    """

    task_zones: dict[str, list[tuple[str, float]]]
    """
    Tasks to instantiate, mapped to the ``(body_name, threshold)`` pairs their success
    condition checks the gripper against.
    """

    def run(self) -> list[AttachedForbiddenZone]:
        """
        :return: One entry per attached zone, across every task.
        """
        loader = RoboCasaDatasetLoader()
        records = []
        for task_name, zones in self.task_zones.items():
            records.extend(self.run_task(loader, task_name, zones))
        return records

    def run_task(
        self,
        loader: RoboCasaDatasetLoader,
        task_name: str,
        zones: list[tuple[str, float]],
    ) -> list[AttachedForbiddenZone]:
        """
        :param loader: Loader used to build the task's environment and parse its world.
        :param task_name: Task to instantiate.
        :param zones: The ``(object_name, threshold)`` pairs to resolve.
        :return: One entry per zone the task resolved.
        """
        environment = loader.build_task_environment(
            task_name,
            layout_id=ALL_LAYOUTS_AND_STYLES,
            style_id=ALL_LAYOUTS_AND_STYLES,
        )
        task = loader.load_task_from_environment(environment)
        records = [
            self.attach(
                task.world,
                GripperExclusionZoneReader.read(
                    environment,
                    name_prefix=task_name,
                    object_name=object_name,
                    radius=radius,
                ),
            )
            for object_name, radius in zones
        ]
        environment.close()
        return records

    @staticmethod
    def attach(world, zone_data) -> AttachedForbiddenZone:
        """
        :param world: The task's world to attach the zone to.
        :param zone_data: The zone to attach.
        :return: The attached zone's summary statistics.
        """
        zone_data.attach_to(world)
        return AttachedForbiddenZone(
            task=zone_data.name.prefix,
            body_name=zone_data.name.name,
            radius=zone_data.radius,
            world_x=float(zone_data.center.x),
            world_y=float(zone_data.center.y),
            world_z=float(zone_data.center.z),
        )


@dataclass
class JointOpennessInterval(ExperimentResult):
    """
    The physical range of a fixture's door joint, together with the sub intervals of it
    RoboCasa treats as open and as closed.

    Unlike a placement region, a joint's admissible range is already a plain interval on
    both the RoboCasa side and the :mod:`random_events` side, so no mesh or other
    intermediate format is needed to compare them.
    """

    task: str
    """
    Task the joint belongs to.
    """

    fixture_type: str
    """
    Class name of the fixture the joint was read from.
    """

    joint_name: str
    """
    Name MuJoCo gave the joint.
    """

    physical_minimum: float
    """
    Smallest value the joint can take.
    """

    physical_maximum: float
    """
    Largest value the joint can take.
    """

    open_threshold: float
    """
    Normalized position, read from the fixture's own ``is_open`` check, above which
    RoboCasa considers the joint open.
    """

    closed_threshold: float
    """
    Normalized position, read from the fixture's own ``is_closed`` check, below which
    RoboCasa considers the joint closed.
    """

    open_minimum: float
    """
    Smallest physical value RoboCasa still considers open.
    """

    open_maximum: float
    """
    Largest physical value the joint can take while still counting as open.
    """

    closed_minimum: float
    """
    Smallest physical value the joint can take while still counting as closed.
    """

    closed_maximum: float
    """
    Largest physical value RoboCasa still considers closed.
    """


@dataclass
class ScalarStateExporter:
    """
    Exports the physical open and closed intervals of the door joints a set of RoboCasa
    tasks resolve.
    """

    task_names: list[str]
    """
    Tasks to instantiate.
    """

    def export(self) -> list[JointOpennessInterval]:
        """
        :return: One entry per door joint the tasks' fixtures resolved.
        """
        loader = RoboCasaDatasetLoader()
        records = []
        for task_name in self.task_names:
            records.extend(self.export_task(loader, task_name))
        return records

    def export_task(
        self, loader: RoboCasaDatasetLoader, task_name: str
    ) -> list[JointOpennessInterval]:
        """
        :param loader: Loader used to build the task's environment.
        :param task_name: Task to instantiate.
        :return: One entry per door joint the task's fixture resolved.
        """
        environment = loader.build_task_environment(
            task_name,
            layout_id=ALL_LAYOUTS_AND_STYLES,
            style_id=ALL_LAYOUTS_AND_STYLES,
        )
        fixture = environment.fxtr
        records = [
            self.joint_interval_of(environment, task_name, fixture, joint_name)
            for joint_name in fixture.door_joint_names
        ]
        environment.close()
        return records

    @staticmethod
    def joint_interval_of(
        environment, task_name: str, fixture, joint_name: str
    ) -> JointOpennessInterval:
        """
        :param environment: The scene the fixture was resolved in.
        :param task_name: Task the scene belongs to.
        :param fixture: The fixture the joint belongs to.
        :param joint_name: Name of the joint to read.
        :return: The joint's physical range and its resolved open and closed intervals.
        """
        joint_id = environment.sim.model.joint_name2id(joint_name)
        minimum, maximum = environment.sim.model.jnt_range[joint_id]
        open_threshold = inspect.signature(fixture.is_open).parameters["th"].default
        closed_threshold = inspect.signature(fixture.is_closed).parameters["th"].default
        span = maximum - minimum
        return JointOpennessInterval(
            task=task_name,
            fixture_type=type(fixture).__name__,
            joint_name=joint_name,
            physical_minimum=float(minimum),
            physical_maximum=float(maximum),
            open_threshold=float(open_threshold),
            closed_threshold=float(closed_threshold),
            open_minimum=float(minimum + open_threshold * span),
            open_maximum=float(maximum),
            closed_minimum=float(minimum),
            closed_maximum=float(minimum + closed_threshold * span),
        )


class StateValueKind(enum.Enum):
    """
    What an extracted device state value tells a candidate representation about its own
    domain.
    """

    BOOLEAN = "boolean domain, exactly {True, False}"
    """
    The value is a boolean, so its domain is established: exactly ``{True, False}``.
    """

    UNCLASSIFIED = "non-boolean value; this extractor does not establish its domain"
    """
    The value is not a boolean.

    This extractor reports the value it observed but does not claim to know the full
    domain it is drawn from.
    """


@dataclass
class DiscreteFixtureState(ExperimentResult):
    """
    One entry of a fixture's device state, as returned by its own ``get_state``.
    """

    task: str
    """
    Task the fixture belongs to.
    """

    fixture_type: str
    """
    Class name of the fixture the state was read from.
    """

    state_key: str
    """
    Name RoboCasa gave this entry in the fixture's state dictionary.
    """

    observed_value: Any
    """
    Value the entry held right after the scene was reset.
    """

    kind: StateValueKind
    """
    Whether this entry's domain is established by this extractor.
    """


@dataclass
class SymbolicStateExporter:
    """
    Exports the discrete device state a set of RoboCasa tasks' fixtures resolve.
    """

    task_fixture_attribute: dict[str, str]
    """
    Tasks to instantiate, mapped to the attribute their built environment exposes the
    relevant fixture under.
    """

    def export(self) -> list[DiscreteFixtureState]:
        """
        :return: One entry per state dictionary key the tasks' fixtures resolved.
        """
        loader = RoboCasaDatasetLoader()
        records = []
        for task_name, attribute_name in self.task_fixture_attribute.items():
            records.extend(self.export_task(loader, task_name, attribute_name))
        return records

    def export_task(
        self, loader: RoboCasaDatasetLoader, task_name: str, attribute_name: str
    ) -> list[DiscreteFixtureState]:
        """
        :param loader: Loader used to build the task's environment.
        :param task_name: Task to instantiate.
        :param attribute_name: Attribute the built environment exposes the fixture
            under.
        :return: One entry per state dictionary key the task's fixture resolved.
        """
        environment = loader.build_task_environment(
            task_name,
            layout_id=ALL_LAYOUTS_AND_STYLES,
            style_id=ALL_LAYOUTS_AND_STYLES,
        )
        fixture = getattr(environment, attribute_name)
        state = self.flatten(self.state_of(fixture, environment))
        records = [
            self.classify(task_name, type(fixture).__name__, key, value)
            for key, value in state.items()
        ]
        environment.close()
        return records

    @staticmethod
    def state_of(fixture, environment) -> dict[str, Any]:
        """
        A fixture's ``get_state`` does not take the same parameters everywhere in
        RoboCasa: some read the scene they were resolved in, most do not need it.

        :param fixture: The fixture to read the state of.
        :param environment: The scene the fixture was resolved in.
        :return: The fixture's state dictionary.
        """
        parameters = inspect.signature(fixture.get_state).parameters
        if "env" in parameters:
            return fixture.get_state(env=environment)
        return fixture.get_state()

    @classmethod
    def flatten(cls, state: dict[Any, Any], prefix: str = "") -> dict[str, Any]:
        """
        :param state: A fixture's state dictionary, possibly nested (for instance one
            entry per slot of a multi slot appliance).
        :param prefix: Dotted path of the keys already descended into.
        :return: The state with every nested dictionary inlined into a single level,
            keyed by the dotted path to each leaf value.
        """
        flattened = {}
        for key, value in state.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flattened.update(cls.flatten(value, path))
            else:
                flattened[path] = value
        return flattened

    @staticmethod
    def classify(
        task_name: str, fixture_type: str, state_key: str, value: Any
    ) -> DiscreteFixtureState:
        """
        :param task_name: Task the fixture belongs to.
        :param fixture_type: Class name of the fixture the state was read from.
        :param state_key: Name of the state dictionary entry.
        :param value: Value the entry held.
        :return: The entry, classified by whether its domain is established.
        """
        kind = (
            StateValueKind.BOOLEAN
            if isinstance(value, (bool, np.bool_))
            else StateValueKind.UNCLASSIFIED
        )
        return DiscreteFixtureState(
            task=task_name,
            fixture_type=fixture_type,
            state_key=state_key,
            observed_value=value,
            kind=kind,
        )


def write_manifest(
    records: list[ExperimentResult], output_directory: pathlib.Path, file_name: str
) -> pathlib.Path:
    """
    :param records: The exported records to summarise.
    :param output_directory: Directory the manifest is written to.
    :param file_name: Name of the manifest file.
    :return: Path the manifest was written to.
    """
    manifest_path = output_directory / file_name
    manifest_path.write_text(
        json.dumps([as_json_value(record.__dict__) for record in records], indent=2)
    )
    return manifest_path


def as_json_value(value: Any) -> Any:
    """
    :param value: A value that may contain enum members or numpy scalars, neither of
        which :mod:`json` can serialise on its own.
    :return: The value with every enum member replaced by its name and every numpy
        scalar replaced by the native Python value it holds.
    """
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: as_json_value(entry) for key, entry in value.items()}
    return value


def main() -> None:
    """
    Exports each track of RoboCasa's ground truth for the surveyed tasks, printing a
    table per track and recording its rows alongside.
    """
    output_directory = pathlib.Path("robocasa_set_export_output")
    output_directory.mkdir(parents=True, exist_ok=True)

    placements = PlacementRegionAttacher(task_names=SURVEYED_TASKS).run()
    write_manifest(placements, output_directory, "placement_regions.json")
    print(f"attached {len(placements)} placement regions")
    print(
        TypstRenderer(ExperimentsTable(placements)).render_figure(
            "Placement regions RoboCasa resolved for the surveyed tasks, attached to "
            "each task's world as semantic_digital_twin Regions."
        )
    )

    placement_fidelity = PlacementRegionFidelityMeasurer(
        task_names=SURVEYED_TASKS
    ).run()
    write_manifest(
        placement_fidelity, output_directory, "placement_region_fidelity.json"
    )
    print(f"measured {len(placement_fidelity)} placement region fidelities")
    print(
        TypstRenderer(ExperimentsTable(placement_fidelity)).render_figure(
            "Intersection over union between each placement region's oriented "
            "footprint and its axis-aligned random_events representation, for the "
            "surveyed tasks."
        )
    )

    forbidden_zones = ForbiddenZoneAttacher(task_zones=GRIPPER_FAR_ZONES).run()
    write_manifest(forbidden_zones, output_directory, "forbidden_zones.json")
    print(f"attached {len(forbidden_zones)} forbidden zones")
    print(
        TypstRenderer(ExperimentsTable(forbidden_zones)).render_figure(
            "Gripper exclusion zones RoboCasa's success conditions resolved for the "
            "surveyed tasks, attached to each task's world as semantic_digital_twin "
            "Regions."
        )
    )

    joint_intervals = ScalarStateExporter(task_names=JOINT_STATE_TASKS).export()
    write_manifest(joint_intervals, output_directory, "joint_openness.json")
    print(f"wrote {len(joint_intervals)} joint openness intervals")
    print(
        TypstRenderer(ExperimentsTable(joint_intervals)).render_figure(
            "Physical open and closed intervals of the door joints RoboCasa resolved "
            "for the surveyed tasks."
        )
    )

    device_states = SymbolicStateExporter(
        task_fixture_attribute=TASK_FIXTURE_ATTRIBUTE
    ).export()
    write_manifest(device_states, output_directory, "discrete_state.json")
    print(f"wrote {len(device_states)} discrete device state entries")
    print(
        TypstRenderer(ExperimentsTable(device_states)).render_figure(
            "Discrete device state RoboCasa resolved for the surveyed tasks."
        )
    )


if __name__ == "__main__":
    main()
