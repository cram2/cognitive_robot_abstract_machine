"""
The process-wide knowledge base: the recorded episode's entities, built once.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from typing_extensions import Any, ClassVar, Dict, List, Optional, Tuple

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.spatial_types import Point3

from cramera.knowledge.architecture_entities import Package, PythonClass, SubPackage
from cramera.knowledge.architecture_scan import ArchitectureScanner, PackageDependency
from cramera.knowledge.detected_events import DetectedEventRecord
from cramera.knowledge.entities import (
    ActionEpisode,
    RecordedArm,
    BenchObject,
    Gripper,
    JointMotion,
    Robot,
)
from cramera.knowledge.enums import JointRegion
from cramera.knowledge.scene_bundle import SceneBundle
from cramera.robot_parts import RobotPartAnnotation, RobotPartRole


@dataclass(init=False)
class EpisodeKnowledgeBase:
    """
    The recorded episode as EQL-queryable entities.

    Built once from the active scene bundle plus a static scan of the CRAM repository;
    every attribute is a plain list of dataclass instances that EQL variables range
    over.
    """

    _instances: ClassVar[Dict[Optional[str], EpisodeKnowledgeBase]] = {}
    """
    One built knowledge base per scene name, keyed by :meth:`of_scene`'s argument, so
    the viewer can hold several onboarded scenes open at once.
    """

    @classmethod
    def of_active_scene(cls) -> EpisodeKnowledgeBase:
        """
        The knowledge base of the active scene, built on first use.
        """
        return cls.of_scene(None)

    @classmethod
    def of_scene(cls, scene: Optional[str]) -> EpisodeKnowledgeBase:
        """
        The knowledge base of one scene, built on first use and cached per scene.

        A cached instance is served only while its bundle is unchanged: the live
        scene's ``scene.json`` is rewritten by the bridge on every attach, and a
        knowledge base built from the old bundle would keep answering for a world that
        no longer exists.

        :param scene: Name of the scene to build against, or None for the active one.
        """
        cached = cls._instances.get(scene)
        if cached is None or cached.bundle_signature != cls._bundle_signature(scene):
            cls._instances[scene] = cls(scene)
        return cls._instances[scene]

    @classmethod
    def _bundle_signature(cls, scene: Optional[str]) -> Optional[int]:
        """
        Modification stamp of a scene's ``scene.json``, or None when it does not exist.

        :param scene: Name of the scene, or None for the active one.
        """
        directory = SceneBundle.directory_of(scene)
        if directory is None:
            return None
        scene_path = directory / "scene.json"
        if not scene_path.is_file():
            return None
        return scene_path.stat().st_mtime_ns

    @classmethod
    def reset(cls) -> None:
        """
        Drop every cached knowledge base so the next access rebuilds it.

        Needed whenever the scenes directory changes, which is what tests do when they
        point ``CRAMERA_SCENES`` at a fixture.
        """
        cls._instances = {}

    scene_name: Optional[str]
    """
    Name of the scene this knowledge base was built from, or None for the active one.
    """
    bundle_signature: Optional[int]
    """
    Modification stamp of the bundle this instance was built from, compared by
    :meth:`of_scene` to serve a rewritten bundle fresh.
    """
    detected_events: List[DetectedEventRecord]
    """
    What the run's detectors saw, or nothing for a scene recorded without them.
    """
    objects: List[BenchObject]
    """
    Objects and placement regions in the recorded scene.
    """
    grippers: List[Gripper]
    """
    Grippers identified by the robot annotations.
    """
    arms: List[RecordedArm]
    """
    Arms available on the recorded robot.
    """
    part_annotations: List[RobotPartAnnotation]
    """
    Typed robot-part facts recorded with the scene.
    """
    robot: Robot
    """
    Robot described by the recording.
    """
    episodes: List[ActionEpisode]
    """
    Action intervals in the recorded plan.
    """
    joints: List[JointMotion]
    """
    Joint motion ranges measured across the trajectory.
    """
    packages: List[Package]
    """
    Packages found in the configured CRAM workspace.
    """
    classes: List[PythonClass]
    """
    Classes declared by the scanned workspace.
    """
    package_dependencies: List[PackageDependency]
    """
    Import dependencies between workspace packages.
    """
    subpackages: List[SubPackage]
    """
    Module groups containing the scanned classes.
    """

    def __init__(self, scene_name: Optional[str] = None) -> None:
        """
        Build every entity list from a scene bundle and a static scan of the CRAM
        architecture.

        :param scene_name: Name of the scene to build against, or None for the active
            one.
        """
        self.scene_name = scene_name
        self.bundle_signature = self._bundle_signature(scene_name)
        bundle = SceneBundle.of_scene(scene_name)
        scene, trajectory = bundle.scene, bundle.trajectory
        frames_per_second = scene.get("framesPerSecond", 30)
        parts = (scene.get("robot") or {}).get("parts") or {}
        robot_name = (scene.get("robot") or {}).get("name", "robot")
        robot_prefix = (scene.get("robot") or {}).get("prefix", "")

        self.objects = self._build_objects(scene)
        objects_by_id = {entity.name: entity for entity in self.objects}
        objects_by_reference = {
            **objects_by_id,
            **{
                entry["key"]: objects_by_id[entry["id"]]
                for entry in scene.get("objects") or []
                if entry.get("key")
            },
        }
        place_area = objects_by_id.get("place_area")

        self.part_annotations = RobotPartAnnotation.of_recording(
            scene.get("robot") or {}
        )

        self.grippers, self.arms = self._build_annotated_arms(
            self.part_annotations, robot_name
        )
        self.robot = Robot(robot_name, arm_count=len(self.arms))
        self.episodes = self._build_episodes(
            scene, frames_per_second, objects_by_reference, place_area
        )
        self.joints = self._build_joint_motions(trajectory, parts, robot_prefix)
        self.detected_events = DetectedEventRecord.of_scene(scene)

        architecture_scan = ArchitectureScanner.of_configured_root().load()
        self.packages = architecture_scan.packages
        self.classes = architecture_scan.classes
        self.package_dependencies = architecture_scan.dependency_edges
        self.subpackages = self._build_subpackages(self.classes)

    @staticmethod
    def _build_objects(scene: Dict[str, Any]) -> List[BenchObject]:
        """
        Scene objects (spawn poses recorded at frame 0) plus the place area.

        :param scene: The active scene bundle's ``scene.json`` content.
        """
        objects = []
        for entry in scene.get("objects") or []:
            objects.append(
                BenchObject(
                    name=entry["id"],
                    kind="object",
                    label=entry["id"].replace("_", " ").title(),
                    height_metres=entry.get("height"),
                    position=Point3(*[round(value, 3) for value in entry["spawn"][:3]]),
                )
            )
        if scene.get("placeTarget"):
            target = scene["placeTarget"]
            objects.append(
                BenchObject(
                    name="place_area",
                    kind="location",
                    label="Place area",
                    height_metres=0.0,  # a target area on a surface, not a solid
                    position=Point3(
                        round(target["position"][0], 3),
                        round(target["position"][1], 3),
                        target.get("z", 0),
                    ),
                )
            )
        return objects

    @classmethod
    def _build_annotated_arms(
        cls, part_annotations: List[RobotPartAnnotation], robot_name: str
    ) -> Tuple[List[Gripper], List[RecordedArm]]:
        """
        Arms and grippers read straight off the recorded sem_dt annotations.

        :param part_annotations: The recorded sem_dt robot-part annotations.
        :param robot_name: Name of the recorded robot, used to build each recorded arm.
        """
        end_effectors = {
            annotation.attached_to: annotation
            for annotation in part_annotations
            if annotation.role is RobotPartRole.END_EFFECTOR
        }
        grippers, arms = [], []
        for annotation in part_annotations:
            if annotation.role is not RobotPartRole.ARM:
                continue
            end_effector = end_effectors.get(annotation.name)
            side = annotation.side
            gripper = Gripper(
                end_effector.name if end_effector else annotation.name + "_ee",
                side,
            )
            grippers.append(gripper)
            arms.append(RecordedArm(annotation.name, side, robot_name, gripper))
        return grippers, arms

    def _build_episodes(
        self,
        scene: Dict[str, Any],
        frames_per_second: int,
        objects_by_reference: Dict[str, BenchObject],
        place_area: Optional[BenchObject],
    ) -> List[ActionEpisode]:
        """
        Action episodes from the recorded plan segments.

        :param scene: The active scene bundle's ``scene.json`` content.
        :param frames_per_second: The recording's frame rate, for episode durations.
        :param objects_by_reference: Scene objects keyed by their serialized key and
            display identity, for picked/placed lookups.
        :param place_area: The scene's place-area object, if any.
        """
        episodes = []
        for index, segment in enumerate(scene.get("segments") or []):
            picks = objects_by_reference.get(segment.get("picks"))
            episodes.append(
                ActionEpisode(
                    name=segment["step"],
                    index=index,
                    start_frame=segment["start"],
                    end_frame=segment["end"],
                    duration_seconds=round(
                        (segment["end"] - segment["start"]) / max(1, frames_per_second),
                        1,
                    ),
                    performed_by=self._arm_of_segment(segment) if picks else None,
                    picks=picks,
                    places_at=place_area if picks else None,
                )
            )
        return episodes

    def _arm_of_segment(self, segment: Dict[str, Any]) -> Optional[RecordedArm]:
        """
        The arm matching a recorded plan segment's side hint, falling back to the first
        arm if the segment picks something but names no side.

        :param segment: The recorded plan segment to match an arm to.
        """
        hint = (segment.get("arm") or "").lower()
        for arm in self.arms:
            if arm.side is not None and arm.side.name.lower() in hint:
                return arm
        return self.arms[0] if self.arms and segment.get("picks") else None

    @classmethod
    def _region_of_joint(
        cls, key: str, robot_prefix: str, link_to_part: Dict[str, str]
    ) -> JointRegion:
        """
        Which region a prefixed joint key belongs to: ``LEFT``/``RIGHT`` for an arm
        joint, ``ENVIRONMENT`` when it isn't the recorded robot's own joint, else
        ``BODY``.

        :param key: The prefixed joint key to classify.
        :param robot_prefix: World-name prefix of the recorded robot's own joints.
        :param link_to_part: Robot link names mapped to the part they belong to.
        """
        prefix, _, joint_name = key.partition("/")
        if "/" not in key:
            prefix, joint_name = "", key
        if robot_prefix and prefix != robot_prefix:
            return JointRegion.ENVIRONMENT
        part = link_to_part.get(joint_name.replace("_joint", "_link"))
        region = cls._side_of_name(part) if part else None
        if region is None:
            region = cls._side_of_name(joint_name)
        if region is Arms.LEFT:
            return JointRegion.LEFT
        if region is Arms.RIGHT:
            return JointRegion.RIGHT
        return JointRegion.BODY

    @classmethod
    def _build_joint_motions(
        cls, trajectory: Dict[str, Any], parts: Dict[str, Any], robot_prefix: str
    ) -> List[JointMotion]:
        """
        Per-joint motion statistics over the whole recorded trajectory.

        :param trajectory: The active scene bundle's ``trajectory.json`` content.
        :param parts: Robot part names to link names, from the recorded robot
            annotation.
        :param robot_prefix: World-name prefix of the recorded robot's own joints.
        """
        minimum: Dict[str, float] = {}
        maximum: Dict[str, float] = {}
        for frame in trajectory.get("frames") or []:
            for joint, value in frame.items():
                if joint not in minimum or value < minimum[joint]:
                    minimum[joint] = value
                if joint not in maximum or value > maximum[joint]:
                    maximum[joint] = value

        link_to_part = {link: part for part, links in parts.items() for link in links}

        return [
            JointMotion(
                name=key.partition("/")[2] or key,
                region=cls._region_of_joint(key, robot_prefix, link_to_part),
                minimum_radians=round(minimum[key], 3),
                maximum_radians=round(maximum[key], 3),
                range_radians=round(maximum[key] - minimum[key], 3),
            )
            for key in sorted(minimum)
        ]

    @staticmethod
    def _build_subpackages(classes: List[PythonClass]) -> List[SubPackage]:
        """
        Subpackage entities aggregated from the scanned classes.

        :param classes: The scanned Python classes to aggregate subpackages from.
        """
        modules = defaultdict(set)
        class_counts = defaultdict(int)
        for entry in classes:
            if entry.subpackage != entry.package:
                modules[(entry.package, entry.subpackage)].add(entry.module)
                class_counts[entry.subpackage] += 1
        return [
            SubPackage(
                name=subpackage,
                package=package,
                module_count=len(modules[(package, subpackage)]),
                class_count=class_counts[subpackage],
            )
            for (package, subpackage) in sorted(modules)
        ]

    @staticmethod
    def _side_of_name(name: str) -> Optional[Arms]:
        """
        Which arm a part/link name encodes, or None when it names neither.

        :param name: The part or link name to inspect.
        """
        lowered = name.lower()
        if "left" in lowered or lowered.startswith("l_"):
            return Arms.LEFT
        if "right" in lowered or lowered.startswith("r_"):
            return Arms.RIGHT
        return None
