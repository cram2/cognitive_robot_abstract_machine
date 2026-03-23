
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, Any, Optional, Union

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.world import World

from pycram.datastructures.enums import Arms

from llmr.pipeline.action_dispatcher import ActionDispatcher, WorldContext
from llmr.workflows.nodes.slot_filler import ActionSlotSchema, run_slot_filler

if TYPE_CHECKING:
    from llmr.planning.motion_precondition_planner import ExecutionState

logger = logging.getLogger(__name__)


# ── World serialiser ───────────────────────────────────────────────────────────


_ROBOT_LINK_SUFFIXES = (
    "_link",
    "_frame",
    "_joint",
    "_screw",
    "_plate",
    "_optical_frame",
    "_motor",
    "_pad",
    "_finger",
)


def _is_robot_link(name: str) -> bool:
    """Return True if *name* looks like a robot kinematic link rather than a scene object."""
    return any(name.endswith(s) for s in _ROBOT_LINK_SUFFIXES)


def _serialise_robot_state(exec_state: "ExecutionState") -> str:
    """Render the robot's current arm occupancy from *exec_state* as a context string."""
    lines = ["\n## Robot Arm State"]

    if exec_state.held_objects:
        all_arms = [Arms.LEFT, Arms.RIGHT]
        for arm in all_arms:
            body = exec_state.held_objects.get(arm)
            if body is not None:
                body_name = str(getattr(getattr(body, "name", None), "name", body))
                lines.append(f"  {arm.name} arm: holding {body_name}")
            else:
                lines.append(f"  {arm.name} arm: empty")
    elif exec_state.last_pickup_arm is not None:
        # Fallback for state objects created before held_objects was introduced.
        body = exec_state.last_pickup_body
        body_name = (
            str(getattr(getattr(body, "name", None), "name", body)) if body else "unknown object"
        )
        lines.append(f"  {exec_state.last_pickup_arm.name} arm: holding {body_name}")
        other = Arms.LEFT if exec_state.last_pickup_arm == Arms.RIGHT else Arms.RIGHT
        lines.append(f"  {other.name} arm: empty")
    else:
        lines.append("  Both arms: empty")

    return "\n".join(lines)


def _serialise_world_for_llm(
    world: World,
    exec_state: Optional["ExecutionState"] = None,
) -> str:
    """Produce a concise string description of the world state for LLM context.

    :param world: The SDT world instance.
    :param exec_state: Optional execution state carrying robot arm occupancy.
        When provided, a ``## Robot Arm State`` section is appended so the LLM
        knows which arm is holding which object before making decisions.
    """
    lines = ["## World State Summary\n"]

    try:
        all_names = [str(getattr(getattr(b, "name", None), "name", b)) for b in world.bodies]
        # Strip robot kinematic links — they dominate the list and are irrelevant
        # to the LLM. Keep only environment objects and surfaces.
        scene_names = [n for n in all_names if not _is_robot_link(n)]
        if scene_names:
            lines.append(f"Scene objects and surfaces: {', '.join(scene_names)}")
        else:
            # Fallback: nothing survived the filter — show raw list capped at 30
            lines.append(f"Bodies present: {', '.join(all_names[:30])}")
            if len(all_names) > 30:
                lines.append(f"  … and {len(all_names) - 30} more.")
    except Exception:
        lines.append("Bodies: unavailable")

    # Semantic annotations — always emit the heading so the LLM knows this
    # section exists even when the world has none.
    # Iterate world.semantic_annotations directly rather than per-body lookup,
    # because body identity breaks after deepcopy/merge.
    lines.append("\n## Semantic annotations")
    try:
        ann_summary: dict[str, list[str]] = {}
        for ann in getattr(world, "semantic_annotations", []):
            ann_type = type(ann).__name__
            try:
                for body in ann.bodies:
                    b_name = str(getattr(getattr(body, "name", None), "name", body))
                    if _is_robot_link(b_name):
                        continue
                    ann_summary.setdefault(b_name, []).append(ann_type)
            except Exception:
                pass
        if ann_summary:
            unique_types = sorted({t for types in ann_summary.values() for t in types})
            lines.append(f"Available types: {', '.join(unique_types)}")
            lines.append("Per body:")
            for body_name, types in ann_summary.items():
                lines.append(f"  {body_name}: {', '.join(types)}")
        else:
            lines.append("  None found in this world.")
    except Exception:
        lines.append("  (unavailable)")

    if exec_state is not None:
        lines.append(_serialise_robot_state(exec_state))

    return "\n".join(lines)


# ── Pipeline ───────────────────────────────────────────────────────────────────


@dataclass
class ActionPipeline:
    """Universal NL → pycram action pipeline.

    Handles any instruction whose action type is registered in ``ActionDispatcher``.
    No subclassing or per-action configuration required.

    :param world: The Semantic Digital Twin world instance.
    :param world_context: Runtime context carrying the robot manipulator.
    """

    world: World
    world_context: WorldContext

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_world(
        cls,
        world: World,
        arm: Optional[Arms] = None,
    ) -> "ActionPipeline":
        """Create a pipeline from a world instance.

        Auto-detects the robot's manipulator for use in grasp descriptions
        (required for PickUpAction; benign no-op for PlaceAction).

        :param world: SDT world with a loaded robot and objects.
        :param arm: Which arm's manipulator to use.  Defaults to RIGHT.
        """
        manipulator = cls._find_manipulator(world, arm or Arms.RIGHT)
        if manipulator is None:
            logger.warning(
                "Could not auto-detect robot manipulator.  "
                "GraspDescription construction for PickUpAction will fail unless "
                "a manipulator is injected manually into world_context."
            )
        return cls(
            world=world,
            world_context=WorldContext(manipulator=manipulator),
        )

    @staticmethod
    def _find_manipulator(world: World, arm: Arms) -> Optional[Manipulator]:
        """Retrieve the ``Manipulator`` annotation for *arm* from the world."""
        try:
            robots = world.get_semantic_annotations_by_type(AbstractRobot)
            if not robots:
                return None
            robot = robots[0]
            if hasattr(robot, "get_manipulator_for_arm"):
                return robot.get_manipulator_for_arm(arm)
            if hasattr(robot, "manipulators"):
                manipulators = robot.manipulators
                if isinstance(manipulators, dict):
                    return manipulators.get(arm)
                if isinstance(manipulators, list) and manipulators:
                    return (
                        manipulators[arm.value]
                        if arm.value < len(manipulators)
                        else manipulators[0]
                    )
        except Exception as exc:
            logger.warning("Could not retrieve manipulator: %s", exc)
        return None

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        instruction: str,
        exec_state: Optional["ExecutionState"] = None,
    ) -> Any:
        """Execute the full pipeline for *any* supported action type.

        :param instruction: Natural language instruction.
        :param exec_state: Optional robot/world state snapshot.  When provided,
            arm occupancy is included in the LLM world-context string so the
            slot-filler and resolver can make state-aware decisions (e.g. knowing
            which arm is already holding an object).
        :return: Fully specified, executable pycram action.
        :raises RuntimeError: On unrecoverable failures in any stage.
        """
        logger.info("ActionPipeline.run: '%s'", instruction)
        schema = self.classify_and_extract(instruction, exec_state=exec_state)
        if schema is None:
            raise RuntimeError("Slot-filler failed.  Check LLM connectivity and API keys.")
        return self.dispatch(schema)

    # ── Step-by-step accessors (for debugging / notebooks) ────────────────────

    def classify_and_extract(
        self,
        instruction: str,
        exec_state: Optional["ExecutionState"] = None,
    ) -> Optional[ActionSlotSchema]:
        """Phase 1: NL instruction → typed slot schema (classify + extract).

        :param exec_state: Optional execution state.  When provided, robot arm
            occupancy is appended to the world context string passed to the LLM.
        :return: ``PickUpSlotSchema`` or ``PlaceSlotSchema``; ``None`` on failure.
        """
        world_ctx_str = _serialise_world_for_llm(self.world, exec_state)
        logger.debug("World context for slot filling:\n%s", world_ctx_str)
        schema = run_slot_filler(instruction=instruction, world_context=world_ctx_str)
        if schema is not None:
            logger.info(
                "classify_and_extract – action_type=%s, object='%s', arm=%s",
                schema.action_type,
                schema.object_description.name,
                schema.arm,
            )
        return schema

    def dispatch(self, schema: ActionSlotSchema) -> Any:
        """Phase 2: typed slot schema → concrete pycram action (ground + resolve).

        :param schema: Output of ``classify_and_extract()``.
        :return: Fully specified pycram action.
        """
        dispatcher = ActionDispatcher(world=self.world, world_context=self.world_context)
        action = dispatcher.dispatch(schema)
        logger.info("ActionPipeline.dispatch complete – %s resolved.", schema.action_type)
        return action
