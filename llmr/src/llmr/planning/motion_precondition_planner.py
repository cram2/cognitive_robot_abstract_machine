
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Any, Dict, List, Optional, Type, Union

from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from semantic_digital_twin.spatial_types.spatial_types import Pose

from pycram.datastructures.enums import Arms
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.designators.location_designator import CostmapLocation, SemanticCostmapLocation
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction

logger = logging.getLogger(__name__)


# ── Shared execution state ──────────────────────────────────────────────────────


@dataclass
class ExecutionState:
    """Mutable state carried across action executions within one instruction sequence."""

    last_pickup_arm: Optional[Arms] = None
    """Arm used in the most recent PickUpAction."""

    last_pickup_body: Optional[Body] = None
    """Object body grasped in the most recent PickUpAction."""

    held_objects: Dict[Arms, Optional[Body]] = field(default_factory=dict)
    """Per-arm occupancy: maps Arms.LEFT / Arms.RIGHT → held Body (or None if empty)."""

    def copy(self) -> "ExecutionState":
        """Return a shallow copy suitable for use as a planning-phase snapshot."""
        return ExecutionState(
            last_pickup_arm=self.last_pickup_arm,
            last_pickup_body=self.last_pickup_body,
            held_objects=dict(self.held_objects),
        )


# ── Precondition result ─────────────────────────────────────────────────────────


@dataclass
class PreconditionResult:
    """Output of a PreconditionProvider."""

    preconditions: List[ActionDescription]
    action: ActionDescription


# ── PreconditionProvider base ───────────────────────────────────────────────────


@dataclass
class PreconditionProvider(ABC):
    """Base class for action-type-specific precondition logic."""

    world: World

    def _get_robot(self) -> Optional[AbstractRobot]:
        """Return the first AbstractRobot annotation found in the world, or None."""
        try:
            robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
            return robots[0] if robots else None
        except Exception:
            return None

    @abstractmethod
    def compute(
        self,
        action: ActionDescription,
        exec_state: ExecutionState,
    ) -> PreconditionResult:
        """Compute preconditions for *action* given the current *exec_state*.

        :param action: The task action that is about to be executed.
        :param exec_state: Accumulated state from previous actions.
        :return: Preconditions to prepend and (possibly updated) action.
        """

    def update_state(
        self,
        action: ActionDescription,
        exec_state: ExecutionState,
    ) -> None:
        """Update *exec_state* after *action* has been executed.

        Override in subclasses that need to persist information for later actions.
        Default implementation is a no-op.
        """


# ── PickUp precondition provider ───────────────────────────────────────────────


@dataclass
class PickUpPreconditionProvider(PreconditionProvider):
    """Prepares the robot to pick up an object (park arms, raise torso, navigate)."""

    def compute(
        self,
        action: PickUpAction,
        exec_state: ExecutionState,
    ) -> PreconditionResult:
        robot = self._get_robot()

        # Correct the arm if the LLM chose one that is already occupied.
        action = self._ensure_free_arm(action, exec_state)

        nav_designator = self._make_nav_designator(action.object_designator, action.arm, robot)

        preconditions: List[Any] = [
            MoveTorsoAction(torso_state=TorsoState.HIGH),
        ]
        # Only park arms that are not currently holding an object — parking a
        # loaded arm would drop whatever is being held.
        arm_to_park = self._free_arms(exec_state)
        if arm_to_park is not None:
            preconditions.insert(0, ParkArmsAction(arm=arm_to_park))

        if nav_designator is not None:
            # Use NavigateAction.description (PartialDesignator) so the plan propagates
            # plan_node into the nested CostmapLocation before it tries to access robot_view.
            preconditions.append(nav_designator)

        return PreconditionResult(preconditions=preconditions, action=action)

    @staticmethod
    def _ensure_free_arm(action: PickUpAction, exec_state: ExecutionState) -> PickUpAction:
        """Return *action* unchanged, or with arm corrected to the free arm.

        The LLM may choose an arm based on spatial position (e.g. object is to
        the right → RIGHT) without knowing that arm is already holding something.
        This method overrides that choice deterministically when needed.
        """
        arm = action.arm
        if arm is None or exec_state.held_objects.get(arm) is None:
            return action  # arm is free or unspecified — nothing to fix

        other = Arms.LEFT if arm == Arms.RIGHT else Arms.RIGHT
        if exec_state.held_objects.get(other) is not None:
            logger.warning(
                "PickUp: both arms occupied — cannot correct arm choice, proceeding with %s.",
                arm.name,
            )
            return action

        logger.info(
            "PickUp: planned arm %s is holding '%s' — switching to free arm %s.",
            arm.name,
            getattr(
                getattr(exec_state.held_objects[arm], "name", None),
                "name",
                exec_state.held_objects[arm],
            ),
            other.name,
        )
        return PickUpAction(
            object_designator=action.object_designator,
            arm=other,
            grasp_description=action.grasp_description,
        )

    @staticmethod
    def _free_arms(exec_state: ExecutionState) -> Optional[Arms]:
        """Return which arm(s) are safe to park (not holding anything).

        Returns ``Arms.BOTH`` when neither arm is occupied, the single free arm
        when only one is held, or ``None`` when both arms are occupied.
        """
        if not exec_state.held_objects:
            return Arms.BOTH
        left_held = exec_state.held_objects.get(Arms.LEFT) is not None
        right_held = exec_state.held_objects.get(Arms.RIGHT) is not None
        if not left_held and not right_held:
            return Arms.BOTH
        if not left_held:
            return Arms.LEFT
        if not right_held:
            return Arms.RIGHT
        return None  # both occupied

    def _make_nav_designator(
        self,
        obj_body: Body,
        arm: Arms,
        robot: Optional[AbstractRobot],
    ) -> Optional[PartialDesignator]:
        """Return a NavigateAction PartialDesignator whose target is a CostmapLocation.

        CostmapLocation.ground() requires robot_view (i.e. plan_node must be set).
        By wrapping it in a PartialDesignator and letting the plan resolve it, the
        plan's plan_node propagation wires up robot_view automatically.

        The target is converted to a PoseStamped so CostmapLocation computes
        reachable base poses from the object's exact world position rather than
        its bounding box geometry (which can shift the nav pose out of reach
        for objects that are off-center relative to the robot).
        """
        target = obj_body.global_pose
        loc = CostmapLocation(
            target=target,
            reachable_for=robot,
            reachable_arm=arm,
        )
        return NavigateAction.description(target_location=loc)

    def update_state(
        self,
        action: PickUpAction,
        exec_state: ExecutionState,
    ) -> None:
        exec_state.last_pickup_arm = action.arm
        exec_state.last_pickup_body = action.object_designator
        exec_state.held_objects[action.arm] = action.object_designator


# ── Place precondition provider ─────────────────────────────────────────────────


@dataclass
class PlacePreconditionProvider(PreconditionProvider):
    """Prepares the robot to place a held object (raise torso, navigate, resolve pose and arm)."""

    def compute(
        self,
        action: PlaceAction,
        exec_state: ExecutionState,
    ) -> PreconditionResult:
        robot = self._get_robot()

        # Look up which arm is holding this specific object.  Falls back to
        # last_pickup_arm (single-pickup case) and then to the LLM's choice.
        obj_body = action.object_designator or exec_state.last_pickup_body
        arm = self._find_holding_arm(obj_body, exec_state) or action.arm

        # Resolve the placement pose (Body → PoseStamped on surface).
        # SemanticCostmapLocation does NOT access robot_view, so .ground() is safe here.
        place_pose = self._resolve_place_pose(action.target_location, obj_body)

        # Build a NavigateAction PartialDesignator whose CostmapLocation will be
        # resolved lazily inside the plan (plan_node propagation sets robot_view).
        nav_designator = self._make_nav_designator(place_pose, arm, robot)

        preconditions: List[Any] = [
            MoveTorsoAction(torso_state=TorsoState.HIGH),
        ]
        if nav_designator is not None:
            preconditions.append(nav_designator)

        # Return an updated PlaceAction with the resolved pose and consistent arm.
        updated_action = PlaceAction(
            object_designator=action.object_designator,
            target_location=place_pose,
            arm=arm,
        )
        return PreconditionResult(preconditions=preconditions, action=updated_action)

    @staticmethod
    def _find_holding_arm(
        object_body: Optional[Body],
        exec_state: ExecutionState,
    ) -> Optional[Arms]:
        """Return the arm currently holding *object_body*.

        Checks ``exec_state.held_objects`` by object identity first (handles
        multi-pickup sequences correctly), then falls back to ``last_pickup_arm``
        for state objects created before ``held_objects`` was populated.
        """
        if object_body is not None:
            for arm, held in exec_state.held_objects.items():
                if held is object_body:
                    return arm
        return exec_state.last_pickup_arm

    def update_state(
        self,
        action: PlaceAction,
        exec_state: ExecutionState,
    ) -> None:
        # The arm is now empty after placing.
        arm = action.arm
        if arm is not None:
            exec_state.held_objects[arm] = None
        # Clear last-pickup tracking if the placed object was the one picked up.
        if action.object_designator is exec_state.last_pickup_body:
            exec_state.last_pickup_arm = None
            exec_state.last_pickup_body = None

    def _resolve_place_pose(
        self,
        target: Any,
        for_object: Optional[Body],
    ) -> Pose:
        """Convert a placement target to a Pose.

        If *target* is already a ``Pose`` it is returned as-is.
        If it is a ``Body`` (as returned by the ActionDispatcher), the pose is
        computed by ``SemanticCostmapLocation`` so the object rests on the surface.
        SemanticCostmapLocation does not access robot_view, so this is safe outside a plan.
        """
        if isinstance(target, Pose):
            return target

        # target is a Body – compute placement pose on its surface.
        try:
            sem_loc = SemanticCostmapLocation(body=target, for_object=for_object)
            grd = sem_loc.ground()
            logger.debug("SemanticCostmapLocation Body: %s", target)
            logger.debug("SemanticCostmapLocation for object: %s", for_object)
            logger.debug("SemanticCostmapLocation: %s", grd)
            return grd
        except Exception as exc:
            logger.warning(
                "SemanticCostmapLocation failed (%s). Falling back to body global pose.", exc
            )
            return target.global_pose

    def _make_nav_designator(
        self,
        place_pose: Pose,
        arm: Arms,
        robot: Optional[AbstractRobot],
    ) -> Optional[PartialDesignator]:
        """Return a NavigateAction PartialDesignator whose target is a CostmapLocation.

        CostmapLocation.ground() requires robot_view (plan_node must be set).
        Wrapping in a PartialDesignator defers resolution until plan execution,
        at which point the plan propagates plan_node into the CostmapLocation.
        """
        loc = CostmapLocation(
            target=place_pose,
            reachable_for=robot,
            reachable_arm=arm,
        )
        return NavigateAction.description(target_location=loc)


# ── MotionPreconditionPlanner ───────────────────────────────────────────────────


class MotionPreconditionPlanner:
    """Registry-based planner mapping action types to their precondition providers."""

    _registry: Dict[Type[ActionDescription], Type[PreconditionProvider]] = {}

    @classmethod
    def register(
        cls,
        action_type: Type[ActionDescription],
        provider_class: Type[PreconditionProvider],
    ) -> None:
        """Register *provider_class* as the precondition provider for *action_type*."""
        cls._registry[action_type] = provider_class
        logger.debug(
            "MotionPreconditionPlanner: registered provider '%s' for '%s'.",
            provider_class.__name__,
            action_type.__name__,
        )

    def __init__(self, world: World) -> None:
        self._world = world

    def compute(
        self,
        action: ActionDescription,
        exec_state: ExecutionState,
    ) -> PreconditionResult:
        """Compute preconditions for *action* given the current *exec_state*.

        If no provider is registered for the action type, returns an empty
        preconditions list and the original action unchanged.
        """
        provider_cls = self._registry.get(type(action))
        if provider_cls is None:
            logger.debug(
                "No precondition provider registered for '%s' – skipping.",
                type(action).__name__,
            )
            return PreconditionResult(preconditions=[], action=action)

        provider = provider_cls(self._world)
        return provider.compute(action, exec_state)

    def update_state(
        self,
        action: ActionDescription,
        exec_state: ExecutionState,
    ) -> None:
        """Update *exec_state* after *action* has been performed."""
        provider_cls = self._registry.get(type(action))
        if provider_cls is not None:
            provider_cls(self._world).update_state(action, exec_state)


# ── Default registrations ───────────────────────────────────────────────────────

MotionPreconditionPlanner.register(PickUpAction, PickUpPreconditionProvider)
MotionPreconditionPlanner.register(PlaceAction, PlacePreconditionProvider)
