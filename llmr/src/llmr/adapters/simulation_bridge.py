"""Connects the llmr CRAM serializer to a live ``semantic_digital_twin.world.World`` for body
lookup, plan construction, and action execution. Pipeline: ad_graph → CRAMToPyCRAMSerializer
→ SimulationBridge → SequentialPlan.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .cram_to_pycram import CRAMEntityInfo

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Body resolver
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_names(entity: CRAMEntityInfo) -> list[str]:
    """Return tag first, then semantic_type as candidate name list."""
    names = []
    if entity.tag:
        names.append(entity.tag)
    if entity.semantic_type and entity.semantic_type != entity.tag:
        names.append(entity.semantic_type)
    return names


def _make_world_body_resolver(world: Any):
    """Return a resolver that maps a CRAMEntityInfo to a Body in *world*."""
    def _resolver(entity: CRAMEntityInfo) -> Optional[Any]:
        if not entity:
            return None
        candidates = _candidate_names(entity)
        if not candidates:
            return None

        for name in candidates:
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                body_name = str(raw.name) if hasattr(raw, "name") else str(raw)
                if body_name == name or body_name.lower() == name.lower():
                    logger.debug("[resolver] Resolved '%s' → %r", name, body)
                    return body

        for name in candidates:
            name_lower = name.lower()
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                body_name = (str(raw.name) if hasattr(raw, "name") else str(raw)).lower()
                if name_lower in body_name or body_name in name_lower:
                    logger.debug("[resolver] Resolved '%s' (substring) → %r", name, body)
                    return body

        for name in candidates:
            normalized = name.replace("_", "").replace("-", "").lower()
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                body_name = (str(raw.name) if hasattr(raw, "name") else str(raw)).lower()
                body_normalized = body_name.replace("_", "").replace("-", "")
                if normalized == body_normalized or normalized in body_normalized or body_normalized in normalized:
                    logger.debug("[resolver] Resolved '%s' (normalized) → %r", name, body)
                    return body

        if entity.semantic_type:
            sem_lower = entity.semantic_type.lower()
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                body_name = (str(raw.name) if hasattr(raw, "name") else str(raw)).lower()
                if sem_lower in body_name:
                    logger.debug("[resolver] Resolved '%s' (fuzzy) → %r", entity.semantic_type, body)
                    return body

        logger.debug("[resolver] Could not resolve '%s' in world.", entity)
        return None

    return _resolver


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _body_name(body: Any) -> str:
    """Return the string name of a ``Body`` object tolerantly."""
    raw = getattr(body, "name", None)
    if raw is None:
        return ""
    # PrefixedName objects stringify to 'prefix:local'; use .name or str()
    if hasattr(raw, "name"):
        return str(raw.name)
    return str(raw)


def _body_pose(body: Any) -> Optional[Any]:
    """Return the global pose of *body*, or ``None`` on failure."""
    try:
        return body.global_pose
    except Exception:
        pass
    try:
        return body.pose
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SimulationBridge
# ─────────────────────────────────────────────────────────────────────────────

class SimulationBridge:
    """Connects CRAM serialization to a live PyCRAM / SemanticDigitalTwin world."""

    def __init__(self, world: Any, robot: Any, ros_node: Any = None) -> None:
        self._world = world
        self._robot = robot
        self._ros_node = ros_node

        # Lazy-import heavy PyCRAM / serializer objects so the module remains
        # importable even without a full PyCRAM installation.
        self._serializer = self._make_serializer()
        self._context = self._make_context()

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def world(self) -> Any:
        """The injected ``World`` object."""
        return self._world

    @property
    def robot(self) -> Any:
        """The injected ``AbstractRobot`` object."""
        return self._robot

    @property
    def context(self) -> Any:
        """The ``Context`` dataclass used to execute plans.

        Re-create it after swapping ``world`` or ``robot`` via
        :meth:`update_world` / :meth:`update_robot`.
        """
        return self._context

    # ── World introspection ────────────────────────────────────────────────

    def list_bodies(self) -> List[str]:
        """Return the names of all bodies currently in the world."""
        return [_body_name(b) for b in getattr(self._world, "bodies", [])]

    def find_body(self, name: str) -> Optional[Any]:
        """Find a Body by name (case-insensitive)."""
        name_lower = name.lower()
        for body in getattr(self._world, "bodies", []):
            bname = _body_name(body)
            if bname == name or bname.lower() == name_lower:
                return body
        return None

    def snapshot(self) -> Dict[str, Any]:
        """Return ``{body_name: pose}`` for every body in the world."""
        result: Dict[str, Any] = {}
        for body in getattr(self._world, "bodies", []):
            bname = _body_name(body)
            result[bname] = _body_pose(body)
        return result

    # ── Runtime world / robot update ───────────────────────────────────────

    def update_world(self, world: Any) -> None:
        """Swap the injected world and rebuild the context."""
        self._world = world
        self._context = self._make_context()
        logger.info("SimulationBridge: world updated to %r", world)

    def update_robot(self, robot: Any) -> None:
        """Swap the robot reference and rebuild the internal context."""
        self._robot = robot
        self._context = self._make_context()
        logger.info("SimulationBridge: robot updated to %r", robot)

    # ── Body resolver ──────────────────────────────────────────────────────

    def make_resolver(self) -> Any:
        """Return a BodyResolver bound to the current world."""
        return _make_world_body_resolver(self._world)

    # ── Serialization helpers ─────────────────────────────────────────────

    def parse(self, cram_string: str) -> Any:
        """Parse a CRAM string into a CRAMActionPlan (no PyCRAM required)."""
        return self._serializer.parse(cram_string)

    def to_partial_designator(
        self,
        cram_string: str,
        arm: Any = None,
        grasp_description: Any = None,
        approach_from: Any = None,
    ) -> Any:
        """Parse a CRAM string and return a PyCRAM PartialDesignator (not yet executed)."""
        resolver = self.make_resolver()
        plan = self._serializer.parse(cram_string)

        # Resolve the effective arm (default to RIGHT when not specified)
        effective_arm = arm
        if effective_arm is None:
            try:
                from pycram.datastructures.enums import Arms
                effective_arm = Arms.RIGHT
            except ImportError:
                pass

        # Auto-compute grasp description from the object body when not provided
        if grasp_description is None and plan.object is not None:
            try:
                from pycram.datastructures.grasp import GraspDescription
                from pycram.datastructures.pose import PoseStamped
                from pycram.view_manager import ViewManager

                object_body = resolver(plan.object)
                if object_body is not None and effective_arm is not None:
                    arm_view = ViewManager.get_arm_view(effective_arm, self._robot)
                    manipulator = arm_view.manipulator
                    object_pose = PoseStamped.from_spatial_type(object_body.global_pose)
                    grasp_descs = GraspDescription.calculate_grasp_descriptions(
                        manipulator, object_pose
                    )
                    if grasp_descs:
                        grasp_description = grasp_descs[0]
                        logger.info(
                            "Auto-computed grasp: approach=%s vertical=%s",
                            grasp_description.approach_direction.name,
                            grasp_description.vertical_alignment.name,
                        )
            except Exception as exc:
                logger.warning("Could not auto-compute grasp description: %s", exc)

        kwargs: Dict[str, Any] = {}
        if effective_arm is not None:
            kwargs["arm"] = effective_arm
        if grasp_description is not None:
            kwargs["grasp_description"] = grasp_description
        if approach_from is not None:
            kwargs["approach_from"] = approach_from

        partial = self._serializer.to_partial_designator(
            plan, body_resolver=resolver, **kwargs
        )
        logger.info(
            "to_partial_designator: action=%s → %r",
            plan.action_type, partial,
        )
        return partial

    def execute(
        self,
        cram_string: str,
        arm: Any = None,
        grasp_description: Any = None,
    ) -> Any:
        """Parse, resolve bodies, and execute a CRAM plan string."""
        partial = self.to_partial_designator(
            cram_string, arm=arm, grasp_description=grasp_description
        )

        try:
            from pycram.language import SequentialPlan
        except ImportError as exc:
            raise RuntimeError(
                "PyCRAM must be installed to execute plans. "
                "Use to_partial_designator() to obtain the designator and "
                "integrate it into your own plan structure."
            ) from exc

        seq_plan = SequentialPlan(self._context, partial)
        logger.info("Executing plan via SequentialPlan …")
        result = seq_plan.perform()
        logger.info("Plan execution finished, result=%r", result)
        return result

    def execute_batch(
        self,
        cram_strings: List[str],
        arm: Any = None,
    ) -> List[Any]:
        """Execute a list of CRAM strings sequentially; auto-injects NavigateAction for placement steps."""
        if not cram_strings:
            return []

        try:
            from pycram.language import SequentialPlan
        except ImportError as exc:
            raise RuntimeError(
                "PyCRAM must be installed to execute plans."
            ) from exc

        # Placement-type action names that require the robot to navigate to the
        # goal location before the action can succeed (same as TransportAction).
        _PLACEMENT_TYPES = {
            "placing", "place", "placeaction",
            "putobject", "transport", "transportaction",
            "pickandplace", "moveandplace", "moveandplaceaction",
        }

        # Resolve the effective arm once (used both for grasp and CostmapLocation)
        effective_arm = arm
        if effective_arm is None:
            try:
                from pycram.datastructures.enums import Arms
                effective_arm = Arms.RIGHT
            except ImportError:
                pass

        resolver = self.make_resolver()

        # Build all designators first, then run them in ONE SequentialPlan so
        # that multi-step actions (e.g. PlaceAction) can look back in the plan
        # tree and find before steps (e.g. the earlier PickUpAction).
        partials = []
        for i, cram_str in enumerate(cram_strings):
            logger.info("execute_batch: building step %d/%d", i + 1, len(cram_strings))

            # Parse the CRAM string to detect placement actions
            plan = self._serializer.parse(cram_str)
            action_norm = (
                plan.action_type.lower().replace("_", "").replace(" ", "")
                if plan.action_type else ""
            )

            placement_nav_pose = None  # will be set for PlaceAction
            if action_norm in _PLACEMENT_TYPES and plan.goal is not None:
                # Auto-inject a NavigateAction to a collision-free base pose near
                # the placement target, resolved before PickUp (milk not yet attached).
                goal_body = resolver(plan.goal)
                if goal_body is not None:
                    placement_nav_pose = self._resolve_placement_nav_pose(goal_body, effective_arm)
                    if placement_nav_pose is not None:
                        from pycram.robot_plans import NavigateActionDescription
                        partials.append(NavigateActionDescription(placement_nav_pose))
                        logger.info(
                            "execute_batch: injected NavigateAction to (%.2f, %.2f) "
                            "before %s",
                            placement_nav_pose.pose.position.x,
                            placement_nav_pose.pose.position.y,
                            plan.action_type,
                        )
                    else:
                        logger.warning(
                            "execute_batch: could not resolve placement nav pose for %r; "
                            "PlaceAction may fail if robot is too far from target.",
                            goal_body,
                        )

            # For PlaceAction, pass nav_pose as approach_from so the placement
            # target is the near edge of the surface (reachable by the arm)
            # rather than the table centre which may be out of reach.
            partials.append(
                self.to_partial_designator(cram_str, arm=arm, approach_from=placement_nav_pose)
            )

        logger.info("execute_batch: executing %d steps in one SequentialPlan", len(partials))
        seq_plan = SequentialPlan(self._context, *partials)
        result = seq_plan.perform()
        # Return a list with one entry per step for API compatibility
        return [result] * len(cram_strings)

    # ── helpers ────────────────────────────────────────────────────

    def _resolve_placement_nav_pose(self, goal_body: Any, arm: Any) -> Any:
        """Return an IK-validated navigation PoseStamped near *goal_body*, or ``None`` on failure."""
        try:
            from pycram.datastructures.pose import PoseStamped
            from pycram.designators.location_designator import CostmapLocation
        except ImportError:
            logger.warning("_resolve_placement_nav_pose: pycram not available.")
            return None

        try:
            import types as _types
            goal_ps = PoseStamped.from_spatial_type(goal_body.global_pose)
            stub = _types.SimpleNamespace(
                plan=_types.SimpleNamespace(world=self._world, robot=self._robot)
            )
            nav_loc = CostmapLocation(
                target=goal_ps,
                reachable_for=self._robot,
                reachable_arm=arm,
            )
            nav_loc.plan_node = stub
            result = next(iter(nav_loc))
            nav_loc.plan_node = None
            return result.pose
        except Exception as exc:
            logger.warning(
                "_resolve_placement_nav_pose: CostmapLocation failed for %r — reason: %s",
                goal_body, exc,
            )
            return None

    def _make_serializer(self) -> Any:
        """Instantiate ``CRAMToPyCRAMSerializer`` from the sibling module."""
        from .cram_to_pycram import CRAMToPyCRAMSerializer
        return CRAMToPyCRAMSerializer()

    def _make_context(self) -> Any:
        """Build a PyCRAM ``Context`` from the injected world + robot.

        Returns ``None`` if ``pycram`` is not installed (parse-only mode).
        """
        try:
            from pycram.datastructures.dataclasses import Context
            ctx = Context(
                world=self._world,
                robot=self._robot,
                ros_node=self._ros_node,
            )
            logger.debug("SimulationBridge: Context built %r", ctx)
            return ctx
        except ImportError:
            logger.warning(
                "pycram not installed — Context not created. "
                "execute() will fail; parse-only mode is available."
            )
            return None


