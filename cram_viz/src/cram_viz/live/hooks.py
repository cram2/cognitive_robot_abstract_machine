"""
Hooks that attach the bridge to a running coraplex/giskardpy demo.

Each hook installs a :class:`LiveHooks` method in place of one CRAM method; that
method forwards what it observes to the bridge, and all of the interpretation lives
there. Installing is idempotent, so the two documented ways of starting live mode can
be combined safely.

.. warning:: Every world access (forward kinematics for poses etc.) happens on
   the *simulation* thread itself, inside the ``Executor.tick`` hook. Reading
   the world from a separate sampler thread corrupts the native solver's heap
   — the HTTP handlers therefore only ever serve the last finished snapshot
   dict.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Callable

from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.plan import Plan
from giskardpy.executor import Executor
from semantic_digital_twin.adapters.mesh import MeshParser
from semantic_digital_twin.world import World

from cram_viz import get_logger
from cram_viz.live.bridge import BRIDGE, Bridge, LiveHook, TaskStatusName
from cram_viz.monkey_patch import MethodPatch

logger = get_logger(__name__)


# %% wrappers forwarded to the bridge
@dataclass(frozen=True)
class LiveHooks:
    """
    Thin per-CRAM-method wrappers that forward what they observe to a bridge.

    Kept as real methods, rather than as closures defined inside the ``install_*``
    functions, so each one is independently callable and testable.
    """

    bridge: Bridge
    """
    The bridge every wrapper forwards its observations to.
    """

    def _observe_tick(
        self,
        original: Callable[..., None],
        executor: Executor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Run the real tick, then let the bridge observe its result.

        :param original: The real, unpatched ``Executor.tick`` bound method.
        :param executor: The executor whose tick is being run.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        result = original(executor, *args, **kwargs)
        if self.bridge.world is None:
            self.bridge.attach(executor.context.world)
        try:
            self.bridge.observe_tick(executor.motion_statechart)
        except Exception:
            # boundary guard: a visualization bug must never take the robot demo
            # down, so this hook swallows everything the bridge raises
            logger.exception("bridge snapshot failed this tick")
        return result

    def _begin_plan(
        self, original: Callable[..., Any], plan: Plan, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Capture the plan the moment it starts performing.

        :param original: The real, unpatched ``Plan.perform`` bound method.
        :param plan: The plan starting to perform.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        self.bridge.begin_plan(plan)
        return original(plan, *args, **kwargs)

    def _track_motion_group(
        self,
        original: Callable[..., None],
        executable: GiskardExecutable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Track this executable's motion group and freeze its final status.

        :param original: The real, unpatched ``GiskardExecutable.execute`` bound method.
        :param executable: The executable whose motion group is tracked.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        self.bridge.bind_motion_group(executable)
        try:
            result = original(executable, *args, **kwargs)
        except BaseException:
            self.bridge.freeze_motion_group(executable, TaskStatusName.FAILED)
            raise
        self.bridge.freeze_motion_group(executable, TaskStatusName.SUCCEEDED)
        return result

    def _remember_mesh_file(
        self,
        original: Callable[..., World],
        mesh_parser: MeshParser,
        *args: Any,
        **kwargs: Any,
    ) -> World:
        """
        Remember this mesh's file path before parsing it.

        :param original: The real, unpatched ``MeshParser.parse`` bound method.
        :param mesh_parser: The parser about to parse the mesh.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if mesh_parser.file_path:
            self.bridge.remember_mesh_file(mesh_parser.file_path)
        return original(mesh_parser, *args, **kwargs)


_LIVE_HOOKS = LiveHooks(bridge=BRIDGE)


# %% world and motion
def install_tick_hook() -> None:
    """
    Bind the bridge to the executing world and snapshot on every simulation tick.
    """
    if not BRIDGE.claim_hook(LiveHook.TICK):
        logger.debug("tick hook already installed")
        return
    MethodPatch(Executor, "tick").install(_LIVE_HOOKS._observe_tick)


# %% the coraplex plan
def install_plan_hooks() -> None:
    """
    Follow the coraplex plan: which plan executes, and which plan nodes the currently
    running giskard executable belongs to.

    Both hooks fire on the thread that runs the plan (the same one that ticks the
    executor), so they may touch plan objects directly.
    """
    if not BRIDGE.claim_hook(LiveHook.PLAN):
        logger.debug("plan hooks already installed")
        return
    MethodPatch(Plan, "perform").install(_LIVE_HOOKS._begin_plan)
    MethodPatch(GiskardExecutable, "execute").install(_LIVE_HOOKS._track_motion_group)


# %% object geometry
def install_mesh_hook() -> None:
    """
    Remember every mesh an object is built from, so the bridge can serve its geometry to
    the viewer.

    All mesh formats go through ``MeshParser.parse``; the body name matches the mesh
    file's basename.
    """
    if not BRIDGE.claim_hook(LiveHook.MESH):
        logger.debug("mesh hook already installed")
        return
    MethodPatch(MeshParser, "parse").install(_LIVE_HOOKS._remember_mesh_file)
