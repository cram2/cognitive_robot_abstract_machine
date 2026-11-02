"""
Tests for holding still before a statechart blocks its tick: a node choosing its child
or a compile only happens once everything the executor drives is at rest, and the
statechart keeps ticking until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
    RepeatUntil,
)
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import (
    ExecutorExtension,
    StatechartExecutor,
    TickingExecutorExtension,
)
from cramph.monitors import CountNodeResets
from cramph.node import EndStatechart, StatechartNode
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.statechart import Statechart

# %% mimics


@dataclass
class ExtensionComingToRest(TickingExecutorExtension):
    """
    An executor extension that reports rest only after it was asked to hold still a few
    times, the way a decelerating robot does.
    """

    questions_until_at_rest: int
    """
    How many more times it answers that it is not at rest yet.
    """

    questions: int = 0
    """
    How often it was asked to hold still.
    """

    def before_recompile(self, executor: StatechartExecutor) -> bool:
        self.questions += 1
        if self.questions_until_at_rest == 0:
            return True
        self.questions_until_at_rest -= 1
        return False


@dataclass
class ChooserNotingTheTick(ChildChooser[CompositeNodeChoosingItsChild]):
    """
    Chooses one prepared child, noting the tick it was asked on.
    """

    executor: StatechartExecutor
    """
    The executor whose ticks are counted.
    """

    child: StatechartNode
    """
    The child to choose.
    """

    asked_on_ticks: list[int] = field(default_factory=list)
    """
    The tick of every question.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        self.asked_on_ticks.append(self.executor.tick_count)
        return self.child


@dataclass
class ExtensionRecordingCompiles(TickingExecutorExtension):
    """
    An executor extension that records in which order it was told about compiling.
    """

    events: list[str] = field(default_factory=list)
    """
    The hooks that ran, in order, each by its method name.
    """

    def before_recompile(self, executor: StatechartExecutor) -> bool:
        self.events.append(self.before_recompile.__name__)
        return True

    def after_compile(self, executor: StatechartExecutor) -> None:
        self.events.append(self.after_compile.__name__)


def _choosing_statechart(
    statechart_context: StatechartContext, extension: ExecutorExtension
) -> tuple[StatechartExecutor, ChooserNotingTheTick, CompositeNodeChoosingItsChild]:
    """
    :return: An executor with `extension`, the chooser its context asks and the node
        choosing its child, compiled.
    """
    executor = StatechartExecutor(context=statechart_context, extensions=[extension])
    chooser = ChooserNotingTheTick(
        executor=executor,
        child=NodeSucceedingOnObservingTrue(
            name="child", observation=ObservationStateValues.TRUE
        ),
    )
    executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=executor.context)
    choosing = CompositeNodeChoosingItsChild(name="choosing")
    statechart.add_node(choosing)
    executor.compile(statechart)
    return executor, chooser, choosing


# %% choosing only at rest


def test_a_node_chooses_its_child_only_once_the_extensions_are_at_rest(
    statechart_context,
) -> None:
    extension = ExtensionComingToRest(questions_until_at_rest=3)
    executor, chooser, _ = _choosing_statechart(statechart_context, extension)

    for _ in range(5):
        executor.tick()

    assert chooser.asked_on_ticks == [3]


def test_the_statechart_keeps_ticking_while_the_extensions_come_to_rest(
    statechart_context,
) -> None:
    extension = ExtensionComingToRest(questions_until_at_rest=3)
    executor, chooser, choosing = _choosing_statechart(statechart_context, extension)

    executor.tick()
    executor.tick()

    assert executor.tick_count == 2
    assert choosing.children == []
    assert extension.questions == 3


def test_the_extensions_are_asked_to_hold_still_once_per_choice(
    statechart_context,
) -> None:
    """
    Choosing the child compiles the statechart again, which must not ask a second time.
    """
    extension = ExtensionRecordingCompiles()

    _choosing_statechart(statechart_context, extension)

    assert extension.events == [
        ExtensionRecordingCompiles.after_compile.__name__,
        ExtensionRecordingCompiles.before_recompile.__name__,
        ExtensionRecordingCompiles.after_compile.__name__,
    ]


# %% resetting while not at rest


@dataclass
class ExtensionBrakingOnceAskedToHoldStill(TickingExecutorExtension):
    """
    An executor extension that starts moving after the first tick and comes to rest on
    the tick after it was first asked to hold still while moving, the way a robot
    braking does.

    It counts every compile it is told about while not at rest.
    """

    is_at_rest: bool = True
    """
    Whether what this extension drives is at rest.
    """

    has_started_moving: bool = False
    """
    Whether what this extension drives started moving.
    """

    is_braking: bool = False
    """
    Whether it was asked to hold still while moving, and comes to rest after this tick.
    """

    compiles_while_not_at_rest: int = 0
    """
    How often the statechart compiled again while this extension was not at rest.
    """

    def before_recompile(self, executor: StatechartExecutor) -> bool:
        if not self.is_at_rest:
            self.is_braking = True
        return self.is_at_rest

    def after_recompile(self, executor: StatechartExecutor) -> None:
        if not self.is_at_rest:
            self.compiles_while_not_at_rest += 1

    def after_tick(self, executor: StatechartExecutor) -> None:
        if self.is_braking:
            self.is_at_rest = True
            self.is_braking = False
        elif not self.has_started_moving:
            self.is_at_rest = False
            self.has_started_moving = True


@dataclass
class ChooserWithoutChildren(ChildChooser[CompositeNodeChoosingItsChild]):
    """
    Says that no child is left whenever it is asked.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        return None


def test_resetting_a_node_choosing_its_child_compiles_nothing_while_moving(
    statechart_context,
) -> None:
    """
    A node that ran out of children is reset to choose again, which must not compile the
    statechart while what the executor drives is not at rest.
    """
    extension = ExtensionBrakingOnceAskedToHoldStill()
    executor = StatechartExecutor(context=statechart_context, extensions=[extension])
    executor.context.add_extension(
        ChildChooserAccess(choosers=[ChooserWithoutChildren()])
    )
    statechart = Statechart(context=executor.context)
    choosing = CompositeNodeChoosingItsChild(name="choosing")
    loop = RepeatUntil(
        name="loop",
        task=choosing,
        stop_retry_monitor=CountNodeResets(name="resets", node=choosing, target=2),
    )
    statechart.add_node(loop)
    statechart.add_node(EndStatechart.when_failed(loop))
    executor.compile(statechart)

    executor.tick_until_end(timeout=60)

    assert loop.life_cycle_state == LifeCycleValues.FAILED
    assert extension.compiles_while_not_at_rest == 0
