from __future__ import annotations

from dataclasses import dataclass

import pytest

from cramph.context import ContextExtension, StatechartContext
from cramph.exceptions import (
    AmbiguousContextExtensionError,
    ConflictingTickDurationError,
    DuplicateContextExtensionError,
    MissingContextExtensionError,
)

# %% context extensions


@dataclass
class ExtensionRecordingItsCleanup(ContextExtension):
    """
    A context extension that remembers whether it was cleaned up.
    """

    cleaned_up: bool = False
    """
    Whether :meth:`cleanup` was called.
    """

    def cleanup(self) -> None:
        self.cleaned_up = True


@dataclass
class SpecializedExtensionRecordingItsCleanup(ExtensionRecordingItsCleanup):
    """
    A context extension standing in for the extension it specializes.
    """


@dataclass
class OtherSpecializedExtensionRecordingItsCleanup(ExtensionRecordingItsCleanup):
    """
    A second context extension standing in for the extension it specializes.
    """


def test_cleaning_up_a_context_cleans_up_its_extensions(
    statechart_context: StatechartContext,
) -> None:
    extension = ExtensionRecordingItsCleanup()
    statechart_context.add_extension(extension)

    statechart_context.cleanup()

    assert extension.cleaned_up


def test_get_extension_returns_none_when_nothing_is_registered(
    statechart_context: StatechartContext,
) -> None:
    assert statechart_context.get_extension(ExtensionRecordingItsCleanup) is None


def test_get_extension_returns_the_registered_extension(
    statechart_context: StatechartContext,
) -> None:
    extension = ExtensionRecordingItsCleanup()
    statechart_context.add_extension(extension)

    assert statechart_context.get_extension(ExtensionRecordingItsCleanup) is extension


def test_require_extension_raises_when_nothing_is_registered(
    statechart_context: StatechartContext,
) -> None:
    with pytest.raises(MissingContextExtensionError):
        statechart_context.require_extension(ExtensionRecordingItsCleanup)


def test_adding_a_second_extension_of_the_same_type_raises(
    statechart_context: StatechartContext,
) -> None:
    statechart_context.add_extension(ExtensionRecordingItsCleanup())

    with pytest.raises(DuplicateContextExtensionError):
        statechart_context.add_extension(ExtensionRecordingItsCleanup())


def test_a_context_created_with_extensions_holds_each_under_its_own_type(
    statechart_context: StatechartContext,
) -> None:
    extension = ExtensionRecordingItsCleanup()
    specialized_extension = SpecializedExtensionRecordingItsCleanup()

    context = StatechartContext.with_extensions(
        statechart_context.world, [extension, specialized_extension]
    )

    assert context.extensions == {
        ExtensionRecordingItsCleanup: extension,
        SpecializedExtensionRecordingItsCleanup: specialized_extension,
    }


def test_creating_a_context_with_two_extensions_of_the_same_type_raises(
    statechart_context: StatechartContext,
) -> None:
    with pytest.raises(DuplicateContextExtensionError):
        StatechartContext.with_extensions(
            statechart_context.world,
            [ExtensionRecordingItsCleanup(), ExtensionRecordingItsCleanup()],
        )


# %% extensions standing in for the type they specialize


def test_an_extension_is_found_by_the_type_it_specializes(
    statechart_context: StatechartContext,
) -> None:
    extension = SpecializedExtensionRecordingItsCleanup()
    statechart_context.add_extension(extension)

    assert statechart_context.get_extension(ExtensionRecordingItsCleanup) is extension
    assert (
        statechart_context.require_extension(ExtensionRecordingItsCleanup) is extension
    )


def test_an_extension_is_not_found_by_a_type_specializing_it(
    statechart_context: StatechartContext,
) -> None:
    statechart_context.add_extension(ExtensionRecordingItsCleanup())

    assert (
        statechart_context.get_extension(SpecializedExtensionRecordingItsCleanup)
        is None
    )


def test_two_extensions_specializing_the_requested_type_are_ambiguous(
    statechart_context: StatechartContext,
) -> None:
    statechart_context.add_extension(SpecializedExtensionRecordingItsCleanup())
    statechart_context.add_extension(OtherSpecializedExtensionRecordingItsCleanup())

    with pytest.raises(AmbiguousContextExtensionError):
        statechart_context.get_extension(ExtensionRecordingItsCleanup)


# %% ensuring an extension


def test_ensuring_an_extension_adds_it_when_none_is_registered(
    statechart_context: StatechartContext,
) -> None:
    extension = ExtensionRecordingItsCleanup()

    assert statechart_context.ensure_extension(extension) is extension
    assert statechart_context.get_extension(ExtensionRecordingItsCleanup) is extension


def test_ensuring_an_extension_keeps_the_registered_one(
    statechart_context: StatechartContext,
) -> None:
    registered = ExtensionRecordingItsCleanup()
    statechart_context.add_extension(registered)

    assert (
        statechart_context.ensure_extension(ExtensionRecordingItsCleanup())
        is registered
    )


# %% tick duration


def test_a_context_without_a_tick_duration_takes_the_one_set(
    statechart_context_without_tick_duration: StatechartContext,
) -> None:
    statechart_context_without_tick_duration.set_tick_duration(0.02)

    assert statechart_context_without_tick_duration.require_tick_duration() == 0.02


def test_setting_the_tick_duration_a_context_already_has_keeps_it(
    statechart_context: StatechartContext,
) -> None:
    tick_duration = statechart_context.require_tick_duration()

    statechart_context.set_tick_duration(tick_duration)

    assert statechart_context.require_tick_duration() == tick_duration


def test_setting_a_different_tick_duration_is_rejected(
    statechart_context: StatechartContext,
) -> None:
    tick_duration = statechart_context.require_tick_duration()

    with pytest.raises(ConflictingTickDurationError):
        statechart_context.set_tick_duration(tick_duration * 2)

    assert statechart_context.require_tick_duration() == tick_duration
