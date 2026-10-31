from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Self, TypeVar

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from cramph.exceptions import (
    AmbiguousContextExtensionError,
    MissingContextExtensionError,
    DuplicateContextExtensionError,
    TickDurationUnknownError,
    ConflictingTickDurationError,
)

from semantic_digital_twin.world import World

# %% context extensions


@dataclass
class ContextExtension:
    """
    Data or a service that the nodes of a statechart read from their
    :class:`StatechartContext`, on top of what every statechart has.

    A node names the extensions it reads in
    :attr:`~cramph.node.StatechartNode.required_context_extensions`, and those it reads
    only while it expands in
    :attr:`~cramph.node.CompositeNode.context_extensions_required_to_expand`.
    """

    def cleanup(self) -> None:
        """
        Releases what the extension acquired while the statechart was running.
        """
        return None


GenericContextExtension = TypeVar("GenericContextExtension", bound=ContextExtension)


# %% statechart context


@dataclass
class StatechartContext:
    """
    Context handed to every node of a statechart while it is built and ticked.
    """

    world: World
    """
    The world in which the statechart is executed.
    """

    tick_duration: float | None = None
    """
    How many seconds one tick stands for, None if ticks do not stand for a fixed time.
    """

    tick_variable: FloatVariable = field(init=False)
    """
    Auxiliary variable counting the ticks, can be used by nodes to implement time-
    dependent actions.
    """

    float_variable_data: FloatVariableData = field(default_factory=FloatVariableData)
    """
    Data structure used to store auxiliary variables.
    """

    extensions: dict[type[ContextExtension], ContextExtension] = field(
        default_factory=dict, repr=False, init=False
    )
    """
    The extensions of this context, each under its own type.

    Executor extensions add the context extensions they need when an executor is
    created, see :meth:`~cramph.executor.ExecutorExtension.extend_context`.
    """

    @classmethod
    def with_extensions(cls, world: World, extensions: list[ContextExtension]) -> Self:
        """
        :param world: The world in which the statechart is executed.
        :param extensions: The extensions the new context holds.
        :return: A new context in `world` holding `extensions`, see
            :meth:`add_extension`.
        :raises DuplicateContextExtensionError: If two of `extensions` are of the same
            type.
        """
        context = cls(world=world)
        for extension in extensions:
            context.add_extension(extension)
        return context

    def set_tick_duration(self, tick_duration: float) -> None:
        """
        Sets how many seconds one tick stands for.

        :param tick_duration: How many seconds one tick stands for.
        :raises ConflictingTickDurationError: If the context already knows a different
            tick duration.
        """
        if self.tick_duration is not None and self.tick_duration != tick_duration:
            raise ConflictingTickDurationError(
                tick_duration=self.tick_duration,
                requested_tick_duration=tick_duration,
            )
        self.tick_duration = tick_duration

    def require_tick_duration(self) -> float:
        """
        :return: How many seconds one tick stands for.
        :raises TickDurationUnknownError: If :attr:`tick_duration` is not known.
        """
        if self.tick_duration is None:
            raise TickDurationUnknownError()
        return self.tick_duration

    @property
    def tick_count(self) -> int:
        """
        :return: The number of ticks run since the statechart started, as held by
            :attr:`tick_variable`.
        """
        return int(self.float_variable_data.get_value(self.tick_variable))

    def require_extension(
        self, extension_type: type[GenericContextExtension]
    ) -> GenericContextExtension:
        """
        :param extension_type: The type of the requested extension.
        :return: The extension :meth:`get_extension` finds for `extension_type`.
        :raises MissingContextExtensionError: If there is none.
        """
        extension = self.get_extension(extension_type)
        if extension is None:
            raise MissingContextExtensionError(expected_extension=extension_type)
        return extension

    def get_extension(
        self, extension_type: type[GenericContextExtension]
    ) -> GenericContextExtension | None:
        """
        :param extension_type: The type of the requested extension.
        :return: The extension that is an instance of `extension_type`, or None if
            there is none.
        :raises AmbiguousContextExtensionError: If several extensions are instances of
            `extension_type`.
        """
        matching_extensions = [
            extension
            for extension in self.extensions.values()
            if isinstance(extension, extension_type)
        ]
        if len(matching_extensions) > 1:
            raise AmbiguousContextExtensionError(
                requested_type=extension_type, matching_extensions=matching_extensions
            )
        return matching_extensions[0] if matching_extensions else None

    def add_extension(self, extension: ContextExtension) -> None:
        """
        :param extension: The extension to add under its own type.
        :raises DuplicateContextExtensionError: If this context already holds an
            extension of that very type.
        """
        extension_type = type(extension)
        if extension_type in self.extensions:
            raise DuplicateContextExtensionError(extension_type=extension_type)
        self.extensions[extension_type] = extension

    def ensure_extension(
        self, extension: GenericContextExtension
    ) -> GenericContextExtension:
        """
        Adds `extension` unless this context already holds one of its very type.

        :param extension: The extension to add if none of its type is there.
        :return: The extension of that type this context holds afterwards.
        """
        return self.extensions.setdefault(type(extension), extension)

    def cleanup(self) -> None:
        """
        Releases what the context and its extensions acquired while the statechart was
        running.
        """
        for extension in self.extensions.values():
            extension.cleanup()
