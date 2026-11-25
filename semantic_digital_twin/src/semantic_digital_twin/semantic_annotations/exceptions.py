"""
Exceptions raised by the semantic annotations of the semantic digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasRootBody

# %% liquid transfer


@dataclass(kw_only=True)
class MissingFillEquationError(UsageError):
    """
    Raised when a liquid transfer is requested from a source that has no outflow
    physics.
    """

    source: HasRootBody
    """
    The annotation that was supposed to act as the liquid source.
    """

    def error_message(self) -> str:
        return (
            f"Cannot transfer liquid from '{self.source.root.name}': it has no fill equation "
            f"or fill connection, so its outflow is undefined."
        )

    def suggest_correction(self) -> str:
        return "call source.initialize_fill_level(world=..., ...) before connecting its outflow."


@dataclass(kw_only=True)
class SourceAlreadyCoupledError(UsageError):
    """
    Raised when a liquid source whose outflow is already coupled to a receiver is
    coupled again.
    """

    source: HasRootBody
    """
    The annotation acting as the liquid source that is already coupled.
    """

    def error_message(self) -> str:
        return (
            f"The outflow of '{self.source.root.name}' is already coupled to a receiver; "
            f"coupling it again would corrupt the existing transfer."
        )

    def suggest_correction(self) -> str:
        return "couple each source to a single receiver."


@dataclass(kw_only=True)
class ReceiverNotInitializedError(UsageError):
    """
    Raised when a container is asked to receive liquid before its fill level was
    initialized.
    """

    receiver: HasRootBody
    """
    The annotation that was supposed to receive the liquid.
    """

    def error_message(self) -> str:
        return (
            f"Cannot pour liquid into '{self.receiver.root.name}': it has no fill "
            f"connection, so it cannot track a fill level."
        )

    def suggest_correction(self) -> str:
        return "call receiver.initialize_fill_level(world=..., ...) before coupling its inflow."


@dataclass(kw_only=True)
class ReceiverAlreadyCoupledError(UsageError):
    """
    Raised when a container whose inflow is already coupled to a source is coupled to
    another source.
    """

    receiver: HasRootBody
    """
    The annotation acting as the liquid receiver that is already coupled.
    """

    def error_message(self) -> str:
        return (
            f"The inflow of '{self.receiver.root.name}' is already coupled to a source; "
            f"coupling it to another source would overwrite the existing transfer and "
            f"lose the first source's outflow."
        )

    def suggest_correction(self) -> str:
        return "couple each receiver to a single source."


@dataclass(kw_only=True)
class FillLevelAlreadyInitializedError(UsageError):
    """
    Raised when a fill level is initialized on a container that already carries one.
    """

    container: HasRootBody
    """
    The annotation whose fill level is already initialized.
    """

    def error_message(self) -> str:
        return (
            f"The fill level of '{self.container.root.name}' is already initialized; "
            f"initializing it again would leave a second fill-level body and fill "
            f"connection in the world."
        )

    def suggest_correction(self) -> str:
        return "initialize each container's fill level exactly once."
