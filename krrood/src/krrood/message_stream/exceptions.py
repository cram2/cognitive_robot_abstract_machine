"""
Exceptions about the messages of a stream.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from krrood.message_stream.messages import StreamPosition


@dataclass(kw_only=True)
class DivergedFromPublisherError(DataclassException):
    """
    Reports that a message of a publisher could not be applied, so that the copy of the
    model it was applied to may differ from the publisher's from then on.

    .. warning:: A message that failed part-way stays applied as far as it got.
    """

    position: StreamPosition
    """
    The message that could not be applied.
    """

    def error_message(self) -> str:
        return (
            f"Message #{self.position.sequence_number} of "
            f"'{self.position.origin.node_name}' could not be applied, so this copy may "
            f"differ from the one of the publisher."
        )

    def suggest_correction(self) -> str:
        return (
            "fetch the model from the publisher again, or stop relying on it being "
            "synchronized with that publisher."
        )
