"""
Optional, task-specific help for the expert while labelling a case.

A :class:`ConclusionHelper` is an injectable collaborator the RDR consults during the
no-ground-truth fit (``ask_for_rule``). Each kind of help is its own mixin declaring the
one method it supplies, so a helper offers only what it implements and a consumer narrows
to the helpers offering what it needs:

* :class:`ConclusionSupportPresenter` renders supporting material (e.g. show the image for
  an MNIST digit, a scene screenshot for a manipulation task, a plot of similar cases).
* :class:`ConclusionSuggester` proposes a candidate conclusion the expert can accept or
  overwrite.

A helper may be both -- a model or an LLM explaining its reasoning as it presents and
proposing a label as it suggests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.interface import CaseContext


class ConclusionHelper(ABC):
    """
    A collaborator the expert may consult while labelling a case.

    Declares no method of its own: what a helper offers is declared by the mixin it
    inherits, so a consumer selects the helpers offering what it needs.
    """


class ConclusionSupportPresenter(ConclusionHelper):
    """
    A helper that supplies supporting material for the case being labelled.
    """

    @abstractmethod
    def present(self, context: CaseContext) -> Optional[str]:
        """
        Render supporting material for the case.

        :param context: Everything known about the case being labelled (the concrete
            instance, the shared variable, the current/target conclusion, the
            classification trace, and the resolved conclusion domain).
        :return: Text to show the expert, or ``None`` to contribute nothing.
        """


class ConclusionSuggester(ConclusionHelper):
    """
    A helper that proposes a conclusion for the case being labelled.
    """

    @abstractmethod
    def suggest(self, context: CaseContext) -> Optional[Any]:
        """
        Propose a candidate conclusion for the case.

        :param context: Everything known about the case being labelled (see
            :meth:`ConclusionSupportPresenter.present`).
        :return: A suggested conclusion value, or ``None`` for no suggestion. The value
            is validated against the conclusion domain before it is offered to the
            expert.
        """
