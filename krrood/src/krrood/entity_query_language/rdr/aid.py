"""
Optional, task-specific assistance for the expert while labelling a case.

A :class:`ConclusionAid` is an injectable collaborator the RDR consults during the
no-ground-truth fit (``ask_for_rule``). It has two optional hooks, both defaulting to a no-op
(return ``None``) so an aid implements only what it needs:

* :meth:`present` renders an information / visual aid (e.g. show the image for an MNIST digit,
  a scene screenshot for a manipulation task, a plot of similar cases). The returned text is
  folded into the shell header and re-shown on demand via the ``%aid`` magic.
* :meth:`suggest` proposes a candidate conclusion the expert can accept or overwrite. The
  suggestion is validated against the conclusion domain and, if valid, pre-seeds the answer.

An informational aid overrides only :meth:`present`; a suggester overrides only
:meth:`suggest`; a smart aid (e.g. a model or LLM) may override both — explaining its
reasoning in :meth:`present` and proposing a label in :meth:`suggest`.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.interface import CaseContext


class ConclusionAid:
    """A pluggable aid that can present help and/or suggest a conclusion for a case."""

    def present(self, context: "CaseContext") -> Optional[str]:
        """
        Render an information / visual aid for the case.

        :param context: Everything known about the case being labelled (the concrete
            instance, the shared variable, the current/target conclusion, the classification
            trace, and the resolved conclusion domain).
        :return: Text to show the expert, or ``None`` to contribute nothing.
        """
        return None

    def suggest(self, context: "CaseContext") -> Optional[Any]:
        """
        Propose a candidate conclusion for the case.

        :param context: Everything known about the case being labelled (see :meth:`present`).
        :return: A suggested conclusion value, or ``None`` for no suggestion. The value is
            validated against the conclusion domain before it is offered to the expert.
        """
        return None
