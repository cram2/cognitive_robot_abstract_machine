"""
The *mechanism* for how to get answers (rule conditions and conclusions) from an expert.

An :class:`ExpertInterface` is the I/O strategy an :class:`~krrood.entity_query_language.rdr.expert.Expert`
delegates to. The :class:`Expert` decides *what* to ask (a condition, or a conclusion +
condition) and *how to validate* it; the interface owns the build-namespace → render →
run → validate → re-prompt loop. Concrete interfaces implement only :meth:`ExpertInterface._run`.

Two value objects carry the request across the boundary:

* :class:`CaseContext` — the concrete case, the shared EQL variable, and the current /
  target conclusions.
* :class:`AnswerRequest` — one named answer the expert must supply, with a validator and
  an example.

The expert writes **EQL expressions**: conditions that are built over the
``case_variable`` (the shared rule-tree variable), while ``case_instance`` is the concrete
object they can inspect and experiment on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.rdr.utils import UNSET
from krrood.entity_query_language.scope import get_definition_scope

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.aid import ConclusionAid
    from krrood.entity_query_language.rdr.conclusion_domain import ConclusionDomain
    from krrood.entity_query_language.rdr.observer import ClassificationTrace

#: Shell name bound to the concrete case (inspect/experiment: ``case_instance.milk``).
CASE_INSTANCE_NAME = "case_instance"

#: Shell name bound to the shared EQL variable (author: ``case_variable.milk == True``).
CASE_VARIABLE_NAME = "case_variable"

#: Shell name of the zero-arg callable the expert calls to leave without answering.
EXIT_NAME = "exit"

#: Private namespace flag set by ``exit()``; checked by the expert interaction loop.
_ABORT_FLAG = "__expert_abort__"


class ExpertAbort(Exception):
    """Raised by :meth:`ExpertInterface.interact` when the expert cancels the session.

    Carries the names of the still-missing required answers so the calling
    :class:`Expert` can raise its own specific exception.
    """

    def __init__(self, missing: List[str]) -> None:
        self.missing = missing
        super().__init__(
            f"Expert cancelled without supplying: {', '.join(missing) or '(nothing)'}"
        )


@dataclass
class CaseContext:
    """The data an expert needs to author a rule for one case."""

    case_instance: Any
    """The concrete case object (e.g. an ``Animal`` instance) to inspect."""
    case_variable: CanBehaveLikeAVariable
    """The shared EQL variable the rule tree ranges over; conditions are built over it."""
    current_conclusion: Any = UNSET
    """What the RDR currently concludes for the case (``_UNSET`` if no rule fired)."""
    target_conclusion: Any = UNSET
    """The known correct conclusion, or absent (sentinel) when the expert must label it."""
    trace: Optional[ClassificationTrace] = None
    """The classification trace for this case, for visualizing the rule tree (``None`` when
    the RDR is empty / no classification was run)."""
    conclusion_domain: Optional["ConclusionDomain"] = None
    """The resolved allowable-value domain of the conclusion attribute, when the expert must
    label the case (``None`` on the conditions-only path, where the conclusion is known)."""
    aids: List["ConclusionAid"] = field(default_factory=list)
    """Optional task-specific aids consulted while labelling the case (presentation and/or
    conclusion suggestion). Empty by default."""

    @property
    def has_target(self) -> bool:
        """:return: True when a ground-truth ``target_conclusion`` was supplied."""
        return self.target_conclusion is not UNSET

    @property
    def has_current_conclusion(self) -> bool:
        """:return: True when the RDR has concluded a conclusion for the case."""
        return self.current_conclusion is not UNSET


@dataclass
class AnswerRequest:
    """One named answer the expert must place in the namespace, with validation."""

    name: str
    """The namespace variable the expert assigns (e.g. ``"conditions"``)."""
    validate: Callable[[Any], Optional[str]]
    """Returns an error message if the assigned value is unacceptable, else ``None``."""
    example: str
    """A copy-pasteable example shown in the header (e.g. ``conditions = ...``)."""
    required: bool = True
    """Whether the session may not complete until a valid value is supplied."""
    default: Any = None
    """The value the answer name is seeded with in the namespace before the expert edits it
    (e.g. ``UNSET`` to distinguish "left unset" from a deliberate ``None``, or a pre-seeded
    suggestion)."""


@dataclass
class ExpertInterface(ABC):
    """The I/O strategy an :class:`Expert` uses as the interaction interface through which answers and questions are
    communicated."""

    def interact(
        self, context: CaseContext, requests: List[AnswerRequest]
    ) -> Dict[str, Any]:
        """
        Drive the request loop until every required answer validates.

        Builds a namespace from the case-definition scope plus ``case_instance`` /
        ``case_variable`` and the EQL factories, runs the (subclass-specific) collection
        step, validates, and re-prompts with an error summary on failure. An explicit
        ``exit()`` raises :class:`ExpertAbort`.

        :return: ``{request.name: value}`` for every request, all validated.
        """
        namespace = self._build_namespace(context, requests)

        def validate() -> Dict[str, str]:
            return self._validate(namespace, requests)

        errors: Dict[str, str] = {}
        while True:
            header = self._render_header(context, requests, errors)
            self._run(namespace, header, validate)
            if namespace.get(_ABORT_FLAG):
                raise ExpertAbort(self._missing_required(namespace, requests))
            errors = validate()
            if not errors:
                return {r.name: namespace.get(r.name) for r in requests}

    def _build_namespace(
        self, context: CaseContext, requests: List[AnswerRequest]
    ) -> Dict[str, Any]:
        namespace = get_definition_scope(context.case_variable)
        namespace[CASE_INSTANCE_NAME] = context.case_instance
        namespace[CASE_VARIABLE_NAME] = context.case_variable
        namespace[_ABORT_FLAG] = False
        namespace[EXIT_NAME] = _make_exit(namespace)
        for request in requests:
            namespace[request.name] = request.default
        return namespace

    @staticmethod
    def _validate(
        namespace: Dict[str, Any], requests: List[AnswerRequest]
    ) -> Dict[str, str]:
        errors: Dict[str, str] = {}
        for request in requests:
            message = request.validate(namespace.get(request.name))
            if message is not None:
                errors[request.name] = message
        return errors

    @classmethod
    def _missing_required(
        cls, namespace: Dict[str, Any], requests: List[AnswerRequest]
    ) -> List[str]:
        errors = cls._validate(namespace, requests)
        return [r.name for r in requests if r.required and r.name in errors]

    def _render_header(
        self,
        context: CaseContext,
        requests: List[AnswerRequest],
        errors: Dict[str, str],
    ) -> str:
        """Plain-text header. Interactive subclasses may override with richer rendering."""
        lines: List[str] = []
        lines.append(f"  {CASE_INSTANCE_NAME}: {context.case_instance!r}")
        lines.append(f"  current: {context.current_conclusion!r}")
        if context.has_target:
            lines.append(f"  target:  {context.target_conclusion!r}")
        for request in requests:
            lines.append(f"  set `{request.name}`   e.g.  {request.example}")
        lines.append(
            "EQL RDR — supply the answer(s) above, then press Ctrl-D to submit "
            f"(call {EXIT_NAME}() to cancel):"
        )
        for name, message in errors.items():
            lines.append(f"[error] {name}: {message}")
        return "\n".join(lines)

    @abstractmethod
    def _run(
        self,
        namespace: Dict[str, Any],
        header: str,
        validate: Callable[[], Dict[str, str]],
    ) -> None:
        """
        Present ``header`` and let the expert populate the answer names in ``namespace``.

        Implementations must leave the expert's assignments (and any ``exit()`` flag)
        visible in ``namespace`` when they return. ``validate`` re-runs the request
        validators against the current namespace and is available to interfaces that want
        to enforce the post-condition before returning (e.g. veto a premature shell exit).
        """
        ...


@dataclass
class FunctionInterface(ExpertInterface):
    """A non-interactive interface that delegates to a plain ``(context, requests) -> dict``.

    Useful for programmatic experts and as a test double: the supplied function returns the
    answers, which are written into the namespace and then validated by the normal loop.
    """

    answer_fn: Callable[[CaseContext, List[AnswerRequest]], Dict[str, Any]]
    """Returns ``{name: value}`` for the requested answers. ``None`` values re-prompt
    (which, for a deterministic function, would loop) — return :func:`abort` semantics by
    raising :class:`ExpertAbort` instead."""

    def __post_init__(self) -> None:
        self._context: Optional[CaseContext] = None
        self._requests: List[AnswerRequest] = []

    def interact(
        self, context: CaseContext, requests: List[AnswerRequest]
    ) -> Dict[str, Any]:
        self._context = context
        self._requests = requests
        return super().interact(context, requests)

    def _run(
        self,
        namespace: Dict[str, Any],
        header: str,
        validate: Callable[[], Dict[str, str]],
    ) -> None:
        answers = self.answer_fn(self._context, self._requests)
        namespace.update(answers)


def _make_exit(namespace: Dict[str, Any]) -> Callable[[], None]:
    def exit() -> None:
        namespace[_ABORT_FLAG] = True

    return exit
