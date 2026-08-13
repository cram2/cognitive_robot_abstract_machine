"""
The probabilistic-models workbench behind the viewer's Models tab.

Wraps :class:`probabilistic_model.gui.controller.ModelController` — the UI-free logic
of the desktop GUI — for the browser: loading a circuit from JSON, building events
from the tab's constraint rows, and answering probability, posterior and mode
requests with JSON payloads (posteriors as ready-to-render Plotly figures).

The probabilistic-model stack is optional, like krrood for EQL: without it the server
still runs and the Models tab explains why it is empty.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field

from typing_extensions import Any, ClassVar, Dict, List, Optional

from cramera.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.io as plotly_io
    from probabilistic_model.gui.controller import ModelController
    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
        ProbabilisticCircuit,
    )
    from random_events.interval import Bound, Interval, SimpleInterval
    from random_events.product_algebra import Event, SimpleEvent
    from random_events.set import Set, SetElement
    from random_events.variable import Continuous, Integer, Symbolic

    PROBABILISTIC_MODELS_AVAILABLE = True
except ImportError:
    PROBABILISTIC_MODELS_AVAILABLE = False

NO_MODELS_MESSAGE = "probabilistic_model not available in this environment"
"""
What the API answers when the probabilistic-model stack is not importable.
"""

UNBOUNDED_SLIDER_LIMIT = 100.0
"""
Slider bound used where a variable's support is infinite, as in the desktop GUI.
"""

POINT_RELATIVE_WIDTH = 1e-6
"""
An interval narrower than this, relative to its magnitude, displays as one value.
"""


class UnknownVariableError(Exception):
    """
    Raised when a constraint names a variable the loaded model does not have.
    """


class NoModelLoadedError(Exception):
    """
    Raised when a computation is requested before any model was loaded.
    """


@dataclass
class ModelWorkbench:
    """
    One loaded probabilistic model and the operations the Models tab offers on it.
    """

    controller: ModelController = field(default_factory=lambda: ModelController())
    """
    The reused desktop-GUI controller holding the model and its variable map.
    """

    model_name: str = ""
    """
    Display name of the loaded model, usually the uploaded file's name.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Serializes API calls; the controller is not thread-safe and the server threads.
    """

    _active: ClassVar[Optional[ModelWorkbench]] = None
    """
    The workbench the server routes talk to, created on first use.
    """

    @classmethod
    def active(cls) -> ModelWorkbench:
        """
        The process-wide workbench, created on first use.
        """
        if cls._active is None:
            cls._active = cls()
        return cls._active

    # %% loading and describing the model

    def load_model(self, data: Dict[str, Any], name: str = "") -> Dict[str, Any]:
        """
        Load a probabilistic circuit from its JSON serialization.

        :param data: The circuit as ``ProbabilisticCircuit.to_json`` wrote it.
        :param name: Display name for the loaded model.
        :return: The workbench state payload after loading.
        """
        with self._lock:
            model = ProbabilisticCircuit.from_json(data)
            self.controller.set_model(model)
            self.model_name = name
        return self.state()

    def state(self) -> Dict[str, Any]:
        """
        What the Models tab needs to render itself: whether a model is loaded and which
        variables it has.
        """
        with self._lock:
            if self.controller.model is None:
                return {"ok": True, "loaded": False, "name": "", "variables": []}
            return {
                "ok": True,
                "loaded": True,
                "name": self.model_name,
                "variables": [
                    self._variable_payload(variable)
                    for variable in self.controller.model.variables
                ],
            }

    def _variable_payload(self, variable: Any) -> Dict[str, Any]:
        """
        One variable as the tab's constraint rows offer it: symbolic variables with
        their elements, numeric ones with the slider bounds of their prior support.

        :param variable: The model variable to describe.
        """
        if isinstance(variable, Symbolic):
            return {
                "name": variable.name,
                "kind": "symbolic",
                "values": [str(element) for element in variable.domain.all_elements],
            }
        kind = "integer" if isinstance(variable, Integer) else "continuous"
        low, high = self._variable_bounds(variable)
        return {"name": variable.name, "kind": kind, "low": low, "high": high}

    def _variable_bounds(self, variable: Any) -> tuple:
        """
        The slider range of a numeric variable: its prior support, with infinite ends
        clamped and a point support widened, as in the desktop GUI.

        :param variable: The numeric variable to bound.
        """
        support = self.controller.priors[variable].support
        low = support.simple_sets[0][variable].simple_sets[0].lower
        high = support.simple_sets[-1][variable].simple_sets[-1].upper
        if low == float("-inf"):
            low = -UNBOUNDED_SLIDER_LIMIT
        if high == float("inf"):
            high = UNBOUNDED_SLIDER_LIMIT
        if low == high:
            low, high = low - 1.0, high + 1.0
        return float(low), float(high)

    # %% events from constraint rows

    def build_event(self, constraints: List[Dict[str, Any]]) -> Event:
        """
        The event the tab's constraint rows describe.

        Mirrors the desktop GUI's semantics: numeric constraints are closed-interval
        unions, symbolic constraints are element selections, and several rows on the
        same variable are united.

        :param constraints: One entry per row: ``{"variable": name}`` plus
            ``"intervals": [[low, high], ...]`` for numeric variables or ``"values":
            [element, ...]`` for symbolic ones.
        :raises UnknownVariableError: When a row names no variable of the model.
        """
        simple_event = SimpleEvent.from_data()
        for constraint in constraints:
            name = constraint.get("variable", "")
            variable = self.controller.variable_map.get(name)
            if variable is None:
                raise UnknownVariableError("unknown variable: %r" % name)
            assignment = self._constraint_assignment(variable, constraint)
            if variable in simple_event:
                simple_event[variable] = simple_event[variable].union_with(assignment)
            else:
                simple_event[variable] = assignment
        if not simple_event:
            return Event()
        event = Event.from_simple_sets(simple_event)
        # unconstrained variables stay at their full domain: truncation needs the
        # event to span the model's whole variable space
        event.fill_missing_variables(self.controller.model.variables)
        return event

    @staticmethod
    def _constraint_assignment(variable: Any, constraint: Dict[str, Any]) -> Any:
        """
        One row's constraint as the assignment the event carries for its variable.

        :param variable: The constrained model variable.
        :param constraint: The row's payload.
        """
        if isinstance(variable, Symbolic):
            selected = [str(value) for value in constraint.get("values", [])]
            all_elements = variable.domain.all_elements
            matched = [element for element in all_elements if str(element) in selected]
            if not matched:
                return Set()
            return Set.from_simple_sets(
                *[SetElement.from_data(element, all_elements) for element in matched]
            )
        intervals = [
            SimpleInterval.from_data(
                min(float(low), float(high)),
                max(float(low), float(high)),
                Bound.CLOSED,
                Bound.CLOSED,
            )
            for low, high in constraint.get("intervals", [])
        ]
        if not intervals:
            return variable.domain
        return Interval.from_simple_sets(*intervals)

    # %% the tab's computations

    def probability(
        self,
        query: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        ``P(query | evidence)`` as the Query section computes it.

        :param query: Constraint rows describing the query event.
        :param evidence: Constraint rows describing the evidence event.
        """
        with self._lock:
            self._require_model()
            value = self.controller.calculate_probability(
                self.build_event(query), self.build_event(evidence)
            )
        return {"ok": True, "probability": value}

    def posterior(
        self,
        variable_names: List[str],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        The posterior marginal of each requested variable, as Plotly figures.

        :param variable_names: The variables whose posterior marginals are plotted.
        :param evidence: Constraint rows describing the evidence event.
        """
        with self._lock:
            self._require_model()
            posterior_model = self.controller.calculate_posterior(
                self.build_event(evidence)
            )
            if posterior_model is None:
                return {"ok": False, "error": "the evidence has zero probability"}
            figures: Dict[str, Any] = {}
            for name in variable_names:
                variable = self.controller.variable_map.get(name)
                if variable is None:
                    raise UnknownVariableError("unknown variable: %r" % name)
                marginal = posterior_model.marginal([variable])
                figure = go.Figure(marginal.plot(), marginal.plotly_layout())
                figures[name] = json.loads(plotly_io.to_json(figure))
        return {"ok": True, "figures": figures}

    def mode(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        The most likely assignments given the evidence, as the Mode section shows.

        :param evidence: Constraint rows describing the evidence event.
        """
        with self._lock:
            self._require_model()
            result = self.controller.calculate_mode(self.build_event(evidence))
            if result is None:
                return {"ok": False, "error": "the evidence has zero probability"}
            mode_event, likelihood = result
            modes = [
                {
                    str(variable.name): self._pretty_assignment(assignment)
                    for variable, assignment in simple_set.items()
                }
                for simple_set in mode_event.simple_sets
            ]
        return {"ok": True, "likelihood": likelihood, "modes": modes}

    @classmethod
    def _pretty_assignment(cls, assignment: Any) -> str:
        """
        One mode assignment as the tab displays it: rounded interval bounds, an interval
        of negligible width as the single value it is, unions joined readably, and
        symbolic selections as their elements.

        :param assignment: The variable's assignment in a mode's simple event.
        """
        if isinstance(assignment, Interval):
            return " ∪ ".join(
                cls._pretty_interval(simple) for simple in assignment.simple_sets
            )
        if isinstance(assignment, Set):
            return ", ".join(str(element) for element in assignment.simple_sets)
        return str(assignment)

    @classmethod
    def _pretty_interval(cls, simple_interval: Any) -> str:
        """
        One simple interval, rounded, as a point when its width is negligible.

        :param simple_interval: The interval to render.
        """
        low, high = float(simple_interval.lower), float(simple_interval.upper)
        magnitude = max(abs(low), abs(high), 1.0)
        if high - low <= magnitude * POINT_RELATIVE_WIDTH:
            return "%.4g" % ((low + high) / 2)
        return "[%.4g, %.4g]" % (low, high)

    def _require_model(self) -> None:
        """
        :raises NoModelLoadedError: When no model was loaded yet.
        """
        if self.controller.model is None:
            raise NoModelLoadedError("no model loaded — upload one first")
