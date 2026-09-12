"""
The two causal-query pipelines under comparison, behind one interface: fit on recorded
attempts, serve ``cause``/``causes_effect`` EQL queries through a model registry, and
report what the fits cost and how well they explain held-out attempts.

Backdoor adjustment needs the circuit it runs on to be support-deterministic over the
cause variable, which a fit guarantees by stratifying its training rows on that
variable's exact value. Stratifying on two variables at once cannot serve both -- two
partitions sharing a value of one of them overlap on it -- so each pipeline keeps one
plain model for everything that is not a causal query, and fits one further model per
cause variable it is asked about, the first time it is asked.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import (
    ModelRegistry,
    RelationalCircuitRegistry,
)
from krrood.parametrization.parameterizer import (
    ModelQueryParameters,
    UnderspecifiedParameters,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel
from typing_extensions import Dict, List, Optional, Sequence, Union

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutteredObject,
    ClutterPickScene,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    FlatTableSchemaMismatchError,
    OneCausePerQueryError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.tracy_clutter_picking.flat_table import (
    FlatTable,
    SceneSchema,
)

# %% what a fit reports


@dataclass(frozen=True)
class CircuitSize:
    """
    How big a fitted circuit is.
    """

    node_count: int
    """
    Number of units, leaves included.
    """

    edge_count: int
    """
    Number of edges between units.
    """

    @classmethod
    def of(cls, circuit: ProbabilisticCircuit) -> CircuitSize:
        """
        :param circuit: The circuit to measure.
        :return: Its size.
        """
        return cls(node_count=len(circuit.nodes()), edge_count=len(circuit.edges()))

    def __add__(self, other: CircuitSize) -> CircuitSize:
        return CircuitSize(
            self.node_count + other.node_count, self.edge_count + other.edge_count
        )


@dataclass
class FitReport:
    """
    What fitting a pipeline cost and produced, over the plain model and every cause-
    specific model fitted on demand.
    """

    training_scene_count: int
    """
    How many attempts the pipeline was fitted on.
    """

    model_count: int = 0
    """
    How many models were fitted.
    """

    training_duration: float = 0.0
    """
    Wall-clock seconds all the fits took together.
    """

    size: CircuitSize = CircuitSize(0, 0)
    """
    The size of every fitted circuit together.
    """

    def record(self, duration: float, size: CircuitSize) -> None:
        """
        Add one model's fit.

        :param duration: Wall-clock seconds the fit took.
        :param size: The size of the fitted circuit.
        """
        self.model_count += 1
        self.training_duration += duration
        self.size = self.size + size


@dataclass(frozen=True)
class LikelihoodReport:
    """
    How well a fitted pipeline explains held-out attempts.
    """

    scene_count: int
    """
    How many attempts were scored.
    """

    covered_scene_count: int
    """
    How many of them lie inside the model's support at all.
    """

    mean_log_likelihood: float
    """
    Mean log-likelihood over the covered attempts; ``nan`` if none is covered.
    """

    log_likelihoods: np.ndarray = field(compare=False, repr=False)
    """
    One log-likelihood per scored attempt, in the attempts' order, ``-inf`` for an
    attempt outside the support.
    """

    @property
    def coverage(self) -> float:
        """
        Share of attempts inside the model's support.
        """
        return self.covered_scene_count / self.scene_count

    @classmethod
    def from_log_likelihoods(cls, log_likelihoods: np.ndarray) -> LikelihoodReport:
        """
        :param log_likelihoods: One log-likelihood per scored attempt, ``-inf`` for an
            attempt outside the support.
        :return: The report.
        """
        finite = log_likelihoods[np.isfinite(log_likelihoods)]
        return cls(
            scene_count=len(log_likelihoods),
            covered_scene_count=len(finite),
            mean_log_likelihood=float(finite.mean()) if len(finite) else float("nan"),
            log_likelihoods=log_likelihoods,
        )


# %% which variable a query marks as its cause


@dataclass(frozen=True)
class CauseStratification:
    """
    What a fit has to be stratified by to register one variable as a cause.
    """

    class_columns: Optional[List[str]]
    """
    The scene-level columns to stratify the class circuit by, or ``None`` to leave it to
    the plain fit.
    """

    neighbour_attributes: Optional[List[str]]
    """
    The neighbour attributes to stratify the neighbour template by, or ``None`` to leave
    it to the plain fit.
    """

    @classmethod
    def for_variable(
        cls, variable_name: str, schema: SceneSchema = SceneSchema()
    ) -> CauseStratification:
        """
        :param variable_name: The cause variable's name, as EQL names it.
        :param schema: How the attempt's attributes are named.
        :return: The stratification that makes a fit support-deterministic over it.
        """
        neighbour_pattern = re.compile(
            rf"^{re.escape(schema.scene_column(schema.neighbours_field))}"
            rf"\[(\d+)\]\.(\w+)$"
        )
        neighbour_match = neighbour_pattern.match(variable_name)
        if neighbour_match is None:
            return cls(class_columns=[variable_name], neighbour_attributes=None)
        return cls(class_columns=None, neighbour_attributes=[neighbour_match.group(2)])


def cause_variable_name(parameters: ModelQueryParameters) -> Optional[str]:
    """
    :param parameters: The parameters extracted from a queried statement.
    :return: The name of the one variable the query marks as its cause, or ``None`` if
        it marks none.
    :raises OneCausePerQueryError: If the query marks more than one cause.
    """
    if not isinstance(parameters, UnderspecifiedParameters):
        return None
    causes = parameters.search_cause_variables
    if not causes:
        return None
    if len(causes) > 1:
        raise OneCausePerQueryError([cause.name for cause in causes])
    return causes[0].name


# %% the shared interface


def unspecified_scene_query(
    neighbour_count: int, schema: SceneSchema = SceneSchema()
) -> Match:
    """
    A query for an attempt with every attribute, its own and its neighbours', left open.

    :param neighbour_count: How many neighbours the attempt has.
    :param schema: How the attempt's attributes are named.
    :return: The query.
    """
    return a(ClutterPickScene)(
        **{name: ... for name in schema.scene_scalar_fields},
        neighbours=[
            a(ClutteredObject)(**{name: ... for name in schema.neighbour_fields})
            for _ in range(neighbour_count)
        ],
    )


@dataclass
class CausalQueryPipeline(ABC):
    """
    One way of turning recorded attempts into models that answer causal EQL queries.
    """

    min_samples_per_leaf: Union[int, float] = 15
    """
    The fewest training rows a leaf of a cause-specific tree may hold: enough for a
    continuous attribute's leaf to span a range rather than pin the single value it saw,
    few enough for a stratum to still split on what else drives the effect.

    See
    :attr:`~probabilistic_model.probabilistic_circuit.relational.rspn.RelationalProbabilisticCircuit.min_samples_per_leaf`.
    """

    plain_min_samples_per_leaf: Union[int, float] = 50
    """
    The fewest training rows a leaf of the plain tree may hold.

    The plain model scores held-out attempts, and a leaf spans only the ranges it saw,
    so wider leaves cover more of them.
    """

    schema: SceneSchema = field(default_factory=SceneSchema)
    """
    How the attempt's attributes are named.
    """

    training_scenes: List[ClutterPickScene] = field(default_factory=list)
    """
    The attempts :meth:`fit` was given, kept for the cause-specific fits.
    """

    fit_report: Optional[FitReport] = None
    """
    What the fits so far cost, once :meth:`fit` ran.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        What the pipeline is called in reports.
        """

    @property
    def registry(self) -> ModelRegistry:
        """
        The registry a
        :class:`~krrood.entity_query_language.backends.ProbabilisticBackend` resolves
        queries against.
        """
        return PipelineRegistry(pipeline=self)

    def fit(self, scenes: Sequence[ClutterPickScene]) -> FitReport:
        """
        Fit the plain model, and keep the attempts for the cause-specific fits.

        :param scenes: The attempts to fit on.
        :return: The report the later fits keep adding to.
        """
        self.training_scenes = list(scenes)
        self.fit_report = FitReport(training_scene_count=len(scenes))
        started = time.perf_counter()
        size = self._fit_plain_model()
        self.fit_report.record(time.perf_counter() - started, size)
        return self.fit_report

    @abstractmethod
    def _fit_plain_model(self) -> CircuitSize:
        """
        Fit the model that serves every query without a cause, on
        :attr:`training_scenes`.

        :return: The size of the fitted circuit.
        """

    @abstractmethod
    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        """
        Fit the model that serves queries marking the named variable as their cause, on
        :attr:`training_scenes`.

        :param cause_name: The cause variable's name, as EQL names it.
        :return: The size of the fitted circuit.
        """

    @abstractmethod
    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        """
        :param cause_name: The cause variable's name, or ``None`` for the plain model.
        :return: The registry over the model fitted for it.
        """

    @abstractmethod
    def _has_cause_model(self, cause_name: str) -> bool:
        """
        :param cause_name: The cause variable's name.
        :return: Whether its model was fitted already.
        """

    def registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        """
        The registry over the model serving queries with the given cause, fitting that
        model first if it was not asked for before.

        :param cause_name: The cause variable's name, or ``None`` for the plain model.
        :return: The registry.
        :raises PipelineNotFittedError: If :meth:`fit` never ran.
        """
        if self.fit_report is None:
            raise PipelineNotFittedError(self.name)
        if cause_name is not None and not self._has_cause_model(cause_name):
            started = time.perf_counter()
            size = self._fit_cause_model(cause_name)
            self.fit_report.record(time.perf_counter() - started, size)
        return self._registry_for(cause_name)

    @abstractmethod
    def observed_attribute_circuit(
        self, neighbour_count: int
    ) -> Optional[ProbabilisticCircuit]:
        """
        The plain model's joint over an attempt's observed attributes -- its own scalars
        and its neighbours' -- for attempts with the given neighbour count, with the
        aggregation statistics marginalized out.

        :param neighbour_count: How many neighbours the attempts have.
        :return: The circuit, or ``None`` if the pipeline cannot model that count.
        """

    def log_likelihood(self, scenes: Sequence[ClutterPickScene]) -> LikelihoodReport:
        """
        Score held-out attempts under the plain model, on their observed attributes.

        :param scenes: The attempts to score.
        :return: The report; an attempt whose neighbour count the pipeline cannot model
            counts as outside the support.
        """
        log_likelihoods = np.full(len(scenes), -np.inf)
        by_count: Dict[int, List[int]] = {}
        for index, scene in enumerate(scenes):
            by_count.setdefault(len(scene.neighbours), []).append(index)
        for neighbour_count, indices in by_count.items():
            circuit = self.observed_attribute_circuit(neighbour_count)
            if circuit is None:
                continue
            table = FlatTable(neighbour_count, self.schema)
            rows = [table.row(scenes[index]) for index in indices]
            events = np.array(
                [
                    [row[variable.name] for variable in circuit.variables]
                    for row in rows
                ],
                dtype=object,
            )
            log_likelihoods[indices] = circuit.log_likelihood(events)
        return LikelihoodReport.from_log_likelihoods(log_likelihoods)

    def _without_aggregations(
        self, circuit: ProbabilisticCircuit
    ) -> Optional[ProbabilisticCircuit]:
        """
        :param circuit: A circuit over an attempt's attributes and aggregation
            statistics.
        :return: The circuit marginalized to everything but the aggregation statistics.
        """
        aggregation_names = set(self.schema.aggregation_columns)
        return circuit.marginal(
            [
                variable
                for variable in circuit.variables
                if variable.name not in aggregation_names
            ]
        )


@dataclass
class PipelineRegistry(ModelRegistry):
    """
    Routes each query to the pipeline's model fitted for the cause it marks.
    """

    pipeline: CausalQueryPipeline
    """
    The pipeline whose models are served.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        return self.pipeline.registry_for(cause_variable_name(parameters)).get_model(
            parameters
        )


# %% relational pipeline


@dataclass
class RelationalPipeline(CausalQueryPipeline):
    """
    A relational probabilistic circuit fitted on the attempts' relational structure and
    grounded per query into a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    monte_carlo_sample_count: int = 20
    """
    How many samples grounding draws for an aggregation statistic a query leaves open.
    """

    plain_model: Optional[RelationalProbabilisticCircuit] = None
    """
    The model fitted without any stratification, once :meth:`fit` ran.
    """

    cause_models: Dict[str, RelationalProbabilisticCircuit] = field(
        default_factory=dict
    )
    """
    The model fitted for each cause variable asked about so far, by variable name.
    """

    @property
    def name(self) -> str:
        return "relational circuit"

    def _new_model(
        self, min_samples_per_leaf: Union[int, float]
    ) -> RelationalProbabilisticCircuit:
        return RelationalProbabilisticCircuit(
            ClutterPickScene,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
            min_samples_per_leaf=min_samples_per_leaf,
        )

    @staticmethod
    def _size_of(model: RelationalProbabilisticCircuit) -> CircuitSize:
        """
        :param model: A fitted relational circuit.
        :return: The size of its class circuit and every part template together.
        """
        size = CircuitSize.of(model.class_probabilistic_circuit)
        for template in model.exchangeable_distribution_templates.values():
            size = size + CircuitSize.of(
                template.template_distribution.class_probabilistic_circuit
            )
        return size

    def _fit_plain_model(self) -> CircuitSize:
        self.plain_model = self._new_model(self.plain_min_samples_per_leaf)
        self.plain_model.fit([to_dao(scene) for scene in self.training_scenes])
        return self._size_of(self.plain_model)

    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        stratification = CauseStratification.for_variable(cause_name, self.schema)
        model = self._new_model(self.min_samples_per_leaf)
        RelationalCausalCircuit().fit(
            model,
            [to_dao(scene) for scene in self.training_scenes],
            stratify_by=stratification.class_columns,
            stratify_parts_by=(
                None
                if stratification.neighbour_attributes is None
                else {self.schema.neighbours_field: stratification.neighbour_attributes}
            ),
        )
        self.cause_models[cause_name] = model
        return self._size_of(model)

    def _has_cause_model(self, cause_name: str) -> bool:
        return cause_name in self.cause_models

    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        model = (
            self.plain_model if cause_name is None else self.cause_models[cause_name]
        )
        return RelationalCircuitRegistry(relational_probabilistic_circuit=model)

    def observed_attribute_circuit(
        self, neighbour_count: int
    ) -> Optional[ProbabilisticCircuit]:
        grounded = self.registry_for(None).get_model(
            UnderspecifiedParameters(
                unspecified_scene_query(neighbour_count, self.schema)
            )
        )
        return self._without_aggregations(grounded)


# %% flat-table pipeline


@dataclass
class FlatTableRegistry(ModelRegistry):
    """
    Serves a circuit fitted on a flat table for queries over exactly that table's
    columns, wrapped as a causal circuit when the query marks a cause.
    """

    circuit: ProbabilisticCircuit
    """
    The fitted circuit.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        """
        :param parameters: The parameters extracted from the queried statement.
        :return: The circuit, renamed to the query's own variables, and wrapped as a
            verified causal circuit if the query marks a cause.
        :raises FlatTableSchemaMismatchError: If the query asks about a column the
            table never had.
        """
        fitted_names = {variable.name for variable in self.circuit.variables}
        missing = sorted(
            name for name in parameters.variables if name not in fitted_names
        )
        if missing:
            raise FlatTableSchemaMismatchError(missing)
        renamed = self.circuit.__deepcopy__()
        renamed.update_variables(
            {
                variable: parameters.variables[variable.name]
                for variable in renamed.variables
                if variable.name in parameters.variables
            }
        )
        if cause_variable_name(parameters) is None:
            return renamed
        return RelationalCausalCircuit().from_grounded_circuit(
            renamed,
            parameters.search_cause_variables,
            list(parameters.effect_variables_from_causes_effect),
            adjustment_variables=parameters.search_confounder_variables,
            trim_to_registered_variables=True,
        )


@dataclass
class FlatTablePipeline(CausalQueryPipeline):
    """
    Joint probability trees fitted on the attempts flattened into one fixed-width table,
    each wrapped as a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    neighbour_count: int = 9
    """
    How many neighbours the table unrolls; the only count the pipeline can be asked
    about.
    """

    plain_circuit: Optional[ProbabilisticCircuit] = None
    """
    The tree fitted without any stratification, once :meth:`fit` ran.
    """

    cause_circuits: Dict[str, ProbabilisticCircuit] = field(default_factory=dict)
    """
    The tree fitted for each cause variable asked about so far, by variable name.
    """

    @property
    def name(self) -> str:
        return "flat-table tree"

    @property
    def table(self) -> FlatTable:
        """
        The table the attempts are flattened into.
        """
        return FlatTable(self.neighbour_count, self.schema)

    def _training_dataframe(self) -> pd.DataFrame:
        return self.table.dataframe(self.training_scenes)

    def _fit_plain_model(self) -> CircuitSize:
        dataframe = self._training_dataframe()
        self.plain_circuit = JointProbabilityTree(
            annotated_variables=infer_variables_from_dataframe(dataframe),
            min_samples_per_leaf=self.plain_min_samples_per_leaf,
        ).fit(dataframe)
        return CircuitSize.of(self.plain_circuit)

    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        dataframe = self._training_dataframe()
        if cause_name not in dataframe.columns:
            raise FlatTableSchemaMismatchError([cause_name])
        self.cause_circuits[cause_name] = (
            RelationalCausalCircuit._fit_stratified_class_circuit(
                dataframe,
                infer_variables_from_dataframe(dataframe),
                cause_name,
                self.min_samples_per_leaf,
            )
        )
        return CircuitSize.of(self.cause_circuits[cause_name])

    def _has_cause_model(self, cause_name: str) -> bool:
        return cause_name in self.cause_circuits

    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        circuit = (
            self.plain_circuit
            if cause_name is None
            else self.cause_circuits[cause_name]
        )
        return FlatTableRegistry(circuit=circuit)

    def observed_attribute_circuit(
        self, neighbour_count: int
    ) -> Optional[ProbabilisticCircuit]:
        if neighbour_count != self.neighbour_count:
            return None
        return self._without_aggregations(self.plain_circuit.__deepcopy__())


def pipelines_for(neighbour_count: int) -> List[CausalQueryPipeline]:
    """
    :param neighbour_count: How many neighbours the attempts to fit on have.
    :return: Both pipelines, unfitted.
    """
    return [RelationalPipeline(), FlatTablePipeline(neighbour_count=neighbour_count)]
