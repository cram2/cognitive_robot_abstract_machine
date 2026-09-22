"""
The causal-query pipelines under comparison, behind one interface: fit on examples,
serve ``cause``/``causes_effect`` EQL queries through a model registry, and report what
the fits cost and how well they explain held-out examples.

Backdoor adjustment needs the circuit it runs on to be support-deterministic over the
cause variable, which a fit guarantees by stratifying its training rows on that
variable's exact value. Stratifying on two variables at once cannot serve both, since
two partitions sharing a value of one of them overlap on it, so each pipeline keeps one
plain model for everything that is not a causal query, and fits one further model per
cause variable it is asked about, the first time it is asked.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
import pandas as pd
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
from probabilistic_model.learning.learning_method import (
    LearningMethod,
    StratifiedLearning,
)
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    ExchangeableDistributionTemplate,
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel
from typing_extensions import Any, Dict, List, Optional, Sequence, Set

from experiments.causal_reasoning.comparison.domain import ExampleView, RelationalDomain
from experiments.causal_reasoning.comparison.exceptions import (
    FlatTableSchemaMismatchError,
    OneCausePerQueryError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.comparison.flat_table import (
    FlatTable,
    PartAttribute,
    Schema,
    TableLayout,
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

    training_example_count: int
    """
    How many examples the pipeline was fitted on.
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
    How well a fitted pipeline explains held-out examples.
    """

    example_count: int
    """
    How many examples were scored.
    """

    covered_example_count: int
    """
    How many of them lie inside the model's support at all.
    """

    mean_log_likelihood: float
    """
    Mean log-likelihood over the covered examples; ``nan`` if none is covered.
    """

    log_likelihoods: np.ndarray = field(compare=False, repr=False)
    """
    One log-likelihood per scored example, in the examples' order, ``-inf`` for an
    example outside the support.
    """

    @property
    def coverage(self) -> float:
        """
        Share of examples inside the model's support.
        """
        return self.covered_example_count / self.example_count

    @classmethod
    def from_log_likelihoods(cls, log_likelihoods: np.ndarray) -> LikelihoodReport:
        """
        :param log_likelihoods: One log-likelihood per scored example, ``-inf`` for a
            example outside the support.
        :return: The report.
        """
        finite = log_likelihoods[np.isfinite(log_likelihoods)]
        return cls(
            example_count=len(log_likelihoods),
            covered_example_count=len(finite),
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
    The example-level columns to stratify the class circuit by, or ``None`` to leave it
    to the plain fit.
    """

    part_attributes: Dict[str, List[str]]
    """
    Per exchangeable-part field, the attributes to stratify that part's template by; a
    part absent from the mapping is left to the plain fit.
    """

    @classmethod
    def for_variable(cls, variable_name: str, schema: Schema) -> CauseStratification:
        """
        :param variable_name: The cause variable's name, as EQL names it.
        :param schema: How the example's attributes are named.
        :return: The stratification that makes a fit support-deterministic over it.
        """
        part = schema.part_attribute(variable_name)
        if part is None:
            return cls(class_columns=[variable_name], part_attributes={})
        return cls(
            class_columns=None, part_attributes={part.part_field: [part.attribute]}
        )


def cause_variable_name(parameters: ModelQueryParameters) -> Optional[str]:
    """
    :param parameters: The parameters extracted from a queried statement.
    :return: The name of the one variable the query marks as its cause, or ``None`` if it
        marks none.
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


def constrained_variable_names(parameters: ModelQueryParameters) -> Set[str]:
    """
    :param parameters: The parameters extracted from a queried statement.
    :return: The names of the variables the query says something about: a value it sets,
        a condition it truncates to, or a cause, confounder or effect it marks. A
        variable the query merely lists and leaves open is not among them.
    """
    if not isinstance(parameters, UnderspecifiedParameters):
        return set(parameters.variables)
    names = {
        variable.name
        for variable in (
            parameters.search_cause_variables
            + parameters.search_confounder_variables
            + parameters.effect_variables_from_causes_effect
            + list(parameters.conditioning_assignments_from_literal_values)
        )
    }
    events = list(parameters.truncation_assignments_from_krrood_variables)
    if parameters.truncation_assignments_from_where_conditions is not None:
        events.append(parameters.truncation_assignments_from_where_conditions)
    for event in events:
        names.update(variable.name for variable in event.variables)
    return names


# %% the shared interface


@dataclass
class CausalQueryPipeline(ABC):
    """
    One way of turning examples into models that answer causal EQL queries.
    """

    min_samples_per_leaf: float = 0.05
    """
    The fewest training rows a leaf of a cause-specific model may hold, as a share of the
    rows it is fitted on: enough for a continuous attribute's leaf to span a range rather
    than pin the values it saw, few enough for a stratum of one cause value to still
    split on what else drives the effect.

    See
    :attr:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree.min_samples_per_leaf`,
    which reads a share below one as a share of its training rows, so the same setting
    holds for a class circuit over a thousand examples and for a part template over their
    tens of thousands of parts.
    """

    plain_min_samples_per_leaf: float = 0.15
    """
    The fewest training rows a leaf of the plain model may hold, as a share of the rows
    it is fitted on.

    The plain model scores held-out examples, and a leaf spans only the ranges it saw,
    so wider leaves cover more of them.
    """

    domain: RelationalDomain = field(kw_only=True)
    """
    The example and its parts.
    """

    training_examples: List[Any] = field(default_factory=list)
    """
    The examples :meth:`fit` was given, kept for the cause-specific fits.
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
    def schema(self) -> Schema:
        """
        How the example's attributes are named.
        """
        return Schema(self.domain)

    @property
    def registry(self) -> ModelRegistry:
        """
        The registry a
        :class:`~krrood.entity_query_language.backends.ProbabilisticBackend` resolves
        queries against.
        """
        return PipelineRegistry(pipeline=self)

    @property
    @abstractmethod
    def table(self) -> FlatTable:
        """
        The examples as rows of what the pipeline's plain model is fitted on.
        """

    @property
    @abstractmethod
    def models_parts(self) -> bool:
        """
        Whether the pipeline models the objects and viewpoints themselves, so that the
        order an example lists them in can matter to it.
        """

    @property
    @abstractmethod
    def order_invariant(self) -> bool:
        """
        Whether fitting on the same examples with their parts in another order gives the
        same model, so that a study over reorderings need fit it once.
        """

    def fit(self, examples: Sequence[Any]) -> FitReport:
        """
        Fit the plain model, and keep the examples for the cause-specific fits.

        :param examples: The examples to fit on.
        :return: The report the later fits keep adding to.
        """
        self.training_examples = list(examples)
        self.fit_report = FitReport(training_example_count=len(examples))
        started = time.perf_counter()
        size = self._fit_plain_model()
        self.fit_report.record(time.perf_counter() - started, size)
        return self.fit_report

    @abstractmethod
    def _fit_plain_model(self) -> CircuitSize:
        """
        Fit the model that serves every query without a cause, on
        :attr:`training_examples`.

        :return: The size of the fitted circuit.
        """

    @abstractmethod
    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        """
        Fit the model that serves queries marking the named variable as their cause, on
        :attr:`training_examples`.

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

    def training_values_of(self, variable_name: str) -> List[Any]:
        """
        The values the training examples hold for a variable, one per row the variable
        is fitted on: one per example for an example attribute or count, one per part for a
        part's own attribute.

        :param variable_name: The variable's name, as EQL names it.
        :return: The values.
        :raises FlatTableSchemaMismatchError: If the pipeline's table has no column for
            the variable.
        """
        part = self.schema.part_attribute(variable_name)
        if part is not None and self.models_parts:
            return self._part_training_values(part)
        if variable_name not in self.table.columns:
            raise FlatTableSchemaMismatchError([variable_name])
        return [
            self.table.row(example)[variable_name] for example in self.training_examples
        ]

    @abstractmethod
    def _part_training_values(self, part: PartAttribute) -> List[Any]:
        """
        :param part: One part's attribute, as a query names it.
        :return: The values the training examples hold for it, one per row the part
            template or column is fitted on.
        """

    @abstractmethod
    def plain_circuit_of(self, view: ExampleView) -> Optional[ProbabilisticCircuit]:
        """
        The plain model's joint over as much of an example as the view asks for.

        :param view: How much of an example to look at.
        :return: The circuit, or ``None`` if the pipeline models less than that.
        :raises PipelineNotFittedError: If :meth:`fit` never ran.
        """

    def log_likelihood(
        self, examples: Sequence[Any], view: ExampleView
    ) -> Optional[LikelihoodReport]:
        """
        Score held-out examples under the plain model, on as much of them as the view
        asks for.

        :param examples: The examples to score.
        :param view: How much of an example to look at.
        :return: The report, or ``None`` if the pipeline models less than the view; a
            example the pipeline's table has no row for counts as outside the support.
        """
        circuit = self.plain_circuit_of(view)
        if circuit is None:
            return None
        log_likelihoods = np.full(len(examples), -np.inf)
        fitting = [
            index for index, example in enumerate(examples) if self.table.fits(example)
        ]
        if fitting:
            rows = [self.table.row(examples[index]) for index in fitting]
            log_likelihoods[fitting] = log_likelihoods_of_rows(circuit, rows)
        return LikelihoodReport.from_log_likelihoods(log_likelihoods)


def log_likelihoods_of_rows(
    circuit: ProbabilisticCircuit, rows: Sequence[Dict[str, Any]]
) -> np.ndarray:
    """
    :param circuit: The circuit to score under.
    :param rows: One value per variable of the circuit, keyed by variable name, per row.
    :return: One log-likelihood per row, ``-inf`` outside the support.
    """
    events = np.array(
        [[row[variable.name] for variable in circuit.variables] for row in rows],
        dtype=object,
    )
    return circuit.log_likelihood(events)


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
    A relational probabilistic circuit fitted on the examples' relational structure and
    grounded per query into a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    monte_carlo_sample_count: int = 2000
    """
    How many samples grounding draws for an aggregation count a query leaves open.
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

    @staticmethod
    def _learning_method(
        min_samples_per_leaf: float, stratified_columns: Optional[List[str]]
    ) -> LearningMethod:
        """
        :param min_samples_per_leaf: The fewest training rows a leaf may hold.
        :param stratified_columns: The columns to stratify by, or ``None`` for a plain
            fit.
        :return: The learning method fitting a circuit that way.
        """
        tree_learning = JointProbabilityTree(min_samples_per_leaf=min_samples_per_leaf)
        if stratified_columns is None:
            return tree_learning
        return StratifiedLearning(variables=stratified_columns, method=tree_learning)

    def _new_model(
        self,
        min_samples_per_leaf: float,
        stratification: CauseStratification = CauseStratification(None, {}),
    ) -> RelationalProbabilisticCircuit:
        """
        :param min_samples_per_leaf: The fewest training rows a leaf of the class circuit
            and of every part template may hold.
        :param stratification: What the fit is stratified by; nothing by default.
        :return: The model, not yet fitted.
        """
        return RelationalProbabilisticCircuit(
            self.domain.example_class,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
            learning_method=self._learning_method(
                min_samples_per_leaf, stratification.class_columns
            ),
            part_learning_methods={
                part_field: self._learning_method(
                    min_samples_per_leaf,
                    stratification.part_attributes.get(part_field),
                )
                for part_field in self.schema.part_fields
            },
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

    def _training_rows(self) -> List[Any]:
        """
        The training examples as data access objects, each with its parts in a canonical
        order, so that the fit is the same whatever order the examples list their parts
        in. The tree learner the templates are fitted with breaks ties by row order, and
        the parts of every example are pooled into its rows.

        :return: One data access object per training example.
        """
        return [to_dao(self._canonical(example)) for example in self.training_examples]

    def _canonical(self, example: Any) -> Any:
        """
        :param example: An example.
        :return: The example with every part list sorted by the parts' attribute values.
        """
        return replace(
            example,
            **{
                part_field: sorted(
                    vars(example)[part_field],
                    key=lambda part: tuple(
                        str(value) if isinstance(value, Enum) else value
                        for value in vars(part).values()
                    ),
                )
                for part_field in self.schema.part_fields
            },
        )

    def _fit_plain_model(self) -> CircuitSize:
        self.plain_model = self._new_model(self.plain_min_samples_per_leaf)
        self.plain_model.fit(self._training_rows())
        return self._size_of(self.plain_model)

    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        stratification = CauseStratification.for_variable(cause_name, self.schema)
        model = self._new_model(self.min_samples_per_leaf, stratification)
        model.fit(self._training_rows())
        self.cause_models[cause_name] = model
        return self._size_of(model)

    def _has_cause_model(self, cause_name: str) -> bool:
        return cause_name in self.cause_models

    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        model = (
            self.plain_model if cause_name is None else self.cause_models[cause_name]
        )
        return RelationalCircuitRegistry(relational_probabilistic_circuit=model)

    def set_monte_carlo_sample_count(self, sample_count: int) -> None:
        """
        Change how many samples grounding draws, on every model fitted so far and on
        every one fitted from now on.

        :param sample_count: The number of samples.
        """
        self.monte_carlo_sample_count = sample_count
        for model in [self.plain_model, *self.cause_models.values()]:
            if model is not None:
                model.monte_carlo_sample_count = sample_count

    @property
    def table(self) -> FlatTable:
        return FlatTable(self.schema, TableLayout.PROPOSITIONAL)

    @property
    def models_parts(self) -> bool:
        return True

    @property
    def order_invariant(self) -> bool:
        return True

    def _part_training_values(self, part: PartAttribute) -> List[Any]:
        return [
            vars(one)[part.attribute]
            for example in self.training_examples
            for one in vars(example)[part.part_field]
        ]

    def plain_circuit_of(self, view: ExampleView) -> Optional[ProbabilisticCircuit]:
        if self.plain_model is None:
            raise PipelineNotFittedError(self.name)
        class_circuit = self.plain_model.class_probabilistic_circuit
        if view is ExampleView.SCALARS:
            scalar_columns = set(self.schema.scalar_columns)
            return class_circuit.marginal(
                [
                    variable
                    for variable in class_circuit.variables
                    if variable.name in scalar_columns
                ]
            )
        return class_circuit

    def log_likelihood(
        self, examples: Sequence[Any], view: ExampleView
    ) -> Optional[LikelihoodReport]:
        """
        Score held-out examples under the plain model, on as much of them as the view
        asks for. A whole example is scored the way the relational circuit factorizes
        it: the class circuit over its scalars and counts, times each part template over
        one object or viewpoint given those counts, the parts taken in canonical order
        so that the sum does not depend on the order the example lists them in.

        :param examples: The examples to score.
        :param view: How much of an example to look at.
        :return: The report.
        """
        if view is not ExampleView.WHOLE:
            return super().log_likelihood(examples, view)
        log_likelihoods = (
            super()
            .log_likelihood(examples, ExampleView.SCALARS_AND_COUNTS)
            .log_likelihoods.copy()
        )
        for part_field in self.plain_model.exchangeable_distribution_templates:
            log_likelihoods += self.part_log_likelihoods(examples, part_field)
        return LikelihoodReport.from_log_likelihoods(log_likelihoods)

    def part_log_likelihoods(
        self, examples: Sequence[Any], part_field: str
    ) -> np.ndarray:
        """
        Score one kind of part of held-out examples under its plain template, given each
        example's counts.

        :param examples: The examples to score.
        :param part_field: The exchangeable-part field.
        :return: One summed log-likelihood per example over its parts of that kind,
            ``-inf`` outside the support.
        :raises PipelineNotFittedError: If :meth:`fit` never ran.
        """
        if self.plain_model is None:
            raise PipelineNotFittedError(self.name)
        template = self.plain_model.exchangeable_distribution_templates[part_field]
        return np.array(
            [
                self._part_log_likelihood(
                    template, example, vars(self._canonical(example))[part_field]
                )
                for example in examples
            ]
        )

    def _part_log_likelihood(
        self,
        template: ExchangeableDistributionTemplate,
        example: Any,
        parts: Sequence[Any],
    ) -> float:
        """
        :param template: The fitted template of one exchangeable part.
        :param example: The example the parts belong to.
        :param parts: The example's parts of that kind.
        :return: The summed log-likelihood of every part given the example's counts.
        """
        circuit = template.template_distribution.class_probabilistic_circuit
        latents = template.latent_variables
        counts = {
            variable.name: self.table.row(example)[variable.name]
            for variable in latents
        }
        if not parts:
            return 0.0
        given_counts = log_likelihoods_of_rows(circuit.marginal(latents), [counts])[0]
        if not np.isfinite(given_counts):
            return -np.inf
        rows = [{**counts, **vars(part)} for part in parts]
        joint = log_likelihoods_of_rows(circuit, rows)
        return float(joint.sum() - len(parts) * given_counts)


# %% flat-table pipeline


@dataclass
class FlatTableRegistry(ModelRegistry):
    """
    Serves a circuit fitted on the flat table for queries over that table's columns,
    wrapped as a causal circuit when the query marks a cause.

    A query may list an example's objects and viewpoints, which the table has no columns
    for, as long as it says nothing about them; the served circuit then simply lacks
    them.
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
        :raises FlatTableSchemaMismatchError: If the query constrains a variable the
            table never had.
        """
        fitted_names = {variable.name for variable in self.circuit.variables}
        missing = sorted(constrained_variable_names(parameters) - fitted_names)
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
    Joint probability trees fitted on the examples flattened into one table, each wrapped
    as a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    layout: TableLayout = TableLayout.PROPOSITIONAL
    """
    What the table the examples are flattened into holds besides their scalars.
    """

    part_widths: Dict[str, int] = field(default_factory=dict)
    """
    Per unrolled part field, how many positions the table has; empty unless the layout
    unrolls the parts.
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
        return f"{self.layout} tree"

    @property
    def table(self) -> FlatTable:
        return FlatTable(self.schema, self.layout, part_widths=self.part_widths)

    @property
    def models_parts(self) -> bool:
        return self.layout.has_parts

    @property
    def order_invariant(self) -> bool:
        return not self.layout.has_parts

    def _part_training_values(self, part: PartAttribute) -> List[Any]:
        column = self.schema.part_column(part)
        return [self.table.row(example)[column] for example in self.training_examples]

    def _training_dataframe(self) -> pd.DataFrame:
        return self.table.dataframe(self.training_examples)

    def _fit_plain_model(self) -> CircuitSize:
        dataframe = self._training_dataframe()
        self.plain_circuit = JointProbabilityTree(
            min_samples_per_leaf=self.plain_min_samples_per_leaf
        ).fit(dataframe, infer_variables_from_dataframe(dataframe))
        return CircuitSize.of(self.plain_circuit)

    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        if cause_name not in self.table.columns:
            raise FlatTableSchemaMismatchError([cause_name])
        dataframe = self._training_dataframe()
        self.cause_circuits[cause_name] = StratifiedLearning(
            variables=[cause_name],
            method=JointProbabilityTree(min_samples_per_leaf=self.min_samples_per_leaf),
        ).fit(dataframe, infer_variables_from_dataframe(dataframe))
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

    def plain_circuit_of(self, view: ExampleView) -> Optional[ProbabilisticCircuit]:
        if self.plain_circuit is None:
            raise PipelineNotFittedError(self.name)
        columns = self.table.columns_of(view)
        if columns is None:
            return None
        column_set = set(columns)
        return self.plain_circuit.marginal(
            [
                variable
                for variable in self.plain_circuit.variables
                if variable.name in column_set
            ]
        )


# %% hybrid pipeline


@dataclass
class HybridRegistry(ModelRegistry):
    """
    Routes a query to the model of the part it constrains: the relational circuit for a
    query about an exchangeable part, the positional tree for everything else.
    """

    pipeline: HybridPipeline
    """
    The pipeline whose models are served.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        cause_name = cause_variable_name(parameters)
        constrained = constrained_variable_names(parameters)
        about_an_exchangeable_part = any(
            (part := self.pipeline.schema.part_attribute(name)) is not None
            and part.part_field in self.pipeline.exchangeable_fields
            for name in constrained
        )
        member = (
            self.pipeline.exchangeable
            if about_an_exchangeable_part
            else self.pipeline.positional
        )
        return member.registry_for(cause_name).get_model(parameters)


@dataclass
class HybridPipeline(CausalQueryPipeline):
    """
    Some parts exchangeable, the rest positional: a relational circuit over the
    exchangeable parts and a tree over the example's scalars, its counts and the
    positional parts by position, for a relation where a position genuinely means the
    same thing in every example.

    A whole example is scored as the tree over scalars, counts and positional parts
    times each exchangeable part's template over the parts given the counts. A question
    about an exchangeable part goes to the relational circuit, whose answers cannot
    depend on the order the parts are listed in; every other question goes to the tree.
    """

    positional_fields: Sequence[str] = ()
    """
    The exchangeable-part fields held by position.
    """

    exchangeable: Optional[RelationalPipeline] = None
    """
    The relational circuit, whose templates over the exchangeable parts are used; built
    at :meth:`fit`.
    """

    positional: Optional[FlatTablePipeline] = None
    """
    The tree over the scalars, the counts and the positional parts, sized to the
    training examples at :meth:`fit`.
    """

    @property
    def name(self) -> str:
        return "hybrid circuit"

    @property
    def exchangeable_fields(self) -> List[str]:
        """
        The exchangeable-part fields the relational circuit keeps as templates.
        """
        return [
            part_field
            for part_field in self.domain.part_fields
            if part_field not in self.positional_fields
        ]

    @property
    def registry(self) -> ModelRegistry:
        return HybridRegistry(pipeline=self)

    @property
    def table(self) -> FlatTable:
        if self.positional is None:
            raise PipelineNotFittedError(self.name)
        return self.positional.table

    @property
    def models_parts(self) -> bool:
        return True

    @property
    def order_invariant(self) -> bool:
        return False

    def fit(self, examples: Sequence[Any]) -> FitReport:
        self.training_examples = list(examples)
        self.positional = FlatTablePipeline(
            domain=self.domain,
            layout=TableLayout.UNROLLED,
            part_widths=FlatTable.unrolled_for(
                self.schema, examples, part_fields=self.positional_fields
            ).part_widths,
            min_samples_per_leaf=self.min_samples_per_leaf,
            plain_min_samples_per_leaf=self.plain_min_samples_per_leaf,
        )
        self.exchangeable = RelationalPipeline(
            domain=self.domain,
            min_samples_per_leaf=self.min_samples_per_leaf,
            plain_min_samples_per_leaf=self.plain_min_samples_per_leaf,
        )
        started = time.perf_counter()
        size = self._fit_plain_model()
        self.fit_report = FitReport(training_example_count=len(examples))
        self.fit_report.record(time.perf_counter() - started, size)
        return self.fit_report

    def _fit_plain_model(self) -> CircuitSize:
        self.positional.fit(self.training_examples)
        self.exchangeable.fit(self.training_examples)
        return self.positional.fit_report.size + self.exchangeable.fit_report.size

    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        raise NotImplementedError("The members fit their own cause models.")

    def _has_cause_model(self, cause_name: str) -> bool:
        raise NotImplementedError("The members keep their own cause models.")

    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        raise NotImplementedError("The registry routes by what the query constrains.")

    def registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        return self.registry

    def _part_training_values(self, part: PartAttribute) -> List[Any]:
        member = (
            self.exchangeable
            if part.part_field in self.exchangeable_fields
            else self.positional
        )
        return member.training_values_of(self.schema.part_column(part))

    def plain_circuit_of(self, view: ExampleView) -> Optional[ProbabilisticCircuit]:
        return self.positional.plain_circuit_of(view)

    def log_likelihood(
        self, examples: Sequence[Any], view: ExampleView
    ) -> Optional[LikelihoodReport]:
        """
        Score held-out examples, on as much of them as the view asks for: a whole
        example as the tree over its scalars, counts and positional parts times each
        exchangeable part's template over the parts given the counts.

        :param examples: The examples to score.
        :param view: How much of an example to look at.
        :return: The report.
        """
        if view is not ExampleView.WHOLE:
            return self.positional.log_likelihood(examples, view)
        log_likelihoods = self.positional.log_likelihood(
            examples, view
        ).log_likelihoods.copy()
        for part_field in self.exchangeable_fields:
            log_likelihoods += self.exchangeable.part_log_likelihoods(
                examples, part_field
            )
        return LikelihoodReport.from_log_likelihoods(log_likelihoods)


def pipelines(
    domain: RelationalDomain,
    examples: Sequence[Any],
    positional_fields: Sequence[str] = (),
) -> List[CausalQueryPipeline]:
    """
    :param domain: The example and its parts.
    :param examples: The examples the pipelines will be fitted on, which size the
        unrolled table.
    :param positional_fields: The part fields a position means the same thing in for
        every example; a hybrid circuit holding them by position is compared too when
        there are any.
    :return: Every pipeline, unfitted: the relational circuit, the hybrid circuit if
        asked for, and one flat-table tree per layout.
    """
    schema = Schema(domain)
    compared: List[CausalQueryPipeline] = [RelationalPipeline(domain=domain)]
    if positional_fields:
        compared.append(
            HybridPipeline(domain=domain, positional_fields=tuple(positional_fields))
        )
    compared += [
        FlatTablePipeline(domain=domain, layout=TableLayout.PROPOSITIONAL),
        FlatTablePipeline(
            domain=domain,
            layout=TableLayout.UNROLLED,
            part_widths=FlatTable.unrolled_for(schema, examples).part_widths,
        ),
        FlatTablePipeline(domain=domain, layout=TableLayout.SCALARS),
    ]
    return compared
