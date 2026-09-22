"""
A permutation invariant neural estimator, as a reference point that is order free
without being a circuit.

The estimator is a deep set. Every part of a collection is encoded on its own, the
encodings are pooled by mean and by maximum so that nothing the network reads depends on
the order the parts were listed in, and a multilayer perceptron reads the pooled
encoding together with the example's own fields. The causal step is the same one the
regression baseline takes, g computation, so that the estimator and not the estimand is
what differs from the circuits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Any, Dict, List, Optional, Sequence, Tuple

from experiments.causal_reasoning.comparison.domain import RelationalDomain
from experiments.causal_reasoning.comparison.evaluation import (
    InterventionalEffect,
    QueryOutcome,
    Refusal,
)
from experiments.causal_reasoning.comparison.flat_table import (
    FlatTable,
    PartAttribute,
    Schema,
    TableLayout,
)
from experiments.causal_reasoning.comparison.pipelines import (
    CircuitSize,
    FitReport,
    cause_variable_name,
)
from experiments.causal_reasoning.comparison.queries import CausalQueryCase

POOLED_SEPARATOR = "."
"""
What separates a part field from the pooled feature taken over it.
"""

CAUSE_COLUMN = "cause"
"""
What the cause is called among the network's features, whatever the query names it.
"""

# %% encoding values


def is_numeric(value: Any) -> bool:
    """
    :param value: One value of a field.
    :return: Whether it is a number the network can read without encoding.
    """
    return isinstance(value, (bool, int, float, np.integer, np.floating))


@dataclass
class RowEncoder:
    """
    Turns a frame of raw field values into a numeric design matrix, one hot for the
    columns whose values are symbols.
    """

    symbolic_columns: List[str] = field(default_factory=list)
    """
    The columns encoded one hot.
    """

    numeric_columns: List[str] = field(default_factory=list)
    """
    The columns read as numbers.
    """

    encoder: Optional[OneHotEncoder] = None
    """
    The fitted one hot encoder, once :meth:`fitted_on` ran.
    """

    @classmethod
    def fitted_on(cls, frame: pd.DataFrame) -> RowEncoder:
        """
        :param frame: The training rows.
        :return: An encoder fitted on them.
        """
        symbolic = [
            column
            for column in frame.columns
            if not all(is_numeric(value) for value in frame[column])
        ]
        numeric = [column for column in frame.columns if column not in symbolic]
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if symbolic:
            encoder.fit(frame[symbolic].astype(str))
        return cls(symbolic_columns=symbolic, numeric_columns=numeric, encoder=encoder)

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        """
        :param frame: Rows with the columns this encoder was fitted on.
        :return: The rows as numbers.
        """
        blocks = [frame[self.numeric_columns].to_numpy(dtype=float)]
        if self.symbolic_columns:
            blocks.append(
                self.encoder.transform(frame[self.symbolic_columns].astype(str))
            )
        return np.hstack(blocks)


def pooled_features(
    parts: Sequence[Any], attributes: Sequence[str], categories: Dict[str, List[Any]]
) -> Dict[str, float]:
    """
    Mean and maximum of every attribute of a collection's parts, which is what makes the
    encoding independent of the order the parts are listed in.

    :param parts: One example's parts of one field.
    :param attributes: The attributes to pool.
    :param categories: Per symbolic attribute, the values seen in training.
    :return: The pooled features, by name.
    """
    pooled: Dict[str, float] = {"count": float(len(parts))}
    for attribute in attributes:
        values = [vars(part)[attribute] for part in parts]
        if attribute in categories:
            for category in categories[attribute]:
                indicators = [float(value == category) for value in values]
                pooled[f"{attribute}={category}.mean"] = (
                    float(np.mean(indicators)) if indicators else 0.0
                )
                pooled[f"{attribute}={category}.max"] = (
                    float(np.max(indicators)) if indicators else 0.0
                )
            continue
        numbers = [float(value) for value in values]
        pooled[f"{attribute}.mean"] = float(np.mean(numbers)) if numbers else 0.0
        pooled[f"{attribute}.max"] = float(np.max(numbers)) if numbers else 0.0
    return pooled


# %% the estimator


@dataclass
class NeuralAdjustmentBaseline:
    """
    A deep set over the parts with g computation on top.
    """

    domain: RelationalDomain
    """
    The example and its parts.
    """

    min_region_support: int = 10
    """
    The fewest training rows a value of the cause may hold for its effect to be read as
    an answer.
    """

    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    """
    The widths of the perceptron's hidden layers.
    """

    max_iterations: int = 1000
    """
    How many optimiser steps the perceptron may take.
    """

    random_seed: int = 0
    """
    Seed of the weight initialisation, so that a fit is reproducible.
    """

    training_examples: List[Any] = field(default_factory=list)
    """
    The examples :meth:`fit` was given.
    """

    example_frame: Optional[pd.DataFrame] = None
    """
    Per training example, its own fields and its aggregation statistics.
    """

    pooled_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    """
    Per part field, the pooled encoding of every training example's parts.
    """

    part_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    """
    Per part field, one row per part of every training example, with the index of the
    example it belongs to.
    """

    fit_report: Optional[FitReport] = None
    """
    What the fits so far cost, once :meth:`fit` ran; one network per question.
    """

    @property
    def name(self) -> str:
        """
        What the estimator is called in reports.
        """
        return "neural adjustment"

    @property
    def order_invariant(self) -> bool:
        """
        Whether its answers are independent of the order the parts are listed in.
        """
        return True

    @property
    def schema(self) -> Schema:
        """
        How the example's attributes are named.
        """
        return Schema(self.domain)

    def fit(self, examples: Sequence[Any]) -> FitReport:
        """
        Encode the examples and their parts; every question fits its own network on
        them.

        :param examples: The examples to fit on.
        :return: The report the questions' fits keep adding to.
        """
        self.training_examples = list(examples)
        started = time.perf_counter()
        schema = self.schema
        self.example_frame = FlatTable(schema, TableLayout.PROPOSITIONAL).dataframe(
            self.training_examples
        )
        self.pooled_frames = {}
        self.part_frames = {}
        for part_field in self.domain.part_fields:
            attributes = list(self.domain.part_attribute_types(part_field))
            categories = self._categories(part_field, attributes)
            self.pooled_frames[part_field] = pd.DataFrame(
                [
                    pooled_features(
                        self.domain.parts_of(example, part_field),
                        attributes,
                        categories,
                    )
                    for example in self.training_examples
                ]
            ).add_prefix(f"{part_field}{POOLED_SEPARATOR}")
            self.part_frames[part_field] = pd.DataFrame(
                [
                    {
                        "example": index,
                        **{name: vars(part)[name] for name in attributes},
                    }
                    for index, example in enumerate(self.training_examples)
                    for part in self.domain.parts_of(example, part_field)
                ]
            )
        self.fit_report = FitReport(training_example_count=len(examples))
        self.fit_report.record(time.perf_counter() - started, CircuitSize(0, 0))
        return self.fit_report

    def _categories(
        self, part_field: str, attributes: Sequence[str]
    ) -> Dict[str, List[Any]]:
        """
        :param part_field: An exchangeable-part field.
        :param attributes: Its parts' attributes.
        :return: Per attribute whose values are symbols, the values seen in training.
        """
        categories: Dict[str, List[Any]] = {}
        for attribute in attributes:
            values = [
                vars(part)[attribute]
                for example in self.training_examples
                for part in self.domain.parts_of(example, part_field)
            ]
            if values and not all(is_numeric(value) for value in values):
                categories[attribute] = sorted(set(values), key=str)
        return categories

    def ask(self, case: CausalQueryCase) -> QueryOutcome:
        """
        Answer one question by g computation on the network.

        :param case: The question.
        :return: What came of it, or a refusal where the question names something the
            encoding does not hold.
        """
        started = time.perf_counter()
        parameters = UnderspecifiedParameters(case.build())
        cause_name = cause_variable_name(parameters)
        confounders = [
            variable.name for variable in parameters.search_confounder_variables
        ]
        [effect_variable] = parameters.effect_variables_from_causes_effect
        schema = self.schema
        effect_part = schema.part_attribute(effect_variable.name)
        cause_part = schema.part_attribute(cause_name)
        if any(schema.part_attribute(name) is not None for name in confounders):
            return self._refused(case, started)
        if cause_part is not None and effect_part is None:
            return self._refused(case, started)
        if effect_part is not None and cause_part is not None:
            if cause_part.part_field != effect_part.part_field:
                return self._refused(case, started)
        [simple_event] = (
            parameters.truncation_assignments_from_where_conditions.simple_sets
        )
        frame, effect = self._rows(
            cause_name,
            cause_part,
            confounders,
            effect_variable,
            simple_event[effect_variable],
            effect_part,
        )
        effects = self._effects(frame, effect, cause_column=CAUSE_COLUMN)
        self.fit_report.record(0.0, CircuitSize(0, 0))
        outcome = QueryOutcome(
            case=case,
            pipeline_name=self.name,
            duration=time.perf_counter() - started,
            min_region_support=self.min_region_support,
            effects=effects,
        )
        best = outcome.most_effective
        if best is not None:
            outcome.best_region = best.cause_region
            outcome.effect_probability_given_best_region = best.adjusted_probability
        return outcome

    def _refused(self, case: CausalQueryCase, started: float) -> QueryOutcome:
        """
        :param case: The question.
        :param started: When it was asked.
        :return: The question refused for want of a feature.
        """
        return QueryOutcome(
            case=case,
            pipeline_name=self.name,
            duration=time.perf_counter() - started,
            min_region_support=self.min_region_support,
            refusal=Refusal.SCHEMA_MISMATCH,
        )

    def _rows(
        self,
        cause_name: str,
        cause_part: Optional[PartAttribute],
        confounders: Sequence[str],
        effect_variable: Any,
        effect_values: Any,
        effect_part: Optional[PartAttribute],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Assemble the rows the network is fitted on.

        A question about the example itself gives one row per example. A question whose
        effect lives on a part gives one row per part of that collection, so that the
        network reads the part it is asked about together with the pooled encoding of
        the example it sits in. Where the cause is an aggregation statistic, the pooled
        encoding of the collection it aggregates is withheld, so that the statistic is
        the only route from the parts to the prediction.

        :param cause_name: The cause variable.
        :param cause_part: The part attribute the cause is, if it is one.
        :param confounders: The confounder variables.
        :param effect_variable: The effect's variable.
        :param effect_values: The values of that variable the effect asks for.
        :param effect_part: The part attribute the effect is, if it is one.
        :return: The rows and, per row, whether the effect holds.
        """
        aggregated = self._aggregated_field(cause_name)
        kept = [
            frame
            for part_field, frame in self.pooled_frames.items()
            if part_field != aggregated
        ]
        pooled = (
            pd.concat(kept, axis=1)
            if kept
            else pd.DataFrame(index=range(len(self.training_examples)))
        )
        if effect_part is None:
            frame = pooled.copy()
            frame[CAUSE_COLUMN] = self.example_frame[cause_name].to_numpy()
            for confounder in confounders:
                frame[confounder] = self.example_frame[confounder].to_numpy()
            effect = self._holds(
                self.example_frame[effect_variable.name], effect_variable, effect_values
            )
            return frame, effect.to_numpy()
        parts = self.part_frames[effect_part.part_field]
        rows = pooled.iloc[parts["example"].to_numpy()].reset_index(drop=True)
        spoken_for = {effect_part.attribute}
        if cause_part is not None:
            spoken_for.add(cause_part.attribute)
        for attribute in self.domain.part_attribute_types(effect_part.part_field):
            if attribute in spoken_for:
                continue
            rows[f"part.{attribute}"] = parts[attribute].to_numpy()
        if cause_part is None:
            rows[CAUSE_COLUMN] = self.example_frame[cause_name].to_numpy()[
                parts["example"].to_numpy()
            ]
        else:
            rows[CAUSE_COLUMN] = parts[cause_part.attribute].to_numpy()
        for confounder in confounders:
            rows[confounder] = self.example_frame[confounder].to_numpy()[
                parts["example"].to_numpy()
            ]
        effect = self._holds(
            parts[effect_part.attribute], effect_variable, effect_values
        )
        return rows, effect.to_numpy()

    def _aggregated_field(self, cause_name: str) -> Optional[str]:
        """
        :param cause_name: The cause variable.
        :return: The part field the cause aggregates over, if the cause is an
            aggregation statistic.
        """
        for part_field in self.domain.part_fields:
            for statistic in self.domain.aggregation_class.aggregation_registry[
                part_field
            ]:
                if self.schema.aggregation_column(statistic.__name__) == cause_name:
                    return part_field
        return None

    @staticmethod
    def _holds(
        values: pd.Series, effect_variable: Any, effect_values: Any
    ) -> pd.Series:
        """
        :param values: The effect field's value per row.
        :param effect_variable: The effect's variable.
        :param effect_values: The values of that variable the effect asks for.
        :return: Per row, whether the effect holds.
        """
        return values.map(
            lambda value: not effect_variable.make_value(value)
            .intersection_with(effect_values)
            .is_empty()
        )

    def _effects(
        self, frame: pd.DataFrame, effect: np.ndarray, cause_column: str
    ) -> List[InterventionalEffect]:
        """
        :param frame: The rows, one column being the cause.
        :param effect: Per row, whether the effect holds.
        :param cause_column: The cause's column.
        :return: The effect's naive and adjusted probability at every value of the
            cause the training rows hold.
        """
        encoder = RowEncoder.fitted_on(frame)
        outcome = effect.astype(int)
        network = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iterations,
            random_state=self.random_seed,
        )
        constant = outcome.min() == outcome.max()
        if not constant:
            network.fit(encoder.design(frame), outcome)

        def predict(rows: pd.DataFrame) -> np.ndarray:
            if constant:
                return np.full(len(rows), float(outcome[0]))
            return network.predict_proba(encoder.design(rows))[:, 1]

        effects = []
        for value in sorted(frame[cause_column].unique(), key=order_of):
            in_region = frame[cause_column] == value
            forced = frame.copy()
            forced[cause_column] = value
            effects.append(
                InterventionalEffect(
                    cause_region=str(value),
                    region_probability=float(in_region.mean()),
                    naive_probability=float(effect[in_region.to_numpy()].mean()),
                    adjusted_probability=float(predict(forced).mean()),
                    support_count=int(in_region.sum()),
                    ordinal=float(value) if is_numeric(value) else None,
                )
            )
        return effects


def order_of(value: Any) -> Tuple[int, float, str]:
    """
    :param value: A value of the cause.
    :return: A sort key putting numbers in numeric order ahead of symbols by name.
    """
    if is_numeric(value):
        return (0, float(value), "")
    return (1, 0.0, str(value))
