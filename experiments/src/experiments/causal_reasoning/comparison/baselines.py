"""
A causal estimator that is not built on circuits, as a reference point for the questions
it can express.

Regression adjustment is the textbook backdoor estimator: fit a model of the effect on
the cause and the confounders, then average its prediction at each value of the cause
over the confounders' distribution in the data. Here the model is a logistic regression
on the propositional table, so it can be asked the same example-level questions as the
propositional tree, and refuses the same ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Any, List, Optional, Sequence

from experiments.causal_reasoning.comparison.domain import RelationalDomain
from experiments.causal_reasoning.comparison.evaluation import (
    InterventionalEffect,
    QueryOutcome,
    Refusal,
)
from experiments.causal_reasoning.comparison.flat_table import (
    FlatTable,
    Schema,
    TableLayout,
)
from experiments.causal_reasoning.comparison.pipelines import (
    CircuitSize,
    FitReport,
    cause_variable_name,
    constrained_variable_names,
)
from experiments.causal_reasoning.comparison.queries import CausalQueryCase


@dataclass
class RegressionAdjustmentBaseline:
    """
    Logistic regression adjustment on the propositional table.
    """

    domain: RelationalDomain
    """
    The example and its parts.
    """

    regularisation: float = 1.0
    """
    The inverse regularisation strength of the logistic regression.
    """

    min_region_support: int = 10
    """
    The fewest training examples a value of the cause may hold for its effect to be read
    as an answer.
    """

    training_examples: List[Any] = field(default_factory=list)
    """
    The examples :meth:`fit` was given.
    """

    dataframe: Optional[pd.DataFrame] = None
    """
    The training examples as rows, once :meth:`fit` ran.
    """

    fit_report: Optional[FitReport] = None
    """
    What the fits so far cost, once :meth:`fit` ran; one regression per question.
    """

    @property
    def name(self) -> str:
        """
        What the baseline is called in reports.
        """
        return "regression adjustment"

    @property
    def order_invariant(self) -> bool:
        """
        Whether its answers are independent of the order the parts are listed in.
        """
        return True

    @property
    def table(self) -> FlatTable:
        """
        The propositional table the examples are flattened into.
        """
        return FlatTable(Schema(self.domain), TableLayout.PROPOSITIONAL)

    def fit(self, examples: Sequence[Any]) -> FitReport:
        """
        Flatten the examples; every question fits its own regression on them.

        :param examples: The examples to fit on.
        :return: The report the questions' fits keep adding to.
        """
        self.training_examples = list(examples)
        started = time.perf_counter()
        self.dataframe = self.table.dataframe(self.training_examples)
        self.fit_report = FitReport(training_example_count=len(examples))
        self.fit_report.record(time.perf_counter() - started, CircuitSize(0, 0))
        return self.fit_report

    def ask(self, case: CausalQueryCase) -> QueryOutcome:
        """
        Answer one question by regression adjustment.

        :param case: The question.
        :return: What came of it: an effect per value of the cause, or a refusal where
            the question names a variable the table has no column for.
        """
        started = time.perf_counter()
        parameters = UnderspecifiedParameters(case.build())
        missing = sorted(
            constrained_variable_names(parameters) - set(self.table.columns)
        )
        if missing:
            return QueryOutcome(
                case=case,
                pipeline_name=self.name,
                duration=time.perf_counter() - started,
                min_region_support=self.min_region_support,
                refusal=Refusal.SCHEMA_MISMATCH,
            )
        cause = cause_variable_name(parameters)
        confounders = [
            variable.name for variable in parameters.search_confounder_variables
        ]
        [effect_variable] = parameters.effect_variables_from_causes_effect
        effect_event = parameters.truncation_assignments_from_where_conditions
        [simple_event] = effect_event.simple_sets
        effect = self.dataframe[effect_variable.name].map(
            lambda value: not effect_variable.make_value(value)
            .intersection_with(simple_event[effect_variable])
            .is_empty()
        )
        effects = self._effects(cause, confounders, effect)
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

    def _effects(
        self, cause: str, confounders: List[str], effect: pd.Series
    ) -> List[InterventionalEffect]:
        """
        :param cause: The cause column.
        :param confounders: The confounder columns.
        :param effect: Per training row, whether the effect holds.
        :return: The effect's naive and adjusted probability at every value of the
            cause the training rows hold.
        """
        features = self.dataframe[[cause] + confounders]
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        symbolic = [
            column for column in features.columns if features[column].dtype == object
        ]
        numeric = [column for column in features.columns if column not in symbolic]

        def design(frame: pd.DataFrame) -> np.ndarray:
            columns = [frame[numeric].to_numpy(dtype=float)]
            if symbolic:
                columns.append(encoder.transform(frame[symbolic].astype(str)))
            return np.hstack(columns)

        if symbolic:
            encoder.fit(features[symbolic].astype(str))
        outcome = effect.to_numpy(dtype=int)
        regression = LogisticRegression(C=self.regularisation, max_iter=1000)
        if outcome.min() != outcome.max():
            regression.fit(design(features), outcome)

        def predict(frame: pd.DataFrame) -> np.ndarray:
            if outcome.min() == outcome.max():
                return np.full(len(frame), float(outcome[0]))
            return regression.predict_proba(design(frame))[:, 1]

        effects = []
        for value in sorted(features[cause].unique(), key=self._order):
            in_region = features[cause] == value
            forced = features.copy()
            forced[cause] = value
            effects.append(
                InterventionalEffect(
                    cause_region=str(value),
                    region_probability=float(in_region.mean()),
                    naive_probability=float(effect[in_region].mean()),
                    adjusted_probability=float(predict(forced).mean()),
                    support_count=int(in_region.sum()),
                    ordinal=(
                        float(value)
                        if isinstance(value, (int, float, np.integer))
                        else None
                    ),
                )
            )
        return effects

    @staticmethod
    def _order(value: Any) -> Any:
        """
        :param value: A value of the cause.
        :return: A sort key putting numbers in numeric order and symbols by name.
        """
        if isinstance(value, (int, float, np.integer)):
            return (0, float(value), "")
        return (1, 0.0, str(value))
