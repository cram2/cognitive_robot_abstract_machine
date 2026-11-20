"""
The probabilistic models learned from trials, which the queries of the entity query
language ask what made the trials succeed and where to try next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
)
from probabilistic_model.causal_structure.learning import (
    CausalStructureLearner,
    LearnedCausalStructure,
    PeterClarkStructureLearner,
    RelationalCausalStructureLearner,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.kernel_density import KernelDensity
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.factories import variable
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from krrood.parametrization.model_registries import (
    CausalCircuitRegistry,
    DictRegistry,
)
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from krrood.parametrization.trials.exceptions import NoSuccessfulTrials
from krrood.parametrization.trials.records import Trial

# %% what the trials of one behaviour tell


@dataclass(kw_only=True)
class TrialModel:
    """
    A joint model of the parameters, measurements and outcomes of trials, with the
    causal structure learned from them, for queries of what made them succeed.

    Parameters come before measurements and measurements before the outcome, so no
    learned edge points back in time.
    """

    trials: list[Trial]
    """
    The trials learned from, all of the same type.
    """

    learning_method: LearningMethod = field(
        default_factory=lambda: JointProbabilityTree(minimum_samples_per_leaf=20)
    )
    """
    What the joint model is fitted with; leaves of at least 20 trials keep a success
    probability from resting on a handful of trials.
    """

    structure_learner: CausalStructureLearner = field(
        default_factory=PeterClarkStructureLearner
    )
    """
    What learns the causal structure, keeping an edge only where its independence test
    rejects independence at its significance level.
    """

    @cached_property
    def trial_type(self) -> type[Trial]:
        """
        The type of the trials.
        """
        return type(self.trials[0])

    @cached_property
    def circuit(self) -> RelationalProbabilisticCircuit:
        """
        The joint model of every feature of the trials.
        """
        return RelationalProbabilisticCircuit(
            self.trial_type, learning_method=self.learning_method
        ).fit(self.trials)

    @cached_property
    def registry(self) -> DictRegistry:
        """
        Answers queries of the probabilities of features of the trials.
        """
        return DictRegistry(
            models={self.trial_type: self.circuit.class_probabilistic_circuit}
        )

    @cached_property
    def causal_registry(self) -> CausalCircuitRegistry:
        """
        Answers causal queries of the trials, with every parameter and measurement as a
        cause that can be searched.
        """
        class_circuit = self.circuit.class_probabilistic_circuit
        trial = variable(self.trial_type, domain=[])
        cause_prefixes = (
            f"{trial.parameters._name_}.",
            f"{trial.measurements._name_}.",
        )
        causes = [
            circuit_variable
            for circuit_variable in class_circuit.variables
            if circuit_variable.name.startswith(cause_prefixes)
        ]
        [outcome] = [
            circuit_variable
            for circuit_variable in class_circuit.variables
            if circuit_variable.name == trial.succeeded._name_
        ]
        return CausalCircuitRegistry(
            circuits={
                self.trial_type: CausalCircuit.from_probabilistic_circuit(
                    class_circuit,
                    MarginalDeterminismTreeNode.from_variable_order(causes),
                    causes,
                    [outcome],
                )
            }
        )

    def domain_knowledge(self) -> DomainCausalKnowledge:
        """
        :return: The order in time of the features: the starting state, then the
            parameters, the measurements and last the outcome.
        """
        trial = variable(self.trial_type, domain=[])
        names = [
            circuit_variable.name
            for circuit_variable in self.circuit.class_probabilistic_circuit.variables
        ]
        return DomainCausalKnowledge(
            tiers=[
                [name for name in names if name == trial.starting_state_id._name_],
                [
                    name
                    for name in names
                    if name.startswith(f"{trial.parameters._name_}.")
                ],
                [
                    name
                    for name in names
                    if name.startswith(f"{trial.measurements._name_}.")
                ],
                [name for name in names if name == trial.succeeded._name_],
            ]
        )

    @cached_property
    def causal_structure(self) -> LearnedCausalStructure:
        """
        The causal structure of the features of the trials.
        """
        return RelationalCausalStructureLearner(
            ClassDiagram(classes=self.classes_of_the_trials()),
            self.structure_learner,
        ).learn(self.circuit, self.trials, self.domain_knowledge())

    def classes_of_the_trials(self) -> list[type]:
        """
        :return: The type of the trials and every class a feature of them belongs to.
        """
        return list(
            dict.fromkeys(
                [
                    self.trial_type,
                    *(
                        feature._owner_class_
                        for feature in self.circuit.feature_extractor.features
                    ),
                ]
            )
        )

    @cached_property
    def causal_backend(self) -> ProbabilisticBackend:
        """
        Answers causal queries of the trials, adjusting by the learned structure; a
        query that marks no cause ranks only the features the structure joins to the
        effect.
        """
        return ProbabilisticBackend(
            model_registry=self.causal_registry,
            causal_structures={self.trial_type: self.causal_structure},
        )

    def registry_near_successes(
        self, *, learning_method: LearningMethod | None = None
    ) -> DictRegistry:
        """
        :param learning_method: What the parameters of the successful trials are
            fitted with; by default a kernel on each, as wide as the
            parameters of all trials vary.
        :return: Draws the parameters of the next trials near those that succeeded.
        :raises NoSuccessfulTrials: If no trial succeeded.
        """
        successful = [trial.parameters for trial in self.trials if trial.succeeded]
        if not successful:
            raise NoSuccessfulTrials(number_of_trials=len(self.trials))
        every_parameters = [trial.parameters for trial in self.trials]
        extractor = FeatureExtractor.from_instances(instances=every_parameters)
        model = (learning_method or KernelDensity()).fit(
            extractor.create_dataframe(instances=successful),
            infer_variables_from_dataframe(
                extractor.create_dataframe(instances=every_parameters)
            ),
        )
        return DictRegistry(models={type(successful[0]): model})
