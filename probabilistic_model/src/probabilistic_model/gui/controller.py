from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import json

from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.product_algebra import VariableMap, Variable, SimpleEvent, Event


@dataclass
class ModelController:
    """
    Controller for managing the state and operations of a ProbabilisticModel.
    """

    model: Optional[ProbabilisticModel] = None
    """
    The probabilistic model being managed.
    """

    variable_map: Dict[str, Variable] = field(default_factory=dict)
    """
    A mapping from variable names to variable objects.
    """

    priors: Optional[VariableMap] = None
    """
    The prior distributions for all variables in the model.
    """

    def __post_init__(self):
        """
        Initializes the controller. If a model is provided, sets it.
        """
        if self.model:
            self.set_model(self.model)

    def set_model(self, model: ProbabilisticModel) -> None:
        """
        Sets the current model and updates related states.
        """
        self.model = model
        self.variable_map = {variable.name: variable for variable in model.variables}
        self.priors = self.calculate_priors()

    def calculate_priors(self) -> VariableMap:
        """
        Calculates the prior distributions (marginals) for all variables in the model.
        """
        if self.model is None:
            return VariableMap()

        prior_distributions = VariableMap()
        for variable in self.model.variables:
            prior_distributions[variable] = self.model.marginal([variable])
        return prior_distributions

    def calculate_probability(self, query: Event, evidence: Event) -> float:
        """
        Calculates P(query | evidence) = P(query AND evidence) / P(evidence).
        """
        if self.model is None:
            return 0.0

        if not evidence.simple_sets:
            # P(query | True) = P(query)
            return self.model.probability(query)

        prob_evidence = self.model.probability(evidence)
        if prob_evidence == 0:
            return 0.0

        # P(query | evidence) = P(query ^ evidence) / P(evidence)
        joint = query.intersection_with(evidence)
        if not joint.simple_sets:
            return 0.0

        prob_joint = self.model.probability(joint)
        return prob_joint / prob_evidence

    def calculate_posterior(self, evidence: Event) -> Optional[ProbabilisticModel]:
        """
        Calculates the posterior distribution given evidence.
        """
        if self.model is None:
            return None

        if not evidence.simple_sets:
            # If evidence is empty, return the model itself
            return self.model

        try:
            conditioned_model, probability = self.model.truncated(evidence)
            if probability == 0:
                return None
            return conditioned_model
        except Exception:
            return None

    def calculate_mode(self, evidence: Event) -> Optional[Tuple[Event, float]]:
        """
        Calculates the mode of the model given evidence.
        """
        if self.model is None:
            return None

        if not evidence.simple_sets:
            # If evidence is empty, return the mode of the original model
            return self.model.mode()

        try:
            conditioned_model, probability = self.model.truncated(evidence)
            if probability == 0:
                return None
            return conditioned_model.mode()
        except Exception:
            return None

    def load_model_from_json_file(self, file_path: str) -> None:
        """
        Loads a ProbabilisticCircuit model from a JSON file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
            # Currently assuming ProbabilisticCircuit as in the original app.py
            model = ProbabilisticCircuit.from_json(data)
            self.set_model(model)
