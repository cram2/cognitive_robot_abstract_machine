from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Any

from probabilistic_model.exceptions import ProbabilisticModelException

if TYPE_CHECKING:
    from random_events.variable import Variable


@dataclass
class CircuitNotFittedError(ProbabilisticModelException):
    """
    Raised when a RelationalProbabilisticCircuit is grounded before it has been fitted.
    """

    def error_message(self) -> str:
        return (
            "A RelationalProbabilisticCircuit must be fitted before it can be grounded."
        )

    def suggest_correction(self) -> str:
        return "Call `fit` with training instances before calling `ground`."


@dataclass
class ClassCircuitGroundingFailedError(ProbabilisticModelException):
    """
    Raised when conditioning the class circuit on aggregation statistics leaves it with
    no nodes at all.
    """

    class_: type
    """The domain class whose class circuit failed to ground."""

    def error_message(self) -> str:
        return (
            f"Grounding the class circuit for {self.class_.__name__} produced an "
            f"empty circuit."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the conditioning aggregation statistics are consistent with "
            "the fitted circuit's support."
        )


@dataclass
class MixedCircuitTypesError(ProbabilisticModelException):
    """
    Raised when the template of an exchangeable relation is fitted in another circuit
    type than the class circuit, which grounding cannot combine.
    """

    class_: type
    """
    The domain class whose relational circuit was fitted.
    """

    exchangeable_part: str
    """
    Field name of the exchangeable relation.
    """

    class_circuit_type: type
    """
    The type of the class circuit.
    """

    part_circuit_type: type
    """
    The type of the class circuit of the template.
    """

    def error_message(self) -> str:
        return (
            f"The relational circuit for {self.class_.__name__} is fitted as "
            f"{self.class_circuit_type.__name__}, but the template of "
            f"{self.exchangeable_part!r} as {self.part_circuit_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            "Fit the class circuit and every part with learning methods that return "
            "the same circuit type."
        )


@dataclass
class UnknownExchangeablePartsError(ProbabilisticModelException):
    """
    Raised when learning methods are given for parts that are not exchangeable parts of
    the class.
    """

    class_: type
    """
    The domain class whose relational circuit is fitted.
    """

    parts: list[str]
    """
    The field names that name no exchangeable part of the class.
    """

    def error_message(self) -> str:
        return (
            f"{self.class_.__name__} has no exchangeable parts named "
            f"{', '.join(map(repr, self.parts))}."
        )

    def suggest_correction(self) -> str:
        return (
            "Key the part learning methods by attribute accesses on a variable of the "
            "class that end at a one-to-many relation."
        )


@dataclass
class NotAnAttributeAccessError(ProbabilisticModelException):
    """
    Raised when a part learning method is keyed by something other than an attribute
    access on an EQL variable.
    """

    part: Any
    """
    The key that is not an attribute access on an EQL variable.
    """

    def error_message(self) -> str:
        return f"Expected an attribute access on an EQL variable, got {self.part!r}."

    def suggest_correction(self) -> str:
        return (
            "Key the part learning methods by attribute accesses such as "
            "`variable(SceneBuilding).floors.rooms`."
        )


@dataclass
class MissingPartLearningMethodError(ProbabilisticModelException):
    """
    Raised when an exchangeable part has no learning method and there is no default.
    """

    path: str
    """
    The path of the part, such as ``"floors.rooms"``.
    """

    def error_message(self) -> str:
        return f"The part {self.path!r} has no learning method and there is no default."

    def suggest_correction(self) -> str:
        return (
            "Give the part a learning method or give `PartLearningMethods` a default."
        )


@dataclass
class PartCircuitGroundingFailedError(ProbabilisticModelException):
    """
    Raised when grounding one exchangeable part leaves its circuit with no nodes at all.
    """

    class_: type
    """The domain class of the exchangeable part that failed to ground."""

    def error_message(self) -> str:
        return (
            f"Grounding the exchangeable part circuit for {self.class_.__name__} "
            f"produced an empty circuit."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the conditioning aggregation statistics are consistent with "
            "the fitted template's support."
        )


@dataclass
class InvalidMonteCarloSampleCountError(ProbabilisticModelException):
    """
    Raised when grounding must integrate out undetermined aggregation statistics but the
    configured Monte-Carlo sample count is not positive.
    """

    sample_count: int
    """
    The invalid, non-positive sample count.
    """

    def error_message(self) -> str:
        return (
            f"Integrating out undetermined aggregation statistics requires a positive "
            f"Monte-Carlo sample count, but got {self.sample_count}."
        )

    def suggest_correction(self) -> str:
        return "Set `monte_carlo_sample_count` to a positive integer."


@dataclass
class UndeterminedLatentsNotModeledError(ProbabilisticModelException):
    """
    Raised when the conditioned class circuit does not model the undetermined
    aggregation statistics, so they cannot be sampled and integrated out.
    """

    undetermined_latents: list[Variable]
    """
    The undetermined latent variables that the conditioned circuit fails to model.
    """

    def error_message(self) -> str:
        names = ", ".join(latent.name for latent in self.undetermined_latents)
        return (
            f"The conditioned class circuit does not model the undetermined "
            f"aggregation statistics [{names}], so they cannot be integrated out."
        )

    def suggest_correction(self) -> str:
        return (
            "Ensure the class circuit is fitted with these aggregation statistics "
            "as latent variables before grounding."
        )


@dataclass
class UndeterminedLatentsNotPartitionedError(ProbabilisticModelException):
    """
    Raised when the undetermined latents' marginal support -- grouped by the fitted
    circuit's own mixture branches -- is not a genuine, pairwise-disjoint partition:
    either the fitted circuit never actually split on these latents at all (a single,
    undifferentiated branch), or the branches it did split into overlap.

    Exact-partition grounding requires at least two mutually exclusive branches so each
    grounded exchangeable instance stays tied to the latent value its own branch
    represents.
    """

    undetermined_latents: list[Variable]
    """
    The undetermined latent variables whose marginal support is not a genuine, disjoint
    partition.
    """

    def error_message(self) -> str:
        names = ", ".join(latent.name for latent in self.undetermined_latents)
        return (
            f"The marginal support of undetermined latents [{names}] is not a "
            f"genuine, pairwise-disjoint partition, so exact-partition grounding "
            f"would not be support-deterministic."
        )

    def suggest_correction(self) -> str:
        return (
            "Refit the template so these latents are split on, or use "
            "GroundingMode.SAMPLED instead."
        )


@dataclass
class VariableNotFoundError(ProbabilisticModelException):
    """
    Raised when a dotted access path does not match any variable in a grounded circuit.
    """

    path: str
    """
    The access path that failed to resolve.
    """

    available_variables: list[Variable]
    """
    The variables the grounded circuit actually has, for diagnosis.
    """

    def error_message(self) -> str:
        names = ", ".join(variable.name for variable in self.available_variables)
        return (
            f"No variable's name matches '{self.path}'. Available variables: [{names}]."
        )

    def suggest_correction(self) -> str:
        return (
            "Check the access path against the grounded circuit's actual variable "
            "names, e.g. 'objects[0].type' or 'chair_count()'."
        )


@dataclass
class AmbiguousVariablePathError(ProbabilisticModelException):
    """
    Raised when a dotted access path matches more than one variable in a grounded
    circuit.
    """

    path: str
    """
    The access path that matched more than one variable.
    """

    matches: list[Variable]
    """
    The variables that matched.
    """

    def error_message(self) -> str:
        names = ", ".join(variable.name for variable in self.matches)
        return f"'{self.path}' matches more than one variable: [{names}]."

    def suggest_correction(self) -> str:
        return "Use a longer, more specific suffix of the variable's full name."


@dataclass
class NoCommonSuperclassError(ProbabilisticModelException):
    """
    Raised when the instances a relational circuit is fitted on share no superclass but
    ``object``.
    """

    classes: list[type]
    """
    The classes of the instances.
    """

    def error_message(self) -> str:
        return (
            f"The classes {', '.join(clazz.__name__ for clazz in self.classes)} have "
            f"no common superclass."
        )

    def suggest_correction(self) -> str:
        return "Fit on instances of one class, or pass the class to `fit`."


@dataclass
class InstanceNotOfClassError(ProbabilisticModelException):
    """
    Raised when a relational circuit is fitted for a class on an instance of another
    class.
    """

    class_: type
    """
    The class the circuit is fitted for.
    """

    instance_class: type
    """
    The class of the instance.
    """

    def error_message(self) -> str:
        return (
            f"An instance of {self.instance_class.__name__} is not an instance of "
            f"{self.class_.__name__}."
        )

    def suggest_correction(self) -> str:
        return "Pass a class that every instance is an instance of."
