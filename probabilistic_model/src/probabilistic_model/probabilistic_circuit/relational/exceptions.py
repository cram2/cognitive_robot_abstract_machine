from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, List, Type

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from random_events.variable import Variable


@dataclass
class CircuitNotFittedError(DataclassException):
    """
    Raised when a RelationalProbabilisticCircuit is grounded before it has been fitted.
    """

    class_: Type
    """
    The domain class whose relational circuit has not been fitted yet.
    """

    def error_message(self) -> str:
        return (
            f"RelationalProbabilisticCircuit for {self.class_.__name__} must be fitted "
            f"before it can be grounded."
        )

    def suggest_correction(self) -> str:
        return "Call `fit` with training instances before calling `ground`."


@dataclass
class InvalidMonteCarloSampleCountError(DataclassException):
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
class UndeterminedLatentsNotModeledError(DataclassException):
    """
    Raised when the conditioned class circuit does not model the undetermined
    aggregation statistics, so they cannot be sampled and integrated out.
    """

    undetermined_latents: List[Variable]
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
class UndeterminedLatentsPartitionOverlapsError(DataclassException):
    """
    Raised when the undetermined latents' exact partition -- their marginal support,
    grouped by the fitted circuit's own mixture branches -- has overlapping branches.

    Exact-partition grounding requires those branches to be mutually exclusive so the
    resulting mixture stays support-deterministic. This is caught internally by
    ``RelationalProbabilisticCircuit`` and triggers a fall back to
    ``GroundingMode.CAUSAL_SAMPLED``; it is not expected to reach a caller of
    ``ground``.
    """

    undetermined_latents: List[Variable]
    """
    The undetermined latent variables whose exact partition overlaps.
    """

    def error_message(self) -> str:
        names = ", ".join(latent.name for latent in self.undetermined_latents)
        return (
            f"The exact partition of undetermined latents [{names}] has overlapping "
            f"branches, so exact-partition grounding would not be support-deterministic."
        )

    def suggest_correction(self) -> str:
        return (
            "Refit the template so these latents are split on, or use "
            "GroundingMode.CAUSAL_SAMPLED instead."
        )
