from dataclasses import dataclass, field

import numpy as np

from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class MonteCarloEstimator:
    """
    This is a wrapper class for using monte carlo estimations of a model that can be
    sampled from and where the likelihood can be evaluated.
    """

    model: ProbabilisticModel
    """
    The wrapped model.
    """

    sample_size: int = field(default=100)
    """
    The number of samples to use for the estimation.
    """

    def total_variation_distance(self, other: ProbabilisticModel) -> float:
        """
        Estimate the total variation distance :math:`\\frac{1}{2} \\int |p(x) - q(x)| dx`
        between the model :math:`p` and another model :math:`q`.

        The estimate is the importance sampling estimate under the equal mixture
        :math:`m = (p + q) / 2`, which draws :attr:`sample_size` samples from each model
        and averages :math:`|p(x) - q(x)| / (p(x) + q(x))`. This is the balance
        heuristic of multiple importance sampling
        :cite:p:`veach1995optimally,elvira2019generalized`. It is unbiased and always
        lies in :math:`[0, 1]`.

        :param other: The other model to compare to.
        :return: The estimated total variation distance between the two models.
        """
        samples = np.concatenate(
            [self.model.sample(self.sample_size), other.sample(self.sample_size)]
        )
        likelihood_of_model = self.model.likelihood(samples)
        likelihood_of_other = other.likelihood(samples)
        return np.mean(
            np.abs(likelihood_of_model - likelihood_of_other)
            / (likelihood_of_model + likelihood_of_other)
        )
