import numpy as np

from .probabilistic_circuit.rx.helper import uniform_measure_of_event
from .probabilistic_model import ProbabilisticModel


class MonteCarloEstimator:
    """
    This is a wrapper class for using monte carlo estimations of a model that can be sampled from and where the
    likelihood can be evaluated.
    """

    model: ProbabilisticModel
    """
    The wrapped model.
    """

    sample_size: int
    """
    The number of samples to use for the estimation.
    """

    def __init__(self, model: ProbabilisticModel, sample_size: int = 100):
        self.model = model
        self.sample_size = sample_size

    def l1_metric_but_with_uniform_measure(self, other: ProbabilisticModel):
        """
        Calculate the L1 metric between the model and another model using a uniform measure over the union of both
        distributions to sample from.

        :param other: The other model to compare to.
        :return: The L1 metric between the two models.
        """

        # get the union of both supports
        supp_of_self = self.model.support
        supp_of_other = other.support
        union_of_supports = supp_of_other | supp_of_self

        # construct uniform measure over the union of both supports
        uniform_model = uniform_measure_of_event(union_of_supports)

        # draw samples from the uniform measure
        samples = uniform_model.sample(self.sample_size)

        # get the density of the uniform model
        density_of_uniform_model = uniform_model.likelihood(samples[:1])[0]

        # compare the likelihoods of the two models on the samples of the uniform model
        p_self = self.model.likelihood(samples)
        p_other = other.likelihood(samples)

        # calculate the L1 metric
        l1_metric = np.mean(np.abs(p_self - p_other)) / density_of_uniform_model
        return l1_metric

    def l1_metric(self, other: ProbabilisticModel, tolerance: float = 10e-8) -> float:
        """
        Estimates the L1 metric between to models.

        :other: the other model.
        :tolerance: Tolerance to use for the comparison of likelihoods.
        Samples that have a likelihood in both models that differs by less than this tolerance are considered to have
        an equal likelihood.
        """
        samples_p = self.model.sample(self.sample_size)
        l_p = self.model.likelihood(samples_p)
        l_q = other.likelihood(samples_p)
        diff = l_p - l_q
        p = (diff > tolerance).mean()
        samples_q = other.sample(self.sample_size)
        l_p = self.model.likelihood(samples_q)
        l_q = other.likelihood(samples_q)
        diff = l_p - l_q
        q = (diff > tolerance).mean()
        return 2 * (p - q)
