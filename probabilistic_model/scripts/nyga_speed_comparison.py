import json

from random_events.interval import closed
from random_events.product_algebra import VariableMap, SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.nyga_induction import NygaInduction
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as NXProbabilisticCircuit,
)
import numpy as np
import jax.numpy as jnp
import tqdm
from probabilistic_model.utils import timeit

import equinox
import os

np.random.seed(69)


# training
number_of_samples_per_component = 10000
number_of_samples_for_evaluation = 5000
number_of_components = 5
min_samples_per_quantile = 10

# performance evaluation
number_of_iterations = 100
warmup_iterations = 10

# model selection
path_prefix = os.path.join(os.path.expanduser("~"), "Documents")
rustworkx_path = os.path.join(path_prefix, "nx_nyga.pm")
jax_model_path = os.path.join(path_prefix, "jax_nyga.pm")
load_from_disc = False
save_to_disc = True


data = []
for component in tqdm.trange(number_of_components, desc="Generating data"):
    samples = np.random.normal(component, 1.0, (number_of_samples_per_component, 1))
    data.append(samples)

data = np.concatenate(data, axis=0)
variable = Continuous("x")

# create models
if not load_from_disc:
    rustworkx_model = NygaInduction(
        variable, min_samples_per_quantile=min_samples_per_quantile
    )
    rustworkx_model.fit(data)
    jax_model = ProbabilisticCircuit.from_rustworkx(rustworkx_model, True)
    if save_to_disc:
        with open(rustworkx_path, "w") as f:
            f.write(json.dumps(rustworkx_model.to_json()))
        with open(jax_model_path, "w") as f:
            f.write(json.dumps(jax_model.to_json()))
else:
    with open(rustworkx_path, "r") as f:
        rustworkx_model = NXProbabilisticCircuit.from_json(json.loads(f.read()))
    with open(jax_model_path, "r") as f:
        jax_model = ProbabilisticCircuit.from_json(json.loads(f.read()))


print("Number of edges:", len(list(rustworkx_model.edges)))
print("Number of parameters:", jax_model.root.number_of_trainable_parameters)
compiled_ll_jax = equinox.filter_jit(jax_model.root.log_likelihood_of_nodes)
# compiled_prob_jax = equinox.filter_jit(model.root.probability_of_simple_event)


def eval_performance(
    rustworkx_method,
    rustworkx_args,
    jax_method,
    jax_args,
    number_of_iterations=15,
    warmup_iterations=10,
):
    """
    Evaluate the performance of two methods by running them multiple times and measuring the time taken for each run.
    :param rustworkx_method: The method to be evaluated in Rustworkx.
    :param rustworkx_args: The arguments to be passed to the Rustworkx method.
    :param jax_method: The method to be evaluated in JAX.
    :param jax_args: The arguments to be passed to the JAX method.
    :param number_of_iterations: The number of iterations to be run for each method.
    :param warmup_iterations: The number of iterations to be run before measuring the performance.
    :return: A tuple containing the times for the Rustworkx method and the times for the JAX method.
    """

    @timeit
    def timed_rustworkx_method():
        return rustworkx_method(*rustworkx_args)

    @timeit
    def timed_jax_method():
        return jax_method(*jax_args)

    times_jax = []
    times_rustworkx = []

    for index in tqdm.trange(number_of_iterations, desc="Evaluating performance"):

        current_log_likelihood_jax, time_jax = timed_jax_method()
        current_log_likelihood_rustworkx, time_rustworkx = timed_rustworkx_method()
        if index >= warmup_iterations:
            times_jax.append(time_jax.total_seconds())
            times_rustworkx.append(time_rustworkx.total_seconds())

    return times_rustworkx, times_jax


data = rustworkx_model.sample(number_of_samples_for_evaluation)
data_jax = jnp.array(data)
# event = SimpleEvent(VariableMap({variable: closed(0, 1) for variable in variables}))

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     jax_model.sample(1000)

# compiled_sample = equinox.filter_jit(jax_model.sample)

# times_nx, times_jax = eval_performance(nx_model.log_likelihood, (data, ), compiled_ll_jax, (data_jax, ), 20, 2)
# times_nx, times_jax = eval_performance(prob_nx, event, prob_jax, event, 15, 10)
times_rustworkx, times_jax = eval_performance(
    rustworkx_model.sample, (1000,), jax_model.sample, (1000,), 5, 10
)

time_jax = np.mean(times_jax), np.std(times_jax)
time_rustworkx = np.mean(times_rustworkx), np.std(times_rustworkx)
print("Jax:", time_jax)
print("Rustworkx:", time_rustworkx)
print("Networkx/Jax ", time_rustworkx[0] / time_jax[0])
