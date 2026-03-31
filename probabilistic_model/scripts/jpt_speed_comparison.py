import json
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RXProbabilisticCircuit,
)
import numpy as np
import jax.numpy as jnp
import tqdm
from probabilistic_model.utils import timeit

import pandas as pd
import equinox
import os

np.random.seed(69)


# training
number_of_variables = 5
number_of_samples_per_component = 100000
number_of_samples_for_evaluation = 5000
number_of_components = 2
min_samples_leaf = 0.0005
min_samples_per_quantile = 10

# performance evaluation
number_of_iterations = 100
warmup_iterations = 10

# model selection
path_prefix = os.path.join(os.path.expanduser("~"), "Documents")
rustworkx_model_path = os.path.join(path_prefix, "nx_model.pm")
jax_model_path = os.path.join(path_prefix, "jax_model.pm")
load_from_disc = False
save_to_disc = True


data = []
for component in tqdm.trange(number_of_components, desc="Generating data"):
    mean = np.full(number_of_variables, component)
    cov = np.random.uniform(0, 1, (number_of_variables, number_of_variables))
    cov = np.dot(cov, cov.T)
    samples = np.random.multivariate_normal(mean, cov, number_of_samples_per_component)
    data.append(samples)

data = np.concatenate(data, axis=0)
df = pd.DataFrame(data, columns=[f"x_{i}" for i in range(number_of_variables)])

variables = infer_variables_from_dataframe(
    df, min_samples_per_quantile=min_samples_per_quantile
)

# create models
if not load_from_disc:
    rustworkx_model = JointProbabilityTree(
        variables, min_samples_per_leaf=min_samples_leaf
    )
    rustworkx_model = rustworkx_model.fit(df)
    jax_model = ProbabilisticCircuit.from_rustworkx(rustworkx_model, True)
    if save_to_disc:
        with open(rustworkx_model_path, "w") as f:
            f.write(json.dumps(rustworkx_model.to_json()))
        with open(jax_model_path, "w") as f:
            f.write(json.dumps(jax_model.to_json()))
else:
    with open(rustworkx_model_path, "r") as f:
        rustworkx_model = RXProbabilisticCircuit.from_json(json.loads(f.read()))
    with open(jax_model_path, "r") as f:
        jax_model = ProbabilisticCircuit.from_json(json.loads(f.read()))


print("Number of edges:", len(list(rustworkx_model.edges())))
print("Number of parameters:", jax_model.root.number_of_trainable_parameters)
compiled_log_likelihood_jax = equinox.filter_jit(jax_model.root.log_likelihood_of_nodes)
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
        current_log_likelihood_rustworkx, times_rustworkx = timed_rustworkx_method()
        if index >= warmup_iterations:
            times_jax.append(time_jax.total_seconds())
            times_rustworkx.append(times_rustworkx.total_seconds())

    return times_rustworkx, times_jax


data = rustworkx_model.sample(number_of_samples_for_evaluation)
data_jax = jnp.array(data)

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     jax_model.sample(1000)

# samples = jax_model.sample(1000)
# ll = jax_model.log_likelihood(samples)
# assert (ll > -jnp.inf).all()

times_rustworkx, times_jax = eval_performance(
    rustworkx_model.log_likelihood,
    (data,),
    compiled_log_likelihood_jax,
    (data_jax,),
    15,
    10,
)

time_jax = np.mean(times_jax), np.std(times_jax)
time_rustworkx = np.mean(times_rustworkx), np.std(times_rustworkx)
print("Jax:", time_jax)
print("Networkx:", time_rustworkx)
print("Networkx/Jax ", time_rustworkx[0] / time_jax[0])
