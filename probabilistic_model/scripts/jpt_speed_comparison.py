import json
import networkx as nx

from random_events.interval import closed
from random_events.product_algebra import VariableMap, SimpleEvent

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit as RXProbabilisticCircuit
import numpy as np
import jax
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
nx_model_path = os.path.join(path_prefix, "nx_model.pm")
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

variables = infer_variables_from_dataframe(df, min_samples_per_quantile=min_samples_per_quantile)

# create models
if not load_from_disc:
    rx_model = JPT(variables, min_samples_leaf=min_samples_leaf)
    rx_model = rx_model.fit(df)
    jax_model = ProbabilisticCircuit.from_nx(rx_model, True)
    if save_to_disc:
        with open(nx_model_path, "w") as f:
            f.write(json.dumps(rx_model.to_json()))
        with open(jax_model_path, "w") as f:
            f.write(json.dumps(jax_model.to_json()))
else:
    with open(nx_model_path, "r") as f:
        rx_model = RXProbabilisticCircuit.from_json(json.loads(f.read()))
    with open(jax_model_path, "r") as f:
        jax_model = ProbabilisticCircuit.from_json(json.loads(f.read()))


print("Number of edges:", len(list(rx_model.edges())))
print("Number of parameters:",  jax_model.root.number_of_trainable_parameters)
compiled_ll_jax = equinox.filter_jit(jax_model.root.log_likelihood_of_nodes)
# compiled_prob_jax = equinox.filter_jit(model.root.probability_of_simple_event)


def eval_performance(rx_method, nx_args, jax_method, jax_args, number_of_iterations=15, warmup_iterations=10):

    @timeit
    def timed_nx_method():
        return rx_method(*nx_args)

    @timeit
    def timed_jax_method():
        return jax_method(*jax_args)

    times_jax = []
    times_nx = []

    for i in tqdm.trange(number_of_iterations, desc="Evaluating performance"):

        current_ll_jax, time_jax = timed_jax_method()
        current_ll_nx, time_nx = timed_nx_method()
        if i >= warmup_iterations:
            times_jax.append(time_jax.total_seconds())
            times_nx.append(time_nx.total_seconds())

    return times_nx, times_jax

data = rx_model.sample(number_of_samples_for_evaluation)
data_jax = jnp.array(data)

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     jax_model.sample(1000)

# samples = jax_model.sample(1000)
# ll = jax_model.log_likelihood(samples)
# assert (ll > -jnp.inf).all()

times_nx, times_jax = eval_performance(rx_model.log_likelihood, (data,), compiled_ll_jax, (data_jax,), 15, 10)

time_jax = np.mean(times_jax), np.std(times_jax)
time_nx = np.mean(times_nx), np.std(times_nx)
print("Jax:", time_jax)
print("Networkx:", time_nx)
print("Networkx/Jax ",time_nx[0]/time_jax[0])
