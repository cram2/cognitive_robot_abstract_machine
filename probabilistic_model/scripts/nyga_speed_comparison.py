import json

from random_events.interval import closed
from random_events.product_algebra import VariableMap, SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit as NXProbabilisticCircuit
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
number_of_samples_per_component = 10000
number_of_samples_for_evaluation = 5000
number_of_components = 5
min_samples_per_quantile = 10

# performance evaluation
number_of_iterations = 100
warmup_iterations = 10

# model selection
path_prefix = os.path.join(os.path.expanduser("~"), "Documents")
nx_model_path = os.path.join(path_prefix, "nx_nyga.pm")
jax_model_path = os.path.join(path_prefix, "jax_nyga.pm")
load_from_disc = False
save_to_disc = True


data = []
for component in tqdm.trange(number_of_components, desc="Generating data"):
    samples = np.random.normal(component, 1., (number_of_samples_per_component, 1))
    data.append(samples)

data = np.concatenate(data, axis=0)
variable = Continuous("x")

# create models
if not load_from_disc:
    nx_model = NygaDistribution(variable, min_samples_per_quantile=min_samples_per_quantile)
    nx_model.fit(data)
    jax_model = ProbabilisticCircuit.from_nx(nx_model, True)
    if save_to_disc:
        with open(nx_model_path, "w") as f:
            f.write(json.dumps(nx_model.to_json()))
        with open(jax_model_path, "w") as f:
            f.write(json.dumps(jax_model.to_json()))
else:
    with open(nx_model_path, "r") as f:
        nx_model = NXProbabilisticCircuit.from_json(json.loads(f.read()))
    with open(jax_model_path, "r") as f:
        jax_model = ProbabilisticCircuit.from_json(json.loads(f.read()))


print("Number of edges:", len(list(nx_model.edges)))
print("Number of parameters:",  jax_model.root.number_of_trainable_parameters)
compiled_ll_jax = equinox.filter_jit(jax_model.root.log_likelihood_of_nodes)
# compiled_prob_jax = equinox.filter_jit(model.root.probability_of_simple_event)


def eval_performance(nx_method, nx_args,  jax_method, jax_args, number_of_iterations=15, warmup_iterations=10):

    @timeit
    def timed_nx_method():
        return nx_method(*nx_args)

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

data = nx_model.sample(number_of_samples_for_evaluation)
data_jax = jnp.array(data)
# event = SimpleEvent(VariableMap({variable: closed(0, 1) for variable in variables}))

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     jax_model.sample(1000)

# compiled_sample = equinox.filter_jit(jax_model.sample)

# times_nx, times_jax = eval_performance(nx_model.log_likelihood, (data, ), compiled_ll_jax, (data_jax, ), 20, 2)
# times_nx, times_jax = eval_performance(prob_nx, event, prob_jax, event, 15, 10)
times_nx, times_jax = eval_performance(nx_model.sample, (1000, ), jax_model.sample, (1000, ), 5, 10)

time_jax = np.mean(times_jax), np.std(times_jax)
time_nx = np.mean(times_nx), np.std(times_nx)
print("Jax:", time_jax)
print("Networkx:", time_nx)
print("Networkx/Jax ",time_nx[0]/time_jax[0])
