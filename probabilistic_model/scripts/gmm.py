import json
import os

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from jax.tree_util import tree_flatten
from matplotlib import pyplot as plt
from random_events.interval import SimpleInterval
from random_events.variable import Continuous

from probabilistic_model.distributions import GaussianDistribution, UniformDistribution
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import GaussianLayer
from probabilistic_model.probabilistic_circuit.jax.uniform_layer import UniformLayer
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit
from probabilistic_model.probabilistic_circuit.rx.distributions import UnivariateContinuousLeaf
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import \
    ProbabilisticCircuit as NXProbabilisticCircuit, ProductUnit, SumUnit
import plotly.express as px
import plotly.graph_objects as go
import jax.profiler

np.random.seed(69)

# training
number_of_variables = 2
number_of_samples_per_component = 5000
number_of_components = 10
number_of_mixtures = 500
number_of_iterations = 1000

# model selection
path_prefix = os.path.join(os.path.expanduser("~"), "Documents")
nx_model_path = os.path.join(path_prefix, "nx_gmm.pm")
jax_model_path = os.path.join(path_prefix, "jax_gmm.pm")
load_from_disc = False
save_to_disc = True

data = []
for component in tqdm.trange(number_of_components, desc="Generating data"):
    mean = np.full(number_of_variables, component)
    cov = np.random.uniform(-1, 1, (number_of_variables, number_of_variables))
    cov = np.dot(cov, cov.T)
    samples = np.random.multivariate_normal(mean, cov, number_of_samples_per_component)
    data.append(samples)

data = np.concatenate(data, axis=0)
np.random.shuffle(data)

fig = px.scatter(x=data[:1000, 0], y=data[:1000, 1], title="Data")
fig.show()

variables = [Continuous(f"x_{i}") for i in range(number_of_variables)]


def generate_gaussian_component(mean, variance):
    result = ProductUnit()
    for i in range(len(mean)):
        distribution = GaussianDistribution(variables[i], mean[i], variance[i])
        # distribution = UniformDistribution(variables[i], SimpleInterval(mean[i] - 2 , mean[i] + 2))
        leaf = UnivariateContinuousLeaf(distribution)
        result.add_subcircuit(leaf)
    return result


# create models
if not load_from_disc:

    result = SumUnit()
    for i in tqdm.trange(number_of_mixtures, desc="Generating model"):
        result.add_subcircuit(generate_gaussian_component(data[i], [1] * number_of_variables), 1.)
    result.normalize()

    nx_model = result.probabilistic_circuit
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

# nx_model.plot_structure()
# plt.show()
fig = go.Figure(nx_model.plot(), nx_model.plotly_layout())
fig.update_layout(title="Initial model guess")
fig.show()

root = jax_model.root
print("Number of edges:", len(list(nx_model.edges)))
print("Number of parameters:", root.number_of_trainable_parameters)

jax_data = jnp.array(data)

jax.config.update("jax_traceback_filtering", "off")

@eqx.filter_jit
def loss(model, x):
    ll = model.log_likelihood_of_nodes(x)
    return -jnp.mean(ll)


optim = optax.adamw(0.01)
opt_state = optim.init(eqx.filter(root, eqx.is_inexact_array))
losses = []
for _ in tqdm.trange(1000, desc="Fitting model"):
    loss_value, grads = eqx.filter_value_and_grad(loss)(root, jax_data)
    losses.append(loss_value)
    grads_of_sum_layer = eqx.filter(tree_flatten(grads), eqx.is_inexact_array)[0][0]

    updates, opt_state = optim.update(grads, opt_state, eqx.filter(root, eqx.is_inexact_array))
    root = eqx.apply_updates(root, updates)

jax.profiler.save_device_memory_profile("memory.prof")

fig = px.line(x=range(len(losses)), y=losses, title="Average negative log likelihood")
fig.show()

jax_model.root = root
nx_model = jax_model.to_nx(True)
fig = go.Figure(nx_model.plot(), nx_model.plotly_layout())
fig.update_layout(title="Fitted model guess")
fig.show()