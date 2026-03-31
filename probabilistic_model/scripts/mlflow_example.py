import json
import tempfile

import mlflow
import numpy as np
import plotly.graph_objects as go
import tqdm
from random_events.variable import Continuous

from probabilistic_model.learning.nyga_induction import NygaInduction
from probabilistic_model.interfaces.mlflow_integration import (
    infer_signature,
    ProbabilisticModelWrapper,
)

np.random.seed(69)

mlflow.set_tracking_uri(uri=" http://127.0.0.1:8080")

# training
number_of_samples_per_component = 1000
number_of_samples_for_evaluation = 5000
number_of_components = 5
min_samples_per_quantile = 60

data = []
for component in tqdm.trange(number_of_components, desc="Generating data"):
    samples = np.random.normal(component, 1.0, (number_of_samples_per_component, 1))
    data.append(samples)

data = np.concatenate(data, axis=0)
variable = Continuous("x")

rustworkx_model = NygaInduction(
    variable, min_samples_per_quantile=min_samples_per_quantile
)

run = mlflow.start_run(run_name="Integration example with mlflow")

# Log the hyperparameters
mlflow.log_params(
    {"min_samples_per_quantile": rustworkx_model.min_samples_per_quantile}
)
rustworkx_model = rustworkx_model.fit(data)
mlflow.set_tag("Training Info", "Basic LR model for iris data")

file = tempfile.NamedTemporaryFile()

with open(file.name, "w") as f:
    json.dump(rustworkx_model.to_json(), f)

# Log the model
model_info = mlflow.pyfunc.log_model(
    artifact_path="mlflow_integration_test",
    artifacts={"model_path": file.name},
    signature=infer_signature(rustworkx_model),
    python_model=ProbabilisticModelWrapper(rustworkx_model),
    registered_model_name="tracking-quickstart",
)

loaded_model = mlflow.pyfunc.load_model(
    model_uri=run.info.artifact_uri + "/mlflow_integration_test"
)
loaded_model = loaded_model.unwrap_python_model().model

fig = go.Figure(loaded_model.plot(), loaded_model.plotly_layout())
fig.show()
