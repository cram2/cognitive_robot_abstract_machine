import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Continuous, Symbolic, Integer
from typing_extensions import Optional
import json

from ..probabilistic_model import ProbabilisticModel


class ProbabilisticModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for a probabilistic model to be used with MLflow.

    The wrapper requires the artifact "model_path" to contain a path to a file that can be parsed using
    the SubclassJSONSerializer.
    """

    model: Optional[ProbabilisticModel] = None

    def __init__(self, model: Optional[ProbabilisticModel] = None):
        self.model = model
        super().__init__()

    def load_context(self, context):
        with open(context.artifacts["model_path"], "r") as f:
            model_dict = json.load(f)
        self.model = SubclassJSONSerializer.from_json(model_dict)

    def predict(self, context, model_input):
        return self.model.log_likelihood(model_input)


def infer_signature(model: ProbabilisticModel) -> ModelSignature:
    """
    Infer the signature of a probabilistic model.
    :param model: The model to infer the signature from.
    :return: The inferred signature.
    """
    inputs = []
    for variable in model.variables:
        if isinstance(variable, Continuous):
            inputs.append(ColSpec(type=DataType.float, name=variable.name, required=False))
        elif isinstance(variable, Symbolic):
            inputs.append(ColSpec(type=DataType.string, name=variable.name, required=False))
        elif isinstance(variable, Integer):
            inputs.append(ColSpec(type=DataType.integer, name=variable.name, required=False))
        else:
            raise ValueError(f"Unknown variable type {type(variable)}")
    result = ModelSignature(inputs=Schema(inputs))
    return result
