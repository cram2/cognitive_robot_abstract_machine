import unittest
import json
import math
import random
import tempfile
from datetime import datetime
from enum import IntEnum, Enum

import numpy as np
import pandas as pd
import sklearn.datasets
from jpt import infer_from_dataframe as old_infer_from_dataframe
from jpt.learning.impurity import Impurity
from jpt.trees import JPT as OldJPT
from matplotlib import pyplot as plt
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Variable, Continuous

from probabilistic_model.distributions import GaussianDistribution
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import (
    ScaledContinuous,
    infer_variables_from_dataframe,
    Integer,
    Symbolic,
)
from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    ProbabilisticCircuit,
    ProductUnit,
    IntegerDistribution,
    SymbolicDistribution,
    UnivariateContinuousLeaf,
    leaf,
)


class SymbolEnum(Enum):
    """
    A simple enum for testing purposes.
    """

    A = 0
    B = 1
    C = 2

    def __hash__(self):
        return hash(self.value)


class VariableTestCase(unittest.TestCase):
    variable: ScaledContinuous = ScaledContinuous("x", 2, 3)

    def test_encode(self):
        self.assertEqual(self.variable.encode(2), 0)
        self.assertEqual(self.variable.encode(5), 1)
        self.assertEqual(self.variable.encode(0), -2 / 3)

    def test_decode(self):
        self.assertEqual(self.variable.decode(0), 2)
        self.assertEqual(self.variable.decode(1), 5)
        self.assertEqual(self.variable.decode(-2 / 3), 0)

    def test_serialization_integer(self):
        variable = Integer("x", 2, 1)
        serialized = variable.to_json()
        deserialized = Variable.from_json(serialized)
        self.assertEqual(variable, deserialized)

    def test_serialization_continuous(self):
        variable = ScaledContinuous("x", 2, 3, 1.0, 0.1, 10)
        serialized = variable.to_json()
        deserialized = Variable.from_json(serialized)
        self.assertEqual(variable, deserialized)


class InferFromDataFrameTestCase(unittest.TestCase):
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        data = pd.DataFrame()
        data["real"] = np.random.normal(2, 4, 100)
        data["integer"] = np.concatenate(
            (np.random.randint(low=0, high=4, size=50), np.random.randint(7, 10, 50))
        )
        data["symbol"] = random.choices(list(SymbolEnum), k=100)
        cls.data = data

    def test_types(self):
        self.assertEqual(self.data.dtypes.iloc[0], float)
        self.assertEqual(self.data.dtypes.iloc[1], int)
        self.assertEqual(self.data.dtypes.iloc[2], Enum)

    def test_infer_from_dataframe_with_scaling(self):
        real, integer, symbol = infer_variables_from_dataframe(
            self.data, scale_continuous_types=True
        )
        self.assertEqual(real.name, "real")
        self.assertIsInstance(real, ScaledContinuous)
        self.assertEqual(integer.name, "integer")
        self.assertEqual(symbol.name, "symbol")
        self.assertLess(real.minimal_distance, 1.0)

    def test_infer_from_dataframe_without_scaling(self):
        real, integer, symbol = infer_variables_from_dataframe(
            self.data, scale_continuous_types=False
        )
        self.assertNotIsInstance(real, ScaledContinuous)

    def test_unknown_type(self):
        df = pd.DataFrame()
        df["time"] = [datetime.now()]
        with self.assertRaises(ValueError):
            infer_variables_from_dataframe(df)


class JPTTestCase(unittest.TestCase):
    data: pd.DataFrame
    real: ScaledContinuous
    integer: Integer
    symbol: Symbolic
    model: JPT

    def setUp(self):
        np.random.seed(69)
        data = pd.DataFrame()
        data["integer"] = np.concatenate(
            (np.random.randint(low=0, high=4, size=50), np.random.randint(7, 10, 50))
        ).astype(int)
        data["real"] = np.random.normal(2, 4, 100).astype(np.float32)
        data["symbol"] = random.choices(list(SymbolEnum), k=100)
        self.data = data
        self.real, self.integer, self.symbol = infer_variables_from_dataframe(
            self.data, scale_continuous_types=False
        )
        self.model = JPT([self.real, self.integer, self.symbol])

    def test_construct_impurity(self):
        impurity = self.model.construct_impurity()
        self.assertIsInstance(impurity, Impurity)

    def test_create_leaf_node(self):
        preprocessed_data = self.model.preprocess_data(self.data)
        leaf_node = self.model.create_leaf_node(preprocessed_data)

        self.assertEqual(len(leaf_node.subcircuits), 3)
        self.assertIsInstance(
            leaf_node.subcircuits[2].distribution, IntegerDistribution
        )
        self.assertIsInstance(leaf_node.subcircuits[1], SumUnit)
        self.assertIsInstance(
            leaf_node.subcircuits[0].distribution, SymbolicDistribution
        )

        # check that all likelihoods are greater than 0
        likelihood = leaf_node.probabilistic_circuit.likelihood(self.data.to_numpy())
        self.assertTrue(all(likelihood > 0))

    def test_fit_without_sum_units(self):
        self.model.min_impurity_improvement = 1
        self.model.fit(self.data)
        self.assertEqual(len(self.model.root.subcircuits), 1)
        self.assertEqual(self.model.root.log_weighted_subcircuits[0][0], 0.0)
        self.assertEqual(len(self.model.root.subcircuits[0].subcircuits), 3)

    def test_fit(self):
        self.model._min_samples_leaf = 10
        pc = self.model.fit(self.data)
        self.assertTrue(
            len(pc.root.subcircuits)
            <= math.floor(len(self.data) / self.model.min_samples_leaf)
        )
        self.assertTrue(
            all([weight > -np.inf for weight, _ in pc.root.log_weighted_subcircuits])
        )

        # check that all likelihoods are greater than 0
        preprocessed_data = self.model.preprocess_data(self.data)
        likelihood = pc.likelihood(preprocessed_data)

        self.assertTrue(all(likelihood > 0))

    def test_preprocessing_and_compare_to_jpt(self):
        variables = old_infer_from_dataframe(
            self.data, scale_numeric_types=False, precision=0.0
        )
        original_jpt = OldJPT(
            variables,
            min_samples_leaf=self.model.min_samples_leaf,
            min_impurity_improvement=self.model.min_impurity_improvement,
        )
        original_preprocessing = original_jpt._preprocess_data(self.data)
        own_preprocessing = self.model.preprocess_data(self.data)

        # Symbolic columns are not preprocessed in order in JPTs. The difference is intended
        self.assertTrue(
            np.all(original_preprocessing[:, :-1] == own_preprocessing[:, :-1])
        )

    def test_fit_and_compare_to_jpt(self):
        self.model._min_samples_leaf = 10
        self.model.keep_sample_indices = True
        self.model.fit(self.data)
        variables = old_infer_from_dataframe(
            self.data, scale_numeric_types=False, precision=0.0
        )
        original_jpt = OldJPT(
            variables,
            min_samples_leaf=self.model.min_samples_leaf,
            min_impurity_improvement=self.model.min_impurity_improvement,
        )
        original_jpt = original_jpt.learn(self.data, keep_samples=True)
        self.assertEqual(len(self.model.root.subcircuits), len(original_jpt.leaves))

    def test_jpt_continuous_variables_only(self):
        data = self.data[["real"]].astype(float)
        variables = infer_variables_from_dataframe(data)
        model = JPT(variables)
        model.fit(data)
        self.assertEqual(len(model.root.subcircuits), 1)

    def test_jpt_integer_variables_only(self):
        data = self.data[["integer"]]
        variables = infer_variables_from_dataframe(data)
        model = JPT(variables)
        model.fit(data)
        self.assertEqual(len(model.root.subcircuits), 1)

    def test_jpt_symbolic_variables_only(self):
        data = self.data[["symbol"]]
        variables = infer_variables_from_dataframe(data)
        model = JPT(variables)
        pc = model.fit(data)
        pc.plot_structure()
        self.assertEqual(len(pc.root.subcircuits), 3)

    def test_variable_dependencies_to_json(self):
        serialized = self.model._variable_dependencies_to_json()
        all_variable_names = [variable.name for variable in self.model.variables]
        self.assertEqual(
            serialized,
            {
                "real": all_variable_names,
                "integer": all_variable_names,
                "symbol": all_variable_names,
            },
        )

    def test_serialization(self):
        self.model._min_samples_leaf = 10
        self.model.fit(self.data)
        serialized = self.model.to_json()
        deserialized = JPT.from_json(serialized)
        self.assertEqual(self.model, deserialized)


class BreastCancerTestCase(unittest.TestCase):
    data: pd.DataFrame
    model: JPT
    pc: ProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        data = sklearn.datasets.load_breast_cancer(as_frame=True)

        df = data.data
        target = data.target.astype(str)
        target[target == "1"] = "malignant"
        target[target == "0"] = "friendly"

        df["malignant"] = target
        cls.data = df
        variables = infer_variables_from_dataframe(
            cls.data, scale_continuous_types=False, min_samples_per_quantile=600
        )

        cls.model = JPT(variables, min_samples_leaf=0.4)
        cls.pc = cls.model.fit(cls.data)

    def test_serialization(self):
        json_dict = self.model.to_json()
        model = JPT.from_json(json_dict)

        self.assertEqual(model, self.model)

        file = tempfile.NamedTemporaryFile()

        with open(file.name, "w") as f:
            json.dump(json_dict, f)

        with open(file.name, "r") as f:
            model_ = json.load(f)
        model_ = JPT.from_json(model_)

        self.assertEqual(model, model_)
        file.close()

    def test_conditional_inference(self):
        evidence = SimpleEvent(
            {variable: variable.domain for variable in self.model.variables}
        ).as_composite_set()
        query = evidence
        conditional_model, evidence_probability = self.pc.truncated(evidence)
        self.assertAlmostEqual(1.0, evidence_probability, delta=1e-5)
        self.assertAlmostEqual(1.0, conditional_model.probability(query), delta=1e-5)

    def test_univariate_continuous_marginal(self):
        marginal = self.pc.marginal(self.model.variables[:1])
        self.assertIsInstance(marginal, ProbabilisticCircuit)

    def test_univariate_symbolic_marginal(self):
        variables = [v for v in self.model.variables if v.name == "malignant"]
        marginal = self.pc.marginal(variables)
        self.assertIsInstance(marginal.root, SumUnit)

    def test_serialization_of_circuit(self):
        json_dict = self.pc.to_json()
        model = ProbabilisticCircuit.from_json(json_dict)
        event = SimpleEvent(
            {variable: variable.domain for variable in self.model.variables}
        ).as_composite_set()
        self.assertAlmostEqual(model.probability(event), 1.0)

    def test_marginal_conditional_chain(self):
        model = self.pc
        marginal = model.marginal(self.model.variables[:2])
        x, y = self.model.variables[:2]
        event = SimpleEvent({x: closed(0, 10)}).as_composite_set()
        conditional, probability = model.truncated(event)

    def test_mode(self):
        mode, likelihood = self.pc.log_mode(check_determinism=False)
        self.assertGreater(len(mode.simple_sets), 0)


class MNISTTestCase(unittest.TestCase):
    model: JPT

    @classmethod
    def setUpClass(cls):
        mnist = sklearn.datasets.load_digits(as_frame=True)
        df = mnist.data
        target = mnist.target
        df["digit"] = target
        df["digit"] = df["digit"].astype(str)

        variables = infer_variables_from_dataframe(
            df, scale_continuous_types=False, min_likelihood_improvement=0.01
        )
        cls.model = JPT(variables, min_samples_leaf=0.1)
        cls.model.fit(df)

    def test_serialization(self):
        json_dict = self.model.to_json()
        # print(json_dict)
        model = JPT.from_json(json_dict)
        self.assertEqual(model, self.model)

        file = tempfile.NamedTemporaryFile()

        with open(file.name, "w") as f:
            json.dump(json_dict, f)

        with open(file.name, "r") as f:
            model_ = json.load(f)
        model_ = JPT.from_json(model_)
        self.assertEqual(model, model_)
        file.close()


import plotly.graph_objects as go


class GaussianJPTTestCase(unittest.TestCase):
    x: Continuous
    y: Continuous

    data: pd.DataFrame
    multivariate_normal: ProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        pc = ProbabilisticCircuit()
        prod = ProductUnit(probabilistic_circuit=pc)
        prod.add_subcircuit(leaf(GaussianDistribution(Continuous("x"), 2, 4), pc))
        prod.add_subcircuit(leaf(GaussianDistribution(Continuous("y"), 2, 4), pc))
        cls.multivariate_normal = pc
        samples = cls.multivariate_normal.sample(1000)
        cls.data = pd.DataFrame(
            samples, columns=[v.name for v in cls.multivariate_normal.variables]
        )

        cls.x, cls.y = infer_variables_from_dataframe(
            cls.data, scale_continuous_types=False, min_samples_per_quantile=20
        )

    def test_plot_2d_jpt(self):
        model = JPT([self.x, self.y], min_samples_leaf=0.9)
        pc = model.fit(self.data)
        fig = go.Figure(pc.plot(500, surface=True), pc.plotly_layout())
        # fig.show()

    def test_plot_2d_gaussian(self):
        fig = go.Figure(
            self.multivariate_normal.plot(500, surface=True),
            self.multivariate_normal.plotly_layout(),
        )
        # fig.show()
