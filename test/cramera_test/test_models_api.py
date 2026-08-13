"""
Tests of the probabilistic-models workbench and its API endpoints.

A small real circuit — a uniform continuous variable times a symbolic one — is loaded
through the API exactly as the Models tab uploads it, and every tool of the tab is
exercised against known probabilities.
"""

from __future__ import annotations

import importlib
import json
import threading
import urllib.request

import numpy as np
import pytest

from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict
from random_events.interval import closed
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from cramera.models_workbench import ModelWorkbench

# %% the model under test


def uniform_color_circuit() -> ProbabilisticCircuit:
    """
    ``x ~ Uniform(0, 1)`` times ``color ~ {red: 0.7, blue: 0.3}``.
    """
    circuit = ProbabilisticCircuit()
    x = Continuous("x")
    color = Symbolic("color", domain=Set.from_iterable(["red", "blue"]))
    probabilities = MissingDict(float, {hash("red"): 0.7, hash("blue"): 0.3})
    uniform = leaf(
        UniformDistribution(variable=x, interval=closed(0, 1).simple_sets[0]), circuit
    )
    colors = leaf(
        SymbolicDistribution(variable=color, probabilities=probabilities), circuit
    )
    product = ProductUnit(probabilistic_circuit=circuit)
    product.add_subcircuit(uniform)
    product.add_subcircuit(colors)
    return circuit


def loaded_workbench() -> ModelWorkbench:
    """
    A fresh workbench with :func:`uniform_color_circuit` loaded through JSON, the way
    the tab uploads a model.
    """
    workbench = ModelWorkbench()
    workbench.load_model(
        json.loads(json.dumps(uniform_color_circuit().to_json())), name="test.json"
    )
    return workbench


# %% the workbench


class TestWorkbenchState:
    def test_an_empty_workbench_reports_nothing_loaded(self):
        state = ModelWorkbench().state()

        assert state == {"ok": True, "loaded": False, "name": "", "variables": []}

    def test_loading_reports_the_variables(self):
        state = loaded_workbench().state()

        assert state["loaded"] is True
        assert state["name"] == "test.json"
        assert state["variables"] == [
            {"name": "color", "kind": "symbolic", "values": ["red", "blue"]},
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]


class TestProbability:
    def test_an_interval_constraint(self):
        result = loaded_workbench().probability(
            query=[{"variable": "x", "intervals": [[0.0, 0.5]]}], evidence=[]
        )

        assert result["ok"] is True
        assert result["probability"] == pytest.approx(0.5)

    def test_a_symbolic_constraint(self):
        result = loaded_workbench().probability(
            query=[{"variable": "color", "values": ["red"]}], evidence=[]
        )

        assert result["probability"] == pytest.approx(0.7)

    def test_conditioning_on_independent_evidence_changes_nothing(self):
        result = loaded_workbench().probability(
            query=[{"variable": "color", "values": ["red"]}],
            evidence=[{"variable": "x", "intervals": [[0.0, 0.5]]}],
        )

        assert result["probability"] == pytest.approx(0.7)

    def test_two_rows_on_one_variable_are_united(self):
        result = loaded_workbench().probability(
            query=[
                {"variable": "x", "intervals": [[0.0, 0.25]]},
                {"variable": "x", "intervals": [[0.75, 1.0]]},
            ],
            evidence=[],
        )

        assert result["probability"] == pytest.approx(0.5)


class TestPosterior:
    def test_each_requested_variable_gets_a_figure(self):
        result = loaded_workbench().posterior(
            variable_names=["x", "color"],
            evidence=[{"variable": "color", "values": ["red"]}],
        )

        assert result["ok"] is True
        assert sorted(result["figures"]) == ["color", "x"]
        for figure in result["figures"].values():
            assert figure["data"], "a figure must carry traces"

    def test_zero_probability_evidence_is_reported(self):
        result = loaded_workbench().posterior(
            variable_names=["x"],
            evidence=[{"variable": "x", "intervals": [[5.0, 6.0]]}],
        )

        assert result == {"ok": False, "error": "the evidence has zero probability"}


class TestMode:
    def test_the_unconditioned_mode_picks_the_likelier_color(self):
        result = loaded_workbench().mode(evidence=[])

        assert result["ok"] is True
        assert result["likelihood"] == pytest.approx(0.7)
        assert result["modes"] == [{"color": "red", "x": "[0, 1]"}]

    def test_a_negligible_interval_displays_as_its_value(self):
        pretty = ModelWorkbench._pretty_interval

        class NarrowInterval:
            lower = 0.024753697216510773
            upper = 0.024753738194704056

        class WideInterval:
            lower = 0.0
            upper = 0.5

        assert pretty(NarrowInterval()) == "0.02475"
        assert pretty(WideInterval()) == "[0, 0.5]"


# %% the API endpoints


@pytest.fixture()
def server(fixture_scene):
    """
    The real server on an ephemeral port, with a fresh workbench.
    """
    from cramera import server as server_module

    importlib.reload(server_module)
    ModelWorkbench._active = None
    httpd = server_module.make_server(0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://localhost:%d" % httpd.server_address[1]
    httpd.shutdown()
    ModelWorkbench._active = None


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


class TestModelsEndpoints:
    def test_the_whole_tab_flow_over_http(self, server):
        assert get_json(server + "/api/models/state")["loaded"] is False

        loaded = post_json(
            server + "/api/models/load",
            {"model": uniform_color_circuit().to_json(), "name": "circuit.json"},
        )
        assert loaded["loaded"] is True
        assert loaded["name"] == "circuit.json"

        probability = post_json(
            server + "/api/models/probability",
            {"query": [{"variable": "x", "intervals": [[0.0, 0.5]]}], "evidence": []},
        )
        assert probability["probability"] == pytest.approx(0.5)

        posterior = post_json(
            server + "/api/models/posterior",
            {"variables": ["x"], "evidence": []},
        )
        assert posterior["ok"] is True and posterior["figures"]["x"]["data"]

        mode = post_json(server + "/api/models/mode", {"evidence": []})
        assert mode["likelihood"] == pytest.approx(0.7)

    def test_a_file_with_infinity_literals_loads_as_text(self, server):
        """
        Circuit files carry ``Infinity`` bounds python writes and the browser's JSON
        parser rejects, so the tab uploads the file verbatim as ``model_text`` and the
        server parses it.
        """
        circuit = ProbabilisticCircuit()
        leaf(
            GaussianDistribution(variable=Continuous("x"), location=0.0, scale=1.0),
            circuit,
        )
        model_text = json.dumps(circuit.to_json())
        assert "Infinity" in model_text

        loaded = post_json(
            server + "/api/models/load",
            {"model_text": model_text, "name": "gaussian.json"},
        )

        assert loaded["loaded"] is True
        assert loaded["variables"] == [
            {"name": "x", "kind": "continuous", "low": -100.0, "high": 100.0}
        ]
        posterior = post_json(
            server + "/api/models/posterior", {"variables": ["x"], "evidence": []}
        )
        assert posterior["ok"] is True and posterior["figures"]["x"]["data"]

    def test_an_unknown_variable_is_reported_as_an_error(self, server):
        post_json(
            server + "/api/models/load",
            {"model": uniform_color_circuit().to_json(), "name": "circuit.json"},
        )

        result = post_json(
            server + "/api/models/probability",
            {"query": [{"variable": "nope", "intervals": [[0, 1]]}], "evidence": []},
        )

        assert result["ok"] is False
        assert "UnknownVariableError" in result["error"]

    def test_computing_without_a_model_is_reported(self, server):
        result = post_json(server + "/api/models/mode", {"evidence": []})

        assert result["ok"] is False
        assert "NoModelLoadedError" in result["error"]
