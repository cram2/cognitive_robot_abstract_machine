import json
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from probabilistic_model.gui.factory import GUIFactory


def main(model_path: str = None):
    """
    Main entry point for starting the Probabilistic Model GUI.

    :param model_path: The path to the model to display in the GUI.
    """
    model_path = "/home/tom_sch/.config/JetBrains/PyCharm2025.3/scratches/model.pm"

    if model_path is None:
        model = None
    else:
        with open(model_path) as f:
            model = ProbabilisticCircuit.from_json(json.load(f))

    factory = GUIFactory(model=model, theme="dark_amber.xml")
    factory.start_gui()


if __name__ == "__main__":
    main()
