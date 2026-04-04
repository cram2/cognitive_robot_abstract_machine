from dataclasses import dataclass
import os
import sys
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from qt_material import apply_stylesheet

from probabilistic_model.probabilistic_model import ProbabilisticModel
from .main_window import MainWindow


@dataclass
class GUIFactory:
    """
    Factory for creating and starting the Probabilistic Model GUI.
    """

    model: Optional[ProbabilisticModel] = None
    """
    The probabilistic model to display.
    """

    theme: str = "dark_amber.xml"
    """
    The qt-material theme to apply.
    """

    def start_gui(self) -> None:
        """
        Starts the GUI with the given model and theme.
        """
        # Recommended for some Linux environments/Docker
        os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

        # Required for QWebEngineView to work in many environments
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

        app = QApplication(sys.argv)
        apply_stylesheet(app, theme=self.theme)

        window = MainWindow(model=self.model)
        window.show()

        sys.exit(app.exec())
