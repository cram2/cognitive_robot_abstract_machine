from dataclasses import dataclass, field, InitVar
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QScrollArea
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from .controller import ModelController
from random_events.interval import Interval
from random_events.set import Set
from typing import Optional


@dataclass
class HomeWidget(QWidget):
    """
    The Home page widget of the GUI.
    Displays the model logo and a list of variables with their support.
    """

    controller: ModelController
    """
    The model controller.
    """

    parent: InitVar[Optional[QWidget]] = None
    """
    The parent widget.
    """

    variable_list_container: QWidget = field(init=False)
    """
    Container for the list of variables.
    """

    variable_list_layout: QVBoxLayout = field(init=False)
    """
    Layout for the list of variables.
    """

    def __post_init__(self, parent: Optional[QWidget]):
        super().__init__(parent)
        self.init_ui()

    def get_logo_path(self) -> str:
        """
        Gets the absolute path to the logo file.
        """
        # The logo is in doc/Logo.svg relative to the project root
        # We can also search for it relative to this file
        current_file_path = os.path.abspath(__file__)
        package_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        )
        logo_path = os.path.join(package_root, "doc", "Logo.svg")
        return logo_path

    def init_ui(self):
        """
        Initializes the user interface.
        """
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Logo
        logo_label = QLabel()
        logo_path = self.get_logo_path()
        logo_pixmap = QPixmap(logo_path)
        if logo_pixmap.isNull():
            # fallback or ignore
            logo_label.setText("[Logo]")
        else:
            logo_label.setPixmap(
                logo_pixmap.scaledToHeight(
                    350, Qt.TransformationMode.SmoothTransformation
                )
            )

        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        # Variable List
        self.variable_list_container = QWidget()
        self.variable_list_layout = QVBoxLayout(self.variable_list_container)
        self.variable_list_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.variable_list_container)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(scroll_area)

        self.refresh_variable_list()

    def refresh_variable_list(self):
        """
        Refreshes the list of variables and their supports.
        """
        # Clear existing
        while self.variable_list_layout.count():
            item = self.variable_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not self.controller.priors:
            return

        for variable, distribution in self.controller.priors.items():
            support = variable.domain

            row_layout = QHBoxLayout()
            row_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            name_label = QLabel(variable.name)
            name_label.setStyleSheet("font-size: 14pt;")

            in_label = QLabel(" ∈ ")
            in_label.setStyleSheet("font-size: 14pt;")

            support_text = ""
            if variable.is_numeric:
                # Replicate Dash logic
                mini = support.simple_sets[0].lower
                maxi = support.simple_sets[-1].upper
                support_text = f"[{round(mini, 3)}, {round(maxi, 3)}]"
            else:
                # Replicate Dash logic for discrete
                # support.all_elements is available on Set
                support_text = str(list(support.all_elements))

            support_label = QLabel(support_text)
            support_label.setStyleSheet("font-size: 14pt;")

            row_layout.addWidget(name_label)
            row_layout.addWidget(in_label)
            row_layout.addWidget(support_label)

            row_widget = QWidget()
            row_widget.setLayout(row_layout)
            self.variable_list_layout.addWidget(row_widget)
