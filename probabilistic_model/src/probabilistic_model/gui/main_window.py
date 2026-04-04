from dataclasses import dataclass, field, InitVar
from typing import Optional
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QToolBar,
    QFileDialog,
    QStackedWidget,
)
from PySide6.QtGui import QAction, QActionGroup

from .controller import ModelController
from .home_widget import HomeWidget
from .query_widget import QueryWidget
from .posterior_widget import PosteriorWidget
from .mode_widget import ModeWidget
from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class MainWindow(QMainWindow):
    """
    Main Window of the Probabilistic Model GUI.
    """

    model: Optional[ProbabilisticModel] = None
    """
    The initial probabilistic model (optional).
    """

    parent: InitVar[Optional[QWidget]] = None
    """
    The parent widget.
    """

    controller: ModelController = field(init=False)
    """
    The model controller.
    """

    home_widget: HomeWidget = field(init=False)
    """
    The home page widget.
    """

    query_widget: QueryWidget = field(init=False)
    """
    The query page widget.
    """

    posterior_widget: PosteriorWidget = field(init=False)
    """
    The posterior page widget.
    """

    mode_widget: ModeWidget = field(init=False)
    """
    The mode page widget.
    """

    central_stack: QStackedWidget = field(init=False)
    """
    The stacked widget for the pages.
    """

    navigation_group: QActionGroup = field(init=False)
    """
    The action group for toolbar navigation.
    """

    def __post_init__(self, parent: Optional[QWidget]):
        super().__init__(parent)
        self.setWindowTitle("Probabilistic Model GUI")
        self.resize(1000, 700)

        self.controller = ModelController(model=self.model)
        self.init_ui()
        if self.model:
            self.refresh_widgets()

    def init_ui(self):
        """
        Initializes the user interface.
        """
        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        load_action = QAction("Load Model", self)
        load_action.triggered.connect(self.load_model)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        self.navigation_group = QActionGroup(self)

        home_action = QAction("Home", self)
        home_action.setCheckable(True)
        home_action.setActionGroup(self.navigation_group)
        home_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.home_widget)
        )
        toolbar.addAction(home_action)
        home_action.setChecked(True)

        query_action = QAction("Query", self)
        query_action.setCheckable(True)
        query_action.setActionGroup(self.navigation_group)
        query_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.query_widget)
        )
        toolbar.addAction(query_action)

        posterior_action = QAction("Posterior", self)
        posterior_action.setCheckable(True)
        posterior_action.setActionGroup(self.navigation_group)
        posterior_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.posterior_widget)
        )
        toolbar.addAction(posterior_action)

        mode_action = QAction("Mode", self)
        mode_action.setCheckable(True)
        mode_action.setActionGroup(self.navigation_group)
        mode_action.triggered.connect(
            lambda: self.central_stack.setCurrentWidget(self.mode_widget)
        )
        toolbar.addAction(mode_action)

        # Central Widget
        self.central_stack = QStackedWidget()
        self.central_stack.currentChanged.connect(self.on_page_changed)
        self.setCentralWidget(self.central_stack)

        self.home_widget = HomeWidget(self.controller)
        self.query_widget = QueryWidget(self.controller)
        self.posterior_widget = PosteriorWidget(self.controller)
        self.mode_widget = ModeWidget(self.controller)

        self.central_stack.addWidget(self.home_widget)
        self.central_stack.addWidget(self.query_widget)
        self.central_stack.addWidget(self.posterior_widget)
        self.central_stack.addWidget(self.mode_widget)

    def on_page_changed(self, index: int):
        """
        Updates the toolbar when the page changes.
        """
        widget = self.central_stack.widget(index)
        for action in self.navigation_group.actions():
            if action.text() == "Home" and widget == self.home_widget:
                action.setChecked(True)
            elif action.text() == "Query" and widget == self.query_widget:
                action.setChecked(True)
            elif action.text() == "Posterior" and widget == self.posterior_widget:
                action.setChecked(True)
            elif action.text() == "Mode" and widget == self.mode_widget:
                action.setChecked(True)

    def load_model(self):
        """
        Opens a file dialog to load a model.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("JSON Files (*.json)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.controller.load_model_from_json_file(file_paths[0])
                self.refresh_widgets()

    def refresh_widgets(self):
        """
        Refreshes all widgets when a new model is loaded.
        """
        self.home_widget.refresh_variable_list()
        # Re-initialize query widget or refresh it
        self.central_stack.removeWidget(self.query_widget)
        self.query_widget.deleteLater()
        self.query_widget = QueryWidget(self.controller)
        self.central_stack.addWidget(self.query_widget)

        self.central_stack.removeWidget(self.posterior_widget)
        self.posterior_widget.deleteLater()
        self.posterior_widget = PosteriorWidget(self.controller)
        self.central_stack.addWidget(self.posterior_widget)

        self.central_stack.removeWidget(self.mode_widget)
        self.mode_widget.deleteLater()
        self.mode_widget = ModeWidget(self.controller)
        self.central_stack.addWidget(self.mode_widget)
