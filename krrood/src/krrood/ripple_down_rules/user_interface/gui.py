from __future__ import annotations

import inspect
import logging
import os.path
from types import MethodType

from krrood.ripple_down_rules import logger

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap, QPainter, QPalette
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QLabel,
        QScrollArea,
        QSizePolicy,
        QToolButton,
        QHBoxLayout,
        QPushButton,
        QMainWindow,
        QGraphicsView,
        QGraphicsScene,
        QGraphicsPixmapItem,
    )
    from qtconsole.inprocess import QtInProcessKernelManager
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
except ImportError as e:
    logging.debug(
        "RDRCaseViewer is not available. GUI features will not work. "
        "Make sure you have PyQt6 installed if you want to use the GUI features."
    )
    raise ImportError(
        "PyQt6 is required for the GUI features. Please install it using 'pip install PyQt6'"
    ) from e

from typing_extensions import Optional, Any, List, Dict, Callable

from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import PromptFor, ExitStatus
from krrood.ripple_down_rules.user_interface.template_file_creator import (
    TemplateFileCreator,
)
from krrood.ripple_down_rules.utils import (
    is_iterable,
    contains_return_statement,
    encapsulate_code_lines_into_a_function,
)
from krrood.ripple_down_rules.user_interface.object_diagram import generate_object_graph


class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.pixmap_item = None

        self._zoom = 0

    def update_image(self, image_path: str):
        if self.pixmap_item is None:
            self.pixmap_item = QGraphicsPixmapItem()
            self.scene().addItem(self.pixmap_item)
        pixmap = QPixmap(image_path)
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())

    def wheelEvent(self, event):
        # Zoom in or out with Ctrl + mouse wheel
        if event.modifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.2 if angle > 0 else 0.8

            self._zoom += 1 if angle > 0 else -1
            if self._zoom > 10:  # max zoom in limit
                self._zoom = 10
                return
            if self._zoom < -10:  # max zoom out limit
                self._zoom = -10
                return

            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


class BackgroundWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(image_path)

        # Layout for buttons
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)

        accept_btn = QPushButton("Accept")
        accept_btn.setStyleSheet(
            "background-color: #4CAF50; color: white;"
        )  # Green button
        edit_btn = QPushButton("Edit")
        edit_btn.setStyleSheet(
            "background-color: #2196F3; color: white;"
        )  # Blue button

        self.layout.addWidget(accept_btn)
        self.layout.addWidget(edit_btn)
        self.layout.addStretch()  # Push buttons to top

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap.isNull():
            # Calculate the vertical space used by buttons
            button_area_height = 0
            for i in range(self.layout.count()):
                item = self.layout.itemAt(i)
                if item.widget():
                    button_area_height += item.widget().height() + self.layout.spacing()

            remaining_height = self.height() - button_area_height
            if remaining_height <= 0:
                return  # No space to draw

            # Scale image to the remaining area (width, height)
            scaled = self.pixmap.scaled(
                self.width(),
                remaining_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            x = (self.width() - scaled.width()) // 2

            # Draw the image starting just below the buttons
            painter.drawPixmap(x, button_area_height + 20, scaled)

    def resizeEvent(self, event):
        self.update()  # Force repaint on resize
        super().resizeEvent(event)


class CollapsibleBox(QWidget):
    def __init__(
        self,
        title="",
        parent=None,
        viewer: Optional[RDRCaseViewer] = None,
        chain_name: Optional[str] = None,
        main_obj: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)

        self.viewer = viewer
        self.chain_name = chain_name
        self.main_obj = main_obj
        self.toggle_button = QToolButton(checkable=True, checked=False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.clicked.connect(self.toggle)
        self.toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                font-weight: bold;
                color: #FFA07A; /* Light orange */
            }
        """)
        self.title_label = QLabel(title)
        self.title_label.setTextFormat(Qt.TextFormat.RichText)  # Enable HTML rendering
        self.title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.title_label.setStyleSheet("QLabel { padding: 1px; color: #FFA07A; }")

        self.content_area = QWidget()
        self.content_area.setVisible(False)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(15, 2, 0, 2)
        self.content_layout.setSpacing(2)

        layout = QVBoxLayout(self)
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.title_label)
        layout.addLayout(header_layout)
        layout.addWidget(self.content_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def toggle(self):
        is_expanded = self.toggle_button.isChecked()
        self.update_object_diagram(is_expanded)
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if is_expanded else Qt.ArrowType.RightArrow
        )
        self.content_area.setVisible(is_expanded)

        self.update_object_diagram(is_expanded)

        # toggle children
        if not is_expanded:
            for i in range(self.content_layout.count()):
                item = self.content_layout.itemAt(i)
                if isinstance(item.widget(), CollapsibleBox):
                    item.widget().toggle_button.setChecked(False)
                    item.widget().toggle()

    def update_object_diagram(self, is_expanded):
        if is_expanded and self.viewer is not None:
            self.viewer.included_attrs.append(self.chain_name)
            main_obj_name = self.chain_name.split(".")[0]
            main_obj = self.main_obj.get(main_obj_name)
            self.viewer.update_object_diagram(main_obj, main_obj_name)
        elif self.viewer is not None and self.chain_name in self.viewer.included_attrs:
            for name in self.viewer.included_attrs:
                if name.startswith(self.chain_name):
                    self.viewer.included_attrs.remove(name)
            main_obj_name = self.chain_name.split(".")[0]
            main_obj = self.main_obj.get(main_obj_name)
            self.viewer.update_object_diagram(main_obj, main_obj_name)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def adjust_size_recursive(self):
        # Trigger resize
        self.adjustSize()

        # Traverse upwards to main window and call adjustSize on it too
        parent = self.parent()
        while parent:
            if isinstance(parent, QWidget):
                parent.layout().activate()  # Force layout refresh
                parent.adjustSize()
            elif isinstance(parent, QScrollArea):
                parent.widget().adjustSize()
                parent.viewport().update()
            if isinstance(parent, BackgroundWidget):
                parent.update()
                parent.updateGeometry()
                parent.repaint()
            if parent.parent() is None:
                top_window = parent.window()  # The main top-level window
                top_window.updateGeometry()
                top_window.repaint()
            parent = parent.parent()


def python_colored_repr(value):
    if isinstance(value, str):
        return f'<span style="color:#90EE90;">"{value}"</span>'
    elif isinstance(value, (int, float)):
        return f'<span style="color:#ADD8E6;">{value}</span>'
    elif isinstance(value, bool) or value is None:
        return f'<span style="color:darkorange;">{value}</span>'
    elif isinstance(value, type):
        return f'<span style="color:#C1BCBB;">{{{value.__name__}}}</span>'
    elif callable(value):
        return ""
    else:
        try:
            return f'<span style="color:white;">{repr(value)}</span>'
        except Exception as e:
            return f'<span style="color:red;">&lt;error: {e}&gt;</span>'


def style(text, color=None, font_size=None, font_weight=None):
    s = '<span style="'
    if color:
        s += f"color:{color_name_to_html(color)};"
    if font_size:
        s += f"font-size:{font_size}px;"
    if font_weight:
        s += f"font-weight:{font_weight};"
    s += '">'
    s += text
    s += "</span>"
    return s


def color_name_to_html(color_name):
    single_char_to_name = {
        "r": "red",
        "g": "green",
        "b": "blue",
        "o": "orange",
        "m": "magenta",
    }
    color_map = {
        "red": "#d6336c",
        "green": "#2eb872",
        "blue": "#007acc",
        "orange": "#FFA07A",
        "magenta": "#a22bcf",
    }
    if len(color_name) == 1:
        color_name = single_char_to_name.get(color_name, color_name)
    return color_map.get(
        color_name.lower(), color_name
    )  # Default to the name itself if not found


class RDRCaseViewer(QMainWindow):
    case_query: Optional[CaseQuery] = None
    prompt_for: Optional[PromptFor] = None
    code_to_modify: Optional[str] = None
    template_file_creator: Optional[TemplateFileCreator] = None
    code_lines: Optional[List[str]] = None
    included_attrs: Optional[List[str]] = None
    main_obj: Optional[Dict[str, Any]] = None
    user_input: Optional[str] = None
    attributes_widget: Optional[QWidget] = None
    save_function: Optional[Callable[str, str], None] = None
    instances: List[RDRCaseViewer] = []
    exit_status: ExitStatus = ExitStatus.CLOSE

    def __init__(self, parent=None):
        super().__init__(parent)
        self.instances.clear()
        self.instances.append(self)
        self.setWindowTitle("RDR Case Viewer")

        self.setBaseSize(1600, 600)  # or your preferred initial size

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.setStyleSheet("background-color: #333333;")
        main_widget.setStyleSheet("background-color: #333333;")

        main_layout = QHBoxLayout(main_widget)  # Horizontal layout to split window

        # === Left: Attributes ===
        self._create_attribute_widget()

        # === Middle: Ipython & Action buttons ===
        middle_widget = QWidget()
        self.middle_widget_layout = QVBoxLayout(middle_widget)

        self.title_label = self.create_label_widget(
            style(
                style(f"Welcome to {style('RDRViewer', 'b')} " f"{style('App', 'g')}"),
                "o",
                28,
                "bold",
            )
        )

        self.buttons_widget = self.create_buttons_widget()

        self.ipython_console = IPythonConsole(parent=self)

        self.middle_widget_layout.addWidget(self.title_label)
        self.middle_widget_layout.addWidget(self.ipython_console)
        self.middle_widget_layout.addWidget(self.buttons_widget)

        # === Right: Object Diagram ===
        self.obj_diagram_viewer = ImageViewer()  # put your image path here

        # Add both to main layout
        main_layout.addWidget(self.attributes_widget, stretch=1)
        main_layout.addWidget(middle_widget, stretch=2)
        main_layout.addWidget(self.obj_diagram_viewer, stretch=2)

    def print(self, msg):
        """
        Print a message to the console.
        """
        self.ipython_console._append_plain_text(msg + "\n", True)

    def update_for_case_query(
        self,
        case_query: CaseQuery,
        prompt_str: Optional[str] = None,
        prompt_for: Optional[PromptFor] = None,
        code_to_modify: Optional[str] = None,
        title: Optional[str] = None,
    ):
        self.case_query = case_query
        self.prompt_for = prompt_for
        self.code_to_modify = code_to_modify
        title_text = title or ""
        case_attr_type = ", ".join([t.__name__ for t in case_query.core_attribute_type])
        case_attr_type = style(f"{case_attr_type}", "g", 28, "bold")
        case_name = style(f"{case_query.name}", "b", 28, "bold")
        title_text = style(
            f"{title_text} {case_name} of type {case_attr_type}", "o", 28, "bold"
        )
        self.update_for_object(
            case_query.case,
            case_query.case_name,
            scope=case_query.scope,
            title_text=title_text,
            header=prompt_str,
        )

    def update_for_object(
        self,
        obj: Any,
        name: str,
        scope: Optional[dict] = None,
        title_text: Optional[str] = None,
        header: Optional[str] = None,
    ):
        self.update_main_obj(obj, name)
        title_text = title_text or style(f"{name}", "o", 28, "bold")
        scope = scope or {}
        scope.update({name: obj})
        self.update_object_diagram(obj, name)
        self.update_attribute_layout(obj, name)
        self.title_label.setText(title_text)
        self.ipython_console.update_namespace(scope)
        if header is not None and len(header) > 0:
            self.ipython_console.print(header)
        self.exit_status = ExitStatus.CLOSE

    def update_main_obj(self, obj, name):
        self.main_obj = {name: obj}
        self.included_attrs = []
        self.user_input = None

    def update_object_diagram(self, obj: Any, name: str):
        self.included_attrs = self.included_attrs or []
        graph = generate_object_graph(obj, name, included_attrs=self.included_attrs)
        file_name = "object_diagram"
        format = "svg"
        filepath = f"{file_name}.{format}"
        try:
            graph.render(file_name, view=False, format=format)
        except FileNotFoundError:
            tmp_filepath = f"{file_name}.dot"
            graph.save(tmp_filepath, directory="./")
            try:
                os.system(f"/usr/bin/dot -T{format} {tmp_filepath} -o {filepath}")
                os.remove(tmp_filepath)
            except Exception as e:
                logger.error(e)
        if os.path.exists(filepath):
            self.obj_diagram_viewer.update_image(filepath)

    def _create_attribute_widget(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)

        expand_btn = QPushButton("Expand All")
        expand_btn.clicked.connect(self.expand_all)
        expand_btn.setStyleSheet(
            f"background-color: white; color: black;"
        )  # Green button
        buttons_layout.addWidget(expand_btn)

        collapse_btn = QPushButton("Collapse All")
        collapse_btn.clicked.connect(self.collapse_all)
        collapse_btn.setStyleSheet(
            f"background-color: white; color: black;"
        )  # Green button
        buttons_layout.addWidget(collapse_btn)

        main_layout.addWidget(buttons_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        attr_widget = QWidget()
        self.attr_widget_layout = QVBoxLayout(attr_widget)
        self.attr_widget_layout.setSpacing(2)
        self.attr_widget_layout.setContentsMargins(6, 6, 6, 6)
        attr_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(attr_widget)
        main_layout.addWidget(scroll)
        self.attributes_widget = main_widget

    def expand_all(self):
        # Expand all collapsible boxes
        for i in range(self.attr_widget_layout.count()):
            item = self.attr_widget_layout.itemAt(i)
            if isinstance(item.widget(), CollapsibleBox):
                self.expand_collapse_all(item.widget(), expand=True)

    def collapse_all(self):
        # Collapse all collapsible boxes
        for i in range(self.attr_widget_layout.count()):
            item = self.attr_widget_layout.itemAt(i)
            if isinstance(item.widget(), CollapsibleBox):
                self.expand_collapse_all(item.widget(), expand=False)

    def expand_collapse_all(self, widget, expand=True, curr_depth=0, max_depth=2):
        if expand and curr_depth < max_depth:
            widget.toggle_button.setChecked(expand)
            widget.toggle()
            # do it for recursive children
            for i in range(widget.content_layout.count()):
                item = widget.content_layout.itemAt(i)
                if isinstance(item.widget(), CollapsibleBox):
                    self.expand_collapse_all(
                        item.widget(),
                        expand=True,
                        curr_depth=curr_depth + 1,
                        max_depth=max_depth,
                    )

    def create_buttons_widget(self):
        button_widget = QWidget()
        button_widget_layout = QVBoxLayout(button_widget)

        row_1_button_widget = QWidget()
        row_1_button_widget_layout = QHBoxLayout(row_1_button_widget)
        row_2_button_widget = QWidget()
        row_2_button_widget_layout = QHBoxLayout(row_2_button_widget)
        button_widget_layout.addWidget(row_1_button_widget)
        button_widget_layout.addWidget(row_2_button_widget)

        accept_btn = QPushButton("Accept")
        accept_btn.clicked.connect(self._accept)
        accept_btn.setStyleSheet(
            f"background-color: {color_name_to_html('g')}; color: white;"
        )  # Green button

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit)
        edit_btn.setStyleSheet(
            f"background-color: {color_name_to_html('o')}; color: white;"
        )  # Orange button

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load)
        load_btn.setStyleSheet(
            f"background-color: {color_name_to_html('b')}; color: white;"
        )  # Blue button

        current_value_btn = QPushButton("Current Value")
        current_value_btn.clicked.connect(self._show_current_value)
        current_value_btn.setStyleSheet(
            f"background-color: {color_name_to_html('m')}; color: white;"
        )

        rule_tree_btn = QPushButton("Rule Tree")
        rule_tree_btn.clicked.connect(
            self._show_rule_tree
        )  # Placeholder for rule tree functionality
        rule_tree_btn.setStyleSheet(
            f"background-color: {color_name_to_html('r')}; color: white;"
        )

        row_1_button_widget_layout.addWidget(edit_btn)
        row_1_button_widget_layout.addWidget(load_btn)
        row_1_button_widget_layout.addWidget(current_value_btn)
        row_2_button_widget_layout.addWidget(rule_tree_btn)
        row_2_button_widget_layout.addWidget(accept_btn)
        return button_widget

    def _accept(self):
        self.exit_status = ExitStatus.SUCCESS
        # close the window
        self.close()

    def _edit(self):
        self.template_file_creator = self.create_template_file_creator()
        self.template_file_creator.edit()

    def _load(self):
        if not self.template_file_creator:
            return
        self.code_lines, updates = self.template_file_creator.load(
            self.template_file_creator.temp_file_path,
            self.template_file_creator.func_name,
            self.template_file_creator.print_func,
        )
        self.ipython_console.kernel.shell.user_ns.update(updates)
        if self.code_lines is not None:
            self.user_input = encapsulate_code_lines_into_a_function(
                self.code_lines,
                self.template_file_creator.func_name,
                self.template_file_creator.function_signature,
                self.template_file_creator.func_doc,
                self.case_query,
            )
            self.case_query.scope.update(updates)

    def _show_current_value(self):
        self.ipython_console.print(self.case_query.current_value_str)

    def _show_rule_tree(self):
        if self.case_query is None:
            self.ipython_console.print("No case query provided.")
            return
        if not self.case_query.rdr:
            self.ipython_console.print("No rule tree available for this case query.")
            return
        self.case_query.render_rule_tree(view=True)

    def create_template_file_creator(self) -> TemplateFileCreator:
        return TemplateFileCreator(
            self.case_query, self.prompt_for, self.code_to_modify, self.print
        )

    def update_attribute_layout(self, obj, name: str):
        # Clear the existing layout
        clear_layout(self.attr_widget_layout)

        # Re-add the collapsible box with the new object
        self.add_collapsible(name, obj, self.attr_widget_layout, 0, 3, chain_name=name)
        self.attr_widget_layout.addStretch()

    def create_label_widget(self, text):
        # Create a QLabel with rich text
        label = QLabel()
        label.setText(text)
        label.setAlignment(Qt.AlignCenter)  # Optional: center the text
        label.setWordWrap(True)  # Optional: allow wrapping if needed
        return label

    def add_attributes(
        self, obj, name, layout, current_depth=0, max_depth=3, chain_name=None
    ):
        if current_depth > max_depth:
            return
        if isinstance(obj, dict):
            items = obj.items()
        elif isinstance(obj, (list, tuple, set)):
            items = enumerate(obj)
        else:
            methods = []
            attributes = []
            iterables = []
            for attr in dir(obj):
                if attr.startswith("_") or attr == "scope":
                    continue
                try:
                    value = getattr(obj, attr)
                    if callable(value):
                        methods.append((attr, value))
                        continue
                    elif is_iterable(value):
                        iterables.append((attr, value))
                        continue
                except Exception as e:
                    value = f"<error: {e}>"
                attributes.append((attr, value))
            items = attributes + iterables + methods
        chain_name = chain_name if chain_name is not None else name
        for attr, value in items:
            attr = f"{attr}"
            try:
                if (
                    is_iterable(value)
                    or hasattr(value, "__dict__")
                    and not inspect.isfunction(value)
                ):
                    self.add_collapsible(
                        attr,
                        value,
                        layout,
                        current_depth + 1,
                        max_depth,
                        chain_name=f"{chain_name}.{attr}",
                    )
                else:
                    self.add_non_collapsible(attr, value, layout)
            except Exception as e:
                err = QLabel(
                    f"<b>{attr}</b>: <span style='color:red;'>&lt;error: {e}&gt;</span>"
                )
                err.setTextFormat(Qt.TextFormat.RichText)
                layout.addWidget(err)

    def add_collapsible(
        self, attr, value, layout, current_depth, max_depth, chain_name=None
    ):
        type_name = type(value) if not isinstance(value, type) else value
        collapsible = CollapsibleBox(
            f'<b><span style="color:#FFA07A;">{attr}</span></b> {python_colored_repr(type_name)}',
            viewer=self,
            chain_name=chain_name,
            main_obj=self.main_obj,
        )
        self.add_attributes(
            value,
            attr,
            collapsible.content_layout,
            current_depth,
            max_depth,
            chain_name=chain_name,
        )
        layout.addWidget(collapsible)

    def add_non_collapsible(self, attr, value, layout):
        type_name = type(value) if not isinstance(value, type) else value
        text = f'<b><span style="color:#FFA07A;">{attr}</span></b> {python_colored_repr(type_name)}: {python_colored_repr(value)}'
        item_label = QLabel()
        item_label.setTextFormat(Qt.TextFormat.RichText)
        item_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        item_label.setStyleSheet("QLabel { padding: 1px; color: #FFA07A; }")
        item_label.setText(text)
        item_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(item_label)


class IPythonConsole(RichJupyterWidget):
    def __init__(self, namespace=None, parent=None):
        super(IPythonConsole, self).__init__(parent)

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = "qt"
        self.command_log = []

        # Monkey patch its run_cell method
        def custom_run_cell(this, raw_cell, **kwargs):
            if contains_return_statement(raw_cell) and "def " not in raw_cell:
                if (
                    self.parent.template_file_creator
                    and self.parent.template_file_creator.func_name in raw_cell
                ):
                    self.command_log = self.parent.code_lines
                self.command_log.append(raw_cell)
                this.history_manager.store_inputs(
                    line_num=this.execution_count, source=raw_cell
                )
                return None
            result = original_run_cell(raw_cell, **kwargs)
            if result.error_in_exec is None and result.error_before_exec is None:
                self.command_log.append(raw_cell)
            return result

        original_run_cell = self.kernel.shell.run_cell
        self.kernel.shell.run_cell = MethodType(custom_run_cell, self.kernel.shell)
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        # Update the user namespace with your custom variables
        if namespace:
            self.update_namespace(namespace)

        # Set the underlying QTextEdit's palette
        palette = QPalette()
        self._control.setPalette(palette)

        # Override the stylesheet to force background and text colors
        self.style_sheet = """\
    QPlainTextEdit, QTextEdit {
        background-color: %(bgcolor)s;
        background-clip: padding;
        color: %(fgcolor)s;
        selection-background-color: %(select)s;
    }
    .inverted {
        background-color: %(fgcolor)s;
        color: %(bgcolor)s;
    }
    .error { color: red; }
    .in-prompt-number { font-weight: bold; color: %(in_prompt_number_color)s} }
    .in-prompt { font-weight: bold; color: %(in_prompt_color)s }
    .out-prompt-number { font-weight: bold; color: %(out_prompt_number_color)s }
    .out-prompt { font-weight: bold; color: %(out_prompt_color)s }
""" % dict(
            bgcolor="#0b0d0b",
            fgcolor="#47d9cc",
            select="#555",
            in_prompt_number_color="lime",
            in_prompt_color="lime",
            out_prompt_number_color="red",
            out_prompt_color="red",
        )

        # Use a dark syntax style like monokai
        self.syntax_style = "monokai"
        # self.set_default_style(colors='lightbg')

        self.exit_requested.connect(self.stop)

    def update_namespace(self, namespace):
        """
        Update the user namespace with new variables.
        """
        self.kernel.shell.user_ns.update(namespace)

    def print(self, msg):
        """
        Custom print function to append messages to the command log.
        """
        self.execute(f'print("\\n\\n{msg}")', hidden=True)

    def execute(self, source=None, hidden=False, interactive=False):
        # Log the command before execution
        source = source if source is not None else self.input_buffer
        # self.command_log.append(source)
        super().execute(source, hidden, interactive)

    def stop(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
        elif item.layout() is not None:
            clear_layout(item.layout())
