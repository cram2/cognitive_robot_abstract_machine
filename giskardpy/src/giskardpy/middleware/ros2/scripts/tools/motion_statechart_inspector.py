import json
import signal
import sys
from typing import List, Dict, Optional, Any

from PyQt5.QtCore import QMutex, QMutexLocker
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QKeySequence
from PyQt5.QtSvg import QGraphicsSvgItem, QSvgRenderer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QShortcut,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    LifeCycleState,
    ObservationState,
)
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz
from json_msgs.action import JsonAction
from json_msgs.action._json_action import JsonAction_FeedbackMessage

compact = False


class SvgGrahpicsView(QGraphicsView):

    def __init__(self, *args):
        QGraphicsView.__init__(self, *args)
        self.mutex = QMutex()  # Mutex for synchronizing access to the widget

        # Set up the graphics scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.svg_item = QGraphicsSvgItem()
        self.scene.addItem(self.svg_item)

        # Configure the view
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        # Only zoom when Ctrl is pressed
        if event.modifiers() & Qt.ControlModifier:
            delta = 1.0 + event.angleDelta().y() / 1200
            transform = self.transform()
            transform.scale(delta, delta)
            self.setTransform(transform)
        elif event.modifiers() & Qt.ShiftModifier:
            # Horizontal scroll when Shift is pressed
            delta = event.angleDelta().y()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta
            )
        else:
            # Pass the event to parent if Ctrl or Shift is not pressed
            super().wheelEvent(event)

    def load(self, svg_path):
        with QMutexLocker(self.mutex):
            renderer = QSvgRenderer(svg_path)
            self.svg_item.setSharedRenderer(renderer)
            self.scene.setSceneRect(self.svg_item.boundingRect())
            self.fitInView(self.svg_item, Qt.KeepAspectRatio)


class DotGraphViewer(QWidget):
    # Add this signal to communicate between threads
    new_message_signal: pyqtSignal = pyqtSignal(object)
    last_goal_id: Optional[int]
    graphs_by_goal: Dict[int, List[Any]]
    motion_statechart: Optional[MotionStatechart]

    def __init__(self):
        super().__init__()
        self.last_goal_id = None
        self.motion_statechart = None

        # Connect the signal to the slot
        self.new_message_signal.connect(self.handle_new_message)

        # Initialize the ROS node
        rospy.init_node("motion_statechart_inspector")

        self.setup_qt()

        # Initialize graph history and goal tracking
        self.graphs_by_goal = {}
        self.goals = []
        self.current_goal_index = -1
        self.current_message_index = -1

    def setup_qt(self):
        self.setup_gui_components()
        self.add_keyboard_shortcuts()
        self.setup_layout()
        self.setup_topic_refresh_timer()

    def setup_topic_refresh_timer(self):
        self.topic_refresh_timer = QTimer(self)
        self.topic_refresh_timer.timeout.connect(self.refresh_topics)
        self.topic_refresh_timer.start(1000)  # Refresh every 5 seconds
        self.refresh_topics()

    def setup_layout(self):
        # Layout for topic selection
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.topic_selector)

        # Layout for navigation buttons and position label
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.first_button)
        nav_layout.addWidget(self.prev_goal_button)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.position_label)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.next_goal_button)
        nav_layout.addWidget(self.latest_button)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.svg_widget)
        layout.addLayout(nav_layout)
        self.setLayout(layout)

        self.setWindowTitle("Motion Statechart Inspector")
        self.resize(800, 600)

    def setup_gui_components(self):
        self.svg_widget = SvgGrahpicsView(self)
        self.svg_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.svg_widget.setMinimumSize(600, 400)

        self.topic_selector = QComboBox(self)
        self.topic_selector.activated.connect(self.on_topic_selector_clicked)

        self.position_label = QLabel(self)
        self.setup_navigation_buttons()

    def setup_navigation_buttons(self):
        self.first_button = QPushButton("First")
        self.first_button.clicked.connect(self.show_first_image)
        self.first_button.setToolTip("Home/(Shift + LeftArrow)")

        self.prev_goal_button = QPushButton("Prev Goal")
        self.prev_goal_button.clicked.connect(self.show_prev_goal)
        self.prev_goal_button.setToolTip("(Ctrl + PageDown)/(Ctrl + LeftArrow)")

        self.prev_button = QPushButton("<")
        self.prev_button.clicked.connect(self.show_previous_image)
        self.prev_button.setToolTip("PageDown/LeftArrow")

        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(self.show_next_image)
        self.next_button.setToolTip("PageUp/RightArrow")

        self.next_goal_button = QPushButton("Next Goal")
        self.next_goal_button.clicked.connect(self.show_next_goal)
        self.next_goal_button.setToolTip("(Ctrl + PageUp)/(Ctrl + RightArrow)")

        self.latest_button = QPushButton("Last")
        self.latest_button.clicked.connect(self.show_latest_image)
        self.latest_button.setToolTip("End/(Shift + RightArrow)")

    def add_keyboard_shortcuts(self):
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.activated.connect(self.show_previous_image)
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.activated.connect(self.show_next_image)
        self.page_down_shortcut = QShortcut(QKeySequence(Qt.Key_PageDown), self)
        self.page_down_shortcut.activated.connect(self.show_previous_image)
        self.page_up_shortcut = QShortcut(QKeySequence(Qt.Key_PageUp), self)
        self.page_up_shortcut.activated.connect(self.show_next_image)

        self.control_left_shortcut = QShortcut(
            QKeySequence(Qt.CTRL + Qt.Key_Left), self
        )
        self.control_left_shortcut.activated.connect(self.show_prev_goal)
        self.control_right_shortcut = QShortcut(
            QKeySequence(Qt.CTRL + Qt.Key_Right), self
        )
        self.control_right_shortcut.activated.connect(self.show_next_goal)
        self.control_page_down_shortcut = QShortcut(
            QKeySequence(Qt.CTRL + Qt.Key_PageDown), self
        )
        self.control_page_down_shortcut.activated.connect(self.show_prev_goal)
        self.control_page_up_shortcut = QShortcut(
            QKeySequence(Qt.CTRL + Qt.Key_PageUp), self
        )
        self.control_page_up_shortcut.activated.connect(self.show_next_goal)

        self.shift_left_shortcut = QShortcut(QKeySequence(Qt.SHIFT + Qt.Key_Left), self)
        self.shift_left_shortcut.activated.connect(self.show_first_image)
        self.shift_right_shortcut = QShortcut(
            QKeySequence(Qt.SHIFT + Qt.Key_Right), self
        )
        self.shift_right_shortcut.activated.connect(self.show_latest_image)
        self.pos1_shortcut = QShortcut(QKeySequence(Qt.Key_Home), self)
        self.pos1_shortcut.activated.connect(self.show_first_image)
        self.end_shortcut = QShortcut(QKeySequence(Qt.Key_End), self)
        self.end_shortcut.activated.connect(self.show_latest_image)

    def refresh_topics(self) -> None:
        if self.topic_selector.currentText() == "":
            # Find all topics of type ExecutionState
            topics = rospy.node.get_topic_names_and_types()
            target_type = "giskard_msgs/action/JsonAction_FeedbackMessage"
            execution_state_topics = [
                name for name, types in topics if target_type in types
            ]

            self.topic_selector.clear()
            self.topic_selector.addItems(execution_state_topics)
            if len(execution_state_topics) > 0:
                self.on_topic_selected(0)

    def on_topic_selector_clicked(self) -> None:
        # Stop refreshing topics once a topic is selected
        if self.topic_selector.currentIndex() != -1:
            self.topic_refresh_timer.stop()
            self.on_topic_selected(self.topic_selector.currentIndex())

    def on_topic_selected(self, index: int) -> None:
        topic_name = self.topic_selector.currentText()
        if topic_name:
            rospy.node.create_subscription(
                msg_type=JsonAction_FeedbackMessage,
                topic=topic_name,
                callback=self.on_new_message_received,
                qos_profile=10,
            )

    def on_new_message_received(self, msg: JsonAction.Feedback) -> None:
        self.new_message_signal.emit(msg)

    def handle_new_message(self, msg: JsonAction_FeedbackMessage) -> None:
        # This runs in the main thread
        json_data = json.loads(msg.feedback.feedback)
        msg_goal_id = json_data["goal_id"]

        self.parse_new_motion_statechart(json_data)
        self.parse_state(json_data)

        at_last_msg = True
        if len(self.goals) > 0:
            at_last_msg = (
                self.current_goal_index == self.goals[-1]
                and self.current_message_index
                == len(self.graphs_by_goal[self.goals[-1]]) - 1
            )

        # Extract goal_id and group graphs by goal_id
        if self.last_goal_id == msg_goal_id:
            goal_id = self.goals[-1]
        else:
            self.last_goal_id = msg_goal_id
            goal_id = len(self.goals)
            self.graphs_by_goal[goal_id] = []
            self.goals.append(goal_id)

        self.plot_motion_statechart(goal_id)

        # Update the display to show the latest graph
        if at_last_msg:
            self.current_goal_index = len(self.goals) - 1
            self.current_message_index = len(self.graphs_by_goal[goal_id]) - 1

        self.update_position_label()

        if at_last_msg:
            self.display_graph(
                self.current_goal_index,
                self.current_message_index,
                update_position_label=False,
            )

    def plot_motion_statechart(self, goal_id: int):
        graph = MotionStatechartGraphviz(self.motion_statechart).to_dot_graph()
        self.graphs_by_goal[goal_id].append(graph)

    def parse_new_motion_statechart(self, json_data: Dict[str, Any]):
        motion_statechart_data = json_data.get("motion_statechart")
        if motion_statechart_data is not None:
            self.motion_statechart = MotionStatechart.from_json(motion_statechart_data)
            self.motion_statechart._add_transitions()

    def parse_state(self, json_data: Dict[str, Any]):
        life_cycle_data = json_data.get("life_cycle_state")
        life_cycle_state = LifeCycleState.from_json(
            life_cycle_data, motion_statechart=self.motion_statechart
        )
        observation_data = json_data.get("observation_state")
        observation_state = ObservationState.from_json(
            observation_data, motion_statechart=self.motion_statechart
        )
        self.motion_statechart.life_cycle_state.data = life_cycle_state.data
        self.motion_statechart.observation_state.data = observation_state.data

    def display_graph(
        self, goal_index: int, message_index: int, update_position_label: bool = True
    ) -> None:
        # Display the graph based on goal and message index
        goal_id = self.goals[goal_index]
        graph = self.graphs_by_goal[goal_id][message_index]
        # Update the position label
        if update_position_label:
            self.update_position_label()

        svg_path = "graph.svg"
        graph.write_svg(svg_path)
        graph.write_pdf("graph.pdf")
        self.svg_widget.load(svg_path)

    def update_position_label(self) -> None:
        goal_count = len(self.goals)
        if goal_count == 0:
            self.position_label.setText("goal 0/0, update 0/0")
            return

        goal_id = self.goals[self.current_goal_index]
        message_count = len(self.graphs_by_goal[goal_id])
        position_text = f"goal {self.current_goal_index + 1}/{goal_count}, update {self.current_message_index + 1}/{message_count}"
        # print(position_text)
        self.position_label.setText(position_text)

    def show_first_image(self) -> None:
        if self.goals:
            self.current_goal_index = 0
            self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_previous_image(self) -> None:
        if self.goals:
            if self.current_message_index > 0:
                self.current_message_index -= 1
            else:
                if self.current_goal_index > 0:
                    self.current_goal_index -= 1
                    self.current_message_index = (
                        len(self.graphs_by_goal[self.goals[self.current_goal_index]])
                        - 1
                    )
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_next_image(self) -> None:
        if self.goals:
            if (
                self.current_message_index
                < len(self.graphs_by_goal[self.goals[self.current_goal_index]]) - 1
            ):
                self.current_message_index += 1
            else:
                if self.current_goal_index < len(self.goals) - 1:
                    self.current_goal_index += 1
                    self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_prev_goal(self) -> None:
        if self.goals and self.current_goal_index > 0:
            self.current_goal_index -= 1
            self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_next_goal(self) -> None:
        if self.goals and self.current_goal_index < len(self.goals) - 1:
            self.current_goal_index += 1
            self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_latest_image(self) -> None:
        if self.goals:
            self.current_goal_index = len(self.goals) - 1
            self.current_message_index = (
                len(self.graphs_by_goal[self.goals[self.current_goal_index]]) - 1
            )
            self.display_graph(self.current_goal_index, self.current_message_index)


def main():
    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, lambda *args: app.quit())

    viewer = DotGraphViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
