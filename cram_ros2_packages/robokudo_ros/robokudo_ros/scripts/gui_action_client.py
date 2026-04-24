#!/usr/bin/env python3
# Needs pyside6 python package
# Might need sudo apt install libxcb-cursor0 as well

"""
ROS 2 Action Caller GUI (PySide6) for robokudo_msgs/Query on /robokudo/query

Features:
- Connect to action server
- Edit Goal as JSON (dict-style)
- Send goal
- Show feedback (JSON)
- Cancel active goal
- Show result (JSON)
- Bottom status bar shows connection state and updates if server dies
"""

import json
import sys
import time

import rclpy
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QLineEdit,
    QMessageBox,
    QGroupBox,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from typing_extensions import Optional, Any, Mapping

from robokudo_msgs.action import Query


def to_builtin(obj: Any) -> Any:
    """
    Recursively convert OrderedDict / mappings to plain dicts and sequences to lists,
    so json.dumps produces clean JSON.
    """
    if isinstance(obj, Mapping):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    return obj


class RosActionThread(QThread):
    # UI-facing signals (Qt thread-safe)
    status = Signal(str)
    connected = Signal(bool, str)  # manual connect result
    conn_state_changed = Signal(bool, str)  # continuous monitoring updates
    goal_accepted = Signal(bool, str)
    feedback = Signal(dict)
    result = Signal(dict)
    cancelled = Signal(bool, str)

    def __init__(self, action_name: str, monitor_period_sec: float = 0.5):
        super().__init__()
        self._action_name = action_name

        self._running = True
        self._node: Optional[Node] = None
        self._client: Optional[ActionClient] = None
        self._goal_handle = None

        self._monitor_period_sec = float(monitor_period_sec)
        self._last_monitor_time = 0.0
        self._last_ready: Optional[bool] = None  # unknown at start

    # ---------- QThread entry ----------
    def run(self):
        try:
            rclpy.init()
        except Exception as e:
            self.connected.emit(False, f"rclpy.init() failed: {e}")
            return

        try:
            self._node = Node("robokudo_query_gui")
            self._client = ActionClient(self._node, Query, self._action_name)
            self.status.emit("ROS node created")

            # Initial short check
            initial_ok = self._client.wait_for_server(timeout_sec=1.0)
            self._last_ready = bool(initial_ok)
            if initial_ok:
                self.connected.emit(True, f"Connected to {self._action_name}")
                self.conn_state_changed.emit(True, self._action_name)
            else:
                self.connected.emit(
                    False, f"Server not available yet: {self._action_name}"
                )
                self.conn_state_changed.emit(False, self._action_name)

            # Spin loop
            while self._running and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.1)
                self._monitor_server()

        except Exception as e:
            self.status.emit(f"ROS thread error: {e}")
        finally:
            try:
                if self._node is not None:
                    self._node.destroy_node()
            except Exception:
                pass
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def stop(self):
        self._running = False

    def _monitor_server(self):
        """Poll ActionClient server readiness and emit if it changes."""
        if not self._client:
            return

        now = time.monotonic()
        if now - self._last_monitor_time < self._monitor_period_sec:
            return
        self._last_monitor_time = now

        try:
            ready = bool(self._client.server_is_ready())
        except Exception:
            # If the underlying client misbehaves, treat as disconnected.
            ready = False

        if self._last_ready is None or ready != self._last_ready:
            self._last_ready = ready
            self.conn_state_changed.emit(ready, self._action_name)

            # If server vanished while we thought we had an active goal handle,
            # cancel UI state by dropping the handle. (Result may never arrive.)
            if not ready and self._goal_handle is not None:
                self._goal_handle = None

    # ---------- Actions called from GUI ----------
    @Slot()
    def connect_server(self):
        if not self._client:
            self.connected.emit(
                False, "ActionClient not ready (ROS thread not initialized yet)"
            )
            return
        try:
            self.status.emit(f"Connecting to {self._action_name} ...")
            ok = self._client.wait_for_server(timeout_sec=2.0)
            self.connected.emit(
                bool(ok),
                f"{'Connected to' if ok else 'Action server not available:'} {self._action_name}",
            )

            # Update monitored state immediately
            self._last_ready = bool(ok)
            self.conn_state_changed.emit(bool(ok), self._action_name)

        except Exception as e:
            self.connected.emit(False, f"Connect failed: {e}")
            self._last_ready = False
            self.conn_state_changed.emit(False, self._action_name)

    @Slot(dict)
    def send_goal_from_dict(self, goal_dict: dict):
        if not self._client:
            self.goal_accepted.emit(False, "ActionClient not ready")
            return
        if not self._client.server_is_ready():
            self.goal_accepted.emit(False, "Action server not ready (disconnected)")
            return
        if self._goal_handle is not None:
            self.goal_accepted.emit(False, "A goal is already active (cancel it first)")
            return

        try:
            goal_msg = Query.Goal()
            set_message_fields(goal_msg, goal_dict)

            self.status.emit("Sending goal...")
            future = self._client.send_goal_async(
                goal_msg, feedback_callback=self._on_feedback
            )
            future.add_done_callback(self._on_goal_response)

        except Exception as e:
            self.goal_accepted.emit(False, f"Failed to build/send goal: {e}")

    @Slot()
    def cancel_active_goal(self):
        if self._goal_handle is None:
            self.cancelled.emit(False, "No active goal to cancel")
            return
        try:
            self.status.emit("Cancelling goal...")
            future = self._goal_handle.cancel_goal_async()
            future.add_done_callback(self._on_cancel_response)
        except Exception as e:
            self.cancelled.emit(False, f"Cancel failed: {e}")

    # ---------- Internal callbacks (run in ROS thread) ----------
    def _on_feedback(self, feedback_msg):
        try:
            fb = dict(to_builtin(message_to_ordereddict(feedback_msg.feedback)))
            self.feedback.emit(fb)
        except Exception as e:
            self.status.emit(f"Feedback parse error: {e}")

    def _on_goal_response(self, future):
        try:
            self._goal_handle = future.result()
            if not self._goal_handle.accepted:
                self._goal_handle = None
                self.goal_accepted.emit(False, "Goal rejected by server")
                return

            self.goal_accepted.emit(True, "Goal accepted")
            self.status.emit("Waiting for result...")

            result_future = self._goal_handle.get_result_async()
            result_future.add_done_callback(self._on_result)

        except Exception as e:
            self._goal_handle = None
            self.goal_accepted.emit(False, f"Goal response error: {e}")

    def _on_result(self, future):
        try:
            res = future.result().result
            res_dict = dict(to_builtin(message_to_ordereddict(res)))
            self.result.emit(res_dict)
            self.status.emit("Result received")
        except Exception as e:
            self.status.emit(f"Result error: {e}")
        finally:
            self._goal_handle = None

    def _on_cancel_response(self, future):
        try:
            response = future.result()
            if response.goals_canceling:
                self.cancelled.emit(True, "Cancel request accepted")
            else:
                self.cancelled.emit(False, "Cancel request rejected")
        except Exception as e:
            self.cancelled.emit(False, f"Cancel response error: {e}")
        finally:
            self._goal_handle = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("robokudo Query Action Caller (ROS 2)")
        self.setMinimumSize(1000, 780)

        # Fixed defaults
        self.action_name = "/robokudo/query"

        self._connected = False
        self._goal_active = False

        self._build_ui()
        self._populate_goal_editor_with_empty_goal()

        # Bottom status bar
        self._set_connection_bar(False, self.action_name)

        # Start ROS thread
        self.ros = RosActionThread(action_name=self.action_name, monitor_period_sec=0.5)
        self._wire_ros_signals()
        self.ros.start()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Connection row
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Action name:"))
        self.action_name_edit = QLineEdit(self.action_name)
        self.action_name_edit.setReadOnly(True)  # fixed for now
        conn_row.addWidget(self.action_name_edit)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        conn_row.addWidget(self.btn_connect)

        layout.addLayout(conn_row)

        self.lbl_status = QLabel("Starting ROS thread...")
        layout.addWidget(self.lbl_status)

        # Goal editor
        goal_group = QGroupBox("Goal (edit as JSON)")
        goal_layout = QVBoxLayout(goal_group)

        self.txt_goal = QTextEdit()
        self.txt_goal.setPlaceholderText(
            "Edit the Goal as JSON here (dict style).\n"
            "Tip: Valid JSON uses double quotes and true/false/null."
        )
        goal_layout.addWidget(self.txt_goal)

        layout.addWidget(goal_group)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_send = QPushButton("Send Goal")
        self.btn_send.setEnabled(False)
        self.btn_send.clicked.connect(self._on_send_clicked)
        btn_row.addWidget(self.btn_send)

        self.btn_cancel = QPushButton("Cancel Goal")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.btn_cancel)

        self.btn_reset_goal = QPushButton("Reset Goal to Empty")
        self.btn_reset_goal.clicked.connect(self._populate_goal_editor_with_empty_goal)
        btn_row.addWidget(self.btn_reset_goal)

        layout.addLayout(btn_row)

        # Feedback / Result
        out_row = QHBoxLayout()

        fb_group = QGroupBox("Feedback")
        fb_layout = QVBoxLayout(fb_group)
        self.txt_feedback = QTextEdit()
        self.txt_feedback.setReadOnly(True)
        fb_layout.addWidget(self.txt_feedback)
        out_row.addWidget(fb_group)

        res_group = QGroupBox("Result")
        res_layout = QVBoxLayout(res_group)
        self.txt_result = QTextEdit()
        self.txt_result.setReadOnly(True)
        res_layout.addWidget(self.txt_result)
        out_row.addWidget(res_group)

        layout.addLayout(out_row)

    def _wire_ros_signals(self):
        self.ros.status.connect(self._set_status)
        self.ros.connected.connect(self._on_connected)
        self.ros.conn_state_changed.connect(self._on_conn_state_changed)
        self.ros.goal_accepted.connect(self._on_goal_accepted)
        self.ros.feedback.connect(self._on_feedback)
        self.ros.result.connect(self._on_result)
        self.ros.cancelled.connect(self._on_cancelled)

    @Slot()
    def _populate_goal_editor_with_empty_goal(self):
        try:
            empty_goal = Query.Goal()
            d = to_builtin(message_to_ordereddict(empty_goal))
            self.txt_goal.setText(json.dumps(d, indent=2))
        except Exception as e:
            QMessageBox.critical(
                self, "Goal Init Error", f"Failed to build empty Goal dict:\n{e}"
            )

    # ---------- Status bar helpers ----------
    def _set_connection_bar(self, connected: bool, action_name: str):
        # QMainWindow has a built-in status bar; create if needed
        sb = self.statusBar()
        if connected:
            sb.showMessage(f"CONNECTED  |  {action_name}")
        else:
            sb.showMessage(f"DISCONNECTED  |  {action_name}")

    # ---------- GUI -> ROS thread ----------
    def _on_connect_clicked(self):
        self.btn_connect.setEnabled(False)
        self._set_status(f"Connecting to {self.action_name} ...")
        self.ros.connect_server()

    def _on_send_clicked(self):
        raw = self.txt_goal.toPlainText().strip()
        if not raw:
            QMessageBox.warning(self, "Goal Error", "Goal text is empty.")
            return

        try:
            goal_dict = json.loads(raw)
            if not isinstance(goal_dict, dict):
                raise ValueError("Top-level JSON must be an object/dict.")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Goal Parse Error",
                "Failed to parse Goal JSON.\n\n"
                "Tip: Use valid JSON (double quotes, true/false/null).\n\n"
                f"Error: {e}",
            )
            return

        self.txt_feedback.clear()
        self.txt_result.clear()
        self.btn_send.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._goal_active = True
        self._set_status("Sending goal...")

        self.ros.send_goal_from_dict(goal_dict)

    def _on_cancel_clicked(self):
        self.btn_cancel.setEnabled(False)
        self._set_status("Cancelling goal...")
        self.ros.cancel_active_goal()

    # ---------- ROS thread -> GUI ----------
    @Slot(str)
    def _set_status(self, msg: str):
        self.lbl_status.setText(msg)

    @Slot(bool, str)
    def _on_connected(self, ok: bool, msg: str):
        # This is the response to pressing "Connect"
        self._connected = ok
        self._set_status(msg)
        self.btn_connect.setEnabled(True)

        # Send enabled only if connected and no active goal
        self.btn_send.setEnabled(ok and not self._goal_active)
        self.btn_cancel.setEnabled(self._goal_active)

        self._set_connection_bar(ok, self.action_name)

    @Slot(bool, str)
    def _on_conn_state_changed(self, ok: bool, action_name: str):
        # This is the continuous monitoring signal (e.g., server dies)
        self._connected = ok
        self._set_connection_bar(ok, action_name)

        # If we lose connection while a goal is "active", disable cancel and re-enable send only when reconnected
        if not ok:
            self.btn_send.setEnabled(False)
            self.btn_cancel.setEnabled(False)
            if self._goal_active:
                self._goal_active = False
                self._set_status(
                    "Disconnected from action server (goal may have been lost)"
                )
        else:
            self.btn_send.setEnabled(not self._goal_active)
            self.btn_cancel.setEnabled(self._goal_active)

    @Slot(bool, str)
    def _on_goal_accepted(self, ok: bool, msg: str):
        self._set_status(msg)
        if not ok:
            self._goal_active = False
            self.btn_send.setEnabled(self._connected)
            self.btn_cancel.setEnabled(False)
            QMessageBox.warning(self, "Goal", msg)

    @Slot(dict)
    def _on_feedback(self, fb: dict):
        self.txt_feedback.setText(json.dumps(fb, indent=2))

    @Slot(dict)
    def _on_result(self, res: dict):
        self.txt_result.setText(json.dumps(res, indent=2))
        self._goal_active = False
        self.btn_cancel.setEnabled(False)
        self.btn_send.setEnabled(self._connected)
        self._set_status("Result received")

    @Slot(bool, str)
    def _on_cancelled(self, ok: bool, msg: str):
        self._set_status(msg)
        self._goal_active = False
        self.btn_cancel.setEnabled(False)
        self.btn_send.setEnabled(self._connected)
        if not ok:
            QMessageBox.warning(self, "Cancel", msg)

    def closeEvent(self, event):
        try:
            self.ros.stop()
            self.ros.quit()
            self.ros.wait(2000)
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
