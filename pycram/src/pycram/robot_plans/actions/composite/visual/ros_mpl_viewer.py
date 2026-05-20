# #!/usr/bin/env python3
# from __future__ import annotations
#
# import argparse
# import threading
# from dataclasses import dataclass, field
# from typing import Dict, Tuple, List
#
# import rclpy
# from rclpy.node import Node
# from visualization_msgs.msg import Marker, MarkerArray
#
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
#
#
# @dataclass
# class MarkerStore:
#     markers: Dict[Tuple[str, int], Marker] = field(default_factory=dict)
#
#     def apply_marker_array(self, msg: MarkerArray):
#         for m in msg.markers:
#             key = (str(m.ns), int(m.id))
#             if m.action == Marker.DELETEALL:
#                 self.markers.clear()
#                 continue
#             if m.action == Marker.DELETE:
#                 self.markers.pop(key, None)
#                 continue
#             self.markers[key] = m
#
#
# class RosMplViewer(Node):
#     def __init__(self, marker_topic: str, seq_topic: str | None = None):
#         super().__init__("ros_mpl_viewer")
#         self._lock = threading.Lock()
#         self._marker_store = MarkerStore()
#         self._seq_store = MarkerStore()
#
#         self.create_subscription(MarkerArray, marker_topic, self._on_marker_array, 10)
#         if seq_topic:
#             self.create_subscription(MarkerArray, seq_topic, self._on_seq_array, 10)
#
#     def _on_marker_array(self, msg: MarkerArray):
#         with self._lock:
#             self._marker_store.apply_marker_array(msg)
#
#     def _on_seq_array(self, msg: MarkerArray):
#         with self._lock:
#             self._seq_store.apply_marker_array(msg)
#
#     def snapshot(self) -> Tuple[List[Marker], List[Marker]]:
#         with self._lock:
#             return (
#                 list(self._marker_store.markers.values()),
#                 list(self._seq_store.markers.values()),
#             )
#
#
# def _color_from_marker(m: Marker):
#     if m.color.a == 0.0:
#         return (1.0, 1.0, 1.0, 1.0)
#     return (m.color.r, m.color.g, m.color.b, m.color.a)
#
#
# def _draw_marker(ax, m: Marker):
#     c = _color_from_marker(m)
#
#     if m.type == Marker.LINE_STRIP:
#         if len(m.points) < 2:
#             return
#         xs = [p.x for p in m.points]
#         ys = [p.y for p in m.points]
#         zs = [p.z for p in m.points]
#         ax.add_line(Line3D(xs, ys, zs, linewidth=max(m.scale.x, 0.001), color=c))
#         return
#
#     if m.type == Marker.LINE_LIST:
#         if len(m.points) < 2:
#             return
#         xs = [p.x for p in m.points]
#         ys = [p.y for p in m.points]
#         zs = [p.z for p in m.points]
#         ax.plot(xs, ys, zs, linewidth=max(m.scale.x, 0.001), color=c)
#         return
#
#     if m.type == Marker.POINTS:
#         if len(m.points) == 0:
#             return
#         xs = [p.x for p in m.points]
#         ys = [p.y for p in m.points]
#         zs = [p.z for p in m.points]
#         size = max(m.scale.x, 0.001) * 200.0
#         ax.scatter(xs, ys, zs, s=size, c=[c])
#         return
#
#     if m.type == Marker.TRIANGLE_LIST:
#         if len(m.points) < 3:
#             return
#         tris = []
#         for i in range(0, len(m.points) - 2, 3):
#             p0 = m.points[i]
#             p1 = m.points[i + 1]
#             p2 = m.points[i + 2]
#             tris.append([(p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z)])
#         if tris:
#             poly = Poly3DCollection(tris, alpha=c[3])
#             poly.set_facecolor(c[:3])
#             poly.set_edgecolor(c[:3])
#             ax.add_collection3d(poly)
#         return
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Matplotlib 3D viewer for ROS MarkerArray topics"
#     )
#     parser.add_argument("--marker-topic", default="/semworld/viz_marker")
#     parser.add_argument("--seq-topic", default="/temporary_pose_seq")
#     parser.add_argument("--interval-ms", type=int, default=100)
#     args = parser.parse_args()
#
#     rclpy.init()
#     node = RosMplViewer(marker_topic=args.marker_topic, seq_topic=args.seq_topic)
#
#     spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     spin_thread.start()
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#
#     def _update(_frame):
#         ax.cla()
#         world_markers, seq_markers = node.snapshot()
#
#         for m in world_markers:
#             _draw_marker(ax, m)
#         for m in seq_markers:
#             _draw_marker(ax, m)
#
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#
#     ani = FuncAnimation(fig, _update, interval=args.interval_ms)
#     plt.show()
#
#     node.destroy_node()
#     rclpy.shutdown()
#
#
# if __name__ == "__main__":
#     main()
