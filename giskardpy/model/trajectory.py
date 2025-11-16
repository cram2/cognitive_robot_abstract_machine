from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from threading import Lock
from typing import Dict, Tuple, Optional, List

import numpy as np
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from line_profiler import profile
from sortedcontainers import SortedDict

from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.utils.utils import cm_to_inch
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_state import WorldState

plot_lock = Lock()


@dataclass
class Trajectory:
    _points: List[WorldState] = field(default_factory=list)

    def clear(self):
        self._points = []

    def __getitem__(self, item: int) -> WorldState:
        return self._points[item]

    def append(self, point: WorldState):
        self._points.append(deepcopy(point))

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return self._points.__iter__()

    def to_dict(
        self,
        world: World,
        normalize_position: Optional[bool] = None,
        filter_0_vel: bool = True,
        sort: bool = True,
    ) -> Dict[Derivatives, Dict[PrefixedName, np.ndarray]]:
        data = defaultdict(lambda: defaultdict(list))
        for time, joint_states in enumerate(self._points):
            for free_variable, joint_state in joint_states.items():
                for derivative, state in enumerate(joint_state):
                    data[derivative][free_variable].append(state)
        for derivative, d_data in data.items():
            for free_variable, trajectory in d_data.items():
                d_data[free_variable] = np.array(trajectory, dtype=float)
                if (
                    free_variable in self.world.degrees_of_freedom
                    and not self.world.get_degree_of_freedom_by_name(
                        free_variable
                    ).has_position_limits()
                ):
                    if normalize_position is None:
                        normalize_position = True
                if normalize_position and derivative == Derivatives.position:
                    d_data[free_variable] -= (
                        d_data[free_variable].max() + d_data[free_variable].min()
                    ) / 2.0
        if filter_0_vel:
            for free_variable, trajectory in list(data[Derivatives.velocity].items()):
                if abs(trajectory.max() - trajectory.min()) < 1e-5:
                    for derivative, d_data in list(data.items()):
                        del d_data[free_variable]
        if sort:
            for derivative, d_data in data.items():
                data[derivative] = SortedDict(sorted(d_data.items()))
        # else:
        #     data = d_data
        return data

    @profile
    def plot_trajectory(
        self,
        path_to_data_folder: str,
        sample_period: float,
        unit: str = "rad or m",
        cm_per_second: float = 2.5,
        normalize_position: bool = False,
        tick_stride: float = 0.5,
        file_name: str = "trajectory.pdf",
        history: int = 5,
        height_per_derivative: float = 6,
        print_last_tick: bool = False,
        legend: bool = True,
        hspace: float = 1,
        y_limits: bool = None,
        sort: bool = True,
        color_map: Optional[Dict[str, Tuple[str, str]]] = None,
        filter_0_vel: bool = True,
        plot0_lines: bool = True,
    ):
        """
        :type tj: Trajectory
        :param controlled_joints: only joints in this list will be added to the plot
        :type controlled_joints: list
        :param velocity_threshold: only joints that exceed this velocity threshold will be added to the plot. Use a negative number if you want to include every joint
        :param cm_per_second: determines how much the x axis is scaled with the length(time) of the trajectory
        :param normalize_position: centers the joint positions around 0 on the y axis
        :param tick_stride: the distance between ticks in the plot. if tick_stride <= 0 pyplot determines the ticks automatically
        """
        # Set matplotlib backend before importing to avoid GUI backend issues
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend that doesn't require GUI

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        cm_per_second = cm_to_inch(cm_per_second)
        height_per_derivative = cm_to_inch(height_per_derivative)
        hspace = cm_to_inch(hspace)
        max_derivative = (
            GiskardBlackboard().executor.qp_controller.config.max_derivative
        )
        with plot_lock:

            def ceil(val, base=0.0, stride=1.0):
                base = base % stride
                return np.ceil((float)(val - base) / stride) * stride + base

            def floor(val, base=0.0, stride=1.0):
                base = base % stride
                return np.floor((float)(val - base) / stride) * stride + base

            if len(self._points) <= 0:
                return
            colors = list(mcolors.TABLEAU_COLORS.keys())
            colors.append("k")
            if color_map is None:
                line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
                graph_styles = list(product(line_styles, colors))
                color_map: Dict[str, Tuple[str, str]] = defaultdict(
                    lambda: graph_styles[len(color_map) + 1]
                )
            data = self.to_dict(
                normalize_position, filter_0_vel=filter_0_vel, sort=sort
            )
            times = np.arange(len(self)) * sample_period

            f, axs = plt.subplots(
                (max_derivative + 1), sharex=True, gridspec_kw={"hspace": hspace}
            )
            f.set_size_inches(
                w=(times[-1] - times[0]) * cm_per_second,
                h=(max_derivative + 1) * height_per_derivative,
            )

            plt.xlim(times[0], times[-1])

            if tick_stride > 0:
                first = ceil(times[0], stride=tick_stride)
                last = floor(times[-1], stride=tick_stride)
                ticks = np.arange(first, last, tick_stride)
                ticks = np.insert(ticks, 0, times[0])
                ticks = np.append(ticks, last)
                if print_last_tick:
                    ticks = np.append(ticks, times[-1])
                for derivative in Derivatives.range(
                    start=Derivatives.position, stop=max_derivative
                ):
                    axs[derivative].set_title(str(derivative.name))
                    axs[derivative].xaxis.set_ticks(ticks)
                    if y_limits is not None:
                        axs[derivative].set_ylim(y_limits)
            else:
                for derivative in Derivatives.range(
                    start=Derivatives.position, stop=max_derivative
                ):
                    axs[derivative].set_title(str(derivative))
                    if y_limits is not None:
                        axs[derivative].set_ylim(y_limits)
            for derivative, d_data in data.items():
                if derivative > max_derivative:
                    continue
                for free_variable, f_data in d_data.items():
                    try:
                        style, color = color_map[str(free_variable)]
                        if plot0_lines or not np.allclose(f_data, 0):
                            if style == "none":
                                continue
                            if not style.startswith("!"):
                                axs[derivative].plot(
                                    times,
                                    f_data,
                                    color=color,
                                    linestyle=style,
                                    label=free_variable,
                                )
                            else:
                                if "above" in style:
                                    y2 = np.array(list(data[derivative].values())).max()
                                else:
                                    y2 = np.array(list(data[derivative].values())).min()
                                # axs[derivative].fill_between(times, f_data, y2=y2, color=color, alpha=0.5, label='Shaded Area')
                                axs[derivative].fill_between(
                                    times,
                                    f_data,
                                    y2=y2,
                                    color="none",
                                    hatch="//",
                                    edgecolor=color,
                                    alpha=1,
                                    label=free_variable,
                                )
                    except KeyError:
                        get_middleware().logwarn(
                            f"Not enough colors to plot all joints, skipping {free_variable}."
                        )
                    except Exception as e:
                        pass
                axs[derivative].grid()

            if legend:
                axs[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")

            axs[-1].set_xlabel("time [s]")
            axs[Derivatives.position].set_ylabel(unit)
            axs[Derivatives.velocity].set_ylabel(unit + r"$/s$")
            if max_derivative >= Derivatives.acceleration:
                axs[Derivatives.acceleration].set_ylabel(unit + r"$/s^2$")
            if max_derivative >= Derivatives.jerk:
                axs[Derivatives.jerk].set_ylabel(unit + r"$/s^3$")

            file_name = path_to_data_folder + file_name
            last_file_name = file_name.replace(".pdf", f"{history}.pdf")

            if os.path.isfile(file_name):
                if os.path.isfile(last_file_name):
                    os.remove(last_file_name)
                for i in np.arange(history, 0, -1):
                    if i == 1:
                        previous_file_name = file_name
                    else:
                        previous_file_name = file_name.replace(".pdf", f"{i - 1}.pdf")
                    current_file_name = file_name.replace(".pdf", f"{i}.pdf")
                    try:
                        os.rename(previous_file_name, current_file_name)
                    except FileNotFoundError:
                        pass
            plt.savefig(file_name, bbox_inches="tight")
            plt.close(f)  # Close the figure to free memory
            get_middleware().loginfo(f"saved {file_name}")
