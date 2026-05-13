from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from os.path import dirname, abspath
from typing import List, Optional, Union
from collections import defaultdict

import pandas as pd
import plotly.express as px

from segmind.datastructures.events import DetectionEvent
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


@dataclass
class EventPlotter:
    """
    A class responsible for plotting events from a timeline.
    """

    def plot(self, events: List[DetectionEvent], show: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot the given events in a timeline.

        :param events: The list of events to plot.
        :param show: Whether to show the plot.
        :param save_path: The path to save the HTML plot.
        """
        if not events:
            logger.debug("No events to plot.")
            return

        data_dict = self._prepare_data(events)
        df = self._create_dataframe(data_dict)
        fig = self._create_figure(df)

        if show:
            fig.show()
        if save_path:
            self._save_plot(fig, save_path)

    def _prepare_data(self, events: List[DetectionEvent]) -> dict:
        """
        Prepare the data for the plot from the list of events.
        """
        data_dict = defaultdict(list)
        for event in events:
            data_dict['event'].append(event.__class__.__name__)

            tracked_objects = getattr(event, 'tracked_objects', [])
            object_name = ", ".join([str(obj.name) for obj in tracked_objects]) if tracked_objects else "None"
            data_dict['object'].append(object_name)

            with_object = getattr(event, 'with_object', None)
            data_dict['with_object'].append(str(with_object.name) if with_object else None)

        return data_dict


    def _create_dataframe(self, data_dict: dict) -> pd.DataFrame:
        """
        Create a pandas DataFrame and normalize timestamps.
        """
        min_start = min(data_dict['start'])
        data_dict['start'] = [x - min_start for x in data_dict['start']]
        data_dict['end'] = [x - min_start for x in data_dict['end']]
        return pd.DataFrame(data_dict)

    def _create_figure(self, df: pd.DataFrame) -> px.timeline:
        """
        Create the plotly timeline figure.
        """
        fig = px.timeline(
            df,
            x_start=pd.to_datetime(df['start'], unit='s'),
            x_end=pd.to_datetime(df['end'], unit='s'),
            y='event',
            color='event',
            hover_data={'object': True, 'with_object': True},
            title="Events Timeline"
        )

        fig.update_xaxes(tickformat='%S')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_layout(
            font_family="Courier New",
            font_color="black",
            font_size=20,
            title_font_family="Times New Roman",
            title_font_color="black",
            title_font_size=30,
            legend_title_font_color="black",
            legend_title_font_size=24,
        )
        return fig

    def _save_plot(self, fig: px.timeline, save_path: str) -> None:
        """
        Save the plot to an HTML file.
        """
        if not os.path.exists(dirname(save_path)):
            os.makedirs(dirname(save_path))
        if not save_path.endswith('.html'):
            save_path += '.html'
        file_path = abspath(save_path)
        fig.write_html(file_path)
        logger.debug(f"Plot saved to {file_path}")