"""
Plotting utilities used by the Streamlit visual sandbox.

This module is the central visualization layer for analysis notebooks and app views in this repository. It provides:

1. **Low-level styling primitives**:
   Shared colors, typography, and helper functions for consistent visual output.

2. **Generic distribution plotting components**:
   Reusable Plotly wrappers (`Visual`, `DistributionPlot`) for scatter-based distribution comparisons with annotation support.

3. **Ground-Duels analysis suites**:
   Higher-level APIs that translate football analysis dataframes into ready-to-render distribution plots for:
    - single CB quality,
    - CB pair quality-fit,
    - anchor-to-companion fit (directional).

4. **Aerial-Duels analysis suites**:
   Higher-level APIs that translate football analysis dataframes into ready-to-render distribution plots for:
    - single CB quality,
    - CB pair quality-fit,
    - anchor-to-companion fit (directional).

5. **Ball-Passing analysis suites**:
   Higher-level APIs that translate football analysis dataframes into ready-to-render distribution plots for:
    - single CB quality,
    - CB pair quality-fit,
    - anchor-to-companion fit (directional).

6. **Global Quality analysis & Radar plots**:
   Higher-level APIs that translate football analysis dataframes into ready-to-render radar distribution plots for:
    - single CB quality,
    - CB pair quality-fit.
    - anchor-to-companion fit (directional).

7. **Personality plotting component**:
   Specialized distribution view for personality-based metrics.

Design goals:
- **Consistency**: identical visual semantics across related analyses.
- **Traceability**: hover payloads and annotations expose interpretable values (raw values, z-scores, rank formatting) for technical and non-technical users.
- **Robustness**: schema validation, deterministic entity resolution, and explicit error messages for ambiguous or invalid user inputs.
- **Extensibility**: resolver utilities and method-level overrides allow new metrics and workflows to be integrated with minimal duplication.
"""


from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import re
import unicodedata
import itertools


from utils.sentences import format_metric
from classes.data_point import Player, Country, Person
from classes.data_source import PlayerStats, CountryStats, PersonStat
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert a hex color string (e.g. `#aabbcc`) to an RGB tuple.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity=1):
    """
    Build a CSS rgba color string from an RGB tuple and opacity.
    """
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color, text, alpha=1.0):
    """
    Wrap text in an HTML span using the given hex color and alpha.
    """
    # color: hexadecimal
    # alpha: transparency value between 0 and 1 (default is 1.0, fully opaque)
    s = (
        "<span style='color:rgba("
        + str(int(color[1:3], 16))
        + ","
        + str(int(color[3:5], 16))
        + ","
        + str(int(color[5:], 16))
        + ","
        + str(alpha)
        + ")'>"
        + str(text)
        + "</span>"
    )
    return s


class Visual:
    """
    Base visual wrapper that applies shared styling to Plotly figures.
    """

    # Can't use streamlit options due to report generation
    dark_green = hex_to_rgb("#002c1c")  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    plot_grid_green = hex_to_rgb("#00663C")
    bright_green = hex_to_rgb("#00A938")  # hex_to_rgb(st.get_option("theme.primaryColor"))
    purple = hex_to_rgb("#800080")
    magenta = hex_to_rgb("#ff00ff")
    pink = hex_to_rgb("#ff69b4")
    bright_orange = hex_to_rgb("#ff4b00")
    brown = hex_to_rgb("#8b4513")
    bright_yellow = hex_to_rgb("#ffcc00")
    gold = hex_to_rgb("#FFD700")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")  # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    silver = hex_to_rgb("#C0C0C0")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    emerald = hex_to_rgb("#50c878")
    ruby = hex_to_rgb("#e0115f")
    sapphire = hex_to_rgb("#0f52ba")
    table_green = hex_to_rgb("#009940")
    table_red = hex_to_rgb("#FF4B00")
    aqua = hex_to_rgb("#00FFD5")
    lime = hex_to_rgb("#C8FF00")

    def __init__(self, pdf=False, plot_type="scout"):
        """
        Initialize the base figure and shared style configuration.

        Args:
            pdf: If true, scale fonts up for PDF-oriented rendering.
            plot_type: Visual mode controlling annotation formatting.
        """
        self.pdf = pdf
        if pdf:
            self.font_size_multiplier = 1.4
        else:
            self.font_size_multiplier = 1.0
        self.fig = go.Figure()
        self._setup_styles()
        self.plot_type = plot_type

        if plot_type == "scout":
            self.annotation_text = (
                "<span style=''>{metric_name}: {data:.2f} </span>" #per 90?
            )
        else:
            # self.annotation_text = "<span style=''>{metric_name}: {data:.0f}/66</span>"  # TODO: this text will not automatically update!
            self.annotation_text = "<span style=''>{metric_name}: {data:.2f}</span>"

    def show(self):
        """
        Render the figure in Streamlit.
        In PDF mode, this could be adapted to save the figure instead.
        """
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def _setup_styles(self):
        """
        Apply common layout, legend, and axis styling.
        Margins and font sizes are scaled based on the `pdf` flag to ensure readability in different contexts.
        """
        side_margin = 60
        top_margin = 75
        pad = 16
        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad),
            paper_bgcolor=rgb_to_color(self.dark_green),
            plot_bgcolor=rgb_to_color(self.dark_green),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 11 * self.font_size_multiplier,
                },
                itemclick=False,
                itemdoubleclick=False,
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="bottom",
                valign="middle",  # Align the text to the middle of the legend
            ),
            xaxis=dict(
                tickfont={
                    "color": rgb_to_color(self.white, 0.5),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            ),
        )

    def add_title(self, title, subtitle):
        """
        Add the main chart title and subtitle.
        """
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(self.white),
                    "size": 12 * self.font_size_multiplier,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.93,
                "yanchor": "top",
            },
        )

    def add_low_center_annotation(self, text):
        """
        Add a low, centered annotation below the plotting area.
        Useful for data source notes or methodological details - content should be concise to avoid cluttering the visual.
        """
        self.fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            text=text,
            showarrow=False,
            font={
                "color": rgb_to_color(self.white, 0.5),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
            },
        )

    def show(self):
        """
        Render the figure in Streamlit.
        In PDF mode, this could be adapted to save the figure instead.
        """
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def close(self):
        """
        No-op placeholder for API symmetry with other visual components.
        """
        pass


class DistributionPlot(Visual):
    """
    Distribution chart for player/country metrics on a shared x-axis.
    Each metric is plotted on a separate horizontal row, with a vertical reference line at x=0 indicating the average performance of the comparison group.
    The background layer shows the distribution of all comparison entities as faint dots, while highlighted players/countries are overlaid with distinct markers and annotations.
    """
    def __init__(self, columns, labels = None, *args, quality_metric_labels = None, quality_metric_value_columns = None, **kwargs):
        """
        Initialize metric columns, marker cycles, and axis labels.

        Args:
            columns: Base metric column names (without suffixes).
            labels: Optional labels for the x-axis ticks.
            *args: Forwarded positional args for `Visual`. 
            quality_metric_labels: Optional metric label mapping/list for annotations.
            quality_metric_value_columns: Optional mapping/list to override values shown in metric annotations.
            **kwargs: Forwarded keyword args for `Visual`.
        """
        self.empty = True
        self.columns = columns
        self.quality_metric_labels = self._normalize_metric_labels(quality_metric_labels)
        self.quality_metric_value_columns = self._normalize_metric_labels(quality_metric_value_columns, value_name = "quality_metric_value_columns")
        
        # Use unique (color, shape) combinations to avoid duplicate marker styles
        # across highlighted entities in the same figure.
        marker_colors = [
            Visual.bright_orange,
            Visual.magenta,
            Visual.bright_yellow,
            Visual.bright_blue,
            Visual.pink,
            Visual.gold,
            Visual.silver,
            Visual.ruby,
            Visual.aqua,
            Visual.emerald,
            Visual.lime,
            Visual.white,
        ]
        marker_shapes = [
            "diamond",
            "circle",
            "hexagon",
            "triangle-left",
            "triangle-right",
            "triangle-down",
            "star",
            "pentagon",
            "cross",
            "x",
            "triangle-up",
            "square",
        ]
        # First assign one-to-one color/shape styles so early highlighted entities
        # always have both unique colors and unique shapes.
        primary_count = min(len(marker_colors), len(marker_shapes))
        step = max(primary_count - 1, 1)
        self._marker_styles = [
            (marker_colors[i], marker_shapes[(i * step) % primary_count])
            for i in range(primary_count)
        ]

        # Then append remaining unique (color, shape) combinations for overflow cases.
        used_styles = set(self._marker_styles)
        for color, shape in itertools.product(marker_colors, marker_shapes):
            if (color, shape) not in used_styles:
                self._marker_styles.append((color, shape))
        self._marker_style_index = 0
        # State used for dynamic multi-entity annotations rendered from add_data_point.
        self._multi_annotation_items = []
        self._multi_annotation_indices = []
        self._legend_group_index = 0
        super().__init__(*args, **kwargs)
        # Allow users to click legend items to hide/show highlighted profiles.
        self.fig.update_layout(
            legend=dict(
                itemclick="toggle",
                groupclick="togglegroup",
            )
        )
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _normalize_metric_labels(self, quality_metric_labels, value_name = "quality_metric_labels"):
        """
        Normalize label input into a dictionary keyed by `self.columns`.

        Accepts `None`, dict, list, or tuple. List/tuple inputs are zipped
        against `self.columns` and must have matching length.
        """
        if quality_metric_labels is None:
            return {}
        if isinstance(quality_metric_labels, dict):
            return quality_metric_labels
        if isinstance(quality_metric_labels, (list, tuple)):
            if len(quality_metric_labels) != len(self.columns):
                raise ValueError(
                    f"`{value_name}` MUST have the same length as columns when passed as a list/tuple"
                )
            # Convert ordered labels into explicit column -> label mapping for direct lookup.
            return dict(zip(self.columns, quality_metric_labels))
        raise TypeError(f"`{value_name}` MUST be a dict, list, tuple, or None")

    def _resolve_annotation_value(self, ser_plot, col):
        """
        Resolve which value to show in metric annotations.

        Priority:
        1) Explicit mapping in `quality_metric_value_columns`
        2) Raw counterpart for `z_`-prefixed metrics when available
        3) The currently plotted column value
        """
        # Explicit mapping has highest priority.
        mapped_col = self.quality_metric_value_columns.get(col)
        if mapped_col is not None and mapped_col in ser_plot.index:
            return ser_plot[mapped_col]

        # Convenience fallback: if plotting `z_xxx` and raw `xxx` exists, show raw value.
        if col.startswith("z_"):
            raw_col = col[2:]
            if raw_col in ser_plot.index:
                return ser_plot[raw_col]

        # Default behavior (backward compatible).
        return ser_plot[col]

    def _is_rank_metric(self, col):
        """
        Detect rank-like metrics based on plotted column name and mapped raw column.
        """
        col_token = str(col).lower()
        mapped_col = self.quality_metric_value_columns.get(col)
        mapped_token = str(mapped_col).lower() if mapped_col is not None else ""
        return ("rank" in col_token) or ("rank" in mapped_token)

    def _format_metric_annotation_value(self, col, value, decimals=2):
        """
        Format annotation value with semantic-aware rendering.

        Rank-like metrics are displayed as integer positions (e.g., `# 1`), while all other numeric metrics preserve decimal precision.
        """
        if pd.isna(value):
            return "N/A"

        if self._is_rank_metric(col):
            try:
                return f"  # {int(round(float(value)))}"
            except (TypeError, ValueError):
                return f"  # {value}"

        return self._format_annotation_value(value, decimals=decimals)

    def _setup_axes(self, labels=["←   Worse", "Average", "Better   →"]):
        """
        Set axis range, ticks, grid style, and center reference line.
        Labels default to directional indicators with padding for visual balance.
        """
        self.fig.update_xaxes(
            range=[-4.5, 4.5],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

        # Add a vertical line at x=0
        self.fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=len(self.columns),
            line=dict(color="gray", width=1, dash="dot"),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        """
        Add background comparison points for each configured metric.

        Args:
            df_plot: DataFrame containing metric columns and hover columns.
            plots: Suffix appended to each metric for x-values (e.g. `"_Z"`).
            names: Hover label values for each row/entity.
            legend: Legacy parameter kept for compatibility.
            hover: Suffix appended for hover-value columns.
            hover_string: Plotly hover template body.
        """

        # One trace per metric so each distribution sits on its own y-row.
        for i, col in enumerate(self.columns):
            self.fig.add_trace(
                go.Scatter(
                    # Suffix-based column addressing keeps plotting logic generic.
                    x=df_plot[col + plots].tolist(), 
                    y=list(np.ones(len(df_plot[col + plots])) * i),
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.dark_green, opacity=0.2),
                        "size": 10,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(self.bright_green),
                    },
                    hovertemplate="%{text}<br>" + hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover].tolist(),
                    showlegend=False,
                )
            )
            

    def add_data_point(self, ser_plot, plots, name, hover="", hover_string="", text=None, add_single_annotations=True, add_multi_annotations=False, add_annotations=None):
        """
        Add one highlighted entity (player/country) across all metrics.

        Args:
            ser_plot: Series-like metric source for a single entity.
            plots: Suffix appended to each metric for x-values (e.g. `"_Z"`).
            name: Legend label for this entity.
            hover: Suffix appended for hover-value columns.
            hover_string: Plotly hover template body.
            text: Optional hover text override.
            add_single_annotations: If true, add one annotation per metric for this single entity.
            add_multi_annotations: If true, include this entity in a combined multi-entity
                annotation line per metric, using marker tokens.
            add_annotations: Backward-compatible alias for `add_single_annotations`.
        """
        if add_annotations is not None:
            add_single_annotations = add_annotations

        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        # We add one trace per metric, but keep only a single legend entry.
        legend = True
        legend_group = f"distribution_highlight_{self._legend_group_index}"
        self._legend_group_index += 1
        color, marker = self._next_marker_style()

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = self.quality_metric_labels.get(col, format_metric(col))

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                    legendgroup=legend_group,
                )
            )
            legend = False

            if add_single_annotations:
                # Place metric labels between distribution rows (rather than on top of points)
                # and render them as badges for readability in dense datasets.
                annotation_y = i + 0.5
                if i == len(self.columns) - 1:
                    # Keep top-most label comfortably inside the plotting area.
                    annotation_y = i + 0.4

                self.fig.add_annotation(
                    # Annotation labels stay near the center reference axis (x=0).
                    x=0,
                    y=annotation_y,
                    text=(
                        f"<span style=''>{metric_name}  →  "
                        f"{self._format_metric_annotation_value(col, self._resolve_annotation_value(ser_plot, col), decimals=2)}"
                        f"</span>"
                    ),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font={
                        "color": rgb_to_color(self.white),
                        "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier,
                    },
                )

        if add_multi_annotations:
            marker_tag = self._build_marker_tag(color=color, marker_symbol=marker)
            self._multi_annotation_items.append(
                {
                    "ser_plot": ser_plot,
                    "marker_tag": marker_tag,
                    "name": name,
                }
            )
            self.add_multi_entity_annotations(
                annotation_items=self._multi_annotation_items,
                value_decimals=2,
                separator="   |   ",
                replace_existing=True,
            )

    def _format_annotation_value(self, value, decimals=2):
        """
        Format annotation values consistently while handling missing data.
        """
        if pd.isna(value):
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    def _build_marker_tag(self, color, marker_symbol):
        """
        Build an HTML marker token used inside combined annotation text.
        """
        marker_symbol_map = {
            "diamond": "◆",
            "circle": "●",
            "hexagon": "⬢",
            "triangle-left": "◀",
            "triangle-right": "▶",
            "triangle-down": "▼",
            "star": "★",
            "pentagon": "⬟",
            "cross": "✚",
            "x": "✕",
            "triangle-up": "▲",
            "square": "■"
        }

        glyph = marker_symbol_map.get(marker_symbol, "●")
        
        return f"<span style='color:{rgb_to_color(color)}'>{glyph}</span>"

    def _next_marker_style(self):
        """
        Return the next unique marker style (color + shape) for highlighted points.
        """
        if self._marker_style_index >= len(self._marker_styles):
            raise ValueError(
                "Exceeded available unique marker style combinations in this plot. "
                f"Maximum unique highlighted entities: {len(self._marker_styles)}."
            )
        color, marker = self._marker_styles[self._marker_style_index]
        self._marker_style_index += 1
        return color, marker

    def _clear_multi_entity_annotations(self):
        """
        Remove previously rendered combined multi-entity annotations.
        """
        if not self._multi_annotation_indices:
            return

        existing_annotations = list(self.fig.layout.annotations or [])
        keep_annotations = [
            ann
            for idx, ann in enumerate(existing_annotations)
            if idx not in self._multi_annotation_indices
        ]
        self.fig.layout.annotations = tuple(keep_annotations)
        self._multi_annotation_indices = []

    def add_multi_entity_annotations(
        self,
        annotation_items=None,
        value_decimals=2,
        x=0,
        separator="   |   ",
        replace_existing=True,
    ):
        """
        Add one combined annotation per metric for multiple highlighted entities.

        Args:
            annotation_items: Iterable of dict-like items with:
                - ser_plot: Series-like metric source for one highlighted entity
                - marker_tag: HTML marker token for this entity (preferred)
                - label: Optional fallback label used in the combined annotation text
            value_decimals: Decimal precision for numeric values.
            x: X-position where combined annotations are anchored.
            separator: Delimiter between per-entity value chunks.
            replace_existing: If true, remove prior combined multi-entity annotations before drawing the updated set.
        """
        if annotation_items is None:
            annotation_items = self._multi_annotation_items

        if not annotation_items:
            return

        valid_items = []
        for item in annotation_items:
            if not isinstance(item, dict):
                continue
            ser_plot = item.get("ser_plot")
            if ser_plot is None:
                continue
            marker_tag = item.get("marker_tag")
            label = item.get("label", item.get("name", "Entity"))
            valid_items.append(
                {
                    "marker_tag": marker_tag,
                    "label": label,
                    "ser_plot": ser_plot,
                }
            )

        if not valid_items:
            return

        if replace_existing:
            self._clear_multi_entity_annotations()

        for i, col in enumerate(self.columns):
            metric_name = self.quality_metric_labels.get(col, format_metric(col))
            chunks = []
            for item in valid_items:
                value = self._resolve_annotation_value(item["ser_plot"], col)
                marker_or_label = item["marker_tag"] if item["marker_tag"] else item["label"]
                chunks.append(
                    f"{marker_or_label} = {self._format_metric_annotation_value(col, value, decimals=value_decimals)}"
                )

            annotation_y = i + 0.5
            if i == len(self.columns) - 1:
                annotation_y = i + 0.4

            self.fig.add_annotation(
                x=x,
                y=annotation_y,
                text=f"<span style=''>{metric_name}  →  {separator.join(chunks)}</span>",
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )
            self._multi_annotation_indices.append(len(self.fig.layout.annotations) - 1)


    def add_player(self, player: Union[Player, Country], n_group, metrics):
        """
        Add a single `Player` or `Country` to the distribution chart.

        Args:
            player: Entity whose `ser_metrics` values are plotted.
            n_group: Group size used in rank hover text.
            metrics: Base metric names used by this view.
        """

        # Keep suffix conventions explicit here for readability and future extensions.
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        # Determine the appropriate attributes for player or country
        if isinstance(player, Player):
            ser_plot = player.ser_metrics
            name = player.name
        elif isinstance(player, Country):  # Adjust this based on your class structure
            ser_plot = (
                player.ser_metrics
            )  # Assuming countries have a similar metric structure
            name = player.name
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_data_point(
            ser_plot=ser_plot,
            plots="_Z",
            name=name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    # def add_players(self, players: PlayerStats, metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_group_data(
    #         df_plot=players.df,
    #         plots="_Z",
    #         names=players.df["player_name"],
    #         hover="_Ranks",
    #         hover_string="Rank: %{customdata}/" + str(len(players.df)),
    #         legend=f"Other players  ",  # space at end is important
    #     )

    def add_players(self, players: Union[PlayerStats, CountryStats], metrics):
        """
        Add comparison-group points for player or country datasets.

        Args:
            players: Dataset wrapper (`PlayerStats` or `CountryStats`).
            metrics: Base metric names used by this view.
        """

        # Keep suffix conventions explicit here for readability and future extensions.
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other players  ",  # space at end is important
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other countries  ",  # space at end is important
            )
        else:
            raise TypeError("Invalid player type: expected Player or Country")

    # def add_title_from_player(self, player: Player):
    #     self.player = player

    #     title = f"Evaluation of {player.name}?"
    #     subtitle = f"Based on {player.minutes_played} minutes played"

    #     self.add_title(title, subtitle)

    def add_title_from_player(self, player: Union[Player, Country]):
        """
        Build and apply an entity-aware title and subtitle.
        """
        self.player = player

        title = f"Evaluation of {player.name}?"
        if isinstance(player, Player):
            subtitle = f"Based on {player.minutes_played} minutes played"
        elif isinstance(player, Country):
            subtitle = f"Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_title(title, subtitle)





# --------------------------------------------------------------------------|
# Ground-Duels Plotting Infrastructure                                      |
# --------------------------------------------------------------------------|
# The classes below form a layered architecture:                            |
# - `_Ground_Duels_Distribution_Resolver`: shared data/selection utilities  |
# - `Single_CB_*`, `CB_Pair_*`, `Anchor_CB_Companion_*`: public plot APIs   |
# --------------------------------------------------------------------------|

class _Ground_Duels_Distribution_Resolver:
    """
    Shared helper utilities used by all Ground-Duels distribution plot wrappers.

    This resolver centralizes cross-cutting concerns such as:
    - robust player/pair name normalization and resolution,
    - rank-aware z-score handling,
    - hover payload formatting,
    - average-point construction.

    Centralizing these behaviors keeps public plotting classes small and ensures
    consistent semantics across single-CB, CB-pair, and companion-fit views.
    """

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        Normalize free-text input for resilient matching.

        The normalization pipeline is intentionally strict and deterministic:
        - converts to lowercase,
        - strips accents/diacritics,
        - removes punctuation/special symbols,
        - collapses repeated whitespace.

        Args:
            value: Any user-provided value that should be interpreted as text.

        Returns:
            Normalized tokenizable text; empty string for null-like values.
        """
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _tokenize(cls, value: Any) -> Tuple[str, ...]:
        """
        Split normalized text into lexical tokens.

        Args:
            value: Input value to normalize and tokenize.

        Returns:
            Tuple of tokens. Empty tuple for empty/invalid normalized text.
        """
        normalized = cls._normalize_text(value)
        if not normalized:
            return tuple()
        return tuple(normalized.split(" "))

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
        """
        Validate dataframe schema requirements.

        Args:
            df: Dataframe to validate.
            required_columns: Columns that must exist in `df`.
            context: Human-readable context included in validation errors.

        Raises:
            KeyError: If any required column is missing.
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required column(s) for {context}: " + ", ".join(missing)
            )

    @staticmethod
    def _to_number_or_none(value: Any) -> Optional[float]:
        """
        Attempt safe scalar numeric conversion.

        Args:
            value: Value to convert.

        Returns:
            `float` when conversion is successful and finite; otherwise `None`.
        """
        try:
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        except Exception:
            return None
        if pd.isna(converted):
            return None
        return float(converted)

    @staticmethod
    def _coerce_positive_rank(rank_value: Any, *, param_name: str) -> int:
        """
        Validate and coerce one rank selector to a positive integer.

        Args:
            rank_value: Candidate rank value.
            param_name: Parameter name used in error messages.

        Returns:
            Rank as a positive integer.
        """
        if not isinstance(rank_value, (int, np.integer)) or int(rank_value) <= 0:
            raise ValueError(f"{param_name} must be a positive integer.")
        return int(rank_value)

    @classmethod
    def _coerce_unique_positive_ranks(
        cls,
        rank_values: Sequence[Any],
        *,
        param_name: str,
    ) -> List[int]:
        """
        Validate, coerce, and deduplicate a rank-sequence selector.

        Args:
            rank_values: Sequence of candidate rank values.
            param_name: Parameter name used in error messages.

        Returns:
            Ordered list of unique positive integers.
        """
        if isinstance(rank_values, (str, bytes)):
            raise ValueError(
                f"{param_name} must be a sequence of positive integers, not a string."
            )

        normalized: List[int] = []
        for rank_value in rank_values:
            rank_int = cls._coerce_positive_rank(rank_value, param_name=param_name)
            if rank_int not in normalized:
                normalized.append(rank_int)
        return normalized

    @classmethod
    def _resolve_row_by_rank(
        cls,
        plot_df: pd.DataFrame,
        *,
        rank_column: str,
        rank_value: Any,
        entity_label: str,
        display_column: str,
    ) -> pd.Series:
        """
        Resolve one row by exact raw-ranking value.

        Args:
            plot_df: Candidate dataframe to search.
            rank_column: Raw rank column name.
            rank_value: Requested rank value.
            entity_label: Human-readable entity label for errors.
            display_column: Column used to list candidate names in ambiguity errors.

        Returns:
            The uniquely resolved row for the requested rank.
        """
        requested_rank = cls._coerce_positive_rank(
            rank_value,
            param_name=f"{entity_label} rank",
        )

        if rank_column not in plot_df.columns:
            raise KeyError(
                f"Cannot resolve {entity_label} by rank because column '{rank_column}' "
                "is not available in the plotting dataframe."
            )

        rank_values = pd.to_numeric(plot_df[rank_column], errors="coerce")
        matches = plot_df[rank_values == float(requested_rank)]

        if matches.empty:
            raise ValueError(
                f"No {entity_label} found at rank #{requested_rank}."
            )
        if len(matches) > 1:
            candidates = (
                matches[display_column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f"Multiple {entity_label} entries found at rank #{requested_rank}: "
                f"{', '.join(candidates)}"
            )
        return matches.iloc[0]

    @classmethod
    def _resolve_entity(cls, candidates: pd.DataFrame, *, entity_value: Any = None, entity_id: Any = None, id_col: str, name_col: str, entity_label: str) -> pd.Series:
        """
        Resolve a single entity row from candidate IDs/names.

        Resolution precedence:
        1. Explicit ID match (numeric-safe first, then string fallback),
        2. Exact normalized full-name match,
        3. Unique token/surname subset match.

        Ambiguity is never auto-resolved: a `ValueError` is raised with candidate
        names so callers can disambiguate explicitly.

        Args:
            candidates: Candidate dataframe containing ID and name columns.
            entity_value: Optional name/surname-like selector.
            entity_id: Optional ID selector.
            id_col: Candidate ID column name.
            name_col: Candidate display-name column name.
            entity_label: Label used in error messages (e.g., "CB player").

        Returns:
            The uniquely resolved candidate row.
        """
        candidates = candidates[[id_col, name_col]].dropna(subset=[id_col, name_col]).drop_duplicates()
        if candidates.empty:
            raise ValueError(f"No {entity_label} candidates available to resolve.")

        if entity_id is None and isinstance(entity_value, (int, np.integer)):
            entity_id = int(entity_value)
            entity_value = None

        if entity_id is not None:
            requested_id_numeric = cls._to_number_or_none(entity_id)
            if requested_id_numeric is not None:
                candidate_id_numeric = pd.to_numeric(candidates[id_col], errors="coerce")
                matches = candidates[candidate_id_numeric == requested_id_numeric]
            else:
                matches = candidates[candidates[id_col].astype(str).str.strip() == str(entity_id).strip()]

            if len(matches) == 1:
                return matches.iloc[0]
            if len(matches) > 1:
                options = sorted(matches[name_col].astype(str).unique().tolist())
                raise ValueError(
                    f"Ambiguous {entity_label} ID '{entity_id}'. Candidates: {', '.join(options)}"
                )
            raise ValueError(f"No {entity_label} found for ID '{entity_id}'.")

        if entity_value is None:
            raise ValueError(f"Please provide either {entity_label} name/surname or {entity_label} ID.")

        query = cls._normalize_text(entity_value)
        if not query:
            raise ValueError(f"Invalid empty {entity_label} name input.")

        working = candidates.copy()
        working["_normalized_name"] = working[name_col].map(cls._normalize_text)
        working["_name_tokens"] = working[name_col].map(cls._tokenize)

        exact_matches = working[working["_normalized_name"] == query]
        if len(exact_matches) == 1:
            return exact_matches.iloc[0]
        if len(exact_matches) > 1:
            options = sorted(exact_matches[name_col].astype(str).unique().tolist())
            raise ValueError(
                f"Ambiguous {entity_label} name '{entity_value}'. Candidates: {', '.join(options)}"
            )

        query_tokens = set(cls._tokenize(query))
        token_matches = working[
            working["_name_tokens"].map(lambda tokens: query_tokens.issubset(set(tokens)))
        ]

        if len(token_matches) == 1:
            return token_matches.iloc[0]
        if len(token_matches) > 1:
            options = sorted(token_matches[name_col].astype(str).unique().tolist())
            raise ValueError(
                f"Ambiguous {entity_label} input '{entity_value}'. Candidates: {', '.join(options)}"
            )

        raise ValueError(f"No {entity_label} found for input '{entity_value}'.")

    @staticmethod
    def _z_standardize(values: Union[pd.Series, Sequence[Any]], *, invert: bool = False, scale: float = 1.0) -> pd.Series:
        """
        Compute z-standardized values with optional sign inversion and scaling.

        This helper is used both for standard metric normalization and rank-based
        transformations (where lower raw rank means better, hence `invert=True`).

        Args:
            values: Numeric-like values to standardize.
            invert: If true, multiply standardized values by `-1`.
            scale: Multiplicative factor applied after standardization.

        Returns:
            Z-standardized series. If variance is zero/undefined, returns all NaN.
        """
        series = pd.to_numeric(pd.Series(values), errors="coerce")
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(float("nan"), index=series.index)
        z_values = (series - series.mean()) / std
        if invert:
            z_values = -z_values
        return z_values * scale

    @classmethod
    def _is_rank_metric(cls, metric_name: str, raw_metric_name: Optional[str]) -> bool:
        """
        Identify rank-like metrics by inspecting z/raw metric names.

        Args:
            metric_name: Plotted metric column (typically z-score column).
            raw_metric_name: Raw companion/value column mapped to that metric.

        Returns:
            True when either name contains the token `rank`.
        """
        metric_token = str(metric_name).lower()
        raw_metric_token = str(raw_metric_name).lower() if raw_metric_name is not None else ""
        return ("rank" in metric_token) or ("rank" in raw_metric_token)

    @classmethod
    def _format_hover_line(cls, metric_name: str, raw_metric_name: Optional[str], metric_label: str, raw_value: Any, z_value: Any) -> str:
        """
        Build one semantic-aware hover text line for a metric.

        Behavior:
        - metrics without a raw column mapping show z-score only,
        - rank-like metrics show integer rank format (`# <int>`),
        - all other metrics show `raw + corresponding z-score`.
        """
        if raw_metric_name is None:
            return f"Z-score = {z_value:.2f}" if pd.notna(z_value) else "Z-score = N/A"
        if cls._is_rank_metric(metric_name, raw_metric_name):
            return f"Rank =   # {int(round(raw_value))}" if pd.notna(raw_value) else "Rank =   # N/A"
        raw_text = f"{raw_value:.2f}" if pd.notna(raw_value) else "N/A"
        z_text = f"{z_value:.2f}" if pd.notna(z_value) else "N/A"
        return f"{metric_label} = {raw_text}<br>Respective Z-score = {z_text}"

    @classmethod
    def _attach_hover_payload(cls, plot_df: pd.DataFrame, *, metric_cols: Sequence[str], metric_labels: Dict[str, str], metric_value_columns: Dict[str, Optional[str]], hover_payload_suffix: str = "_hover_payload", fill_missing_rank_z: bool = True, rank_z_scale: float = 2.0) -> pd.DataFrame:
        """
        Attach per-metric hover payload tuples used by `DistributionPlot`.

        For each metric, this method optionally reconstructs missing/all-NaN
        rank z-score columns from the corresponding raw rank column, then creates
        payload tuples of the form:
        `(metric_label, raw_value, z_value, formatted_hover_line)`.

        Args:
            plot_df: Plotting dataframe to enrich in place.
            metric_cols: Metrics expected in the distribution plot.
            metric_labels: Human-readable labels per metric.
            metric_value_columns: Raw-value column mapping per metric.
            hover_payload_suffix: Suffix for generated payload columns.
            fill_missing_rank_z: Whether rank z-score fallback should be applied.
            rank_z_scale: Scaling factor used when reconstructing rank z-scores.

        Returns:
            The same dataframe instance, enriched with hover payload columns.
        """
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))

            can_backfill_rank_z = (
                fill_missing_rank_z
                and raw_metric is not None
                and cls._is_rank_metric(metric, raw_metric)
                and raw_metric in plot_df.columns
            )
            metric_missing = metric not in plot_df.columns
            metric_all_nan = False
            if not metric_missing:
                metric_all_nan = pd.to_numeric(plot_df[metric], errors="coerce").isna().all()

            if metric_missing or (can_backfill_rank_z and metric_all_nan):
                if can_backfill_rank_z:
                    plot_df[metric] = cls._z_standardize(
                        plot_df[raw_metric],
                        invert=True,
                        scale=rank_z_scale,
                    )
                else:
                    plot_df[metric] = float("nan")

            metric_label = metric_labels.get(metric, format_metric(metric))
            default_series = pd.Series(float("nan"), index=plot_df.index)
            raw_values = pd.to_numeric(
                plot_df.get(raw_metric, default_series),
                errors="coerce",
            )
            z_values = pd.to_numeric(
                plot_df.get(metric, default_series),
                errors="coerce",
            )

            hover_lines = [
                cls._format_hover_line(
                    metric,
                    raw_metric,
                    metric_label,
                    raw_value,
                    z_value,
                )
                for raw_value, z_value in zip(raw_values, z_values)
            ]

            plot_df[f"{metric}{hover_payload_suffix}"] = list(
                zip(
                    [metric_label] * len(plot_df),
                    raw_values,
                    z_values,
                    hover_lines,
                )
            )

        return plot_df

    @classmethod
    def _split_pair_string(cls, pair_value: str) -> Tuple[str, str]:
        """
        Parse user-friendly pair strings into two entity selectors.

        Accepted separators:
        - `+`
        - `and`
        - `&`

        Args:
            pair_value: Pair input such as `"A + B"` or `"A and B"`.

        Returns:
            Tuple of two stripped entity strings.
        """
        if not isinstance(pair_value, str):
            raise ValueError("CB_Pair must be a string such as 'A + B', 'A and B', or 'A & B'.")

        parts = re.split(r"\s*(?:\+|&|\band\b)\s*", pair_value.strip(), maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                "Could not parse CB pair input. Use formats like 'A + B', 'A and B', or 'A & B'."
            )
        return parts[0].strip(), parts[1].strip()

    @classmethod
    def _build_average_point(cls, *, plot_df: pd.DataFrame, metric_cols: Sequence[str], metric_labels: Dict[str, str], metric_value_columns: Dict[str, Optional[str]], display_name_col: str, display_name_value: str, hover_payload_suffix: str = "_hover_payload") -> pd.Series:
        """
        Build a synthetic cohort-average point compatible with `DistributionPlot`.

        The returned series contains:
        - average z-scores for each plotted metric,
        - average raw values for mapped raw columns (when available),
        - per-metric hover payload entries matching normal row payload shape,
        - a display-name field to support legend/labels.

        Args:
            plot_df: Source dataframe defining the cohort.
            metric_cols: Metrics used in the plot.
            metric_labels: Human-readable labels per metric.
            metric_value_columns: Raw-value mapping per metric.
            display_name_col: Column name used by the plot as display label.
            display_name_value: Display label for the synthetic average point.
            hover_payload_suffix: Suffix used for hover payload columns.

        Returns:
            Series representing one synthetic average row.
        """
        avg_raw_metrics = pd.Series(
            {
                raw_col: pd.to_numeric(plot_df[raw_col], errors="coerce").mean()
                for raw_col in set(metric_value_columns.values())
                if raw_col is not None and raw_col in plot_df.columns
            }
        )
        avg_z_scores = pd.Series(
            {
                z_col: pd.to_numeric(plot_df[z_col], errors="coerce").mean()
                if z_col in plot_df.columns
                else float("nan")
                for z_col in metric_cols
            }
        )

        average_point = avg_z_scores.copy()
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))
            metric_label = metric_labels.get(metric, format_metric(metric))
            avg_raw_value = avg_raw_metrics.get(raw_metric, float("nan"))
            avg_z_value = average_point.get(metric, float("nan"))

            if raw_metric is not None:
                average_point[raw_metric] = avg_raw_value

            average_point[f"{metric}{hover_payload_suffix}"] = [
                metric_label,
                avg_raw_value,
                avg_z_value,
                cls._format_hover_line(
                    metric,
                    raw_metric,
                    metric_label,
                    avg_raw_value,
                    avg_z_value,
                ),
            ]

        average_point[display_name_col] = display_name_value
        return average_point

    @classmethod
    def _resolve_metric_config(
        cls,
        *,
        default_metrics: Sequence[str],
        default_metric_labels: Mapping[str, str],
        default_metric_value_columns: Mapping[str, Optional[str]],
        configured_metrics: Optional[Sequence[str]] = None,
        configured_metric_labels: Optional[Mapping[str, str]] = None,
        configured_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        override_metrics: Optional[Sequence[str]] = None,
        override_metric_labels: Optional[Mapping[str, str]] = None,
        override_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve the metric configuration for one plotting call.

        Precedence:
        1. Method-level overrides (`override_*`)
        2. Constructor-level settings (`configured_*`)
        3. Class defaults (`default_*`)

        Returns only the mappings required by the selected metrics.
        """
        chosen_metrics = override_metrics
        if chosen_metrics is None:
            chosen_metrics = configured_metrics
        if chosen_metrics is None:
            chosen_metrics = default_metrics

        ordered_metrics: List[str] = []
        for metric in chosen_metrics:
            metric_str = str(metric)
            if metric_str not in ordered_metrics:
                ordered_metrics.append(metric_str)
        if not ordered_metrics:
            raise ValueError("At least one metric must be provided for the distribution plot.")

        resolved_labels = dict(default_metric_labels)
        if configured_metric_labels:
            resolved_labels.update(dict(configured_metric_labels))
        if override_metric_labels:
            resolved_labels.update(dict(override_metric_labels))
        metric_labels = {
            metric: resolved_labels.get(metric, format_metric(metric))
            for metric in ordered_metrics
        }

        resolved_value_columns: Dict[str, Optional[str]] = dict(default_metric_value_columns)
        if configured_metric_value_columns:
            resolved_value_columns.update(dict(configured_metric_value_columns))
        if override_metric_value_columns:
            resolved_value_columns.update(dict(override_metric_value_columns))
        metric_value_columns = {
            metric: resolved_value_columns.get(metric, metric.replace("z_", "", 1))
            for metric in ordered_metrics
        }

        return ordered_metrics, metric_labels, metric_value_columns

    @staticmethod
    def _show_figures_side_by_side(
        figures: Sequence[go.Figure],
        *,
        column_gap: str = "16px",
        min_column_width_px: int = 520,
    ) -> bool:
        """
        Display Plotly figures side-by-side in notebook environments.

        Returns `True` when HTML notebook rendering succeeds. Callers can fall back to sequential `.show()` rendering when `False` is returned.
        """
        if not figures:
            return False

        try:
            from IPython.display import HTML, display  # type: ignore
        except Exception:
            return False

        try:
            blocks = []
            for idx, fig in enumerate(figures):
                include_plotlyjs = "cdn" if idx == 0 else False
                fig_html = pio.to_html(
                    fig,
                    include_plotlyjs=include_plotlyjs,
                    full_html=False,
                )
                blocks.append(
                    (
                        "<div style='flex:1 1 "
                        f"{min_column_width_px}px;min-width:{min_column_width_px}px'>"
                        f"{fig_html}</div>"
                    )
                )

            container_html = (
                "<div style='display:flex;flex-wrap:wrap;"
                f"gap:{column_gap};align-items:flex-start'>{''.join(blocks)}</div>"
            )
            display(HTML(container_html))
            return True
        except Exception:
            return False


class Single_CB_Ground_Duels_Distribution_Plot(_Ground_Duels_Distribution_Resolver):
    """
    Build Ground-Duels distribution plots for:
    1. One selected CB against the full cohort
    2. Multiple selected CBs in comparison mode

    The class ships with sensible defaults for metrics, labels, and raw-value mappings. Users can still override configuration at construction time or per plotting call.
    """

    DEFAULT_METRICS = [
        "z_duels_per90",
        "z_card_discipline",
        "z_discipline",
        "z_interceptions_per90",
        "z_duel_success_rate",
        "z_possession_win_rate",
        "CB_ground_duels_quality_z_score",
        "CB_rank_for_ground_duels_quality_z_score"
    ]


    DEFAULT_METRIC_LABELS = {
        "z_duels_per90": "Ground Duels per-90",
        "z_card_discipline": "Card Discipline",
        "z_discipline": "(Overall) Discipline",
        "z_interceptions_per90": "Interceptions per-90",
        "z_duel_success_rate": "Ground Duel Success Rate",
        "z_possession_win_rate": "Possession Win Rate",
        "CB_ground_duels_quality_z_score": "CB's (Overall) Ground Duels Quality Score",
        "CB_rank_for_ground_duels_quality_z_score": "CB's (Ground Duels Quality) Ranking",
        "CB_rank_for_ground_duels_quality": "CB's (Ground Duels Quality) Ranking"
    }


    DEFAULT_METRIC_VALUE_COLUMNS = {
        "z_duels_per90": "duels_per90",
        "z_card_discipline": "card_discipline",
        "z_discipline": "discipline",
        "z_interceptions_per90": "interceptions_per90",
        "z_duel_success_rate": "duel_success_rate",
        "z_possession_win_rate": "possession_win_rate",
        "CB_ground_duels_quality_z_score": None,
        "CB_rank_for_ground_duels_quality_z_score": "CB_rank_for_ground_duels_quality",
    }


    RANK_Z_SCORE_SCALE = 2.0
    RAW_RANK_COLUMN = "CB_rank_for_ground_duels_quality"


    def __init__(
        self,
        *,
        duel_summary: pd.DataFrame,
        z_scores: pd.DataFrame,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, str]] = None,
        min_num_duels_involved_in_threshold: Any,
        min_minutes_played_threshold: Any,
    ) -> None:
        """
        Args:
            duel_summary: Raw single-CB metric dataframe.
            z_scores: Z-scored single-CB metric dataframe.
            metrics: Optional constructor-level metric list.
            metric_labels: Optional constructor-level metric label map.
            metric_value_columns: Optional constructor-level raw-value column map.
            min_num_duels_involved_in_threshold: Threshold shown in subtitles.
            min_minutes_played_threshold: Threshold shown in subtitles.
        """
        self.duel_summary = duel_summary
        self.z_scores = z_scores
        self._configured_metrics = list(metrics) if metrics is not None else None
        self._configured_metric_labels = (
            dict(metric_labels) if metric_labels is not None else None
        )
        self._configured_metric_value_columns = (
            dict(metric_value_columns) if metric_value_columns is not None else None
        )
        self.min_num_duels_involved_in_threshold = min_num_duels_involved_in_threshold
        self.min_minutes_played_threshold = min_minutes_played_threshold

    def _resolve_current_metric_config(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve call-ready metric configuration with precedence:
        method override > constructor config > class defaults.
        """
        return self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_METRIC_VALUE_COLUMNS,
            configured_metrics=self._configured_metrics,
            configured_metric_labels=self._configured_metric_labels,
            configured_metric_value_columns=self._configured_metric_value_columns,
            override_metrics=metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )

    def _build_plot_df(
        self,
        *,
        metrics: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Build the merged plotting dataframe and attach hover payload columns.
        """
        z_scores_df = self.z_scores.reset_index()
        self._ensure_columns(
            z_scores_df,
            ["player.id", "player.name"],
            "single-CB z-score dataframe",
        )

        raw_metric_columns = [
            col for col in dict.fromkeys(metric_value_columns.values()) if col is not None
        ]
        duel_summary_df = self.duel_summary.reset_index()
        always_include_raw_columns: List[str] = []
        if self.RAW_RANK_COLUMN in duel_summary_df.columns:
            always_include_raw_columns.append(self.RAW_RANK_COLUMN)
        fallback_metric_columns = [
            metric
            for metric in metrics
            if metric not in z_scores_df.columns and metric in duel_summary_df.columns
        ]
        required_duel_summary_columns = list(
            dict.fromkeys(
                raw_metric_columns + fallback_metric_columns + always_include_raw_columns
            )
        )
        self._ensure_columns(
            duel_summary_df,
            ["player.id", "player.name"] + required_duel_summary_columns,
            "single-CB raw metrics dataframe",
        )

        raw_metrics_df = duel_summary_df[
            ["player.id", "player.name"] + required_duel_summary_columns
        ].copy()
        plot_df = z_scores_df.merge(
            raw_metrics_df,
            on=["player.id", "player.name"],
            how="left",
            validate="one_to_one",
        )

        # Spread rank-based points for readability while keeping raw rank values in labels/tooltips via `metric_value_columns`.
        rank_metric_col = "CB_rank_for_ground_duels_quality_z_score"
        if rank_metric_col in plot_df.columns:
            plot_df[rank_metric_col] = (
                pd.to_numeric(plot_df[rank_metric_col], errors="coerce")
                * self.RANK_Z_SCORE_SCALE
            )

        return self._attach_hover_payload(
            plot_df,
            metric_cols=metrics,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=self.RANK_Z_SCORE_SCALE,
        )

    def _create_plot(
        self,
        *,
        metrics: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        subtitle_style: str = "single",
    ) -> DistributionPlot:
        """
        Instantiate and title the distribution chart with the resolved config.
        """
        if subtitle_style == "comparison":
            subtitle = (
                f"Based on {len(self.z_scores)} CB players with ≥ {self.min_num_duels_involved_in_threshold} duels & playing time ≥ {self.min_minutes_played_threshold} minutes"
            )
        else:
            subtitle = (
                f"Based on {len(self.z_scores)} CB players with ≥ {self.min_num_duels_involved_in_threshold} duels & playing time ≥ {self.min_minutes_played_threshold} minutes"
            )

        dist_plot = DistributionPlot(
            columns=list(metrics),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        dist_plot.add_title(
            title="CB Ground Duel Quality Distribution",
            subtitle=subtitle,
        )
        return dist_plot

    def _resolve_single_cb_by_rank(
        self,
        plot_df: pd.DataFrame,
        *,
        CB_Rank: Any,
    ) -> pd.Series:
        """
        Resolve one CB row by raw Ground-Duels-quality ranking value.

        Args:
            plot_df: Prepared plotting dataframe.
            CB_Rank: Requested rank in the raw ranking column.

        Returns:
            One resolved row for the requested rank.
        """
        return self._resolve_row_by_rank(
            plot_df,
            rank_column=self.RAW_RANK_COLUMN,
            rank_value=CB_Rank,
            entity_label="CB player",
            display_column="player.name",
        )

    def _resolve_single_cb(
        self,
        plot_df: pd.DataFrame,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Any = None,
    ) -> pd.Series:
        """
        Resolve a single CB row from the prepared single-CB plotting dataframe.

        If neither `CB`, `CB_ID`, nor `CB_Rank` is provided, the first row is selected as a
        deterministic fallback to keep quick exploratory plotting simple.

        Args:
            plot_df: Prepared plotting dataframe containing `player.id/name`.
            CB: Optional player name/surname selector.
            CB_ID: Optional player ID selector.
            CB_Rank: Optional raw ranking selector.

        Returns:
            One resolved plotting row for the selected CB.
        """
        candidates = plot_df[["player.id", "player.name"]].drop_duplicates()
        if CB is None and CB_ID is None and CB_Rank is None:
            return plot_df.iloc[0]

        if CB_Rank is not None:
            if CB is not None or CB_ID is not None:
                raise ValueError(
                    "Please provide either CB_Rank or CB/CB_ID, not both."
                )
            return self._resolve_single_cb_by_rank(plot_df, CB_Rank=CB_Rank)

        resolved = self._resolve_entity(
            candidates,
            entity_value=CB,
            entity_id=CB_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        selected_rows = plot_df[plot_df["player.id"] == resolved["player.id"]]
        if selected_rows.empty:
            raise ValueError(f"Resolved CB '{resolved['player.name']}' is not available in plotting dataframe.")
        return selected_rows.iloc[0]

    def Plot_Single_CB(
        self,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Optional[int] = None,
        include_league_average: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one selected CB against the full single-CB Ground-Duels distribution.

        Selection options:
        - `CB` (name/surname),
        - `CB_ID`,
        - `CB_Rank` (raw quality ranking value).
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        plot_df = self._build_plot_df(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
            subtitle_style="single",
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["player.name"],
            legend="All players",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        selected_cb = self._resolve_single_cb(
            plot_df,
            CB=CB,
            CB_ID=CB_ID,
            CB_Rank=CB_Rank,
        )
        dist_plot.add_data_point(
            ser_plot=selected_cb,
            plots="",
            name=selected_cb["player.name"],
            hover=hover_payload_suffix,
            hover_string=hover_string,
            add_annotations=True,
        )

        if include_league_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="player.name",
                display_name_value="CBs' League Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["player.id"] = -1
            dist_plot.add_data_point(
                ser_plot=average_point,
                plots="",
                name="CBs' League Average",
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=False,
            )

        if show:
            dist_plot.fig.show()
        return dist_plot

    def Plot_CBs_Comparison(
        self,
        *,
        CBs: Optional[Sequence[Any]] = None,
        CB_IDs: Optional[Sequence[Any]] = None,
        CB_Rank: Optional[int] = None,
        CB_Ranks: Optional[Sequence[int]] = None,
        include_league_average: bool = True,
        multi_annotations: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one or more selected CBs in comparison mode.

        CBs can be selected by:
        - `CBs` (names/surnames),
        - `CB_IDs`,
        - `CB_Rank` / `CB_Ranks` (raw quality ranking values).
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        plot_df = self._build_plot_df(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
            subtitle_style="comparison",
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["player.name"],
            legend="All players",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        requested_items: List[Tuple[str, Any, Any]] = []
        for cb_id in (CB_IDs or []):
            requested_items.append(("id", None, cb_id))
        for cb_name in (CBs or []):
            requested_items.append(("name", cb_name, None))

        # Rank selectors are optional and additive in comparison mode.
        # They can be used alone or mixed with explicit name/ID selectors.
        if CB_Ranks is not None:
            if CB_Rank is not None:
                raise ValueError(
                    "Please provide either CB_Rank or CB_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Ranks,
                param_name="CB_Ranks",
            ):
                requested_items.append(("rank", rank_value, None))
        if CB_Rank is not None:
            requested_items.append(
                (
                    "rank",
                    self._coerce_positive_rank(CB_Rank, param_name="CB_Rank"),
                    None,
                )
            )

        if not requested_items:
            raise ValueError(
                "Please provide at least one CB selector via CBs, CB_IDs, CB_Rank, or CB_Ranks."
            )

        # Deduplicate by player ID so overlapping selectors (e.g., name + rank
        # pointing to the same CB) do not create duplicate highlighted traces.
        used_ids = set()
        for selection_mode, cb_name_or_rank, cb_id in requested_items:
            if selection_mode == "rank":
                selected_cb = self._resolve_single_cb(
                    plot_df,
                    CB_Rank=cb_name_or_rank,
                )
            else:
                selected_cb = self._resolve_single_cb(
                    plot_df,
                    CB=cb_name_or_rank,
                    CB_ID=cb_id,
                )
            cb_unique_id = selected_cb["player.id"]
            if cb_unique_id in used_ids:
                continue
            used_ids.add(cb_unique_id)

            if multi_annotations:
                dist_plot.add_data_point(
                    ser_plot=selected_cb,
                    plots="",
                    name=selected_cb["player.name"],
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_single_annotations=False,
                    add_multi_annotations=True,
                )
            else:
                dist_plot.add_data_point(
                    ser_plot=selected_cb,
                    plots="",
                    name=selected_cb["player.name"],
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_annotations=True,
                )

        if include_league_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="player.name",
                display_name_value="CBs' League Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["player.id"] = -1
            if multi_annotations:
                dist_plot.add_data_point(
                    ser_plot=average_point,
                    plots="",
                    name="CBs' League Average",
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_single_annotations=False,
                    add_multi_annotations=True,
                )
            else:
                dist_plot.add_data_point(
                    ser_plot=average_point,
                    plots="",
                    name="CBs' League Average",
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_annotations=False,
                )

        if show:
            dist_plot.fig.show()
        return dist_plot


class CB_Pair_Ground_Duels_Distribution_Plot(_Ground_Duels_Distribution_Resolver):
    """
    Build Ground-Duels distribution plots for CB-pair analyses.

    By default, CB-pair plots are split into two side-by-side figures:
    1. Fit/composition metrics (left)
    2. Individual-quality metrics (right)

    Set `split_view=False` in plotting methods to render the legacy single-figure
    layout.
    """

    DEFAULT_PAIR_METRIC_LABELS = {
        "z_duels_per90": "Ground Duels per-90",
        "z_card_discipline": "Card Discipline",
        "z_discipline": "Discipline",
        "z_interceptions_per90": "Interceptions per-90",
        "z_duel_success_rate": "Duel Success Rate",
        "z_possession_win_rate": "Possession Win Rate",
        "floor_z": "CB-Pair Weak-Link (i.e. Floor) Protection",
        "complement_z": "CB-Pair Complementarity (i.e. Deficit-Coverage Gain)",
        "quality_z": "CB-Pair Weighted Avg. Quality",
        "CB_pair_fit_z_score": "CB-Pair (Overall) Fit Score",
        "CB_pair_fit_rank_z_score": "CB-Pair Fit Ranking (Within This Sample)",
        "CB_pair_fit_rank": "CB-Pair Fit Ranking (Within This Sample)",
    }


    DEFAULT_PAIR_METRIC_VALUE_COLUMNS = {
        "z_duels_per90": "duels_per90",
        "z_card_discipline": "card_discipline",
        "z_discipline": "discipline",
        "z_interceptions_per90": "interceptions_per90",
        "z_duel_success_rate": "duel_success_rate",
        "z_possession_win_rate": "possession_win_rate",
        "floor_z": "floor_raw",
        "complement_z": "complement_raw",
        "quality_z": "quality_raw",
        "CB_pair_fit_z_score": None,
        "CB_pair_fit_rank_z_score": "CB_pair_fit_rank",
    }


    DEFAULT_LEFT_METRICS = [
        "floor_z",
        "complement_z",
        "quality_z",
        "CB_pair_fit_z_score",
        "CB_pair_fit_rank_z_score",
    ]


    DEFAULT_RIGHT_METRICS = [
        "z_duels_per90",
        "z_card_discipline",
        "z_discipline",
        "z_interceptions_per90",
        "z_duel_success_rate",
        "z_possession_win_rate",
    ]

    DEFAULT_METRICS = DEFAULT_RIGHT_METRICS + DEFAULT_LEFT_METRICS


    RANK_Z_SCALE = 2.5
    RAW_RANK_COLUMN = "CB_pair_fit_rank"


    def __init__(
        self,
        *,
        df_ground_duel_pairs: pd.DataFrame,
        plot_metric_cols: Optional[Sequence[str]] = None,
        pair_metric_labels: Optional[Mapping[str, str]] = None,
        pair_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        min_num_duels_involved_in_threshold: Any,
        min_minutes_played_threshold: Any,
    ) -> None:
        """
        Args:
            df_ground_duel_pairs: CB-pair dataframe used for plotting.
            plot_metric_cols: Optional legacy single-view metric list.
            pair_metric_labels: Optional constructor-level label map.
            pair_metric_value_columns: Optional constructor-level raw-value map.
            left_metrics: Optional constructor-level left split metrics.
            right_metrics: Optional constructor-level right split metrics.
            min_num_duels_involved_in_threshold: Threshold shown in subtitles.
            min_minutes_played_threshold: Threshold shown in subtitles.
        """
        self.df_ground_duel_pairs = df_ground_duel_pairs
        self._configured_plot_metric_cols = (
            list(plot_metric_cols) if plot_metric_cols is not None else None
        )
        self._configured_pair_metric_labels = (
            dict(pair_metric_labels) if pair_metric_labels is not None else None
        )
        self._configured_pair_metric_value_columns = (
            dict(pair_metric_value_columns)
            if pair_metric_value_columns is not None
            else None
        )
        self._configured_left_metrics = (
            list(left_metrics) if left_metrics is not None else None
        )
        self._configured_right_metrics = (
            list(right_metrics) if right_metrics is not None else None
        )
        self.min_num_duels_involved_in_threshold = min_num_duels_involved_in_threshold
        self.min_minutes_played_threshold = min_minutes_played_threshold

    def _resolve_pair_metric_configs(
        self,
        *,
        split_view: bool,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Dict[str, Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]]:
        """
        Resolve split or single metric configs with method > constructor > default precedence.
        """
        constructor_labels = self._configured_pair_metric_labels
        constructor_value_columns = self._configured_pair_metric_value_columns

        if split_view:
            # Split mode resolves each side independently, then builds a combined
            # superset config used for shared data preparation/hover payloads.
            configured_left = self._configured_left_metrics
            configured_right = self._configured_right_metrics

            if configured_left is None and configured_right is None and self._configured_plot_metric_cols is not None:
                ordered = list(self._configured_plot_metric_cols)
                derived_left = [m for m in ordered if m in self.DEFAULT_LEFT_METRICS]
                derived_right = [m for m in ordered if m not in derived_left]
                configured_left = derived_left or list(self.DEFAULT_LEFT_METRICS)
                configured_right = derived_right or list(self.DEFAULT_RIGHT_METRICS)

            left_cfg = self._resolve_metric_config(
                default_metrics=self.DEFAULT_LEFT_METRICS,
                default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
                default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
                configured_metrics=configured_left,
                configured_metric_labels=constructor_labels,
                configured_metric_value_columns=constructor_value_columns,
                override_metrics=left_metrics,
                override_metric_labels=metric_labels,
                override_metric_value_columns=metric_value_columns,
            )
            right_cfg = self._resolve_metric_config(
                default_metrics=self.DEFAULT_RIGHT_METRICS,
                default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
                default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
                configured_metrics=configured_right,
                configured_metric_labels=constructor_labels,
                configured_metric_value_columns=constructor_value_columns,
                override_metrics=right_metrics,
                override_metric_labels=metric_labels,
                override_metric_value_columns=metric_value_columns,
            )

            combined_metrics: List[str] = []
            for metric in left_cfg[0] + right_cfg[0]:
                if metric not in combined_metrics:
                    combined_metrics.append(metric)

            combined_labels = dict(left_cfg[1])
            combined_labels.update(right_cfg[1])
            combined_value_columns = dict(left_cfg[2])
            combined_value_columns.update(right_cfg[2])
            combined_cfg = (combined_metrics, combined_labels, combined_value_columns)

            return {
                "left": left_cfg,
                "right": right_cfg,
                "combined": combined_cfg,
            }

        override_metrics: Optional[List[str]] = None
        if left_metrics is not None or right_metrics is not None:
            override_metrics = []
            for metric in list(left_metrics or []) + list(right_metrics or []):
                metric_str = str(metric)
                if metric_str not in override_metrics:
                    override_metrics.append(metric_str)

        configured_single_metrics = self._configured_plot_metric_cols
        if configured_single_metrics is None and (
            self._configured_left_metrics is not None or self._configured_right_metrics is not None
        ):
            configured_single_metrics = []
            for metric in list(self._configured_left_metrics or []) + list(self._configured_right_metrics or []):
                metric_str = str(metric)
                if metric_str not in configured_single_metrics:
                    configured_single_metrics.append(metric_str)

        single_cfg = self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
            configured_metrics=configured_single_metrics,
            configured_metric_labels=constructor_labels,
            configured_metric_value_columns=constructor_value_columns,
            override_metrics=override_metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )
        return {"single": single_cfg, "combined": single_cfg}

    def _prepare_plot_df(
        self,
        *,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Prepare the CB-pair dataframe and inject hover payload columns.
        """
        plot_df = self.df_ground_duel_pairs.copy()
        self._ensure_columns(
            plot_df,
            ["pair_name", "CB_pair_fit_z_score"],
            "CB-pair plotting dataframe",
        )

        # Always scale rank-derived z-scores with the CB-pair-specific spread factor
        # so ranking rows are readable even when the source dataframe already
        # contains a precomputed rank z-score column.
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))
            if (
                raw_metric is not None
                and self._is_rank_metric(metric, raw_metric)
                and raw_metric in plot_df.columns
            ):
                plot_df[metric] = self._z_standardize(
                    plot_df[raw_metric],
                    invert=True,
                    scale=self.RANK_Z_SCALE,
                )

        return self._attach_hover_payload(
            plot_df,
            metric_cols=metric_cols,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=self.RANK_Z_SCALE,
        )

    def _create_plot(
        self,
        *,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        section_title: Optional[str] = None,
    ) -> DistributionPlot:
        """
        Instantiate and title one CB-pair distribution figure.
        """
        dist_plot = DistributionPlot(
            columns=list(metric_cols),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        title = "CB-Pair Ground Duels Quality Fit Distribution"
        if section_title:
            title = f"{title} <br> ⇒  {section_title}"
        dist_plot.add_title(
            title=title,
            subtitle=(
                f"All {len(self.df_ground_duel_pairs)} unordered CB pairs, with ≥ {self.min_num_duels_involved_in_threshold} duels & playing time ≥ {self.min_minutes_played_threshold} minutes (each individual CB)"
                "<br>"
                "All metrics are CB-Pair-level (i.e. the weighted average of both players for that metric)"
            ),
        )
        return dist_plot

    def _player_candidates_from_pairs(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a normalized unique player-candidate table from pair columns.

        Args:
            plot_df: CB-pair dataframe containing CB1/CB2 ID and name columns.

        Returns:
            Two-column dataframe (`player.id`, `player.name`) with unique players.
        """
        self._ensure_columns(
            plot_df,
            ["player.id_CB1", "player.name_CB1", "player.id_CB2", "player.name_CB2"],
            "CB-pair player resolution",
        )
        cb1 = plot_df[["player.id_CB1", "player.name_CB1"]].rename(
            columns={"player.id_CB1": "player.id", "player.name_CB1": "player.name"}
        )
        cb2 = plot_df[["player.id_CB2", "player.name_CB2"]].rename(
            columns={"player.id_CB2": "player.id", "player.name_CB2": "player.name"}
        )
        return pd.concat([cb1, cb2], ignore_index=True).drop_duplicates()

    def _resolve_pair_row(self, plot_df: pd.DataFrame, *, CB_Pair: Any = None, CB_1: Any = None, CB_2: Any = None, CB_1_ID: Any = None, CB_2_ID: Any = None) -> Optional[pd.Series]:
        """
        Resolve one pair row from flexible pair selectors.

        Supported selection modes:
        - `CB_Pair` string formats (`A + B`, `A and B`, `A & B`),
        - `CB_Pair` tuple/list `(CB_1, CB_2)`,
        - dict-based specs with `CB_1`/`CB_2` and optional IDs,
        - explicit `CB_1`/`CB_2` + optional IDs.

        Returns:
            Matching pair row, or `None` when no selector was provided.
        """
        if (
            CB_Pair is None
            and CB_1 is None
            and CB_2 is None
            and CB_1_ID is None
            and CB_2_ID is None
        ):
            return None

        if CB_Pair is not None:
            if isinstance(CB_Pair, str):
                CB_1, CB_2 = self._split_pair_string(CB_Pair)
            elif isinstance(CB_Pair, (list, tuple)) and len(CB_Pair) == 2:
                CB_1, CB_2 = CB_Pair[0], CB_Pair[1]
            elif isinstance(CB_Pair, dict):
                CB_1 = CB_Pair.get("CB_1", CB_1)
                CB_2 = CB_Pair.get("CB_2", CB_2)
                CB_1_ID = CB_Pair.get("CB_1_ID", CB_1_ID)
                CB_2_ID = CB_Pair.get("CB_2_ID", CB_2_ID)
                if CB_Pair.get("CB_Pair") is not None:
                    CB_1, CB_2 = self._split_pair_string(CB_Pair["CB_Pair"])
            else:
                raise ValueError(
                    "CB_Pair must be a string ('A + B', 'A and B', 'A & B'), "
                    "a tuple/list of two CB inputs, or a dict with CB_1/CB_2 keys."
                )

        if (CB_1 is None and CB_1_ID is None) or (CB_2 is None and CB_2_ID is None):
            raise ValueError("Please provide both CB_1 and CB_2 (name/surname and/or ID).")

        candidates = self._player_candidates_from_pairs(plot_df)
        resolved_cb1 = self._resolve_entity(
            candidates,
            entity_value=CB_1,
            entity_id=CB_1_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        resolved_cb2 = self._resolve_entity(
            candidates,
            entity_value=CB_2,
            entity_id=CB_2_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )

        id_1 = resolved_cb1["player.id"]
        id_2 = resolved_cb2["player.id"]
        pair_mask = (
            ((plot_df["player.id_CB1"] == id_1) & (plot_df["player.id_CB2"] == id_2))
            | ((plot_df["player.id_CB1"] == id_2) & (plot_df["player.id_CB2"] == id_1))
        )
        matches = plot_df[pair_mask]
        if matches.empty:
            raise ValueError(
                f"No CB pair found for '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple rows found for pair '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        return matches.iloc[0]

    def _resolve_pair_row_by_rank(
        self,
        plot_df: pd.DataFrame,
        *,
        CB_Pair_Rank: Any,
    ) -> pd.Series:
        """
        Resolve one CB-pair row by its raw fit-ranking value.

        Args:
            plot_df: Prepared CB-pair plotting dataframe.
            CB_Pair_Rank: Requested pair rank in `CB_pair_fit_rank`.

        Returns:
            One resolved pair row for the requested rank.
        """
        return self._resolve_row_by_rank(
            plot_df,
            rank_column=self.RAW_RANK_COLUMN,
            rank_value=CB_Pair_Rank,
            entity_label="CB pair",
            display_column="pair_name",
        )

    def _pair_key(self, row: pd.Series) -> str:
        """
        Build a stable deduplication key for a pair-like row.

        Prefers explicit `pair_key` when available, then falls back to `pair_name`.
        """
        if "pair_key" in row.index and pd.notna(row["pair_key"]):
            return str(row["pair_key"])
        return str(row.get("pair_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
        """
        Add one highlighted pair row to a distribution plot.

        This wrapper keeps annotation mode wiring centralized so callers can
        switch between single-entity and multi-entity annotation rendering
        without duplicating plotting code.
        """
        if multi_annotations:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_single_annotations=False,
                add_multi_annotations=True,
            )
        else:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=add_single_annotations,
            )

    def _build_pair_plot(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        highlighted_rows: Sequence[Tuple[pd.Series, str, bool]],
        multi_annotations: bool,
        section_title: Optional[str] = None,
    ) -> DistributionPlot:
        """
        Build one CB-pair figure from pre-resolved highlighted rows.
        """
        dist_plot = self._create_plot(
            metric_cols=metric_cols,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
            section_title=section_title,
        )
        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["pair_name"],
            legend="All pairs",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        for row, display_name, add_single_annotations in highlighted_rows:
            self._add_row_point(
                dist_plot,
                row=row,
                display_name=display_name,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=add_single_annotations,
            )

        return dist_plot

    def _collect_single_pair_highlights(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
        CB_Pair_Rank: Optional[int] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for a single CB-pair plot scenario.

        A pair can be selected either by identity selectors (`CB_Pair` or
        `CB_1`/`CB_2`) or by `CB_Pair_Rank`.
        """
        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()

        # In single-pair mode, rank-based selection and identity-based selection
        # are intentionally mutually exclusive to keep intent unambiguous.
        if CB_Pair_Rank is not None and any(
            value is not None for value in [CB_Pair, CB_1, CB_2, CB_1_ID, CB_2_ID]
        ):
            raise ValueError(
                "Please provide either CB_Pair_Rank or CB_Pair/CB_1/CB_2 selectors, not both."
            )

        if CB_Pair_Rank is not None:
            selected_pair = self._resolve_pair_row_by_rank(
                plot_df,
                CB_Pair_Rank=CB_Pair_Rank,
            )
        else:
            selected_pair = self._resolve_pair_row(
                plot_df,
                CB_Pair=CB_Pair,
                CB_1=CB_1,
                CB_2=CB_2,
                CB_1_ID=CB_1_ID,
                CB_2_ID=CB_2_ID,
            )
        if selected_pair is not None:
            selected_key = self._pair_key(selected_pair)
            used_keys.add(selected_key)
            highlighted_rows.append((selected_pair, selected_pair["pair_name"], True))

        if include_best:
            best_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=False).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                highlighted_rows.append((best_pair, best_pair["pair_name"], True))

        if include_worst:
            worst_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=True).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                highlighted_rows.append((worst_pair, worst_pair["pair_name"], True))

        if include_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=metric_cols,
                metric_labels=dict(metric_labels),
                metric_value_columns=dict(metric_value_columns),
                display_name_col="pair_name",
                display_name_value="CB-Pairs' League Average",
                hover_payload_suffix="_hover_payload",
            )
            average_point["pair_key"] = "__pair_average__"
            highlighted_rows.append((average_point, "CB-Pairs' League Average", False))

        return highlighted_rows

    def _collect_comparison_pair_highlights(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        CB_Pairs: Optional[Sequence[Union[str, Sequence[Any], Dict[str, Any]]]] = None,
        CB_Pair_Rank: Optional[int] = None,
        CB_Pair_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for CB-pair comparison plots.

        Comparison selectors can include explicit pair identities and/or raw
        ranking selectors (`CB_Pair_Rank`, `CB_Pair_Ranks`).
        """
        specs: List[Dict[str, Any]] = []
        for item in (CB_Pairs or []):
            if isinstance(item, str):
                specs.append({"CB_Pair": item})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                specs.append({"CB_1": item[0], "CB_2": item[1]})
            elif isinstance(item, dict):
                specs.append(dict(item))
            else:
                raise ValueError(
                    "Each item in CB_Pairs must be either a pair string, a tuple/list of length 2, or a dict with CB_1/CB_2."
                )

        # Rank-based pair selection is additive in comparison mode.
        # It can be used alone or combined with explicit pair selectors.
        if CB_Pair_Ranks is not None:
            if CB_Pair_Rank is not None:
                raise ValueError(
                    "Please provide either CB_Pair_Rank or CB_Pair_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Pair_Ranks,
                param_name="CB_Pair_Ranks",
            ):
                specs.append({"CB_Pair_Rank": rank_value})
        if CB_Pair_Rank is not None:
            specs.append(
                {
                    "CB_Pair_Rank": self._coerce_positive_rank(
                        CB_Pair_Rank,
                        param_name="CB_Pair_Rank",
                    )
                }
            )

        if not specs and not any([include_best, include_worst, include_average]):
            raise ValueError(
                "Please provide at least one pair selector via CB_Pairs, CB_Pair_Rank, or CB_Pair_Ranks; "
                "or enable at least one of include_best/include_worst/include_average."
            )

        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()
        for spec in specs:
            if spec.get("CB_Pair_Rank") is not None:
                selected_pair = self._resolve_pair_row_by_rank(
                    plot_df,
                    CB_Pair_Rank=spec.get("CB_Pair_Rank"),
                )
            else:
                selected_pair = self._resolve_pair_row(
                    plot_df,
                    CB_Pair=spec.get("CB_Pair"),
                    CB_1=spec.get("CB_1"),
                    CB_2=spec.get("CB_2"),
                    CB_1_ID=spec.get("CB_1_ID"),
                    CB_2_ID=spec.get("CB_2_ID"),
                )
                if selected_pair is None:
                    continue

            key = self._pair_key(selected_pair)
            if key in used_keys:
                continue
            used_keys.add(key)
            highlighted_rows.append((selected_pair, selected_pair["pair_name"], True))

        if include_best:
            best_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=False).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                highlighted_rows.append((best_pair, best_pair["pair_name"], True))

        if include_worst:
            worst_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=True).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                highlighted_rows.append((worst_pair, worst_pair["pair_name"], True))

        if include_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=metric_cols,
                metric_labels=dict(metric_labels),
                metric_value_columns=dict(metric_value_columns),
                display_name_col="pair_name",
                display_name_value="CB-Pairs' League Average",
                hover_payload_suffix="_hover_payload",
            )
            average_point["pair_key"] = "__pair_average__"
            highlighted_rows.append((average_point, "CB-Pairs' League Average", False))

        return highlighted_rows

    def Plot_CB_Pair(
        self,
        *,
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
        CB_Pair_Rank: Optional[int] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
        multi_annotations: bool = False,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        split_view: bool = True,
        render_side_by_side: bool = True,
        show: bool = True,
    ) -> Union[DistributionPlot, Tuple[DistributionPlot, DistributionPlot]]:
        """
        Plot one selected CB-pair (optional) plus optional best/worst/average references.

        Selection options:
        - `CB_Pair` / `CB_1` + `CB_2` (with optional IDs),
        - `CB_Pair_Rank` (raw pair-fit ranking value).
        """
        cfg = self._resolve_pair_metric_configs(
            split_view=split_view,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
        )
        combined_metrics, combined_labels, combined_values = cfg["combined"]
        plot_df = self._prepare_plot_df(
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
        )

        highlighted_rows = self._collect_single_pair_highlights(
            plot_df=plot_df,
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
            CB_Pair=CB_Pair,
            CB_1=CB_1,
            CB_2=CB_2,
            CB_1_ID=CB_1_ID,
            CB_2_ID=CB_2_ID,
            CB_Pair_Rank=CB_Pair_Rank,
            include_best=include_best,
            include_worst=include_worst,
            include_average=include_average,
        )

        if split_view:
            left_metrics_cfg, left_labels_cfg, left_values_cfg = cfg["left"]
            right_metrics_cfg, right_labels_cfg, right_values_cfg = cfg["right"]
            left_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=left_metrics_cfg,
                metric_labels=left_labels_cfg,
                metric_value_columns=left_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Fit / Composition Metrics",
            )
            right_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=right_metrics_cfg,
                metric_labels=right_labels_cfg,
                metric_value_columns=right_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Individual Quality Metrics",
            )

            if show:
                rendered_side_by_side = False
                if render_side_by_side:
                    rendered_side_by_side = self._show_figures_side_by_side(
                        [left_plot.fig, right_plot.fig]
                    )
                if not rendered_side_by_side:
                    left_plot.fig.show()
                    right_plot.fig.show()
            return left_plot, right_plot

        single_metrics, single_labels, single_values = cfg["single"]
        single_plot = self._build_pair_plot(
            plot_df=plot_df,
            metric_cols=single_metrics,
            metric_labels=single_labels,
            metric_value_columns=single_values,
            highlighted_rows=highlighted_rows,
            multi_annotations=multi_annotations,
            section_title=None,
        )
        if show:
            single_plot.fig.show()
        return single_plot

    def Plot_CB_Pairs_Comparison(
        self,
        *,
        CB_Pairs: Optional[Sequence[Union[str, Sequence[Any], Dict[str, Any]]]] = None,
        CB_Pair_Rank: Optional[int] = None,
        CB_Pair_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
        multi_annotations: bool = True,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        split_view: bool = True,
        render_side_by_side: bool = True,
        show: bool = True,
    ) -> Union[DistributionPlot, Tuple[DistributionPlot, DistributionPlot]]:
        """
        Compare multiple user-selected CB-pairs plus optional best/worst/average overlays.

        Pair selection supports:
        - explicit pair selectors (`CB_Pairs`),
        - rank selectors (`CB_Pair_Rank`, `CB_Pair_Ranks`).
        """
        cfg = self._resolve_pair_metric_configs(
            split_view=split_view,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
        )
        combined_metrics, combined_labels, combined_values = cfg["combined"]
        plot_df = self._prepare_plot_df(
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
        )

        highlighted_rows = self._collect_comparison_pair_highlights(
            plot_df=plot_df,
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
            CB_Pairs=CB_Pairs,
            CB_Pair_Rank=CB_Pair_Rank,
            CB_Pair_Ranks=CB_Pair_Ranks,
            include_best=include_best,
            include_worst=include_worst,
            include_average=include_average,
        )

        if split_view:
            left_metrics_cfg, left_labels_cfg, left_values_cfg = cfg["left"]
            right_metrics_cfg, right_labels_cfg, right_values_cfg = cfg["right"]
            left_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=left_metrics_cfg,
                metric_labels=left_labels_cfg,
                metric_value_columns=left_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Fit / Composition Metrics",
            )
            right_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=right_metrics_cfg,
                metric_labels=right_labels_cfg,
                metric_value_columns=right_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Individual Quality Metrics",
            )

            if show:
                rendered_side_by_side = False
                if render_side_by_side:
                    rendered_side_by_side = self._show_figures_side_by_side(
                        [left_plot.fig, right_plot.fig]
                    )
                if not rendered_side_by_side:
                    left_plot.fig.show()
                    right_plot.fig.show()
            return left_plot, right_plot

        single_metrics, single_labels, single_values = cfg["single"]
        single_plot = self._build_pair_plot(
            plot_df=plot_df,
            metric_cols=single_metrics,
            metric_labels=single_labels,
            metric_value_columns=single_values,
            highlighted_rows=highlighted_rows,
            multi_annotations=multi_annotations,
            section_title=None,
        )
        if show:
            single_plot.fig.show()
        return single_plot


class Anchor_CB_Companion_Fit_Ground_Duels_Distribution_Plot(_Ground_Duels_Distribution_Resolver):
    """
    Build directional anchor-to-companion Ground-Duels distribution plots.

    The class includes default companion-fit metrics/labels/raw mappings but still
    supports constructor-level and per-call metric overrides.
    """

    DEFAULT_METRICS = [
        "coverage_gain_z_within_anchor",
        "CB_pair_fit_z_score",
        "companion_fit_score_z_within_anchor",
        "companion_rank_for_anchor_z_score",
    ]
    DEFAULT_METRIC_LABELS = {
        "coverage_gain_z_within_anchor": "Deficit-Coverage Gain (Within Anchor CB)",
        "CB_pair_fit_z_score": "CB-Pair (Overall) Fit Score",
        "companion_fit_score_z_within_anchor": "CB-Companion Fit Score (Within Anchor CB)",
        "companion_rank_for_anchor_z_score": "CB-Companion Fit Ranking (Within Anchor CB's Sample)",
        "companion_rank_for_anchor": "CB-Companion Fit Ranking (Within Anchor CB's Sample)",
    }
    DEFAULT_METRIC_VALUE_COLUMNS = {
        "coverage_gain_z_within_anchor": "coverage_gain_raw",
        "CB_pair_fit_z_score": None,
        "companion_fit_score_z_within_anchor": "companion_fit_score",
        "companion_rank_for_anchor_z_score": "companion_rank_for_anchor",
    }

    def __init__(
        self,
        *,
        df_ground_duel_companion_fit: pd.DataFrame,
        companion_plot_metric_cols: Optional[Sequence[str]] = None,
        companion_metric_labels: Optional[Mapping[str, str]] = None,
        companion_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        """
        Args:
            df_ground_duel_companion_fit: Directional anchor-to-partner fit dataframe.
            companion_plot_metric_cols: Optional constructor-level metric list.
            companion_metric_labels: Optional constructor-level label map.
            companion_metric_value_columns: Optional constructor-level raw-value map.
        """
        self.df_ground_duel_companion_fit = df_ground_duel_companion_fit
        self._configured_companion_plot_metric_cols = (
            list(companion_plot_metric_cols)
            if companion_plot_metric_cols is not None
            else None
        )
        self._configured_companion_metric_labels = (
            dict(companion_metric_labels)
            if companion_metric_labels is not None
            else None
        )
        self._configured_companion_metric_value_columns = (
            dict(companion_metric_value_columns)
            if companion_metric_value_columns is not None
            else None
        )
        self._anchor_player_id: Optional[Any] = None
        self._anchor_player_name: Optional[str] = None

    def _resolve_current_metric_config(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve call-ready metric config with precedence:
        method override > constructor config > class defaults.
        """
        return self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_METRIC_VALUE_COLUMNS,
            configured_metrics=self._configured_companion_plot_metric_cols,
            configured_metric_labels=self._configured_companion_metric_labels,
            configured_metric_value_columns=self._configured_companion_metric_value_columns,
            override_metrics=metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )

    def _anchor_candidates(self) -> pd.DataFrame:
        """
        Build candidate anchor table for anchor selection resolution.

        Returns:
            Dataframe with `anchor_player_id` and `anchor_player_name`.
            If IDs are unavailable in source data, synthetic IDs are generated.
        """
        self._ensure_columns(
            self.df_ground_duel_companion_fit,
            ["anchor_player_name"],
            "companion-fit anchor resolution",
        )
        if "anchor_player_id" in self.df_ground_duel_companion_fit.columns:
            return self.df_ground_duel_companion_fit[
                ["anchor_player_id", "anchor_player_name"]
            ].drop_duplicates()

        candidates = self.df_ground_duel_companion_fit[["anchor_player_name"]].drop_duplicates().copy()
        candidates["anchor_player_id"] = np.arange(len(candidates))
        return candidates

    def _resolve_anchor(self, *, Anchor_CB: Any = None, Anchor_CB_ID: Any = None) -> pd.Series:
        """
        Resolve the active anchor CB from name/surname and/or ID input.

        Behavior:
        - if no input is provided, reuses previously initialized anchor when possible; otherwise falls back to first alphabetical anchor.
        - if ID is provided but the dataset has no anchor ID column, raises.
        """
        candidates = self._anchor_candidates()

        if Anchor_CB is None and Anchor_CB_ID is None:
            if self._anchor_player_name is not None:
                cached_by_name = candidates[
                    candidates["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(self._anchor_player_name)
                ]
                if not cached_by_name.empty:
                    return cached_by_name.iloc[0]
            return candidates.sort_values("anchor_player_name").iloc[0]

        if (
            Anchor_CB_ID is not None
            and "anchor_player_id" not in self.df_ground_duel_companion_fit.columns
        ):
            raise ValueError(
                "Anchor_CB_ID was provided, but `anchor_player_id` is not available in the companion dataframe."
            )

        return self._resolve_entity(
            candidates,
            entity_value=Anchor_CB,
            entity_id=Anchor_CB_ID,
            id_col="anchor_player_id",
            name_col="anchor_player_name",
            entity_label="anchor CB",
        )

    def Initialize_Desired_Anchor_CB(self, *, Anchor_CB: Any = None, Anchor_CB_ID: Any = None) -> pd.Series:
        """
        Resolve and cache the anchor CB used by subsequent companion plots.
        """
        resolved_anchor = self._resolve_anchor(Anchor_CB=Anchor_CB, Anchor_CB_ID=Anchor_CB_ID)
        self._anchor_player_id = resolved_anchor.get("anchor_player_id", None)
        self._anchor_player_name = str(resolved_anchor["anchor_player_name"])
        return resolved_anchor

    def _build_anchor_plot_df(
        self,
        *,
        anchor_name: str,
        anchor_id: Any = None,
        top_n: Optional[int] = None,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Build anchor-filtered companion-fit dataframe and attach hover payloads.
        """
        plot_df = self.df_ground_duel_companion_fit.copy()
        self._ensure_columns(
            plot_df,
            ["anchor_player_name", "partner_player_name", "companion_fit_score"],
            "companion-fit plotting dataframe",
        )

        if anchor_id is not None and "anchor_player_id" in plot_df.columns:
            filtered = plot_df[plot_df["anchor_player_id"] == anchor_id].copy()
            if filtered.empty:
                filtered = plot_df[
                    plot_df["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(anchor_name)
                ].copy()
        else:
            filtered = plot_df[
                plot_df["anchor_player_name"].map(self._normalize_text)
                == self._normalize_text(anchor_name)
            ].copy()

        if filtered.empty:
            raise ValueError(f"No companion-fit rows found for anchor CB '{anchor_name}'.")

        if "coverage_gain_z_within_anchor" not in filtered.columns and "coverage_gain_raw" in filtered.columns:
            filtered["coverage_gain_z_within_anchor"] = self._z_standardize(filtered["coverage_gain_raw"])

        if "companion_fit_score_z_within_anchor" not in filtered.columns:
            filtered["companion_fit_score_z_within_anchor"] = self._z_standardize(
                filtered["companion_fit_score"]
            )

        if "companion_rank_for_anchor_z_score" not in filtered.columns:
            if "companion_rank_for_anchor" in filtered.columns:
                filtered["companion_rank_for_anchor_z_score"] = self._z_standardize(
                    filtered["companion_rank_for_anchor"],
                    invert=True,
                    scale=2.0,
                )
            else:
                filtered["companion_rank_for_anchor_z_score"] = float("nan")

        filtered = filtered.sort_values(
            ["companion_fit_score", "partner_player_name"],
            ascending=[False, True],
        ).reset_index(drop=True)

        if top_n is not None:
            if not isinstance(top_n, (int, np.integer)) or int(top_n) <= 0:
                raise ValueError("top_n must be a positive integer or None.")
            filtered = filtered.head(int(top_n)).copy().reset_index(drop=True)

        return self._attach_hover_payload(
            filtered,
            metric_cols=metric_cols,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=2.0,
        )

    def _resolve_companion_row(
        self,
        plot_df: pd.DataFrame,
        *,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
    ) -> Optional[pd.Series]:
        """
        Resolve one companion row by either:
        1) companion name/surname,
        2) companion ID, or
        3) companion rank within the current anchor sample.
        """
        if Companion_CB is None and Companion_CB_ID is None and Companion_Rank is None:
            return None

        if Companion_Rank is not None:
            if Companion_CB is not None or Companion_CB_ID is not None:
                raise ValueError(
                    "Please specify either Companion_Rank or Companion_CB/Companion_CB_ID, not both."
                )
            if not isinstance(Companion_Rank, (int, np.integer)) or int(Companion_Rank) <= 0:
                raise ValueError("Companion_Rank must be a positive integer.")

            requested_rank = int(Companion_Rank)
            if "companion_rank_for_anchor" in plot_df.columns:
                rank_values = pd.to_numeric(
                    plot_df["companion_rank_for_anchor"], errors="coerce"
                )
                matches = plot_df[rank_values == float(requested_rank)]
            else:
                # Fallback when rank column is missing: use current deterministic order
                # (already sorted by companion_fit_score desc, partner name asc).
                if requested_rank > len(plot_df):
                    matches = plot_df.iloc[0:0]
                else:
                    matches = plot_df.iloc[[requested_rank - 1]]

            if matches.empty:
                top_n_hint = (
                    " The requested rank may be outside the currently filtered set (e.g., due to top_n)."
                    if len(plot_df) > 0
                    else ""
                )
                raise ValueError(
                    f"No companion found at rank #{requested_rank} for the current anchor CB's sample."
                    f"{top_n_hint}"
                )
            if len(matches) > 1:
                candidates = matches["partner_player_name"].astype(str).tolist()
                raise ValueError(
                    f"Multiple companions found for rank #{requested_rank}: {', '.join(candidates)}"
                )
            return matches.iloc[0]

        self._ensure_columns(
            plot_df,
            ["partner_player_name"],
            "companion selection",
        )

        has_partner_id = "partner_player_id" in plot_df.columns
        if Companion_CB_ID is not None and not has_partner_id:
            raise ValueError(
                "Companion_CB_ID was provided, but `partner_player_id` is not available in the companion dataframe."
            )

        if has_partner_id:
            candidates = plot_df[["partner_player_id", "partner_player_name"]].drop_duplicates().rename(
                columns={"partner_player_id": "player.id", "partner_player_name": "player.name"}
            )
            resolved = self._resolve_entity(
                candidates,
                entity_value=Companion_CB,
                entity_id=Companion_CB_ID,
                id_col="player.id",
                name_col="player.name",
                entity_label="companion CB",
            )
            matches = plot_df[plot_df["partner_player_id"] == resolved["player.id"]]
        else:
            candidates = plot_df[["partner_player_name"]].drop_duplicates().copy()
            candidates["player.id"] = np.arange(len(candidates))
            candidates = candidates.rename(columns={"partner_player_name": "player.name"})
            resolved = self._resolve_entity(
                candidates,
                entity_value=Companion_CB,
                entity_id=None,
                id_col="player.id",
                name_col="player.name",
                entity_label="companion CB",
            )
            matches = plot_df[
                plot_df["partner_player_name"].map(self._normalize_text)
                == self._normalize_text(resolved["player.name"])
            ]

        if matches.empty:
            raise ValueError(f"No companion fit found for '{Companion_CB}'.")
        return matches.iloc[0]

    def _companion_key(self, row: pd.Series) -> str:
        """
        Build a stable deduplication key for companion rows.
        """
        if "partner_player_id" in row.index and pd.notna(row["partner_player_id"]):
            return str(row["partner_player_id"])
        return self._normalize_text(row.get("partner_player_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
        """
        Add one highlighted companion row to the anchor-companion distribution.

        Args mirror the pair-level `_add_row_point` helper and are intentionally
        aligned for maintenance consistency across plot suites.
        """
        if multi_annotations:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_single_annotations=False,
                add_multi_annotations=True,
            )
        else:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=add_single_annotations,
            )

    def _create_plot(
        self,
        *,
        anchor_name: str,
        num_candidates: int,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> DistributionPlot:
        """
        Instantiate and title one anchor-companion distribution figure.
        """
        dist_plot = DistributionPlot(
            columns=list(metric_cols),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        dist_plot.add_title(
            title=f"CB-Companion Ground Duels Potential Fits Distribution   →   {anchor_name} acting as the Anchor CB",
            subtitle=(
                f"All {num_candidates} potential companions for {anchor_name} (directional A → B)"
                "<br>"
                "Metrics are at the CB-Companion-level (i.e. within the Anchor CB's sample),"
                "<br>"
                "showing the expected contribution of the companion to the CB-Pair's overall fit with the specified Anchor CB."
            ),
        )
        return dist_plot

    def Plot_Companion_of_Anchor_CB(
        self,
        *,
        Anchor_CB: Any = None,
        Anchor_CB_ID: Any = None,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
        Companion_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_anchor_CB_pool_average: bool = False,
        top_n: Optional[int] = None,
        multi_annotations: bool = False,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot potential companions for one anchor CB, with optional highlighted rows.

        The highlighted rows can include:
        - an explicitly selected companion
        - an explicitly selected companion rank within the anchor CB's sample
        - multiple explicitly selected companion ranks within the anchor CB's sample
        - best/worst companion for the anchor
        - the anchor's companion-pool average
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        resolved_anchor = self.Initialize_Desired_Anchor_CB(
            Anchor_CB=Anchor_CB,
            Anchor_CB_ID=Anchor_CB_ID,
        )
        anchor_name = str(resolved_anchor["anchor_player_name"])
        anchor_id = resolved_anchor.get("anchor_player_id", None)

        plot_df = self._build_anchor_plot_df(
            anchor_name=anchor_name,
            anchor_id=anchor_id,
            top_n=top_n,
            metric_cols=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            anchor_name=anchor_name,
            num_candidates=len(plot_df),
            metric_cols=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["partner_player_name"],
            legend="All companions",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        used_keys = set()

        selected_companions: List[pd.Series] = []

        # Selection mode A: multiple explicit ranks provided by the caller.
        # Each rank is resolved independently in current anchor/top_n context.
        if Companion_Ranks is not None:
            if Companion_Rank is not None:
                raise ValueError(
                    "Please provide either Companion_Rank or Companion_Ranks, not both."
                )
            if isinstance(Companion_Ranks, (str, bytes)):
                raise ValueError(
                    "Companion_Ranks must be a sequence of positive integers, not a string."
                )

            unique_ranks: List[int] = []
            for rank_value in Companion_Ranks:
                if not isinstance(rank_value, (int, np.integer)) or int(rank_value) <= 0:
                    raise ValueError(
                        f"Invalid rank '{rank_value}' in Companion_Ranks. All ranks must be positive integers."
                    )
                rank_int = int(rank_value)
                if rank_int not in unique_ranks:
                    unique_ranks.append(rank_int)

            for rank_int in unique_ranks:
                rank_selected_companion = self._resolve_companion_row(
                    plot_df,
                    Companion_Rank=rank_int,
                )
                if rank_selected_companion is not None:
                    selected_companions.append(rank_selected_companion)

        # Selection mode B: one explicit companion selector (name/ID/rank).
        single_selected_companion = self._resolve_companion_row(
            plot_df,
            Companion_CB=Companion_CB,
            Companion_CB_ID=Companion_CB_ID,
            Companion_Rank=Companion_Rank,
        )
        if single_selected_companion is not None:
            selected_companions.append(single_selected_companion)

        # Deduplicate selected rows so the same companion is not plotted twice
        # when selection criteria overlap (e.g., explicit rank equals best row).
        for selected_companion in selected_companions:
            selected_key = self._companion_key(selected_companion)
            if selected_key in used_keys:
                continue
            used_keys.add(selected_key)
            selected_label = f"{anchor_name} + {selected_companion['partner_player_name']}"
            self._add_row_point(
                dist_plot,
                row=selected_companion,
                display_name=selected_label,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=True,
            )

        if include_best:
            best_companion = plot_df.sort_values("companion_fit_score", ascending=False).iloc[0]
            best_key = self._companion_key(best_companion)
            if best_key not in used_keys:
                used_keys.add(best_key)
                best_label = f"{anchor_name} + {best_companion['partner_player_name']}"
                self._add_row_point(
                    dist_plot,
                    row=best_companion,
                    display_name=best_label,
                    hover_payload_suffix=hover_payload_suffix,
                    hover_string=hover_string,
                    multi_annotations=multi_annotations,
                    add_single_annotations=True,
                )

        if include_worst:
            worst_companion = plot_df.sort_values("companion_fit_score", ascending=True).iloc[0]
            worst_key = self._companion_key(worst_companion)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                worst_label = f"{anchor_name} + {worst_companion['partner_player_name']}"
                self._add_row_point(
                    dist_plot,
                    row=worst_companion,
                    display_name=worst_label,
                    hover_payload_suffix=hover_payload_suffix,
                    hover_string=hover_string,
                    multi_annotations=multi_annotations,
                    add_single_annotations=True,
                )

        if include_anchor_CB_pool_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="partner_player_name",
                display_name_value="Companion Pool Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["partner_player_id"] = -1
            average_label = f"{anchor_name}'s Companion Pool Average"
            self._add_row_point(
                dist_plot,
                row=average_point,
                display_name=average_label,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=False,
            )

        if show:
            dist_plot.fig.show()
        return dist_plot





# --------------------------------------------------------------------------|
# Aerial-Duels Plotting Infrastructure                                      |
# --------------------------------------------------------------------------|
# The classes below form a layered architecture:                            |
# - `_Aerial_Duels_Distribution_Resolver`: shared data/selection utilities  |
# - `Single_CB_*`, `CB_Pair_*`, `Anchor_CB_Companion_*`: public plot APIs   |
# --------------------------------------------------------------------------|

class _Aerial_Duels_Distribution_Resolver:
    """
    Shared helper utilities used by all Aerial-Duels distribution plot wrappers.

    This resolver centralizes cross-cutting concerns such as:
    - robust player/pair name normalization and resolution,
    - rank-aware z-score handling,
    - hover payload formatting,
    - average-point construction.

    Centralizing these behaviors keeps public plotting classes small and ensures consistent semantics across single-CB, CB-pair, and companion-fit views.
    """

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        Normalize free-text input for resilient matching.

        The normalization pipeline is intentionally strict and deterministic:
        - converts to lowercase,
        - strips accents/diacritics,
        - removes punctuation/special symbols,
        - collapses repeated whitespace.

        Args:
            value: Any user-provided value that should be interpreted as text.

        Returns:
            Normalized tokenizable text; empty string for null-like values.
        """
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _tokenize(cls, value: Any) -> Tuple[str, ...]:
        """
        Split normalized text into lexical tokens.

        Args:
            value: Input value to normalize and tokenize.

        Returns:
            Tuple of tokens. Empty tuple for empty/invalid normalized text.
        """
        normalized = cls._normalize_text(value)
        if not normalized:
            return tuple()
        return tuple(normalized.split(" "))

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
        """
        Validate dataframe schema requirements.

        Args:
            df: Dataframe to validate.
            required_columns: Columns that must exist in `df`.
            context: Human-readable context included in validation errors.

        Raises:
            KeyError: If any required column is missing.
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required column(s) for {context}: " + ", ".join(missing)
            )

    @staticmethod
    def _to_number_or_none(value: Any) -> Optional[float]:
        """
        Attempt safe scalar numeric conversion.

        Args:
            value: Value to convert.

        Returns:
            `float` when conversion is successful and finite; otherwise `None`.
        """
        try:
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        except Exception:
            return None
        if pd.isna(converted):
            return None
        return float(converted)

    @staticmethod
    def _coerce_positive_rank(rank_value: Any, *, param_name: str) -> int:
        """
        Validate and coerce one rank selector to a positive integer.

        Args:
            rank_value: Candidate rank value.
            param_name: Parameter name used in error messages.

        Returns:
            Rank as a positive integer.
        """
        if not isinstance(rank_value, (int, np.integer)) or int(rank_value) <= 0:
            raise ValueError(f"{param_name} must be a positive integer.")
        return int(rank_value)

    @classmethod
    def _coerce_unique_positive_ranks(
        cls,
        rank_values: Sequence[Any],
        *,
        param_name: str,
    ) -> List[int]:
        """
        Validate, coerce, and deduplicate a rank-sequence selector.

        Args:
            rank_values: Sequence of candidate rank values.
            param_name: Parameter name used in error messages.

        Returns:
            Ordered list of unique positive integers.
        """
        if isinstance(rank_values, (str, bytes)):
            raise ValueError(
                f"{param_name} must be a sequence of positive integers, not a string."
            )

        normalized: List[int] = []
        for rank_value in rank_values:
            rank_int = cls._coerce_positive_rank(rank_value, param_name=param_name)
            if rank_int not in normalized:
                normalized.append(rank_int)
        return normalized

    @classmethod
    def _resolve_row_by_rank(
        cls,
        plot_df: pd.DataFrame,
        *,
        rank_column: str,
        rank_value: Any,
        entity_label: str,
        display_column: str,
    ) -> pd.Series:
        """
        Resolve one row by exact raw-ranking value.

        Args:
            plot_df: Candidate dataframe to search.
            rank_column: Raw rank column name.
            rank_value: Requested rank value.
            entity_label: Human-readable entity label for errors.
            display_column: Column used to list candidate names in ambiguity errors.

        Returns:
            The uniquely resolved row for the requested rank.
        """
        requested_rank = cls._coerce_positive_rank(
            rank_value,
            param_name=f"{entity_label} rank",
        )

        if rank_column not in plot_df.columns:
            raise KeyError(
                f"Cannot resolve {entity_label} by rank because column '{rank_column}' "
                "is not available in the plotting dataframe."
            )

        rank_values = pd.to_numeric(plot_df[rank_column], errors="coerce")
        matches = plot_df[rank_values == float(requested_rank)]

        if matches.empty:
            raise ValueError(
                f"No {entity_label} found at rank #{requested_rank}."
            )
        if len(matches) > 1:
            candidates = (
                matches[display_column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f"Multiple {entity_label} entries found at rank #{requested_rank}: "
                f"{', '.join(candidates)}"
            )
        return matches.iloc[0]

    @classmethod
    def _resolve_entity(cls, candidates: pd.DataFrame, *, entity_value: Any = None, entity_id: Any = None, id_col: str, name_col: str, entity_label: str) -> pd.Series:
        """
        Resolve a single entity row from candidate IDs/names.

        Resolution precedence:
        1. Explicit ID match (numeric-safe first, then string fallback),
        2. Exact normalized full-name match,
        3. Unique token/surname subset match.

        Ambiguity is never auto-resolved: a `ValueError` is raised with candidate
        names so callers can disambiguate explicitly.

        Args:
            candidates: Candidate dataframe containing ID and name columns.
            entity_value: Optional name/surname-like selector.
            entity_id: Optional ID selector.
            id_col: Candidate ID column name.
            name_col: Candidate display-name column name.
            entity_label: Label used in error messages (e.g., "CB player").

        Returns:
            The uniquely resolved candidate row.
        """
        candidates = candidates[[id_col, name_col]].dropna(subset=[id_col, name_col]).drop_duplicates()
        if candidates.empty:
            raise ValueError(f"No {entity_label} candidates available to resolve.")

        if entity_id is None and isinstance(entity_value, (int, np.integer)):
            entity_id = int(entity_value)
            entity_value = None

        if entity_id is not None:
            requested_id_numeric = cls._to_number_or_none(entity_id)
            if requested_id_numeric is not None:
                candidate_id_numeric = pd.to_numeric(candidates[id_col], errors="coerce")
                matches = candidates[candidate_id_numeric == requested_id_numeric]
            else:
                matches = candidates[candidates[id_col].astype(str).str.strip() == str(entity_id).strip()]

            if len(matches) == 1:
                return matches.iloc[0]
            if len(matches) > 1:
                options = sorted(matches[name_col].astype(str).unique().tolist())
                raise ValueError(
                    f"Ambiguous {entity_label} ID '{entity_id}'. Candidates: {', '.join(options)}"
                )
            raise ValueError(f"No {entity_label} found for ID '{entity_id}'.")

        if entity_value is None:
            raise ValueError(f"Please provide either {entity_label} name/surname or {entity_label} ID.")

        query = cls._normalize_text(entity_value)
        if not query:
            raise ValueError(f"Invalid empty {entity_label} name input.")

        working = candidates.copy()
        working["_normalized_name"] = working[name_col].map(cls._normalize_text)
        working["_name_tokens"] = working[name_col].map(cls._tokenize)

        exact_matches = working[working["_normalized_name"] == query]
        if len(exact_matches) == 1:
            return exact_matches.iloc[0]
        if len(exact_matches) > 1:
            options = sorted(exact_matches[name_col].astype(str).unique().tolist())
            raise ValueError(
                f"Ambiguous {entity_label} name '{entity_value}'. Candidates: {', '.join(options)}"
            )

        query_tokens = set(cls._tokenize(query))
        token_matches = working[
            working["_name_tokens"].map(lambda tokens: query_tokens.issubset(set(tokens)))
        ]

        if len(token_matches) == 1:
            return token_matches.iloc[0]
        if len(token_matches) > 1:
            options = sorted(token_matches[name_col].astype(str).unique().tolist())
            raise ValueError(
                f"Ambiguous {entity_label} input '{entity_value}'. Candidates: {', '.join(options)}"
            )

        raise ValueError(f"No {entity_label} found for input '{entity_value}'.")

    @staticmethod
    def _z_standardize(values: Union[pd.Series, Sequence[Any]], *, invert: bool = False, scale: float = 1.0) -> pd.Series:
        """
        Compute z-standardized values with optional sign inversion and scaling.

        This helper is used both for standard metric normalization and rank-based
        transformations (where lower raw rank means better, hence `invert=True`).

        Args:
            values: Numeric-like values to standardize.
            invert: If true, multiply standardized values by `-1`.
            scale: Multiplicative factor applied after standardization.

        Returns:
            Z-standardized series. If variance is zero/undefined, returns all NaN.
        """
        series = pd.to_numeric(pd.Series(values), errors="coerce")
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(float("nan"), index=series.index)
        z_values = (series - series.mean()) / std
        if invert:
            z_values = -z_values
        return z_values * scale

    @classmethod
    def _is_rank_metric(cls, metric_name: str, raw_metric_name: Optional[str]) -> bool:
        """
        Identify rank-like metrics by inspecting z/raw metric names.

        Args:
            metric_name: Plotted metric column (typically z-score column).
            raw_metric_name: Raw companion/value column mapped to that metric.

        Returns:
            True when either name contains the token `rank`.
        """
        metric_token = str(metric_name).lower()
        raw_metric_token = str(raw_metric_name).lower() if raw_metric_name is not None else ""
        return ("rank" in metric_token) or ("rank" in raw_metric_token)

    @classmethod
    def _format_hover_line(cls, metric_name: str, raw_metric_name: Optional[str], metric_label: str, raw_value: Any, z_value: Any) -> str:
        """
        Build one semantic-aware hover text line for a metric.

        Behavior:
        - metrics without a raw column mapping show z-score only,
        - rank-like metrics show integer rank format (`# <int>`),
        - all other metrics show `raw + corresponding z-score`.
        """
        if raw_metric_name is None:
            return f"Z-score = {z_value:.2f}" if pd.notna(z_value) else "Z-score = N/A"
        if cls._is_rank_metric(metric_name, raw_metric_name):
            return f"Rank =   # {int(round(raw_value))}" if pd.notna(raw_value) else "Rank =   # N/A"
        raw_text = f"{raw_value:.2f}" if pd.notna(raw_value) else "N/A"
        z_text = f"{z_value:.2f}" if pd.notna(z_value) else "N/A"
        return f"{metric_label} = {raw_text}<br>Respective Z-score = {z_text}"

    @classmethod
    def _attach_hover_payload(cls, plot_df: pd.DataFrame, *, metric_cols: Sequence[str], metric_labels: Dict[str, str], metric_value_columns: Dict[str, Optional[str]], hover_payload_suffix: str = "_hover_payload", fill_missing_rank_z: bool = True, rank_z_scale: float = 2.0) -> pd.DataFrame:
        """
        Attach per-metric hover payload tuples used by `DistributionPlot`.

        For each metric, this method optionally reconstructs missing/all-NaN
        rank z-score columns from the corresponding raw rank column, then creates
        payload tuples of the form:
        `(metric_label, raw_value, z_value, formatted_hover_line)`.

        Args:
            plot_df: Plotting dataframe to enrich in place.
            metric_cols: Metrics expected in the distribution plot.
            metric_labels: Human-readable labels per metric.
            metric_value_columns: Raw-value column mapping per metric.
            hover_payload_suffix: Suffix for generated payload columns.
            fill_missing_rank_z: Whether rank z-score fallback should be applied.
            rank_z_scale: Scaling factor used when reconstructing rank z-scores.

        Returns:
            The same dataframe instance, enriched with hover payload columns.
        """
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))

            can_backfill_rank_z = (
                fill_missing_rank_z
                and raw_metric is not None
                and cls._is_rank_metric(metric, raw_metric)
                and raw_metric in plot_df.columns
            )
            metric_missing = metric not in plot_df.columns
            metric_all_nan = False
            if not metric_missing:
                metric_all_nan = pd.to_numeric(plot_df[metric], errors="coerce").isna().all()

            if metric_missing or (can_backfill_rank_z and metric_all_nan):
                if can_backfill_rank_z:
                    plot_df[metric] = cls._z_standardize(
                        plot_df[raw_metric],
                        invert=True,
                        scale=rank_z_scale,
                    )
                else:
                    plot_df[metric] = float("nan")

            metric_label = metric_labels.get(metric, format_metric(metric))
            default_series = pd.Series(float("nan"), index=plot_df.index)
            raw_values = pd.to_numeric(
                plot_df.get(raw_metric, default_series),
                errors="coerce",
            )
            z_values = pd.to_numeric(
                plot_df.get(metric, default_series),
                errors="coerce",
            )

            hover_lines = [
                cls._format_hover_line(
                    metric,
                    raw_metric,
                    metric_label,
                    raw_value,
                    z_value,
                )
                for raw_value, z_value in zip(raw_values, z_values)
            ]

            plot_df[f"{metric}{hover_payload_suffix}"] = list(
                zip(
                    [metric_label] * len(plot_df),
                    raw_values,
                    z_values,
                    hover_lines,
                )
            )

        return plot_df

    @classmethod
    def _split_pair_string(cls, pair_value: str) -> Tuple[str, str]:
        """
        Parse user-friendly pair strings into two entity selectors.

        Accepted separators:
        - `+`
        - `and`
        - `&`

        Args:
            pair_value: Pair input such as `"A + B"` or `"A and B"`.

        Returns:
            Tuple of two stripped entity strings.
        """
        if not isinstance(pair_value, str):
            raise ValueError("CB_Pair must be a string such as 'A + B', 'A and B', or 'A & B'.")

        parts = re.split(r"\s*(?:\+|&|\band\b)\s*", pair_value.strip(), maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                "Could not parse CB pair input. Use formats like 'A + B', 'A and B', or 'A & B'."
            )
        return parts[0].strip(), parts[1].strip()

    @classmethod
    def _build_average_point(cls, *, plot_df: pd.DataFrame, metric_cols: Sequence[str], metric_labels: Dict[str, str], metric_value_columns: Dict[str, Optional[str]], display_name_col: str, display_name_value: str, hover_payload_suffix: str = "_hover_payload") -> pd.Series:
        """
        Build a synthetic cohort-average point compatible with `DistributionPlot`.

        The returned series contains:
        - average z-scores for each plotted metric,
        - average raw values for mapped raw columns (when available),
        - per-metric hover payload entries matching normal row payload shape,
        - a display-name field to support legend/labels.

        Args:
            plot_df: Source dataframe defining the cohort.
            metric_cols: Metrics used in the plot.
            metric_labels: Human-readable labels per metric.
            metric_value_columns: Raw-value mapping per metric.
            display_name_col: Column name used by the plot as display label.
            display_name_value: Display label for the synthetic average point.
            hover_payload_suffix: Suffix used for hover payload columns.

        Returns:
            Series representing one synthetic average row.
        """
        avg_raw_metrics = pd.Series(
            {
                raw_col: pd.to_numeric(plot_df[raw_col], errors="coerce").mean()
                for raw_col in set(metric_value_columns.values())
                if raw_col is not None and raw_col in plot_df.columns
            }
        )
        avg_z_scores = pd.Series(
            {
                z_col: pd.to_numeric(plot_df[z_col], errors="coerce").mean()
                if z_col in plot_df.columns
                else float("nan")
                for z_col in metric_cols
            }
        )

        average_point = avg_z_scores.copy()
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))
            metric_label = metric_labels.get(metric, format_metric(metric))
            avg_raw_value = avg_raw_metrics.get(raw_metric, float("nan"))
            avg_z_value = average_point.get(metric, float("nan"))

            if raw_metric is not None:
                average_point[raw_metric] = avg_raw_value

            average_point[f"{metric}{hover_payload_suffix}"] = [
                metric_label,
                avg_raw_value,
                avg_z_value,
                cls._format_hover_line(
                    metric,
                    raw_metric,
                    metric_label,
                    avg_raw_value,
                    avg_z_value,
                ),
            ]

        average_point[display_name_col] = display_name_value
        return average_point

    @classmethod
    def _resolve_metric_config(
        cls,
        *,
        default_metrics: Sequence[str],
        default_metric_labels: Mapping[str, str],
        default_metric_value_columns: Mapping[str, Optional[str]],
        configured_metrics: Optional[Sequence[str]] = None,
        configured_metric_labels: Optional[Mapping[str, str]] = None,
        configured_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        override_metrics: Optional[Sequence[str]] = None,
        override_metric_labels: Optional[Mapping[str, str]] = None,
        override_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve the metric configuration for one plotting call.

        Precedence:
        1. Method-level overrides (`override_*`)
        2. Constructor-level settings (`configured_*`)
        3. Class defaults (`default_*`)

        Returns only the mappings required by the selected metrics.
        """
        chosen_metrics = override_metrics
        if chosen_metrics is None:
            chosen_metrics = configured_metrics
        if chosen_metrics is None:
            chosen_metrics = default_metrics

        ordered_metrics: List[str] = []
        for metric in chosen_metrics:
            metric_str = str(metric)
            if metric_str not in ordered_metrics:
                ordered_metrics.append(metric_str)
        if not ordered_metrics:
            raise ValueError("At least one metric must be provided for the distribution plot.")

        resolved_labels = dict(default_metric_labels)
        if configured_metric_labels:
            resolved_labels.update(dict(configured_metric_labels))
        if override_metric_labels:
            resolved_labels.update(dict(override_metric_labels))
        metric_labels = {
            metric: resolved_labels.get(metric, format_metric(metric))
            for metric in ordered_metrics
        }

        resolved_value_columns: Dict[str, Optional[str]] = dict(default_metric_value_columns)
        if configured_metric_value_columns:
            resolved_value_columns.update(dict(configured_metric_value_columns))
        if override_metric_value_columns:
            resolved_value_columns.update(dict(override_metric_value_columns))
        metric_value_columns = {
            metric: resolved_value_columns.get(metric, metric.replace("z_", "", 1))
            for metric in ordered_metrics
        }

        return ordered_metrics, metric_labels, metric_value_columns

    @staticmethod
    def _show_figures_side_by_side(
        figures: Sequence[go.Figure],
        *,
        column_gap: str = "16px",
        min_column_width_px: int = 520,
    ) -> bool:
        """
        Display Plotly figures side-by-side in notebook environments.

        Returns `True` when HTML notebook rendering succeeds. Callers can fall back to sequential `.show()` rendering when `False` is returned.
        """
        if not figures:
            return False

        try:
            from IPython.display import HTML, display  # type: ignore
        except Exception:
            return False

        try:
            blocks = []
            for idx, fig in enumerate(figures):
                include_plotlyjs = "cdn" if idx == 0 else False
                fig_html = pio.to_html(
                    fig,
                    include_plotlyjs=include_plotlyjs,
                    full_html=False,
                )
                blocks.append(
                    (
                        "<div style='flex:1 1 "
                        f"{min_column_width_px}px;min-width:{min_column_width_px}px'>"
                        f"{fig_html}</div>"
                    )
                )

            container_html = (
                "<div style='display:flex;flex-wrap:wrap;"
                f"gap:{column_gap};align-items:flex-start'>{''.join(blocks)}</div>"
            )
            display(HTML(container_html))
            return True
        except Exception:
            return False


class Single_CB_Aerial_Duels_Distribution_Plot(_Aerial_Duels_Distribution_Resolver):
    """
    Build Aerial-Duels distribution plots for:
    1. One selected CB against the full cohort
    2. Multiple selected CBs in comparison mode

    The class ships with sensible defaults for metrics, labels, and raw-value mappings. Users can still override configuration at construction time or per plotting call.
    """

    DEFAULT_METRICS = [
        "z_duels_per90",
        "z_card_discipline",
        "z_discipline",
        "z_aerial_wins_per90",
        "z_duel_success_rate",
        "z_possession_win_rate",
        "CB_aerial_duels_quality_z_score",
        "CB_rank_for_aerial_duels_quality_z_score"
    ]


    DEFAULT_METRIC_LABELS = {
        "z_duels_per90": "Aerial Duels per-90",
        "z_card_discipline": "Card Discipline",
        "z_discipline": "(Overall) Discipline",
        "z_aerial_wins_per90": "Aerial Wins per-90",
        "z_duel_success_rate": "Aerial Duel Success Rate",
        "z_possession_win_rate": "Possession Win Rate",
        "CB_aerial_duels_quality_z_score": "CB's (Overall) Aerial Duels Quality Score",
        "CB_rank_for_aerial_duels_quality_z_score": "CB's (Aerial Duels Quality) Ranking",
        "CB_rank_for_aerial_duels_quality": "CB's (Aerial Duels Quality) Ranking"
    }


    DEFAULT_METRIC_VALUE_COLUMNS = {
        "z_duels_per90": "duels_per90",
        "z_card_discipline": "card_discipline",
        "z_discipline": "discipline",
        "z_aerial_wins_per90": "aerial_wins_per90",
        "z_duel_success_rate": "duel_success_rate",
        "z_possession_win_rate": "possession_win_rate",
        "CB_aerial_duels_quality_z_score": None,  # No separate raw column for the overall aerial duel quality z-score, as it's a composite metric derived from the z-scores of the individual quality metrics, so we just use the z-score column for both the value and the label in this case - no raw value or label in the plot's tooltip.
        "CB_rank_for_aerial_duels_quality_z_score": "CB_rank_for_aerial_duels_quality"   # we can show the raw rank value in the hover tooltip for context, even though the x-axis position is based on the z-scored rank (where higher is better fit)
    }


    RANK_Z_SCORE_SCALE = 2.0
    RAW_RANK_COLUMN = "CB_rank_for_aerial_duels_quality"


    def __init__(
        self,
        *,
        duel_summary: pd.DataFrame,
        z_scores: pd.DataFrame,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, str]] = None,
        min_num_duels_involved_in_threshold: Any,
        min_minutes_played_threshold: Any,
    ) -> None:
        """
        Args:
            duel_summary: Raw single-CB metric dataframe.
            z_scores: Z-scored single-CB metric dataframe.
            metrics: Optional constructor-level metric list.
            metric_labels: Optional constructor-level metric label map.
            metric_value_columns: Optional constructor-level raw-value column map.
            min_num_duels_involved_in_threshold: Threshold shown in subtitles.
            min_minutes_played_threshold: Threshold shown in subtitles.
        """
        self.duel_summary = duel_summary
        self.z_scores = z_scores
        self._configured_metrics = list(metrics) if metrics is not None else None
        self._configured_metric_labels = (
            dict(metric_labels) if metric_labels is not None else None
        )
        self._configured_metric_value_columns = (
            dict(metric_value_columns) if metric_value_columns is not None else None
        )
        self.min_num_duels_involved_in_threshold = min_num_duels_involved_in_threshold
        self.min_minutes_played_threshold = min_minutes_played_threshold

    def _resolve_current_metric_config(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve call-ready metric configuration with precedence:
        method override > constructor config > class defaults.
        """
        return self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_METRIC_VALUE_COLUMNS,
            configured_metrics=self._configured_metrics,
            configured_metric_labels=self._configured_metric_labels,
            configured_metric_value_columns=self._configured_metric_value_columns,
            override_metrics=metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )

    def _build_plot_df(
        self,
        *,
        metrics: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Build the merged plotting dataframe and attach hover payload columns.
        """
        z_scores_df = self.z_scores.reset_index()
        self._ensure_columns(
            z_scores_df,
            ["player.id", "player.name"],
            "single-CB z-score dataframe",
        )

        raw_metric_columns = [
            col for col in dict.fromkeys(metric_value_columns.values()) if col is not None
        ]
        duel_summary_df = self.duel_summary.reset_index()
        always_include_raw_columns: List[str] = []
        if self.RAW_RANK_COLUMN in duel_summary_df.columns:
            always_include_raw_columns.append(self.RAW_RANK_COLUMN)
        fallback_metric_columns = [
            metric
            for metric in metrics
            if metric not in z_scores_df.columns and metric in duel_summary_df.columns
        ]
        required_duel_summary_columns = list(
            dict.fromkeys(
                raw_metric_columns + fallback_metric_columns + always_include_raw_columns
            )
        )
        self._ensure_columns(
            duel_summary_df,
            ["player.id", "player.name"] + required_duel_summary_columns,
            "single-CB raw metrics dataframe",
        )

        raw_metrics_df = duel_summary_df[
            ["player.id", "player.name"] + required_duel_summary_columns
        ].copy()
        plot_df = z_scores_df.merge(
            raw_metrics_df,
            on=["player.id", "player.name"],
            how="left",
            validate="one_to_one",
        )

        # Spread rank-based points for readability while keeping raw rank values in labels/tooltips via `metric_value_columns`.
        rank_metric_col = "CB_rank_for_aerial_duels_quality_z_score"
        if rank_metric_col in plot_df.columns:
            plot_df[rank_metric_col] = (
                pd.to_numeric(plot_df[rank_metric_col], errors="coerce")
                * self.RANK_Z_SCORE_SCALE
            )

        return self._attach_hover_payload(
            plot_df,
            metric_cols=metrics,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=self.RANK_Z_SCORE_SCALE,
        )

    def _create_plot(
        self,
        *,
        metrics: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        subtitle_style: str = "single",
    ) -> DistributionPlot:
        """
        Instantiate and title the distribution chart with the resolved config.
        """
        if subtitle_style == "comparison":
            subtitle = (
                f"Based on {len(self.z_scores)} CB players with ≥ {self.min_num_duels_involved_in_threshold} duels "
                f"& playing time ≥ {self.min_minutes_played_threshold} minutes"
            )
        else:
            subtitle = (
                f"Based on {len(self.z_scores)} CB players with ≥ {self.min_num_duels_involved_in_threshold} duels "
                f"& playing time ≥ {self.min_minutes_played_threshold} minutes"
            )

        dist_plot = DistributionPlot(
            columns=list(metrics),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        dist_plot.add_title(
            title="CB Aerial Duel Quality Distribution",
            subtitle=subtitle,
        )
        return dist_plot

    def _resolve_single_cb_by_rank(
        self,
        plot_df: pd.DataFrame,
        *,
        CB_Rank: Any,
    ) -> pd.Series:
        """
        Resolve one CB row by raw Aerial-Duels-quality ranking value.

        Args:
            plot_df: Prepared plotting dataframe.
            CB_Rank: Requested rank in the raw ranking column.

        Returns:
            One resolved row for the requested rank.
        """
        return self._resolve_row_by_rank(
            plot_df,
            rank_column=self.RAW_RANK_COLUMN,
            rank_value=CB_Rank,
            entity_label="CB player",
            display_column="player.name",
        )

    def _resolve_single_cb(
        self,
        plot_df: pd.DataFrame,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Any = None,
    ) -> pd.Series:
        """
        Resolve a single CB row from the prepared single-CB plotting dataframe.

        If neither `CB`, `CB_ID`, nor `CB_Rank` is provided, the first row is selected as a
        deterministic fallback to keep quick exploratory plotting simple.

        Args:
            plot_df: Prepared plotting dataframe containing `player.id/name`.
            CB: Optional player name/surname selector.
            CB_ID: Optional player ID selector.
            CB_Rank: Optional raw ranking selector.

        Returns:
            One resolved plotting row for the selected CB.
        """
        candidates = plot_df[["player.id", "player.name"]].drop_duplicates()
        if CB is None and CB_ID is None and CB_Rank is None:
            return plot_df.iloc[0]

        if CB_Rank is not None:
            if CB is not None or CB_ID is not None:
                raise ValueError(
                    "Please provide either CB_Rank or CB/CB_ID, not both."
                )
            return self._resolve_single_cb_by_rank(plot_df, CB_Rank=CB_Rank)

        resolved = self._resolve_entity(
            candidates,
            entity_value=CB,
            entity_id=CB_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        selected_rows = plot_df[plot_df["player.id"] == resolved["player.id"]]
        if selected_rows.empty:
            raise ValueError(f"Resolved CB '{resolved['player.name']}' is not available in plotting dataframe.")
        return selected_rows.iloc[0]

    def Plot_Single_CB(
        self,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Optional[int] = None,
        include_league_average: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one selected CB against the full single-CB Aerial-Duels distribution.

        Selection options:
        - `CB` (name/surname),
        - `CB_ID`,
        - `CB_Rank` (raw quality ranking value).
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        plot_df = self._build_plot_df(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
            subtitle_style="single",
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["player.name"],
            legend="All players",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        selected_cb = self._resolve_single_cb(
            plot_df,
            CB=CB,
            CB_ID=CB_ID,
            CB_Rank=CB_Rank,
        )
        dist_plot.add_data_point(
            ser_plot=selected_cb,
            plots="",
            name=selected_cb["player.name"],
            hover=hover_payload_suffix,
            hover_string=hover_string,
            add_annotations=True,
        )

        if include_league_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="player.name",
                display_name_value="CBs' League Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["player.id"] = -1
            dist_plot.add_data_point(
                ser_plot=average_point,
                plots="",
                name="CBs' League Average",
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=False,
            )

        if show:
            dist_plot.fig.show()
        return dist_plot

    def Plot_CBs_Comparison(
        self,
        *,
        CBs: Optional[Sequence[Any]] = None,
        CB_IDs: Optional[Sequence[Any]] = None,
        CB_Rank: Optional[int] = None,
        CB_Ranks: Optional[Sequence[int]] = None,
        include_league_average: bool = True,
        multi_annotations: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one or more selected CBs in comparison mode.

        CBs can be selected by:
        - `CBs` (names/surnames),
        - `CB_IDs`,
        - `CB_Rank` / `CB_Ranks` (raw quality ranking values).
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        plot_df = self._build_plot_df(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
            subtitle_style="comparison",
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["player.name"],
            legend="All players",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        requested_items: List[Tuple[str, Any, Any]] = []
        for cb_id in (CB_IDs or []):
            requested_items.append(("id", None, cb_id))
        for cb_name in (CBs or []):
            requested_items.append(("name", cb_name, None))

        # Rank selectors are optional and additive in comparison mode.
        # They can be used alone or mixed with explicit name/ID selectors.
        if CB_Ranks is not None:
            if CB_Rank is not None:
                raise ValueError(
                    "Please provide either CB_Rank or CB_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Ranks,
                param_name="CB_Ranks",
            ):
                requested_items.append(("rank", rank_value, None))
        if CB_Rank is not None:
            requested_items.append(
                (
                    "rank",
                    self._coerce_positive_rank(CB_Rank, param_name="CB_Rank"),
                    None,
                )
            )

        if not requested_items:
            raise ValueError(
                "Please provide at least one CB selector via CBs, CB_IDs, CB_Rank, or CB_Ranks."
            )

        # Deduplicate by player ID so overlapping selectors (e.g., name + rank
        # pointing to the same CB) do not create duplicate highlighted traces.
        used_ids = set()
        for selection_mode, cb_name_or_rank, cb_id in requested_items:
            if selection_mode == "rank":
                selected_cb = self._resolve_single_cb(
                    plot_df,
                    CB_Rank=cb_name_or_rank,
                )
            else:
                selected_cb = self._resolve_single_cb(
                    plot_df,
                    CB=cb_name_or_rank,
                    CB_ID=cb_id,
                )
            cb_unique_id = selected_cb["player.id"]
            if cb_unique_id in used_ids:
                continue
            used_ids.add(cb_unique_id)

            if multi_annotations:
                dist_plot.add_data_point(
                    ser_plot=selected_cb,
                    plots="",
                    name=selected_cb["player.name"],
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_single_annotations=False,
                    add_multi_annotations=True,
                )
            else:
                dist_plot.add_data_point(
                    ser_plot=selected_cb,
                    plots="",
                    name=selected_cb["player.name"],
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_annotations=True,
                )

        if include_league_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="player.name",
                display_name_value="CBs' League Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["player.id"] = -1
            if multi_annotations:
                dist_plot.add_data_point(
                    ser_plot=average_point,
                    plots="",
                    name="CBs' League Average",
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_single_annotations=False,
                    add_multi_annotations=True,
                )
            else:
                dist_plot.add_data_point(
                    ser_plot=average_point,
                    plots="",
                    name="CBs' League Average",
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_annotations=False,
                )

        if show:
            dist_plot.fig.show()
        return dist_plot


class CB_Pair_Aerial_Duels_Distribution_Plot(_Aerial_Duels_Distribution_Resolver):
    """
    Build Aerial-Duels distribution plots for CB-pair analyses.

    By default, CB-pair plots are split into two side-by-side figures:
    1. Fit/composition metrics (left)
    2. Individual-quality metrics (right)

    Set `split_view=False` in plotting methods to render the legacy single-figure
    layout.
    """

    DEFAULT_PAIR_METRIC_LABELS = {
        "z_duels_per90": "Aerial Duels per-90",
        "z_card_discipline": "Card Discipline",
        "z_discipline": "Discipline",
        "z_aerial_wins_per90": "Aerial Wins per-90",
        "z_duel_success_rate": "Duel Success Rate",
        "z_possession_win_rate": "Possession Win Rate",
        "floor_z": "CB-Pair Weak-Link (i.e. Floor) Protection",
        "complement_z": "CB-Pair Complementarity (i.e. Deficit-Coverage Gain)",
        "quality_z": "CB-Pair Weighted Avg. Quality",
        "CB_pair_fit_z_score": "CB-Pair (Overall) Fit Score",
        "CB_pair_fit_rank_z_score": "CB-Pair Fit Ranking (Within This Sample)",
        "CB_pair_fit_rank": "CB-Pair Fit Ranking (Within This Sample)"
    }


    DEFAULT_PAIR_METRIC_VALUE_COLUMNS = {
        "z_duels_per90": "duels_per90",
        "z_card_discipline": "card_discipline",
        "z_discipline": "discipline",
        "z_aerial_wins_per90": "aerial_wins_per90",
        "z_duel_success_rate": "duel_success_rate",
        "z_possession_win_rate": "possession_win_rate",
        "floor_z": "floor_raw",
        "complement_z": "complement_raw",
        "quality_z": "quality_raw",
        "CB_pair_fit_z_score": None,  # No separate raw column for the CB-pair overall fit score, as it's a composite metric derived from the quality_z, complement_z, and floor_z components, so we just use the z-score column for both the value and the label in this case - no raw value or label in the plot's tooltip.
        "CB_pair_fit_rank_z_score": "CB_pair_fit_rank"    # we can show the raw rank value in the hover tooltip for context, even though the x-axis position is based on the z-scored rank (where higher is better fit)
    }


    DEFAULT_LEFT_METRICS = [
        "floor_z",
        "complement_z",
        "quality_z",
        "CB_pair_fit_z_score",
        "CB_pair_fit_rank_z_score",
    ]


    DEFAULT_RIGHT_METRICS = [
        "z_duels_per90",
        "z_card_discipline",
        "z_discipline",
        "z_aerial_wins_per90",
        "z_duel_success_rate",
        "z_possession_win_rate",
    ]

    DEFAULT_METRICS = DEFAULT_RIGHT_METRICS + DEFAULT_LEFT_METRICS


    RANK_Z_SCALE = 2.5
    RAW_RANK_COLUMN = "CB_pair_fit_rank"


    def __init__(
        self,
        *,
        df_aerial_duel_pairs: Optional[pd.DataFrame] = None,
        df_ground_duel_pairs: Optional[pd.DataFrame] = None,
        plot_metric_cols: Optional[Sequence[str]] = None,
        pair_metric_labels: Optional[Mapping[str, str]] = None,
        pair_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        min_num_duels_involved_in_threshold: Any,
        min_minutes_played_threshold: Any,
    ) -> None:
        """
        Args:
            df_aerial_duel_pairs: Canonical Aerial-Duels CB-pair dataframe used for plotting.
            df_ground_duel_pairs: Backward-compatible alias for legacy call sites.
            plot_metric_cols: Optional legacy single-view metric list.
            pair_metric_labels: Optional constructor-level label map.
            pair_metric_value_columns: Optional constructor-level raw-value map.
            left_metrics: Optional constructor-level left split metrics.
            right_metrics: Optional constructor-level right split metrics.
            min_num_duels_involved_in_threshold: Threshold shown in subtitles.
            min_minutes_played_threshold: Threshold shown in subtitles.
        """
        if df_aerial_duel_pairs is None and df_ground_duel_pairs is None:
            raise ValueError(
                "Please provide `df_aerial_duel_pairs` (preferred) or "
                "`df_ground_duel_pairs` (legacy alias)."
            )
        if df_aerial_duel_pairs is not None and df_ground_duel_pairs is not None:
            raise ValueError(
                "Provide only one of `df_aerial_duel_pairs` or `df_ground_duel_pairs`, not both."
            )

        resolved_pair_df = (
            df_aerial_duel_pairs
            if df_aerial_duel_pairs is not None
            else df_ground_duel_pairs
        )
        self.df_aerial_duel_pairs = resolved_pair_df
        # Legacy alias kept for backward compatibility with any external call
        # sites that still reference the old attribute name.
        self.df_ground_duel_pairs = resolved_pair_df
        self._configured_plot_metric_cols = (
            list(plot_metric_cols) if plot_metric_cols is not None else None
        )
        self._configured_pair_metric_labels = (
            dict(pair_metric_labels) if pair_metric_labels is not None else None
        )
        self._configured_pair_metric_value_columns = (
            dict(pair_metric_value_columns)
            if pair_metric_value_columns is not None
            else None
        )
        self._configured_left_metrics = (
            list(left_metrics) if left_metrics is not None else None
        )
        self._configured_right_metrics = (
            list(right_metrics) if right_metrics is not None else None
        )
        self.min_num_duels_involved_in_threshold = min_num_duels_involved_in_threshold
        self.min_minutes_played_threshold = min_minutes_played_threshold

    def _resolve_pair_metric_configs(
        self,
        *,
        split_view: bool,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Dict[str, Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]]:
        """
        Resolve split or single metric configs with method > constructor > default precedence.
        """
        constructor_labels = self._configured_pair_metric_labels
        constructor_value_columns = self._configured_pair_metric_value_columns

        if split_view:
            # Split mode resolves each side independently, then builds a combined
            # superset config used for shared data preparation/hover payloads.
            configured_left = self._configured_left_metrics
            configured_right = self._configured_right_metrics

            if configured_left is None and configured_right is None and self._configured_plot_metric_cols is not None:
                ordered = list(self._configured_plot_metric_cols)
                derived_left = [m for m in ordered if m in self.DEFAULT_LEFT_METRICS]
                derived_right = [m for m in ordered if m not in derived_left]
                configured_left = derived_left or list(self.DEFAULT_LEFT_METRICS)
                configured_right = derived_right or list(self.DEFAULT_RIGHT_METRICS)

            left_cfg = self._resolve_metric_config(
                default_metrics=self.DEFAULT_LEFT_METRICS,
                default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
                default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
                configured_metrics=configured_left,
                configured_metric_labels=constructor_labels,
                configured_metric_value_columns=constructor_value_columns,
                override_metrics=left_metrics,
                override_metric_labels=metric_labels,
                override_metric_value_columns=metric_value_columns,
            )
            right_cfg = self._resolve_metric_config(
                default_metrics=self.DEFAULT_RIGHT_METRICS,
                default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
                default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
                configured_metrics=configured_right,
                configured_metric_labels=constructor_labels,
                configured_metric_value_columns=constructor_value_columns,
                override_metrics=right_metrics,
                override_metric_labels=metric_labels,
                override_metric_value_columns=metric_value_columns,
            )

            combined_metrics: List[str] = []
            for metric in left_cfg[0] + right_cfg[0]:
                if metric not in combined_metrics:
                    combined_metrics.append(metric)

            combined_labels = dict(left_cfg[1])
            combined_labels.update(right_cfg[1])
            combined_value_columns = dict(left_cfg[2])
            combined_value_columns.update(right_cfg[2])
            combined_cfg = (combined_metrics, combined_labels, combined_value_columns)

            return {
                "left": left_cfg,
                "right": right_cfg,
                "combined": combined_cfg,
            }

        override_metrics: Optional[List[str]] = None
        if left_metrics is not None or right_metrics is not None:
            override_metrics = []
            for metric in list(left_metrics or []) + list(right_metrics or []):
                metric_str = str(metric)
                if metric_str not in override_metrics:
                    override_metrics.append(metric_str)

        configured_single_metrics = self._configured_plot_metric_cols
        if configured_single_metrics is None and (
            self._configured_left_metrics is not None or self._configured_right_metrics is not None
        ):
            configured_single_metrics = []
            for metric in list(self._configured_left_metrics or []) + list(self._configured_right_metrics or []):
                metric_str = str(metric)
                if metric_str not in configured_single_metrics:
                    configured_single_metrics.append(metric_str)

        single_cfg = self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
            configured_metrics=configured_single_metrics,
            configured_metric_labels=constructor_labels,
            configured_metric_value_columns=constructor_value_columns,
            override_metrics=override_metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )
        return {"single": single_cfg, "combined": single_cfg}

    def _prepare_plot_df(
        self,
        *,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Prepare the CB-pair dataframe and inject hover payload columns.
        """
        plot_df = self.df_aerial_duel_pairs.copy()
        self._ensure_columns(
            plot_df,
            ["pair_name", "CB_pair_fit_z_score"],
            "CB-pair plotting dataframe",
        )

        # Always scale rank-derived z-scores with the CB-pair-specific spread factor
        # so ranking rows are readable even when the source dataframe already
        # contains a precomputed rank z-score column.
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))
            if (
                raw_metric is not None
                and self._is_rank_metric(metric, raw_metric)
                and raw_metric in plot_df.columns
            ):
                plot_df[metric] = self._z_standardize(
                    plot_df[raw_metric],
                    invert=True,
                    scale=self.RANK_Z_SCALE,
                )

        return self._attach_hover_payload(
            plot_df,
            metric_cols=metric_cols,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=self.RANK_Z_SCALE,
        )

    def _create_plot(
        self,
        *,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        section_title: Optional[str] = None,
    ) -> DistributionPlot:
        """
        Instantiate and title one CB-pair distribution figure.
        """
        dist_plot = DistributionPlot(
            columns=list(metric_cols),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        title = "CB-Pair Aerial Duels Quality Fit Distribution"
        if section_title:
            title = f"{title} <br> ⇒  {section_title}"
        dist_plot.add_title(
            title=title,
            subtitle=(
                f"All {len(self.df_aerial_duel_pairs)} unordered CB pairs, with ≥ {self.min_num_duels_involved_in_threshold} duels & playing time ≥ {self.min_minutes_played_threshold} minutes (each individual CB)"
                "<br>"
                "All metrics are CB-Pair-level (i.e. the weighted average of both players for that metric)"
            ),
        )
        return dist_plot

    def _player_candidates_from_pairs(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a normalized unique player-candidate table from pair columns.

        Args:
            plot_df: CB-pair dataframe containing CB1/CB2 ID and name columns.

        Returns:
            Two-column dataframe (`player.id`, `player.name`) with unique players.
        """
        self._ensure_columns(
            plot_df,
            ["player.id_CB1", "player.name_CB1", "player.id_CB2", "player.name_CB2"],
            "CB-pair player resolution",
        )
        cb1 = plot_df[["player.id_CB1", "player.name_CB1"]].rename(
            columns={"player.id_CB1": "player.id", "player.name_CB1": "player.name"}
        )
        cb2 = plot_df[["player.id_CB2", "player.name_CB2"]].rename(
            columns={"player.id_CB2": "player.id", "player.name_CB2": "player.name"}
        )
        return pd.concat([cb1, cb2], ignore_index=True).drop_duplicates()

    def _resolve_pair_row(self, plot_df: pd.DataFrame, *, CB_Pair: Any = None, CB_1: Any = None, CB_2: Any = None, CB_1_ID: Any = None, CB_2_ID: Any = None) -> Optional[pd.Series]:
        """
        Resolve one pair row from flexible pair selectors.

        Supported selection modes:
        - `CB_Pair` string formats (`A + B`, `A and B`, `A & B`),
        - `CB_Pair` tuple/list `(CB_1, CB_2)`,
        - dict-based specs with `CB_1`/`CB_2` and optional IDs,
        - explicit `CB_1`/`CB_2` + optional IDs.

        Returns:
            Matching pair row, or `None` when no selector was provided.
        """
        if (
            CB_Pair is None
            and CB_1 is None
            and CB_2 is None
            and CB_1_ID is None
            and CB_2_ID is None
        ):
            return None

        if CB_Pair is not None:
            if isinstance(CB_Pair, str):
                CB_1, CB_2 = self._split_pair_string(CB_Pair)
            elif isinstance(CB_Pair, (list, tuple)) and len(CB_Pair) == 2:
                CB_1, CB_2 = CB_Pair[0], CB_Pair[1]
            elif isinstance(CB_Pair, dict):
                CB_1 = CB_Pair.get("CB_1", CB_1)
                CB_2 = CB_Pair.get("CB_2", CB_2)
                CB_1_ID = CB_Pair.get("CB_1_ID", CB_1_ID)
                CB_2_ID = CB_Pair.get("CB_2_ID", CB_2_ID)
                if CB_Pair.get("CB_Pair") is not None:
                    CB_1, CB_2 = self._split_pair_string(CB_Pair["CB_Pair"])
            else:
                raise ValueError(
                    "CB_Pair must be a string ('A + B', 'A and B', 'A & B'), "
                    "a tuple/list of two CB inputs, or a dict with CB_1/CB_2 keys."
                )

        if (CB_1 is None and CB_1_ID is None) or (CB_2 is None and CB_2_ID is None):
            raise ValueError("Please provide both CB_1 and CB_2 (name/surname and/or ID).")

        candidates = self._player_candidates_from_pairs(plot_df)
        resolved_cb1 = self._resolve_entity(
            candidates,
            entity_value=CB_1,
            entity_id=CB_1_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        resolved_cb2 = self._resolve_entity(
            candidates,
            entity_value=CB_2,
            entity_id=CB_2_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )

        id_1 = resolved_cb1["player.id"]
        id_2 = resolved_cb2["player.id"]
        pair_mask = (
            ((plot_df["player.id_CB1"] == id_1) & (plot_df["player.id_CB2"] == id_2))
            | ((plot_df["player.id_CB1"] == id_2) & (plot_df["player.id_CB2"] == id_1))
        )
        matches = plot_df[pair_mask]
        if matches.empty:
            raise ValueError(
                f"No CB pair found for '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple rows found for pair '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        return matches.iloc[0]

    def _resolve_pair_row_by_rank(
        self,
        plot_df: pd.DataFrame,
        *,
        CB_Pair_Rank: Any,
    ) -> pd.Series:
        """
        Resolve one CB-pair row by its raw fit-ranking value.

        Args:
            plot_df: Prepared CB-pair plotting dataframe.
            CB_Pair_Rank: Requested pair rank in `CB_pair_fit_rank`.

        Returns:
            One resolved pair row for the requested rank.
        """
        return self._resolve_row_by_rank(
            plot_df,
            rank_column=self.RAW_RANK_COLUMN,
            rank_value=CB_Pair_Rank,
            entity_label="CB pair",
            display_column="pair_name",
        )

    def _pair_key(self, row: pd.Series) -> str:
        """
        Build a stable deduplication key for a pair-like row.

        Prefers explicit `pair_key` when available, then falls back to `pair_name`.
        """
        if "pair_key" in row.index and pd.notna(row["pair_key"]):
            return str(row["pair_key"])
        return str(row.get("pair_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
        """
        Add one highlighted pair row to a distribution plot.

        This wrapper keeps annotation mode wiring centralized so callers can
        switch between single-entity and multi-entity annotation rendering
        without duplicating plotting code.
        """
        if multi_annotations:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_single_annotations=False,
                add_multi_annotations=True,
            )
        else:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=add_single_annotations,
            )

    def _build_pair_plot(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        highlighted_rows: Sequence[Tuple[pd.Series, str, bool]],
        multi_annotations: bool,
        section_title: Optional[str] = None,
    ) -> DistributionPlot:
        """
        Build one CB-pair figure from pre-resolved highlighted rows.
        """
        dist_plot = self._create_plot(
            metric_cols=metric_cols,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
            section_title=section_title,
        )
        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["pair_name"],
            legend="All pairs",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        for row, display_name, add_single_annotations in highlighted_rows:
            self._add_row_point(
                dist_plot,
                row=row,
                display_name=display_name,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=add_single_annotations,
            )

        return dist_plot

    def _collect_single_pair_highlights(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
        CB_Pair_Rank: Optional[int] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for a single CB-pair plot scenario.

        A pair can be selected either by identity selectors (`CB_Pair` or
        `CB_1`/`CB_2`) or by `CB_Pair_Rank`.
        """
        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()

        # In single-pair mode, rank-based selection and identity-based selection
        # are intentionally mutually exclusive to keep intent unambiguous.
        if CB_Pair_Rank is not None and any(
            value is not None for value in [CB_Pair, CB_1, CB_2, CB_1_ID, CB_2_ID]
        ):
            raise ValueError(
                "Please provide either CB_Pair_Rank or CB_Pair/CB_1/CB_2 selectors, not both."
            )

        if CB_Pair_Rank is not None:
            selected_pair = self._resolve_pair_row_by_rank(
                plot_df,
                CB_Pair_Rank=CB_Pair_Rank,
            )
        else:
            selected_pair = self._resolve_pair_row(
                plot_df,
                CB_Pair=CB_Pair,
                CB_1=CB_1,
                CB_2=CB_2,
                CB_1_ID=CB_1_ID,
                CB_2_ID=CB_2_ID,
            )
        if selected_pair is not None:
            selected_key = self._pair_key(selected_pair)
            used_keys.add(selected_key)
            highlighted_rows.append((selected_pair, selected_pair["pair_name"], True))

        if include_best:
            best_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=False).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                highlighted_rows.append((best_pair, best_pair["pair_name"], True))

        if include_worst:
            worst_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=True).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                highlighted_rows.append((worst_pair, worst_pair["pair_name"], True))

        if include_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=metric_cols,
                metric_labels=dict(metric_labels),
                metric_value_columns=dict(metric_value_columns),
                display_name_col="pair_name",
                display_name_value="CB-Pairs' League Average",
                hover_payload_suffix="_hover_payload",
            )
            average_point["pair_key"] = "__pair_average__"
            highlighted_rows.append((average_point, "CB-Pairs' League Average", False))

        return highlighted_rows

    def _collect_comparison_pair_highlights(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        CB_Pairs: Optional[Sequence[Union[str, Sequence[Any], Dict[str, Any]]]] = None,
        CB_Pair_Rank: Optional[int] = None,
        CB_Pair_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for CB-pair comparison plots.

        Comparison selectors can include explicit pair identities and/or raw
        ranking selectors (`CB_Pair_Rank`, `CB_Pair_Ranks`).
        """
        specs: List[Dict[str, Any]] = []
        for item in (CB_Pairs or []):
            if isinstance(item, str):
                specs.append({"CB_Pair": item})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                specs.append({"CB_1": item[0], "CB_2": item[1]})
            elif isinstance(item, dict):
                specs.append(dict(item))
            else:
                raise ValueError(
                    "Each item in CB_Pairs must be either a pair string, a tuple/list of length 2, or a dict with CB_1/CB_2."
                )

        # Rank-based pair selection is additive in comparison mode.
        # It can be used alone or combined with explicit pair selectors.
        if CB_Pair_Ranks is not None:
            if CB_Pair_Rank is not None:
                raise ValueError(
                    "Please provide either CB_Pair_Rank or CB_Pair_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Pair_Ranks,
                param_name="CB_Pair_Ranks",
            ):
                specs.append({"CB_Pair_Rank": rank_value})
        if CB_Pair_Rank is not None:
            specs.append(
                {
                    "CB_Pair_Rank": self._coerce_positive_rank(
                        CB_Pair_Rank,
                        param_name="CB_Pair_Rank",
                    )
                }
            )

        if not specs and not any([include_best, include_worst, include_average]):
            raise ValueError(
                "Please provide at least one pair selector via CB_Pairs, CB_Pair_Rank, or CB_Pair_Ranks; "
                "or enable at least one of include_best/include_worst/include_average."
            )

        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()
        for spec in specs:
            if spec.get("CB_Pair_Rank") is not None:
                selected_pair = self._resolve_pair_row_by_rank(
                    plot_df,
                    CB_Pair_Rank=spec.get("CB_Pair_Rank"),
                )
            else:
                selected_pair = self._resolve_pair_row(
                    plot_df,
                    CB_Pair=spec.get("CB_Pair"),
                    CB_1=spec.get("CB_1"),
                    CB_2=spec.get("CB_2"),
                    CB_1_ID=spec.get("CB_1_ID"),
                    CB_2_ID=spec.get("CB_2_ID"),
                )
                if selected_pair is None:
                    continue

            key = self._pair_key(selected_pair)
            if key in used_keys:
                continue
            used_keys.add(key)
            highlighted_rows.append((selected_pair, selected_pair["pair_name"], True))

        if include_best:
            best_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=False).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                highlighted_rows.append((best_pair, best_pair["pair_name"], True))

        if include_worst:
            worst_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=True).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                highlighted_rows.append((worst_pair, worst_pair["pair_name"], True))

        if include_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=metric_cols,
                metric_labels=dict(metric_labels),
                metric_value_columns=dict(metric_value_columns),
                display_name_col="pair_name",
                display_name_value="CB-Pairs' League Average",
                hover_payload_suffix="_hover_payload",
            )
            average_point["pair_key"] = "__pair_average__"
            highlighted_rows.append((average_point, "CB-Pairs' League Average", False))

        return highlighted_rows

    def Plot_CB_Pair(
        self,
        *,
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
        CB_Pair_Rank: Optional[int] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
        multi_annotations: bool = False,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        split_view: bool = True,
        render_side_by_side: bool = True,
        show: bool = True,
    ) -> Union[DistributionPlot, Tuple[DistributionPlot, DistributionPlot]]:
        """
        Plot one selected CB-pair (optional) plus optional best/worst/average references.

        Selection options:
        - `CB_Pair` / `CB_1` + `CB_2` (with optional IDs),
        - `CB_Pair_Rank` (raw pair-fit ranking value).
        """
        cfg = self._resolve_pair_metric_configs(
            split_view=split_view,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
        )
        combined_metrics, combined_labels, combined_values = cfg["combined"]
        plot_df = self._prepare_plot_df(
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
        )

        highlighted_rows = self._collect_single_pair_highlights(
            plot_df=plot_df,
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
            CB_Pair=CB_Pair,
            CB_1=CB_1,
            CB_2=CB_2,
            CB_1_ID=CB_1_ID,
            CB_2_ID=CB_2_ID,
            CB_Pair_Rank=CB_Pair_Rank,
            include_best=include_best,
            include_worst=include_worst,
            include_average=include_average,
        )

        if split_view:
            left_metrics_cfg, left_labels_cfg, left_values_cfg = cfg["left"]
            right_metrics_cfg, right_labels_cfg, right_values_cfg = cfg["right"]
            left_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=left_metrics_cfg,
                metric_labels=left_labels_cfg,
                metric_value_columns=left_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Fit / Composition Metrics",
            )
            right_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=right_metrics_cfg,
                metric_labels=right_labels_cfg,
                metric_value_columns=right_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Individual Quality Metrics",
            )

            if show:
                rendered_side_by_side = False
                if render_side_by_side:
                    rendered_side_by_side = self._show_figures_side_by_side(
                        [left_plot.fig, right_plot.fig]
                    )
                if not rendered_side_by_side:
                    left_plot.fig.show()
                    right_plot.fig.show()
            return left_plot, right_plot

        single_metrics, single_labels, single_values = cfg["single"]
        single_plot = self._build_pair_plot(
            plot_df=plot_df,
            metric_cols=single_metrics,
            metric_labels=single_labels,
            metric_value_columns=single_values,
            highlighted_rows=highlighted_rows,
            multi_annotations=multi_annotations,
            section_title=None,
        )
        if show:
            single_plot.fig.show()
        return single_plot

    def Plot_CB_Pairs_Comparison(
        self,
        *,
        CB_Pairs: Optional[Sequence[Union[str, Sequence[Any], Dict[str, Any]]]] = None,
        CB_Pair_Rank: Optional[int] = None,
        CB_Pair_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
        multi_annotations: bool = True,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        split_view: bool = True,
        render_side_by_side: bool = True,
        show: bool = True,
    ) -> Union[DistributionPlot, Tuple[DistributionPlot, DistributionPlot]]:
        """
        Compare multiple user-selected CB-pairs plus optional best/worst/average overlays.

        Pair selection supports:
        - explicit pair selectors (`CB_Pairs`),
        - rank selectors (`CB_Pair_Rank`, `CB_Pair_Ranks`).
        """
        cfg = self._resolve_pair_metric_configs(
            split_view=split_view,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
        )
        combined_metrics, combined_labels, combined_values = cfg["combined"]
        plot_df = self._prepare_plot_df(
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
        )

        highlighted_rows = self._collect_comparison_pair_highlights(
            plot_df=plot_df,
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
            CB_Pairs=CB_Pairs,
            CB_Pair_Rank=CB_Pair_Rank,
            CB_Pair_Ranks=CB_Pair_Ranks,
            include_best=include_best,
            include_worst=include_worst,
            include_average=include_average,
        )

        if split_view:
            left_metrics_cfg, left_labels_cfg, left_values_cfg = cfg["left"]
            right_metrics_cfg, right_labels_cfg, right_values_cfg = cfg["right"]
            left_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=left_metrics_cfg,
                metric_labels=left_labels_cfg,
                metric_value_columns=left_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Fit / Composition Metrics",
            )
            right_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=right_metrics_cfg,
                metric_labels=right_labels_cfg,
                metric_value_columns=right_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Individual Quality Metrics",
            )

            if show:
                rendered_side_by_side = False
                if render_side_by_side:
                    rendered_side_by_side = self._show_figures_side_by_side(
                        [left_plot.fig, right_plot.fig]
                    )
                if not rendered_side_by_side:
                    left_plot.fig.show()
                    right_plot.fig.show()
            return left_plot, right_plot

        single_metrics, single_labels, single_values = cfg["single"]
        single_plot = self._build_pair_plot(
            plot_df=plot_df,
            metric_cols=single_metrics,
            metric_labels=single_labels,
            metric_value_columns=single_values,
            highlighted_rows=highlighted_rows,
            multi_annotations=multi_annotations,
            section_title=None,
        )
        if show:
            single_plot.fig.show()
        return single_plot


class Anchor_CB_Companion_Fit_Aerial_Duels_Distribution_Plot(_Aerial_Duels_Distribution_Resolver):
    """
    Build directional anchor-to-companion Aerial-Duels distribution plots.

    The class includes default companion-fit metrics/labels/raw mappings but still
    supports constructor-level and per-call metric overrides.
    """

    DEFAULT_METRICS = [
        "coverage_gain_z_within_anchor",
        "CB_pair_fit_z_score",
        "companion_fit_score_z_within_anchor",
        "companion_rank_for_anchor_z_score",
    ]
    DEFAULT_METRIC_LABELS = {
        "coverage_gain_z_within_anchor": "Deficit-Coverage Gain (Within Anchor CB)",
        "CB_pair_fit_z_score": "CB-Pair (Overall) Fit Score",
        "companion_fit_score_z_within_anchor": "CB-Companion Fit Score (Within Anchor CB)",
        "companion_rank_for_anchor_z_score": "CB-Companion Fit Ranking (Within Anchor CB's Sample)",
        "companion_rank_for_anchor": "CB-Companion Fit Ranking (Within Anchor CB's Sample)",
    }
    DEFAULT_METRIC_VALUE_COLUMNS = {
        "coverage_gain_z_within_anchor": "coverage_gain_raw",
        "CB_pair_fit_z_score": None,
        "companion_fit_score_z_within_anchor": "companion_fit_score",
        "companion_rank_for_anchor_z_score": "companion_rank_for_anchor",
    }

    def __init__(
        self,
        *,
        df_aerial_duel_companion_fit: Optional[pd.DataFrame] = None,
        df_ground_duel_companion_fit: Optional[pd.DataFrame] = None,
        companion_plot_metric_cols: Optional[Sequence[str]] = None,
        companion_metric_labels: Optional[Mapping[str, str]] = None,
        companion_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        """
        Args:
            df_aerial_duel_companion_fit: Canonical Aerial-Duels directional anchor-to-partner fit dataframe.
            df_ground_duel_companion_fit: Backward-compatible alias for legacy call sites.
            companion_plot_metric_cols: Optional constructor-level metric list.
            companion_metric_labels: Optional constructor-level label map.
            companion_metric_value_columns: Optional constructor-level raw-value map.
        """
        if (
            df_aerial_duel_companion_fit is None
            and df_ground_duel_companion_fit is None
        ):
            raise ValueError(
                "Please provide `df_aerial_duel_companion_fit` (preferred) or "
                "`df_ground_duel_companion_fit` (legacy alias)."
            )
        if (
            df_aerial_duel_companion_fit is not None
            and df_ground_duel_companion_fit is not None
        ):
            raise ValueError(
                "Provide only one of `df_aerial_duel_companion_fit` or "
                "`df_ground_duel_companion_fit`, not both."
            )

        resolved_companion_df = (
            df_aerial_duel_companion_fit
            if df_aerial_duel_companion_fit is not None
            else df_ground_duel_companion_fit
        )
        self.df_aerial_duel_companion_fit = resolved_companion_df
        # Legacy alias kept for backward compatibility with any external call
        # sites that still reference the old attribute name.
        self.df_ground_duel_companion_fit = resolved_companion_df
        self._configured_companion_plot_metric_cols = (
            list(companion_plot_metric_cols)
            if companion_plot_metric_cols is not None
            else None
        )
        self._configured_companion_metric_labels = (
            dict(companion_metric_labels)
            if companion_metric_labels is not None
            else None
        )
        self._configured_companion_metric_value_columns = (
            dict(companion_metric_value_columns)
            if companion_metric_value_columns is not None
            else None
        )
        self._anchor_player_id: Optional[Any] = None
        self._anchor_player_name: Optional[str] = None

    def _resolve_current_metric_config(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve call-ready metric config with precedence:
        method override > constructor config > class defaults.
        """
        return self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_METRIC_VALUE_COLUMNS,
            configured_metrics=self._configured_companion_plot_metric_cols,
            configured_metric_labels=self._configured_companion_metric_labels,
            configured_metric_value_columns=self._configured_companion_metric_value_columns,
            override_metrics=metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )

    def _anchor_candidates(self) -> pd.DataFrame:
        """
        Build candidate anchor table for anchor selection resolution.

        Returns:
            Dataframe with `anchor_player_id` and `anchor_player_name`.
            If IDs are unavailable in source data, synthetic IDs are generated.
        """
        self._ensure_columns(
            self.df_aerial_duel_companion_fit,
            ["anchor_player_name"],
            "companion-fit anchor resolution",
        )
        if "anchor_player_id" in self.df_aerial_duel_companion_fit.columns:
            return self.df_aerial_duel_companion_fit[
                ["anchor_player_id", "anchor_player_name"]
            ].drop_duplicates()

        candidates = self.df_aerial_duel_companion_fit[["anchor_player_name"]].drop_duplicates().copy()
        candidates["anchor_player_id"] = np.arange(len(candidates))
        return candidates

    def _resolve_anchor(self, *, Anchor_CB: Any = None, Anchor_CB_ID: Any = None) -> pd.Series:
        """
        Resolve the active anchor CB from name/surname and/or ID input.

        Behavior:
        - if no input is provided, reuses previously initialized anchor when
          possible; otherwise falls back to first alphabetical anchor.
        - if ID is provided but the dataset has no anchor ID column, raises.
        """
        candidates = self._anchor_candidates()

        if Anchor_CB is None and Anchor_CB_ID is None:
            if self._anchor_player_name is not None:
                cached_by_name = candidates[
                    candidates["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(self._anchor_player_name)
                ]
                if not cached_by_name.empty:
                    return cached_by_name.iloc[0]
            return candidates.sort_values("anchor_player_name").iloc[0]

        if (
            Anchor_CB_ID is not None
            and "anchor_player_id" not in self.df_aerial_duel_companion_fit.columns
        ):
            raise ValueError(
                "Anchor_CB_ID was provided, but `anchor_player_id` is not available in the companion dataframe."
            )

        return self._resolve_entity(
            candidates,
            entity_value=Anchor_CB,
            entity_id=Anchor_CB_ID,
            id_col="anchor_player_id",
            name_col="anchor_player_name",
            entity_label="anchor CB",
        )

    def Initialize_Desired_Anchor_CB(self, *, Anchor_CB: Any = None, Anchor_CB_ID: Any = None) -> pd.Series:
        """
        Resolve and cache the anchor CB used by subsequent companion plots.
        """
        resolved_anchor = self._resolve_anchor(Anchor_CB=Anchor_CB, Anchor_CB_ID=Anchor_CB_ID)
        self._anchor_player_id = resolved_anchor.get("anchor_player_id", None)
        self._anchor_player_name = str(resolved_anchor["anchor_player_name"])
        return resolved_anchor

    def _build_anchor_plot_df(
        self,
        *,
        anchor_name: str,
        anchor_id: Any = None,
        top_n: Optional[int] = None,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Build anchor-filtered companion-fit dataframe and attach hover payloads.
        """
        plot_df = self.df_aerial_duel_companion_fit.copy()
        self._ensure_columns(
            plot_df,
            ["anchor_player_name", "partner_player_name", "companion_fit_score"],
            "companion-fit plotting dataframe",
        )

        if anchor_id is not None and "anchor_player_id" in plot_df.columns:
            filtered = plot_df[plot_df["anchor_player_id"] == anchor_id].copy()
            if filtered.empty:
                filtered = plot_df[
                    plot_df["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(anchor_name)
                ].copy()
        else:
            filtered = plot_df[
                plot_df["anchor_player_name"].map(self._normalize_text)
                == self._normalize_text(anchor_name)
            ].copy()

        if filtered.empty:
            raise ValueError(f"No companion-fit rows found for anchor CB '{anchor_name}'.")

        if "coverage_gain_z_within_anchor" not in filtered.columns and "coverage_gain_raw" in filtered.columns:
            filtered["coverage_gain_z_within_anchor"] = self._z_standardize(filtered["coverage_gain_raw"])

        if "companion_fit_score_z_within_anchor" not in filtered.columns:
            filtered["companion_fit_score_z_within_anchor"] = self._z_standardize(
                filtered["companion_fit_score"]
            )

        if "companion_rank_for_anchor_z_score" not in filtered.columns:
            if "companion_rank_for_anchor" in filtered.columns:
                filtered["companion_rank_for_anchor_z_score"] = self._z_standardize(
                    filtered["companion_rank_for_anchor"],
                    invert=True,
                    scale=2.0,
                )
            else:
                filtered["companion_rank_for_anchor_z_score"] = float("nan")

        filtered = filtered.sort_values(
            ["companion_fit_score", "partner_player_name"],
            ascending=[False, True],
        ).reset_index(drop=True)

        if top_n is not None:
            if not isinstance(top_n, (int, np.integer)) or int(top_n) <= 0:
                raise ValueError("top_n must be a positive integer or None.")
            filtered = filtered.head(int(top_n)).copy().reset_index(drop=True)

        return self._attach_hover_payload(
            filtered,
            metric_cols=metric_cols,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=2.0,
        )

    def _resolve_companion_row(
        self,
        plot_df: pd.DataFrame,
        *,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
    ) -> Optional[pd.Series]:
        """
        Resolve one companion row by either:
        1) companion name/surname,
        2) companion ID, or
        3) companion rank within the current anchor sample.
        """
        if Companion_CB is None and Companion_CB_ID is None and Companion_Rank is None:
            return None

        if Companion_Rank is not None:
            if Companion_CB is not None or Companion_CB_ID is not None:
                raise ValueError(
                    "Please specify either Companion_Rank or Companion_CB/Companion_CB_ID, not both."
                )
            if not isinstance(Companion_Rank, (int, np.integer)) or int(Companion_Rank) <= 0:
                raise ValueError("Companion_Rank must be a positive integer.")

            requested_rank = int(Companion_Rank)
            if "companion_rank_for_anchor" in plot_df.columns:
                rank_values = pd.to_numeric(
                    plot_df["companion_rank_for_anchor"], errors="coerce"
                )
                matches = plot_df[rank_values == float(requested_rank)]
            else:
                # Fallback when rank column is missing: use current deterministic order
                # (already sorted by companion_fit_score desc, partner name asc).
                if requested_rank > len(plot_df):
                    matches = plot_df.iloc[0:0]
                else:
                    matches = plot_df.iloc[[requested_rank - 1]]

            if matches.empty:
                top_n_hint = (
                    " The requested rank may be outside the currently filtered set (e.g., due to top_n)."
                    if len(plot_df) > 0
                    else ""
                )
                raise ValueError(
                    f"No companion found at rank #{requested_rank} for the current anchor CB's sample."
                    f"{top_n_hint}"
                )
            if len(matches) > 1:
                candidates = matches["partner_player_name"].astype(str).tolist()
                raise ValueError(
                    f"Multiple companions found for rank #{requested_rank}: {', '.join(candidates)}"
                )
            return matches.iloc[0]

        self._ensure_columns(
            plot_df,
            ["partner_player_name"],
            "companion selection",
        )

        has_partner_id = "partner_player_id" in plot_df.columns
        if Companion_CB_ID is not None and not has_partner_id:
            raise ValueError(
                "Companion_CB_ID was provided, but `partner_player_id` is not available in the companion dataframe."
            )

        if has_partner_id:
            candidates = plot_df[["partner_player_id", "partner_player_name"]].drop_duplicates().rename(
                columns={"partner_player_id": "player.id", "partner_player_name": "player.name"}
            )
            resolved = self._resolve_entity(
                candidates,
                entity_value=Companion_CB,
                entity_id=Companion_CB_ID,
                id_col="player.id",
                name_col="player.name",
                entity_label="companion CB",
            )
            matches = plot_df[plot_df["partner_player_id"] == resolved["player.id"]]
        else:
            candidates = plot_df[["partner_player_name"]].drop_duplicates().copy()
            candidates["player.id"] = np.arange(len(candidates))
            candidates = candidates.rename(columns={"partner_player_name": "player.name"})
            resolved = self._resolve_entity(
                candidates,
                entity_value=Companion_CB,
                entity_id=None,
                id_col="player.id",
                name_col="player.name",
                entity_label="companion CB",
            )
            matches = plot_df[
                plot_df["partner_player_name"].map(self._normalize_text)
                == self._normalize_text(resolved["player.name"])
            ]

        if matches.empty:
            raise ValueError(f"No companion fit found for '{Companion_CB}'.")
        return matches.iloc[0]

    def _companion_key(self, row: pd.Series) -> str:
        """
        Build a stable deduplication key for companion rows.
        """
        if "partner_player_id" in row.index and pd.notna(row["partner_player_id"]):
            return str(row["partner_player_id"])
        return self._normalize_text(row.get("partner_player_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
        """
        Add one highlighted companion row to the anchor-companion distribution.

        Args mirror the pair-level `_add_row_point` helper and are intentionally
        aligned for maintenance consistency across plot suites.
        """
        if multi_annotations:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_single_annotations=False,
                add_multi_annotations=True,
            )
        else:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=add_single_annotations,
            )

    def _create_plot(
        self,
        *,
        anchor_name: str,
        num_candidates: int,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> DistributionPlot:
        """
        Instantiate and title one anchor-companion distribution figure.
        """
        dist_plot = DistributionPlot(
            columns=list(metric_cols),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        dist_plot.add_title(
            title=f"CB-Companion Aerial Duels Potential Fits Distribution   →   {anchor_name} acting as the Anchor CB",
            subtitle=(
                f"All {num_candidates} potential companions for {anchor_name} (directional A → B)"
                "<br>"
                "Metrics are at the CB-Companion-level (i.e. within the Anchor CB's sample),"
                "<br>"
                "showing the expected contribution of the companion to the CB-Pair's overall fit with the specified Anchor CB."
            ),
        )
        return dist_plot

    def Plot_Companion_of_Anchor_CB(
        self,
        *,
        Anchor_CB: Any = None,
        Anchor_CB_ID: Any = None,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
        Companion_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_anchor_CB_pool_average: bool = False,
        top_n: Optional[int] = None,
        multi_annotations: bool = False,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot potential companions for one anchor CB, with optional highlighted rows.

        The highlighted rows can include:
        - an explicitly selected companion
        - an explicitly selected companion rank within the anchor CB's sample
        - multiple explicitly selected companion ranks within the anchor CB's sample
        - best/worst companion for the anchor
        - the anchor's companion-pool average
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        resolved_anchor = self.Initialize_Desired_Anchor_CB(
            Anchor_CB=Anchor_CB,
            Anchor_CB_ID=Anchor_CB_ID,
        )
        anchor_name = str(resolved_anchor["anchor_player_name"])
        anchor_id = resolved_anchor.get("anchor_player_id", None)

        plot_df = self._build_anchor_plot_df(
            anchor_name=anchor_name,
            anchor_id=anchor_id,
            top_n=top_n,
            metric_cols=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            anchor_name=anchor_name,
            num_candidates=len(plot_df),
            metric_cols=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["partner_player_name"],
            legend="All companions",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        used_keys = set()

        selected_companions: List[pd.Series] = []

        # Selection mode A: multiple explicit ranks provided by the caller.
        # Each rank is resolved independently in current anchor/top_n context.
        if Companion_Ranks is not None:
            if Companion_Rank is not None:
                raise ValueError(
                    "Please provide either Companion_Rank or Companion_Ranks, not both."
                )
            if isinstance(Companion_Ranks, (str, bytes)):
                raise ValueError(
                    "Companion_Ranks must be a sequence of positive integers, not a string."
                )

            unique_ranks: List[int] = []
            for rank_value in Companion_Ranks:
                if not isinstance(rank_value, (int, np.integer)) or int(rank_value) <= 0:
                    raise ValueError(
                        f"Invalid rank '{rank_value}' in Companion_Ranks. All ranks must be positive integers."
                    )
                rank_int = int(rank_value)
                if rank_int not in unique_ranks:
                    unique_ranks.append(rank_int)

            for rank_int in unique_ranks:
                rank_selected_companion = self._resolve_companion_row(
                    plot_df,
                    Companion_Rank=rank_int,
                )
                if rank_selected_companion is not None:
                    selected_companions.append(rank_selected_companion)

        # Selection mode B: one explicit companion selector (name/ID/rank).
        single_selected_companion = self._resolve_companion_row(
            plot_df,
            Companion_CB=Companion_CB,
            Companion_CB_ID=Companion_CB_ID,
            Companion_Rank=Companion_Rank,
        )
        if single_selected_companion is not None:
            selected_companions.append(single_selected_companion)

        # Deduplicate selected rows so the same companion is not plotted twice
        # when selection criteria overlap (e.g., explicit rank equals best row).
        for selected_companion in selected_companions:
            selected_key = self._companion_key(selected_companion)
            if selected_key in used_keys:
                continue
            used_keys.add(selected_key)
            selected_label = f"{anchor_name} + {selected_companion['partner_player_name']}"
            self._add_row_point(
                dist_plot,
                row=selected_companion,
                display_name=selected_label,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=True,
            )

        if include_best:
            best_companion = plot_df.sort_values("companion_fit_score", ascending=False).iloc[0]
            best_key = self._companion_key(best_companion)
            if best_key not in used_keys:
                used_keys.add(best_key)
                best_label = f"{anchor_name} + {best_companion['partner_player_name']}"
                self._add_row_point(
                    dist_plot,
                    row=best_companion,
                    display_name=best_label,
                    hover_payload_suffix=hover_payload_suffix,
                    hover_string=hover_string,
                    multi_annotations=multi_annotations,
                    add_single_annotations=True,
                )

        if include_worst:
            worst_companion = plot_df.sort_values("companion_fit_score", ascending=True).iloc[0]
            worst_key = self._companion_key(worst_companion)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                worst_label = f"{anchor_name} + {worst_companion['partner_player_name']}"
                self._add_row_point(
                    dist_plot,
                    row=worst_companion,
                    display_name=worst_label,
                    hover_payload_suffix=hover_payload_suffix,
                    hover_string=hover_string,
                    multi_annotations=multi_annotations,
                    add_single_annotations=True,
                )

        if include_anchor_CB_pool_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="partner_player_name",
                display_name_value="Companion Pool Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["partner_player_id"] = -1
            average_label = f"{anchor_name}'s Companion Pool Average"
            self._add_row_point(
                dist_plot,
                row=average_point,
                display_name=average_label,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=False,
            )

        if show:
            dist_plot.fig.show()
        return dist_plot





# --------------------------------------------------------------------------|
# Ball-Passing Plotting Infrastructure                                      |
# --------------------------------------------------------------------------|
# The classes below form a layered architecture:                            |
# - `_Ball_Passing_Distribution_Resolver`: shared data/selection utilities  |
# - `Single_CB_*`, `CB_Pair_*`, `Anchor_CB_Companion_*`: public plot APIs   |
# --------------------------------------------------------------------------|

class _Ball_Passing_Distribution_Resolver:
    """
    Shared helper utilities used by all Ball-Passing distribution plot wrappers.

    This resolver centralizes cross-cutting concerns such as:
    - robust player/pair name normalization and resolution,
    - rank-aware z-score handling,
    - hover payload formatting,
    - average-point construction.

    Centralizing these behaviors keeps public plotting classes small and ensures
    consistent semantics across single-CB, CB-pair, and companion-fit views.
    """

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        Normalize free-text input for resilient matching.

        The normalization pipeline is intentionally strict and deterministic:
        - converts to lowercase,
        - strips accents/diacritics,
        - removes punctuation/special symbols,
        - collapses repeated whitespace.

        Args:
            value: Any user-provided value that should be interpreted as text.

        Returns:
            Normalized tokenizable text; empty string for null-like values.
        """
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _tokenize(cls, value: Any) -> Tuple[str, ...]:
        """
        Split normalized text into lexical tokens.

        Args:
            value: Input value to normalize and tokenize.

        Returns:
            Tuple of tokens. Empty tuple for empty/invalid normalized text.
        """
        normalized = cls._normalize_text(value)
        if not normalized:
            return tuple()
        return tuple(normalized.split(" "))

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
        """
        Validate dataframe schema requirements.

        Args:
            df: Dataframe to validate.
            required_columns: Columns that must exist in `df`.
            context: Human-readable context included in validation errors.

        Raises:
            KeyError: If any required column is missing.
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required column(s) for {context}: " + ", ".join(missing)
            )

    @staticmethod
    def _to_number_or_none(value: Any) -> Optional[float]:
        """
        Attempt safe scalar numeric conversion.

        Args:
            value: Value to convert.

        Returns:
            `float` when conversion is successful and finite; otherwise `None`.
        """
        try:
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        except Exception:
            return None
        if pd.isna(converted):
            return None
        return float(converted)

    @staticmethod
    def _coerce_positive_rank(rank_value: Any, *, param_name: str) -> int:
        """
        Validate and coerce one rank selector to a positive integer.

        Args:
            rank_value: Candidate rank value.
            param_name: Parameter name used in error messages.

        Returns:
            Rank as a positive integer.
        """
        if not isinstance(rank_value, (int, np.integer)) or int(rank_value) <= 0:
            raise ValueError(f"{param_name} must be a positive integer.")
        return int(rank_value)

    @classmethod
    def _coerce_unique_positive_ranks(
        cls,
        rank_values: Sequence[Any],
        *,
        param_name: str,
    ) -> List[int]:
        """
        Validate, coerce, and deduplicate a rank-sequence selector.

        Args:
            rank_values: Sequence of candidate rank values.
            param_name: Parameter name used in error messages.

        Returns:
            Ordered list of unique positive integers.
        """
        if isinstance(rank_values, (str, bytes)):
            raise ValueError(
                f"{param_name} must be a sequence of positive integers, not a string."
            )

        normalized: List[int] = []
        for rank_value in rank_values:
            rank_int = cls._coerce_positive_rank(rank_value, param_name=param_name)
            if rank_int not in normalized:
                normalized.append(rank_int)
        return normalized

    @classmethod
    def _resolve_row_by_rank(
        cls,
        plot_df: pd.DataFrame,
        *,
        rank_column: str,
        rank_value: Any,
        entity_label: str,
        display_column: str,
    ) -> pd.Series:
        """
        Resolve one row by exact raw-ranking value.

        Args:
            plot_df: Candidate dataframe to search.
            rank_column: Raw rank column name.
            rank_value: Requested rank value.
            entity_label: Human-readable entity label for errors.
            display_column: Column used to list candidate names in ambiguity errors.

        Returns:
            The uniquely resolved row for the requested rank.
        """
        requested_rank = cls._coerce_positive_rank(
            rank_value,
            param_name=f"{entity_label} rank",
        )

        if rank_column not in plot_df.columns:
            raise KeyError(
                f"Cannot resolve {entity_label} by rank because column '{rank_column}' "
                "is not available in the plotting dataframe."
            )

        rank_values = pd.to_numeric(plot_df[rank_column], errors="coerce")
        matches = plot_df[rank_values == float(requested_rank)]

        if matches.empty:
            raise ValueError(
                f"No {entity_label} found at rank #{requested_rank}."
            )
        if len(matches) > 1:
            candidates = (
                matches[display_column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f"Multiple {entity_label} entries found at rank #{requested_rank}: "
                f"{', '.join(candidates)}"
            )
        return matches.iloc[0]

    @classmethod
    def _resolve_entity(cls, candidates: pd.DataFrame, *, entity_value: Any = None, entity_id: Any = None, id_col: str, name_col: str, entity_label: str) -> pd.Series:
        """
        Resolve a single entity row from candidate IDs/names.

        Resolution precedence:
        1. Explicit ID match (numeric-safe first, then string fallback),
        2. Exact normalized full-name match,
        3. Unique token/surname subset match.

        Ambiguity is never auto-resolved: a `ValueError` is raised with candidate
        names so callers can disambiguate explicitly.

        Args:
            candidates: Candidate dataframe containing ID and name columns.
            entity_value: Optional name/surname-like selector.
            entity_id: Optional ID selector.
            id_col: Candidate ID column name.
            name_col: Candidate display-name column name.
            entity_label: Label used in error messages (e.g., "CB player").

        Returns:
            The uniquely resolved candidate row.
        """
        candidates = candidates[[id_col, name_col]].dropna(subset=[id_col, name_col]).drop_duplicates()
        if candidates.empty:
            raise ValueError(f"No {entity_label} candidates available to resolve.")

        if entity_id is None and isinstance(entity_value, (int, np.integer)):
            entity_id = int(entity_value)
            entity_value = None

        if entity_id is not None:
            requested_id_numeric = cls._to_number_or_none(entity_id)
            if requested_id_numeric is not None:
                candidate_id_numeric = pd.to_numeric(candidates[id_col], errors="coerce")
                matches = candidates[candidate_id_numeric == requested_id_numeric]
            else:
                matches = candidates[candidates[id_col].astype(str).str.strip() == str(entity_id).strip()]

            if len(matches) == 1:
                return matches.iloc[0]
            if len(matches) > 1:
                options = sorted(matches[name_col].astype(str).unique().tolist())
                raise ValueError(
                    f"Ambiguous {entity_label} ID '{entity_id}'. Candidates: {', '.join(options)}"
                )
            raise ValueError(f"No {entity_label} found for ID '{entity_id}'.")

        if entity_value is None:
            raise ValueError(f"Please provide either {entity_label} name/surname or {entity_label} ID.")

        query = cls._normalize_text(entity_value)
        if not query:
            raise ValueError(f"Invalid empty {entity_label} name input.")

        working = candidates.copy()
        working["_normalized_name"] = working[name_col].map(cls._normalize_text)
        working["_name_tokens"] = working[name_col].map(cls._tokenize)

        exact_matches = working[working["_normalized_name"] == query]
        if len(exact_matches) == 1:
            return exact_matches.iloc[0]
        if len(exact_matches) > 1:
            options = sorted(exact_matches[name_col].astype(str).unique().tolist())
            raise ValueError(
                f"Ambiguous {entity_label} name '{entity_value}'. Candidates: {', '.join(options)}"
            )

        query_tokens = set(cls._tokenize(query))
        token_matches = working[
            working["_name_tokens"].map(lambda tokens: query_tokens.issubset(set(tokens)))
        ]

        if len(token_matches) == 1:
            return token_matches.iloc[0]
        if len(token_matches) > 1:
            options = sorted(token_matches[name_col].astype(str).unique().tolist())
            raise ValueError(
                f"Ambiguous {entity_label} input '{entity_value}'. Candidates: {', '.join(options)}"
            )

        raise ValueError(f"No {entity_label} found for input '{entity_value}'.")

    @staticmethod
    def _z_standardize(values: Union[pd.Series, Sequence[Any]], *, invert: bool = False, scale: float = 1.0) -> pd.Series:
        """
        Compute z-standardized values with optional sign inversion and scaling.

        This helper is used both for standard metric normalization and rank-based
        transformations (where lower raw rank means better, hence `invert=True`).

        Args:
            values: Numeric-like values to standardize.
            invert: If true, multiply standardized values by `-1`.
            scale: Multiplicative factor applied after standardization.

        Returns:
            Z-standardized series. If variance is zero/undefined, returns all NaN.
        """
        series = pd.to_numeric(pd.Series(values), errors="coerce")
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(float("nan"), index=series.index)
        z_values = (series - series.mean()) / std
        if invert:
            z_values = -z_values
        return z_values * scale

    @classmethod
    def _is_rank_metric(cls, metric_name: str, raw_metric_name: Optional[str]) -> bool:
        """
        Identify rank-like metrics by inspecting z/raw metric names.

        Args:
            metric_name: Plotted metric column (typically z-score column).
            raw_metric_name: Raw companion/value column mapped to that metric.

        Returns:
            True when either name contains the token `rank`.
        """
        metric_token = str(metric_name).lower()
        raw_metric_token = str(raw_metric_name).lower() if raw_metric_name is not None else ""
        return ("rank" in metric_token) or ("rank" in raw_metric_token)

    @classmethod
    def _format_hover_line(cls, metric_name: str, raw_metric_name: Optional[str], metric_label: str, raw_value: Any, z_value: Any) -> str:
        """
        Build one semantic-aware hover text line for a metric.

        Behavior:
        - metrics without a raw column mapping show z-score only,
        - rank-like metrics show integer rank format (`# <int>`),
        - all other metrics show `raw + corresponding z-score`.
        """
        if raw_metric_name is None:
            return f"Z-score = {z_value:.2f}" if pd.notna(z_value) else "Z-score = N/A"
        if cls._is_rank_metric(metric_name, raw_metric_name):
            return f"Rank =   # {int(round(raw_value))}" if pd.notna(raw_value) else "Rank =   # N/A"
        raw_text = f"{raw_value:.2f}" if pd.notna(raw_value) else "N/A"
        z_text = f"{z_value:.2f}" if pd.notna(z_value) else "N/A"
        return f"{metric_label} = {raw_text}<br>Respective Z-score = {z_text}"

    @classmethod
    def _attach_hover_payload(cls, plot_df: pd.DataFrame, *, metric_cols: Sequence[str], metric_labels: Dict[str, str], metric_value_columns: Dict[str, Optional[str]], hover_payload_suffix: str = "_hover_payload", fill_missing_rank_z: bool = True, rank_z_scale: float = 2.0) -> pd.DataFrame:
        """
        Attach per-metric hover payload tuples used by `DistributionPlot`.

        For each metric, this method optionally reconstructs missing/all-NaN
        rank z-score columns from the corresponding raw rank column, then creates
        payload tuples of the form:
        `(metric_label, raw_value, z_value, formatted_hover_line)`.

        Args:
            plot_df: Plotting dataframe to enrich in place.
            metric_cols: Metrics expected in the distribution plot.
            metric_labels: Human-readable labels per metric.
            metric_value_columns: Raw-value column mapping per metric.
            hover_payload_suffix: Suffix for generated payload columns.
            fill_missing_rank_z: Whether rank z-score fallback should be applied.
            rank_z_scale: Scaling factor used when reconstructing rank z-scores.

        Returns:
            The same dataframe instance, enriched with hover payload columns.
        """
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))

            can_backfill_rank_z = (
                fill_missing_rank_z
                and raw_metric is not None
                and cls._is_rank_metric(metric, raw_metric)
                and raw_metric in plot_df.columns
            )
            metric_missing = metric not in plot_df.columns
            metric_all_nan = False
            if not metric_missing:
                metric_all_nan = pd.to_numeric(plot_df[metric], errors="coerce").isna().all()

            if metric_missing or (can_backfill_rank_z and metric_all_nan):
                if can_backfill_rank_z:
                    plot_df[metric] = cls._z_standardize(
                        plot_df[raw_metric],
                        invert=True,
                        scale=rank_z_scale,
                    )
                else:
                    plot_df[metric] = float("nan")

            metric_label = metric_labels.get(metric, format_metric(metric))
            default_series = pd.Series(float("nan"), index=plot_df.index)
            raw_values = pd.to_numeric(
                plot_df.get(raw_metric, default_series),
                errors="coerce",
            )
            z_values = pd.to_numeric(
                plot_df.get(metric, default_series),
                errors="coerce",
            )

            hover_lines = [
                cls._format_hover_line(
                    metric,
                    raw_metric,
                    metric_label,
                    raw_value,
                    z_value,
                )
                for raw_value, z_value in zip(raw_values, z_values)
            ]

            plot_df[f"{metric}{hover_payload_suffix}"] = list(
                zip(
                    [metric_label] * len(plot_df),
                    raw_values,
                    z_values,
                    hover_lines,
                )
            )

        return plot_df

    @classmethod
    def _split_pair_string(cls, pair_value: str) -> Tuple[str, str]:
        """
        Parse user-friendly pair strings into two entity selectors.

        Accepted separators:
        - `+`
        - `and`
        - `&`

        Args:
            pair_value: Pair input such as `"A + B"` or `"A and B"`.

        Returns:
            Tuple of two stripped entity strings.
        """
        if not isinstance(pair_value, str):
            raise ValueError("CB_Pair must be a string such as 'A + B', 'A and B', or 'A & B'.")

        parts = re.split(r"\s*(?:\+|&|\band\b)\s*", pair_value.strip(), maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                "Could not parse CB pair input. Use formats like 'A + B', 'A and B', or 'A & B'."
            )
        return parts[0].strip(), parts[1].strip()

    @classmethod
    def _build_average_point(cls, *, plot_df: pd.DataFrame, metric_cols: Sequence[str], metric_labels: Dict[str, str], metric_value_columns: Dict[str, Optional[str]], display_name_col: str, display_name_value: str, hover_payload_suffix: str = "_hover_payload") -> pd.Series:
        """
        Build a synthetic cohort-average point compatible with `DistributionPlot`.

        The returned series contains:
        - average z-scores for each plotted metric,
        - average raw values for mapped raw columns (when available),
        - per-metric hover payload entries matching normal row payload shape,
        - a display-name field to support legend/labels.

        Args:
            plot_df: Source dataframe defining the cohort.
            metric_cols: Metrics used in the plot.
            metric_labels: Human-readable labels per metric.
            metric_value_columns: Raw-value mapping per metric.
            display_name_col: Column name used by the plot as display label.
            display_name_value: Display label for the synthetic average point.
            hover_payload_suffix: Suffix used for hover payload columns.

        Returns:
            Series representing one synthetic average row.
        """
        avg_raw_metrics = pd.Series(
            {
                raw_col: pd.to_numeric(plot_df[raw_col], errors="coerce").mean()
                for raw_col in set(metric_value_columns.values())
                if raw_col is not None and raw_col in plot_df.columns
            }
        )
        avg_z_scores = pd.Series(
            {
                z_col: pd.to_numeric(plot_df[z_col], errors="coerce").mean()
                if z_col in plot_df.columns
                else float("nan")
                for z_col in metric_cols
            }
        )

        average_point = avg_z_scores.copy()
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))
            metric_label = metric_labels.get(metric, format_metric(metric))
            avg_raw_value = avg_raw_metrics.get(raw_metric, float("nan"))
            avg_z_value = average_point.get(metric, float("nan"))

            if raw_metric is not None:
                average_point[raw_metric] = avg_raw_value

            average_point[f"{metric}{hover_payload_suffix}"] = [
                metric_label,
                avg_raw_value,
                avg_z_value,
                cls._format_hover_line(
                    metric,
                    raw_metric,
                    metric_label,
                    avg_raw_value,
                    avg_z_value,
                ),
            ]

        average_point[display_name_col] = display_name_value
        return average_point

    @classmethod
    def _resolve_metric_config(
        cls,
        *,
        default_metrics: Sequence[str],
        default_metric_labels: Mapping[str, str],
        default_metric_value_columns: Mapping[str, Optional[str]],
        configured_metrics: Optional[Sequence[str]] = None,
        configured_metric_labels: Optional[Mapping[str, str]] = None,
        configured_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        override_metrics: Optional[Sequence[str]] = None,
        override_metric_labels: Optional[Mapping[str, str]] = None,
        override_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve the metric configuration for one plotting call.

        Precedence:
        1. Method-level overrides (`override_*`)
        2. Constructor-level settings (`configured_*`)
        3. Class defaults (`default_*`)

        Returns only the mappings required by the selected metrics.
        """
        chosen_metrics = override_metrics
        if chosen_metrics is None:
            chosen_metrics = configured_metrics
        if chosen_metrics is None:
            chosen_metrics = default_metrics

        ordered_metrics: List[str] = []
        for metric in chosen_metrics:
            metric_str = str(metric)
            if metric_str not in ordered_metrics:
                ordered_metrics.append(metric_str)
        if not ordered_metrics:
            raise ValueError("At least one metric must be provided for the distribution plot.")

        resolved_labels = dict(default_metric_labels)
        if configured_metric_labels:
            resolved_labels.update(dict(configured_metric_labels))
        if override_metric_labels:
            resolved_labels.update(dict(override_metric_labels))
        metric_labels = {
            metric: resolved_labels.get(metric, format_metric(metric))
            for metric in ordered_metrics
        }

        resolved_value_columns: Dict[str, Optional[str]] = dict(default_metric_value_columns)
        if configured_metric_value_columns:
            resolved_value_columns.update(dict(configured_metric_value_columns))
        if override_metric_value_columns:
            resolved_value_columns.update(dict(override_metric_value_columns))
        metric_value_columns = {
            metric: resolved_value_columns.get(metric, metric.replace("z_", "", 1))
            for metric in ordered_metrics
        }

        return ordered_metrics, metric_labels, metric_value_columns

    @staticmethod
    def _show_figures_side_by_side(
        figures: Sequence[go.Figure],
        *,
        column_gap: str = "16px",
        min_column_width_px: int = 520,
    ) -> bool:
        """
        Display Plotly figures side-by-side in notebook environments.

        Returns `True` when HTML notebook rendering succeeds. Callers can fall back to sequential `.show()` rendering when `False` is returned.
        """
        if not figures:
            return False

        try:
            from IPython.display import HTML, display  # type: ignore
        except Exception:
            return False

        try:
            blocks = []
            for idx, fig in enumerate(figures):
                include_plotlyjs = "cdn" if idx == 0 else False
                fig_html = pio.to_html(
                    fig,
                    include_plotlyjs=include_plotlyjs,
                    full_html=False,
                )
                blocks.append(
                    (
                        "<div style='flex:1 1 "
                        f"{min_column_width_px}px;min-width:{min_column_width_px}px'>"
                        f"{fig_html}</div>"
                    )
                )

            container_html = (
                "<div style='display:flex;flex-wrap:wrap;"
                f"gap:{column_gap};align-items:flex-start'>{''.join(blocks)}</div>"
            )
            display(HTML(container_html))
            return True
        except Exception:
            return False


class Single_CB_Ball_Passing_Distribution_Plot(_Ball_Passing_Distribution_Resolver):
    """
    Build Ball-Passing distribution plots for:
    1. One selected CB against the full cohort
    2. Multiple selected CBs in comparison mode

    The class ships with sensible defaults for metrics, labels, and raw-value mappings. Users can still override configuration at construction time or per plotting call.
    """

    DEFAULT_METRICS = [
    "z_passes_per90",
    "z_long_pass_accuracy",
    "z_progressive_pass_accuracy",
    "z_final_third_passes_per90",
    "z_progressive_passes_per90",
    "z_pass_accuracy_rate",
    "CB_ball_passing_quality_z_score",
    "CB_rank_for_ball_passing_quality_z_score"
    ]


    DEFAULT_METRIC_LABELS = {
    "z_passes_per90": "Number of Passes per-90",
    "z_long_pass_accuracy": "Long Pass Accuracy",
    "z_progressive_pass_accuracy": "Progressive Pass Accuracy",
    "z_final_third_passes_per90": "Passes to Final Third per-90",
    "z_progressive_passes_per90": "Progressive Passes per-90",
    "z_pass_accuracy_rate": "Pass Accuracy Rate",
    "CB_ball_passing_quality_z_score": "CB's (Overall) Ball Passing Quality Score",
    "CB_rank_for_ball_passing_quality_z_score": "CB's (Ball Passing Quality) Ranking",
    "CB_rank_for_ball_passing_quality": "CB's (Ball Passing Quality) Ranking"
    }


    DEFAULT_METRIC_VALUE_COLUMNS = {
    "z_passes_per90": "passes_per90",
    "z_long_pass_accuracy": "long_pass_accuracy",
    "z_progressive_pass_accuracy": "progressive_pass_accuracy",
    "z_final_third_passes_per90": "final_third_passes_per90",
    "z_progressive_passes_per90": "progressive_passes_per90",
    "z_pass_accuracy_rate": "pass_accuracy_rate",
    "CB_ball_passing_quality_z_score": None,  # No separate raw column for the overall ball passing quality z-score, as it's a composite metric derived from the z-scores of the individual quality metrics, so we just use the z-score column for both the value and the label in this case - no raw value or label in the plot's tooltip.
    "CB_rank_for_ball_passing_quality_z_score": "CB_rank_for_ball_passing_quality"   # we can show the raw rank value in the hover tooltip for context, even though the x-axis position is based on the z-scored rank (where higher is better fit)
    }


    RANK_Z_SCORE_SCALE = 2.0
    RAW_RANK_COLUMN = "CB_rank_for_ball_passing_quality"

    def __init__(
        self,
        *,
        duel_summary: pd.DataFrame,
        z_scores: pd.DataFrame,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, str]] = None,
        min_num_passes_involved_in_threshold: Any = None,
        min_num_duels_involved_in_threshold: Any = None,
        min_minutes_played_threshold: Any,
    ) -> None:
        """
        Args:
            duel_summary: Raw single-CB metric dataframe.
            z_scores: Z-scored single-CB metric dataframe.
            metrics: Optional constructor-level metric list.
            metric_labels: Optional constructor-level metric label map.
            metric_value_columns: Optional constructor-level raw-value column map.
            min_num_passes_involved_in_threshold: Threshold shown in subtitles (canonical).
            min_num_duels_involved_in_threshold: Legacy alias for backward compatibility.
            min_minutes_played_threshold: Threshold shown in subtitles.
        """
        self.duel_summary = duel_summary
        self.z_scores = z_scores
        self._configured_metrics = list(metrics) if metrics is not None else None
        self._configured_metric_labels = (
            dict(metric_labels) if metric_labels is not None else None
        )
        self._configured_metric_value_columns = (
            dict(metric_value_columns) if metric_value_columns is not None else None
        )

        if min_num_passes_involved_in_threshold is not None and min_num_duels_involved_in_threshold is not None:
            raise ValueError(
                "Please provide only one of `min_num_passes_involved_in_threshold` or `min_num_duels_involved_in_threshold`."
            )
        if min_num_passes_involved_in_threshold is None and min_num_duels_involved_in_threshold is None:
            raise ValueError(
                "Please provide `min_num_passes_involved_in_threshold` (or legacy `min_num_duels_involved_in_threshold`)."
            )
        self.min_num_passes_involved_in_threshold = (
            min_num_passes_involved_in_threshold
            if min_num_passes_involved_in_threshold is not None
            else min_num_duels_involved_in_threshold
        )
        self.min_minutes_played_threshold = min_minutes_played_threshold


    def _resolve_current_metric_config(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve call-ready metric configuration with precedence:
        method override > constructor config > class defaults.
        """
        return self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_METRIC_VALUE_COLUMNS,
            configured_metrics=self._configured_metrics,
            configured_metric_labels=self._configured_metric_labels,
            configured_metric_value_columns=self._configured_metric_value_columns,
            override_metrics=metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )

    def _build_plot_df(
        self,
        *,
        metrics: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Build the merged plotting dataframe and attach hover payload columns.
        """
        z_scores_df = self.z_scores.reset_index()
        self._ensure_columns(
            z_scores_df,
            ["player.id", "player.name"],
            "single-CB z-score dataframe",
        )

        raw_metric_columns = [
            col for col in dict.fromkeys(metric_value_columns.values()) if col is not None
        ]
        duel_summary_df = self.duel_summary.reset_index()
        always_include_raw_columns: List[str] = []
        if self.RAW_RANK_COLUMN in duel_summary_df.columns:
            always_include_raw_columns.append(self.RAW_RANK_COLUMN)
        fallback_metric_columns = [
            metric
            for metric in metrics
            if metric not in z_scores_df.columns and metric in duel_summary_df.columns
        ]
        required_duel_summary_columns = list(
            dict.fromkeys(
                raw_metric_columns + fallback_metric_columns + always_include_raw_columns
            )
        )
        self._ensure_columns(
            duel_summary_df,
            ["player.id", "player.name"] + required_duel_summary_columns,
            "single-CB raw metrics dataframe",
        )

        raw_metrics_df = duel_summary_df[
            ["player.id", "player.name"] + required_duel_summary_columns
        ].copy()
        plot_df = z_scores_df.merge(
            raw_metrics_df,
            on=["player.id", "player.name"],
            how="left",
            validate="one_to_one",
        )

        # Spread rank-based points for readability while keeping raw rank values in labels/tooltips via `metric_value_columns`.
        rank_metric_col = "CB_rank_for_ball_passing_quality_z_score"
        if rank_metric_col in plot_df.columns:
            plot_df[rank_metric_col] = (
                pd.to_numeric(plot_df[rank_metric_col], errors="coerce")
                * self.RANK_Z_SCORE_SCALE
            )

        return self._attach_hover_payload(
            plot_df,
            metric_cols=metrics,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=self.RANK_Z_SCORE_SCALE,
        )

    def _create_plot(
        self,
        *,
        metrics: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        subtitle_style: str = "single",
    ) -> DistributionPlot:
        """
        Instantiate and title the distribution chart with the resolved config.
        """
        if subtitle_style == "comparison":
            subtitle = (
                f"Based on {len(self.z_scores)} CB players with ≥ {self.min_num_passes_involved_in_threshold} passes & playing time ≥ {self.min_minutes_played_threshold} minutes"
            )
        else:
            subtitle = (
                f"Based on {len(self.z_scores)} CB players with ≥ {self.min_num_passes_involved_in_threshold} passes & playing time ≥ {self.min_minutes_played_threshold} minutes"
            )

        dist_plot = DistributionPlot(
            columns=list(metrics),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        dist_plot.add_title(
            title="CB Ball Passing Quality Distribution",
            subtitle=subtitle,
        )
        return dist_plot

    def _resolve_single_cb_by_rank(
        self,
        plot_df: pd.DataFrame,
        *,
        CB_Rank: Any,
    ) -> pd.Series:
        """
        Resolve one CB row by raw Ball-Passing-quality ranking value.

        Args:
            plot_df: Prepared plotting dataframe.
            CB_Rank: Requested rank in the raw ranking column.

        Returns:
            One resolved row for the requested rank.
        """
        return self._resolve_row_by_rank(
            plot_df,
            rank_column=self.RAW_RANK_COLUMN,
            rank_value=CB_Rank,
            entity_label="CB player",
            display_column="player.name",
        )

    def _resolve_single_cb(
        self,
        plot_df: pd.DataFrame,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Any = None,
    ) -> pd.Series:
        """
        Resolve a single CB row from the prepared single-CB plotting dataframe.

        If neither `CB`, `CB_ID`, nor `CB_Rank` is provided, the first row is selected as a
        deterministic fallback to keep quick exploratory plotting simple.

        Args:
            plot_df: Prepared plotting dataframe containing `player.id/name`.
            CB: Optional player name/surname selector.
            CB_ID: Optional player ID selector.
            CB_Rank: Optional raw ranking selector.

        Returns:
            One resolved plotting row for the selected CB.
        """
        candidates = plot_df[["player.id", "player.name"]].drop_duplicates()
        if CB is None and CB_ID is None and CB_Rank is None:
            return plot_df.iloc[0]

        if CB_Rank is not None:
            if CB is not None or CB_ID is not None:
                raise ValueError(
                    "Please provide either CB_Rank or CB/CB_ID, not both."
                )
            return self._resolve_single_cb_by_rank(plot_df, CB_Rank=CB_Rank)

        resolved = self._resolve_entity(
            candidates,
            entity_value=CB,
            entity_id=CB_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        selected_rows = plot_df[plot_df["player.id"] == resolved["player.id"]]
        if selected_rows.empty:
            raise ValueError(f"Resolved CB '{resolved['player.name']}' is not available in plotting dataframe.")
        return selected_rows.iloc[0]

    def Plot_Single_CB(
        self,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Optional[int] = None,
        include_league_average: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one selected CB against the full single-CB Ball-Passing distribution.

        Selection options:
        - `CB` (name/surname),
        - `CB_ID`,
        - `CB_Rank` (raw quality ranking value).
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        plot_df = self._build_plot_df(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
            subtitle_style="single",
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["player.name"],
            legend="All players",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        selected_cb = self._resolve_single_cb(
            plot_df,
            CB=CB,
            CB_ID=CB_ID,
            CB_Rank=CB_Rank,
        )
        dist_plot.add_data_point(
            ser_plot=selected_cb,
            plots="",
            name=selected_cb["player.name"],
            hover=hover_payload_suffix,
            hover_string=hover_string,
            add_annotations=True,
        )

        if include_league_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="player.name",
                display_name_value="CBs' League Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["player.id"] = -1
            dist_plot.add_data_point(
                ser_plot=average_point,
                plots="",
                name="CBs' League Average",
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=False,
            )

        if show:
            dist_plot.fig.show()
        return dist_plot

    def Plot_CBs_Comparison(
        self,
        *,
        CBs: Optional[Sequence[Any]] = None,
        CB_IDs: Optional[Sequence[Any]] = None,
        CB_Rank: Optional[int] = None,
        CB_Ranks: Optional[Sequence[int]] = None,
        include_league_average: bool = True,
        multi_annotations: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one or more selected CBs in comparison mode.

        CBs can be selected by:
        - `CBs` (names/surnames),
        - `CB_IDs`,
        - `CB_Rank` / `CB_Ranks` (raw quality ranking values).
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        plot_df = self._build_plot_df(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            metrics=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
            subtitle_style="comparison",
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["player.name"],
            legend="All players",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        requested_items: List[Tuple[str, Any, Any]] = []
        for cb_id in (CB_IDs or []):
            requested_items.append(("id", None, cb_id))
        for cb_name in (CBs or []):
            requested_items.append(("name", cb_name, None))

        # Rank selectors are optional and additive in comparison mode.
        # They can be used alone or mixed with explicit name/ID selectors.
        if CB_Ranks is not None:
            if CB_Rank is not None:
                raise ValueError(
                    "Please provide either CB_Rank or CB_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Ranks,
                param_name="CB_Ranks",
            ):
                requested_items.append(("rank", rank_value, None))
        if CB_Rank is not None:
            requested_items.append(
                (
                    "rank",
                    self._coerce_positive_rank(CB_Rank, param_name="CB_Rank"),
                    None,
                )
            )

        if not requested_items:
            raise ValueError(
                "Please provide at least one CB selector via CBs, CB_IDs, CB_Rank, or CB_Ranks."
            )

        # Deduplicate by player ID so overlapping selectors (e.g., name + rank
        # pointing to the same CB) do not create duplicate highlighted traces.
        used_ids = set()
        for selection_mode, cb_name_or_rank, cb_id in requested_items:
            if selection_mode == "rank":
                selected_cb = self._resolve_single_cb(
                    plot_df,
                    CB_Rank=cb_name_or_rank,
                )
            else:
                selected_cb = self._resolve_single_cb(
                    plot_df,
                    CB=cb_name_or_rank,
                    CB_ID=cb_id,
                )
            cb_unique_id = selected_cb["player.id"]
            if cb_unique_id in used_ids:
                continue
            used_ids.add(cb_unique_id)

            if multi_annotations:
                dist_plot.add_data_point(
                    ser_plot=selected_cb,
                    plots="",
                    name=selected_cb["player.name"],
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_single_annotations=False,
                    add_multi_annotations=True,
                )
            else:
                dist_plot.add_data_point(
                    ser_plot=selected_cb,
                    plots="",
                    name=selected_cb["player.name"],
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_annotations=True,
                )

        if include_league_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="player.name",
                display_name_value="CBs' League Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["player.id"] = -1
            if multi_annotations:
                dist_plot.add_data_point(
                    ser_plot=average_point,
                    plots="",
                    name="CBs' League Average",
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_single_annotations=False,
                    add_multi_annotations=True,
                )
            else:
                dist_plot.add_data_point(
                    ser_plot=average_point,
                    plots="",
                    name="CBs' League Average",
                    hover=hover_payload_suffix,
                    hover_string=hover_string,
                    add_annotations=False,
                )

        if show:
            dist_plot.fig.show()
        return dist_plot


class CB_Pair_Ball_Passing_Distribution_Plot(_Ball_Passing_Distribution_Resolver):
    """
    Build Ball-Passing distribution plots for CB-pair analyses.

    By default, CB-pair plots are split into two side-by-side figures:
    1. Fit/composition metrics (left)
    2. Individual-quality metrics (right)

    Set `split_view=False` in plotting methods to render the legacy single-figure
    layout.
    """

    DEFAULT_PAIR_METRIC_LABELS = {
    "z_passes_per90": "Number of Passes per-90",
    "z_long_pass_accuracy": "Long Pass Accuracy",
    "z_progressive_pass_accuracy": "Progressive Pass Accuracy",
    "z_final_third_passes_per90": "Passes to Final Third per-90",
    "z_progressive_passes_per90": "Progressive Passes per-90",
    "z_pass_accuracy_rate": "Pass Accuracy Rate",
    "floor_z": "CB-Pair Weak-Link (i.e. Floor) Protection",
    "complement_z": "CB-Pair Complementarity (i.e. Deficit-Coverage Gain)",
    "quality_z": "CB-Pair Weighted Avg. Quality",
    "CB_pair_fit_z_score": "CB-Pair (Overall) Fit Score",
    "CB_pair_fit_rank_z_score": "CB-Pair Fit Ranking (Within This Sample)",
    "CB_pair_fit_rank": "CB-Pair Fit Ranking (Within This Sample)",
    }


    DEFAULT_PAIR_METRIC_VALUE_COLUMNS = {
    "z_passes_per90": "passes_per90",
    "z_long_pass_accuracy": "long_pass_accuracy",
    "z_progressive_pass_accuracy": "progressive_pass_accuracy",
    "z_final_third_passes_per90": "final_third_passes_per90",
    "z_progressive_passes_per90": "progressive_passes_per90",
    "z_pass_accuracy_rate": "pass_accuracy_rate",
    "floor_z": "floor_raw",
    "complement_z": "complement_raw",
    "quality_z": "quality_raw",
    "CB_pair_fit_z_score": None,  # No separate raw column for the overall ball passing quality z-score, as it's a composite metric derived from the z-scores of the individual quality metrics, so we just use the z-score column for both the value and the label in this case - no raw value or label in the plot's tooltip.
    "CB_pair_fit_rank_z_score": "CB_pair_fit_rank",   # we can show the raw rank value in the hover tooltip for context, even though the x-axis position is based on the z-scored rank (where higher is better fit)
    }


    DEFAULT_LEFT_METRICS = [
    "floor_z",
    "complement_z",
    "quality_z",
    "CB_pair_fit_z_score",
    "CB_pair_fit_rank_z_score",
    ]


    DEFAULT_RIGHT_METRICS = [
    "z_passes_per90",
    "z_long_pass_accuracy",
    "z_progressive_pass_accuracy",
    "z_final_third_passes_per90",
    "z_progressive_passes_per90",
    "z_pass_accuracy_rate",
    ]

    DEFAULT_METRICS = DEFAULT_RIGHT_METRICS + DEFAULT_LEFT_METRICS


    RANK_Z_SCALE = 2.5
    RAW_RANK_COLUMN = "CB_pair_fit_rank"

    def __init__(
        self,
        *,
        df_ball_passing_pairs: Optional[pd.DataFrame] = None,
        df_ground_duel_pairs: Optional[pd.DataFrame] = None,
        plot_metric_cols: Optional[Sequence[str]] = None,
        pair_metric_labels: Optional[Mapping[str, str]] = None,
        pair_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        min_num_passes_involved_in_threshold: Any = None,
        min_num_duels_involved_in_threshold: Any = None,
        min_minutes_played_threshold: Any,
    ) -> None:
        """
        Args:
            df_ball_passing_pairs: Canonical CB-pair dataframe used for plotting.
            df_ground_duel_pairs: Legacy alias for backward compatibility.
            plot_metric_cols: Optional legacy single-view metric list.
            pair_metric_labels: Optional constructor-level label map.
            pair_metric_value_columns: Optional constructor-level raw-value map.
            left_metrics: Optional constructor-level left split metrics.
            right_metrics: Optional constructor-level right split metrics.
            min_num_passes_involved_in_threshold: Threshold shown in subtitles (canonical).
            min_num_duels_involved_in_threshold: Legacy alias for backward compatibility.
            min_minutes_played_threshold: Threshold shown in subtitles.
        """
        if df_ball_passing_pairs is not None and df_ground_duel_pairs is not None:
            raise ValueError(
                "Please provide only one of `df_ball_passing_pairs` or `df_ground_duel_pairs`."
            )
        if df_ball_passing_pairs is None and df_ground_duel_pairs is None:
            raise ValueError(
                "Please provide `df_ball_passing_pairs` (or legacy `df_ground_duel_pairs`)."
            )
        self.df_ball_passing_pairs = (
            df_ball_passing_pairs
            if df_ball_passing_pairs is not None
            else df_ground_duel_pairs
        )

        self._configured_plot_metric_cols = (
            list(plot_metric_cols) if plot_metric_cols is not None else None
        )
        self._configured_pair_metric_labels = (
            dict(pair_metric_labels) if pair_metric_labels is not None else None
        )
        self._configured_pair_metric_value_columns = (
            dict(pair_metric_value_columns)
            if pair_metric_value_columns is not None
            else None
        )
        self._configured_left_metrics = (
            list(left_metrics) if left_metrics is not None else None
        )
        self._configured_right_metrics = (
            list(right_metrics) if right_metrics is not None else None
        )

        if min_num_passes_involved_in_threshold is not None and min_num_duels_involved_in_threshold is not None:
            raise ValueError(
                "Please provide only one of `min_num_passes_involved_in_threshold` or `min_num_duels_involved_in_threshold`."
            )
        if min_num_passes_involved_in_threshold is None and min_num_duels_involved_in_threshold is None:
            raise ValueError(
                "Please provide `min_num_passes_involved_in_threshold` (or legacy `min_num_duels_involved_in_threshold`)."
            )
        self.min_num_passes_involved_in_threshold = (
            min_num_passes_involved_in_threshold
            if min_num_passes_involved_in_threshold is not None
            else min_num_duels_involved_in_threshold
        )
        self.min_minutes_played_threshold = min_minutes_played_threshold


    def _resolve_pair_metric_configs(
        self,
        *,
        split_view: bool,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Dict[str, Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]]:
        """
        Resolve split or single metric configs with method > constructor > default precedence.
        """
        constructor_labels = self._configured_pair_metric_labels
        constructor_value_columns = self._configured_pair_metric_value_columns

        if split_view:
            # Split mode resolves each side independently, then builds a combined
            # superset config used for shared data preparation/hover payloads.
            configured_left = self._configured_left_metrics
            configured_right = self._configured_right_metrics

            if configured_left is None and configured_right is None and self._configured_plot_metric_cols is not None:
                ordered = list(self._configured_plot_metric_cols)
                derived_left = [m for m in ordered if m in self.DEFAULT_LEFT_METRICS]
                derived_right = [m for m in ordered if m not in derived_left]
                configured_left = derived_left or list(self.DEFAULT_LEFT_METRICS)
                configured_right = derived_right or list(self.DEFAULT_RIGHT_METRICS)

            left_cfg = self._resolve_metric_config(
                default_metrics=self.DEFAULT_LEFT_METRICS,
                default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
                default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
                configured_metrics=configured_left,
                configured_metric_labels=constructor_labels,
                configured_metric_value_columns=constructor_value_columns,
                override_metrics=left_metrics,
                override_metric_labels=metric_labels,
                override_metric_value_columns=metric_value_columns,
            )
            right_cfg = self._resolve_metric_config(
                default_metrics=self.DEFAULT_RIGHT_METRICS,
                default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
                default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
                configured_metrics=configured_right,
                configured_metric_labels=constructor_labels,
                configured_metric_value_columns=constructor_value_columns,
                override_metrics=right_metrics,
                override_metric_labels=metric_labels,
                override_metric_value_columns=metric_value_columns,
            )

            combined_metrics: List[str] = []
            for metric in left_cfg[0] + right_cfg[0]:
                if metric not in combined_metrics:
                    combined_metrics.append(metric)

            combined_labels = dict(left_cfg[1])
            combined_labels.update(right_cfg[1])
            combined_value_columns = dict(left_cfg[2])
            combined_value_columns.update(right_cfg[2])
            combined_cfg = (combined_metrics, combined_labels, combined_value_columns)

            return {
                "left": left_cfg,
                "right": right_cfg,
                "combined": combined_cfg,
            }

        override_metrics: Optional[List[str]] = None
        if left_metrics is not None or right_metrics is not None:
            override_metrics = []
            for metric in list(left_metrics or []) + list(right_metrics or []):
                metric_str = str(metric)
                if metric_str not in override_metrics:
                    override_metrics.append(metric_str)

        configured_single_metrics = self._configured_plot_metric_cols
        if configured_single_metrics is None and (
            self._configured_left_metrics is not None or self._configured_right_metrics is not None
        ):
            configured_single_metrics = []
            for metric in list(self._configured_left_metrics or []) + list(self._configured_right_metrics or []):
                metric_str = str(metric)
                if metric_str not in configured_single_metrics:
                    configured_single_metrics.append(metric_str)

        single_cfg = self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_PAIR_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_PAIR_METRIC_VALUE_COLUMNS,
            configured_metrics=configured_single_metrics,
            configured_metric_labels=constructor_labels,
            configured_metric_value_columns=constructor_value_columns,
            override_metrics=override_metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )
        return {"single": single_cfg, "combined": single_cfg}

    def _prepare_plot_df(
        self,
        *,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Prepare the CB-pair dataframe and inject hover payload columns.
        """
        plot_df = self.df_ball_passing_pairs.copy()
        self._ensure_columns(
            plot_df,
            ["pair_name", "CB_pair_fit_z_score"],
            "CB-pair plotting dataframe",
        )

        # Always scale rank-derived z-scores with the CB-pair-specific spread factor
        # so ranking rows are readable even when the source dataframe already
        # contains a precomputed rank z-score column.
        for metric in metric_cols:
            raw_metric = metric_value_columns.get(metric, metric.replace("z_", "", 1))
            if (
                raw_metric is not None
                and self._is_rank_metric(metric, raw_metric)
                and raw_metric in plot_df.columns
            ):
                plot_df[metric] = self._z_standardize(
                    plot_df[raw_metric],
                    invert=True,
                    scale=self.RANK_Z_SCALE,
                )

        return self._attach_hover_payload(
            plot_df,
            metric_cols=metric_cols,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=self.RANK_Z_SCALE,
        )

    def _create_plot(
        self,
        *,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        section_title: Optional[str] = None,
    ) -> DistributionPlot:
        """
        Instantiate and title one CB-pair distribution figure.
        """
        dist_plot = DistributionPlot(
            columns=list(metric_cols),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        title = "CB-Pair Ball Passing Quality Fit Distribution"
        if section_title:
            title = f"{title} <br> ⇒  {section_title}"
        dist_plot.add_title(
            title=title,
            subtitle=(
                f"All {len(self.df_ball_passing_pairs)} unordered CB pairs, with ≥ {self.min_num_passes_involved_in_threshold} passes & playing time ≥ {self.min_minutes_played_threshold} minutes (each individual CB)"
                "<br>"
                "All metrics are CB-Pair-level (i.e. the weighted average of both players for that metric)"
            ),
        )
        return dist_plot

    def _player_candidates_from_pairs(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a normalized unique player-candidate table from pair columns.

        Args:
            plot_df: CB-pair dataframe containing CB1/CB2 ID and name columns.

        Returns:
            Two-column dataframe (`player.id`, `player.name`) with unique players.
        """
        self._ensure_columns(
            plot_df,
            ["player.id_CB1", "player.name_CB1", "player.id_CB2", "player.name_CB2"],
            "CB-pair player resolution",
        )
        cb1 = plot_df[["player.id_CB1", "player.name_CB1"]].rename(
            columns={"player.id_CB1": "player.id", "player.name_CB1": "player.name"}
        )
        cb2 = plot_df[["player.id_CB2", "player.name_CB2"]].rename(
            columns={"player.id_CB2": "player.id", "player.name_CB2": "player.name"}
        )
        return pd.concat([cb1, cb2], ignore_index=True).drop_duplicates()

    def _resolve_pair_row(self, plot_df: pd.DataFrame, *, CB_Pair: Any = None, CB_1: Any = None, CB_2: Any = None, CB_1_ID: Any = None, CB_2_ID: Any = None) -> Optional[pd.Series]:
        """
        Resolve one pair row from flexible pair selectors.

        Supported selection modes:
        - `CB_Pair` string formats (`A + B`, `A and B`, `A & B`),
        - `CB_Pair` tuple/list `(CB_1, CB_2)`,
        - dict-based specs with `CB_1`/`CB_2` and optional IDs,
        - explicit `CB_1`/`CB_2` + optional IDs.

        Returns:
            Matching pair row, or `None` when no selector was provided.
        """
        if (
            CB_Pair is None
            and CB_1 is None
            and CB_2 is None
            and CB_1_ID is None
            and CB_2_ID is None
        ):
            return None

        if CB_Pair is not None:
            if isinstance(CB_Pair, str):
                CB_1, CB_2 = self._split_pair_string(CB_Pair)
            elif isinstance(CB_Pair, (list, tuple)) and len(CB_Pair) == 2:
                CB_1, CB_2 = CB_Pair[0], CB_Pair[1]
            elif isinstance(CB_Pair, dict):
                CB_1 = CB_Pair.get("CB_1", CB_1)
                CB_2 = CB_Pair.get("CB_2", CB_2)
                CB_1_ID = CB_Pair.get("CB_1_ID", CB_1_ID)
                CB_2_ID = CB_Pair.get("CB_2_ID", CB_2_ID)
                if CB_Pair.get("CB_Pair") is not None:
                    CB_1, CB_2 = self._split_pair_string(CB_Pair["CB_Pair"])
            else:
                raise ValueError(
                    "CB_Pair must be a string ('A + B', 'A and B', 'A & B'), "
                    "a tuple/list of two CB inputs, or a dict with CB_1/CB_2 keys."
                )

        if (CB_1 is None and CB_1_ID is None) or (CB_2 is None and CB_2_ID is None):
            raise ValueError("Please provide both CB_1 and CB_2 (name/surname and/or ID).")

        candidates = self._player_candidates_from_pairs(plot_df)
        resolved_cb1 = self._resolve_entity(
            candidates,
            entity_value=CB_1,
            entity_id=CB_1_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        resolved_cb2 = self._resolve_entity(
            candidates,
            entity_value=CB_2,
            entity_id=CB_2_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )

        id_1 = resolved_cb1["player.id"]
        id_2 = resolved_cb2["player.id"]
        pair_mask = (
            ((plot_df["player.id_CB1"] == id_1) & (plot_df["player.id_CB2"] == id_2))
            | ((plot_df["player.id_CB1"] == id_2) & (plot_df["player.id_CB2"] == id_1))
        )
        matches = plot_df[pair_mask]
        if matches.empty:
            raise ValueError(
                f"No CB pair found for '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple rows found for pair '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        return matches.iloc[0]

    def _resolve_pair_row_by_rank(
        self,
        plot_df: pd.DataFrame,
        *,
        CB_Pair_Rank: Any,
    ) -> pd.Series:
        """
        Resolve one CB-pair row by its raw fit-ranking value.

        Args:
            plot_df: Prepared CB-pair plotting dataframe.
            CB_Pair_Rank: Requested pair rank in `CB_pair_fit_rank`.

        Returns:
            One resolved pair row for the requested rank.
        """
        return self._resolve_row_by_rank(
            plot_df,
            rank_column=self.RAW_RANK_COLUMN,
            rank_value=CB_Pair_Rank,
            entity_label="CB pair",
            display_column="pair_name",
        )

    def _pair_key(self, row: pd.Series) -> str:
        """
        Build a stable deduplication key for a pair-like row.

        Prefers explicit `pair_key` when available, then falls back to `pair_name`.
        """
        if "pair_key" in row.index and pd.notna(row["pair_key"]):
            return str(row["pair_key"])
        return str(row.get("pair_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
        """
        Add one highlighted pair row to a distribution plot.

        This wrapper keeps annotation mode wiring centralized so callers can
        switch between single-entity and multi-entity annotation rendering
        without duplicating plotting code.
        """
        if multi_annotations:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_single_annotations=False,
                add_multi_annotations=True,
            )
        else:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=add_single_annotations,
            )

    def _build_pair_plot(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        highlighted_rows: Sequence[Tuple[pd.Series, str, bool]],
        multi_annotations: bool,
        section_title: Optional[str] = None,
    ) -> DistributionPlot:
        """
        Build one CB-pair figure from pre-resolved highlighted rows.
        """
        dist_plot = self._create_plot(
            metric_cols=metric_cols,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
            section_title=section_title,
        )
        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["pair_name"],
            legend="All pairs",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        for row, display_name, add_single_annotations in highlighted_rows:
            self._add_row_point(
                dist_plot,
                row=row,
                display_name=display_name,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=add_single_annotations,
            )

        return dist_plot

    def _collect_single_pair_highlights(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
        CB_Pair_Rank: Optional[int] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for a single CB-pair plot scenario.

        A pair can be selected either by identity selectors (`CB_Pair` or
        `CB_1`/`CB_2`) or by `CB_Pair_Rank`.
        """
        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()

        # In single-pair mode, rank-based selection and identity-based selection
        # are intentionally mutually exclusive to keep intent unambiguous.
        if CB_Pair_Rank is not None and any(
            value is not None for value in [CB_Pair, CB_1, CB_2, CB_1_ID, CB_2_ID]
        ):
            raise ValueError(
                "Please provide either CB_Pair_Rank or CB_Pair/CB_1/CB_2 selectors, not both."
            )

        if CB_Pair_Rank is not None:
            selected_pair = self._resolve_pair_row_by_rank(
                plot_df,
                CB_Pair_Rank=CB_Pair_Rank,
            )
        else:
            selected_pair = self._resolve_pair_row(
                plot_df,
                CB_Pair=CB_Pair,
                CB_1=CB_1,
                CB_2=CB_2,
                CB_1_ID=CB_1_ID,
                CB_2_ID=CB_2_ID,
            )
        if selected_pair is not None:
            selected_key = self._pair_key(selected_pair)
            used_keys.add(selected_key)
            highlighted_rows.append((selected_pair, selected_pair["pair_name"], True))

        if include_best:
            best_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=False).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                highlighted_rows.append((best_pair, best_pair["pair_name"], True))

        if include_worst:
            worst_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=True).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                highlighted_rows.append((worst_pair, worst_pair["pair_name"], True))

        if include_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=metric_cols,
                metric_labels=dict(metric_labels),
                metric_value_columns=dict(metric_value_columns),
                display_name_col="pair_name",
                display_name_value="CB-Pairs' League Average",
                hover_payload_suffix="_hover_payload",
            )
            average_point["pair_key"] = "__pair_average__"
            highlighted_rows.append((average_point, "CB-Pairs' League Average", False))

        return highlighted_rows

    def _collect_comparison_pair_highlights(
        self,
        *,
        plot_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
        CB_Pairs: Optional[Sequence[Union[str, Sequence[Any], Dict[str, Any]]]] = None,
        CB_Pair_Rank: Optional[int] = None,
        CB_Pair_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for CB-pair comparison plots.

        Comparison selectors can include explicit pair identities and/or raw
        ranking selectors (`CB_Pair_Rank`, `CB_Pair_Ranks`).
        """
        specs: List[Dict[str, Any]] = []
        for item in (CB_Pairs or []):
            if isinstance(item, str):
                specs.append({"CB_Pair": item})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                specs.append({"CB_1": item[0], "CB_2": item[1]})
            elif isinstance(item, dict):
                specs.append(dict(item))
            else:
                raise ValueError(
                    "Each item in CB_Pairs must be either a pair string, a tuple/list of length 2, or a dict with CB_1/CB_2."
                )

        # Rank-based pair selection is additive in comparison mode.
        # It can be used alone or combined with explicit pair selectors.
        if CB_Pair_Ranks is not None:
            if CB_Pair_Rank is not None:
                raise ValueError(
                    "Please provide either CB_Pair_Rank or CB_Pair_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Pair_Ranks,
                param_name="CB_Pair_Ranks",
            ):
                specs.append({"CB_Pair_Rank": rank_value})
        if CB_Pair_Rank is not None:
            specs.append(
                {
                    "CB_Pair_Rank": self._coerce_positive_rank(
                        CB_Pair_Rank,
                        param_name="CB_Pair_Rank",
                    )
                }
            )

        if not specs and not any([include_best, include_worst, include_average]):
            raise ValueError(
                "Please provide at least one pair selector via CB_Pairs, CB_Pair_Rank, or CB_Pair_Ranks; "
                "or enable at least one of include_best/include_worst/include_average."
            )

        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()
        for spec in specs:
            if spec.get("CB_Pair_Rank") is not None:
                selected_pair = self._resolve_pair_row_by_rank(
                    plot_df,
                    CB_Pair_Rank=spec.get("CB_Pair_Rank"),
                )
            else:
                selected_pair = self._resolve_pair_row(
                    plot_df,
                    CB_Pair=spec.get("CB_Pair"),
                    CB_1=spec.get("CB_1"),
                    CB_2=spec.get("CB_2"),
                    CB_1_ID=spec.get("CB_1_ID"),
                    CB_2_ID=spec.get("CB_2_ID"),
                )
                if selected_pair is None:
                    continue

            key = self._pair_key(selected_pair)
            if key in used_keys:
                continue
            used_keys.add(key)
            highlighted_rows.append((selected_pair, selected_pair["pair_name"], True))

        if include_best:
            best_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=False).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                highlighted_rows.append((best_pair, best_pair["pair_name"], True))

        if include_worst:
            worst_pair = plot_df.sort_values("CB_pair_fit_z_score", ascending=True).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                highlighted_rows.append((worst_pair, worst_pair["pair_name"], True))

        if include_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=metric_cols,
                metric_labels=dict(metric_labels),
                metric_value_columns=dict(metric_value_columns),
                display_name_col="pair_name",
                display_name_value="CB-Pairs' League Average",
                hover_payload_suffix="_hover_payload",
            )
            average_point["pair_key"] = "__pair_average__"
            highlighted_rows.append((average_point, "CB-Pairs' League Average", False))

        return highlighted_rows

    def Plot_CB_Pair(
        self,
        *,
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
        CB_Pair_Rank: Optional[int] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
        multi_annotations: bool = False,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        split_view: bool = True,
        render_side_by_side: bool = True,
        show: bool = True,
    ) -> Union[DistributionPlot, Tuple[DistributionPlot, DistributionPlot]]:
        """
        Plot one selected CB-pair (optional) plus optional best/worst/average references.

        Selection options:
        - `CB_Pair` / `CB_1` + `CB_2` (with optional IDs),
        - `CB_Pair_Rank` (raw pair-fit ranking value).
        """
        cfg = self._resolve_pair_metric_configs(
            split_view=split_view,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
        )
        combined_metrics, combined_labels, combined_values = cfg["combined"]
        plot_df = self._prepare_plot_df(
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
        )

        highlighted_rows = self._collect_single_pair_highlights(
            plot_df=plot_df,
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
            CB_Pair=CB_Pair,
            CB_1=CB_1,
            CB_2=CB_2,
            CB_1_ID=CB_1_ID,
            CB_2_ID=CB_2_ID,
            CB_Pair_Rank=CB_Pair_Rank,
            include_best=include_best,
            include_worst=include_worst,
            include_average=include_average,
        )

        if split_view:
            left_metrics_cfg, left_labels_cfg, left_values_cfg = cfg["left"]
            right_metrics_cfg, right_labels_cfg, right_values_cfg = cfg["right"]
            left_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=left_metrics_cfg,
                metric_labels=left_labels_cfg,
                metric_value_columns=left_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Fit / Composition Metrics",
            )
            right_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=right_metrics_cfg,
                metric_labels=right_labels_cfg,
                metric_value_columns=right_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Individual Quality Metrics",
            )

            if show:
                rendered_side_by_side = False
                if render_side_by_side:
                    rendered_side_by_side = self._show_figures_side_by_side(
                        [left_plot.fig, right_plot.fig]
                    )
                if not rendered_side_by_side:
                    left_plot.fig.show()
                    right_plot.fig.show()
            return left_plot, right_plot

        single_metrics, single_labels, single_values = cfg["single"]
        single_plot = self._build_pair_plot(
            plot_df=plot_df,
            metric_cols=single_metrics,
            metric_labels=single_labels,
            metric_value_columns=single_values,
            highlighted_rows=highlighted_rows,
            multi_annotations=multi_annotations,
            section_title=None,
        )
        if show:
            single_plot.fig.show()
        return single_plot

    def Plot_CB_Pairs_Comparison(
        self,
        *,
        CB_Pairs: Optional[Sequence[Union[str, Sequence[Any], Dict[str, Any]]]] = None,
        CB_Pair_Rank: Optional[int] = None,
        CB_Pair_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
        multi_annotations: bool = True,
        left_metrics: Optional[Sequence[str]] = None,
        right_metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        split_view: bool = True,
        render_side_by_side: bool = True,
        show: bool = True,
    ) -> Union[DistributionPlot, Tuple[DistributionPlot, DistributionPlot]]:
        """
        Compare multiple user-selected CB-pairs plus optional best/worst/average overlays.

        Pair selection supports:
        - explicit pair selectors (`CB_Pairs`),
        - rank selectors (`CB_Pair_Rank`, `CB_Pair_Ranks`).
        """
        cfg = self._resolve_pair_metric_configs(
            split_view=split_view,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            metric_labels=metric_labels,
            metric_value_columns=metric_value_columns,
        )
        combined_metrics, combined_labels, combined_values = cfg["combined"]
        plot_df = self._prepare_plot_df(
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
        )

        highlighted_rows = self._collect_comparison_pair_highlights(
            plot_df=plot_df,
            metric_cols=combined_metrics,
            metric_labels=combined_labels,
            metric_value_columns=combined_values,
            CB_Pairs=CB_Pairs,
            CB_Pair_Rank=CB_Pair_Rank,
            CB_Pair_Ranks=CB_Pair_Ranks,
            include_best=include_best,
            include_worst=include_worst,
            include_average=include_average,
        )

        if split_view:
            left_metrics_cfg, left_labels_cfg, left_values_cfg = cfg["left"]
            right_metrics_cfg, right_labels_cfg, right_values_cfg = cfg["right"]
            left_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=left_metrics_cfg,
                metric_labels=left_labels_cfg,
                metric_value_columns=left_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Fit / Composition Metrics",
            )
            right_plot = self._build_pair_plot(
                plot_df=plot_df,
                metric_cols=right_metrics_cfg,
                metric_labels=right_labels_cfg,
                metric_value_columns=right_values_cfg,
                highlighted_rows=highlighted_rows,
                multi_annotations=multi_annotations,
                section_title="Individual Quality Metrics",
            )

            if show:
                rendered_side_by_side = False
                if render_side_by_side:
                    rendered_side_by_side = self._show_figures_side_by_side(
                        [left_plot.fig, right_plot.fig]
                    )
                if not rendered_side_by_side:
                    left_plot.fig.show()
                    right_plot.fig.show()
            return left_plot, right_plot

        single_metrics, single_labels, single_values = cfg["single"]
        single_plot = self._build_pair_plot(
            plot_df=plot_df,
            metric_cols=single_metrics,
            metric_labels=single_labels,
            metric_value_columns=single_values,
            highlighted_rows=highlighted_rows,
            multi_annotations=multi_annotations,
            section_title=None,
        )
        if show:
            single_plot.fig.show()
        return single_plot


class Anchor_CB_Companion_Fit_Ball_Passing_Distribution_Plot(_Ball_Passing_Distribution_Resolver):
    """
    Build directional anchor-to-companion Ball-Passing distribution plots.

    The class includes default companion-fit metrics/labels/raw mappings but still
    supports constructor-level and per-call metric overrides.
    """

    DEFAULT_METRICS = [
        "coverage_gain_z_within_anchor",
        "CB_pair_fit_z_score",
        "companion_fit_score_z_within_anchor",
        "companion_rank_for_anchor_z_score",
    ]
    DEFAULT_METRIC_LABELS = {
        "coverage_gain_z_within_anchor": "Deficit-Coverage Gain (Within Anchor CB)",
        "CB_pair_fit_z_score": "CB-Pair (Overall) Fit Score",
        "companion_fit_score_z_within_anchor": "CB-Companion Fit Score (Within Anchor CB)",
        "companion_rank_for_anchor_z_score": "CB-Companion Fit Ranking (Within Anchor CB's Sample)",
        "companion_rank_for_anchor": "CB-Companion Fit Ranking (Within Anchor CB's Sample)",
    }
    DEFAULT_METRIC_VALUE_COLUMNS = {
        "coverage_gain_z_within_anchor": "coverage_gain_raw",
        "CB_pair_fit_z_score": None,
        "companion_fit_score_z_within_anchor": "companion_fit_score",
        "companion_rank_for_anchor_z_score": "companion_rank_for_anchor",
    }

    def __init__(
        self,
        *,
        df_ball_passing_companion_fit: Optional[pd.DataFrame] = None,
        df_ground_duel_companion_fit: Optional[pd.DataFrame] = None,
        companion_plot_metric_cols: Optional[Sequence[str]] = None,
        companion_metric_labels: Optional[Mapping[str, str]] = None,
        companion_metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        """
        Args:
            df_ball_passing_companion_fit: Canonical directional anchor-to-partner fit dataframe.
            df_ground_duel_companion_fit: Legacy alias for backward compatibility.
            companion_plot_metric_cols: Optional constructor-level metric list.
            companion_metric_labels: Optional constructor-level label map.
            companion_metric_value_columns: Optional constructor-level raw-value map.
        """
        if (
            df_ball_passing_companion_fit is not None
            and df_ground_duel_companion_fit is not None
        ):
            raise ValueError(
                "Please provide only one of `df_ball_passing_companion_fit` or `df_ground_duel_companion_fit`."
            )
        if (
            df_ball_passing_companion_fit is None
            and df_ground_duel_companion_fit is None
        ):
            raise ValueError(
                "Please provide `df_ball_passing_companion_fit` (or legacy `df_ground_duel_companion_fit`)."
            )

        self.df_ball_passing_companion_fit = (
            df_ball_passing_companion_fit
            if df_ball_passing_companion_fit is not None
            else df_ground_duel_companion_fit
        )
        self._configured_companion_plot_metric_cols = (
            list(companion_plot_metric_cols)
            if companion_plot_metric_cols is not None
            else None
        )
        self._configured_companion_metric_labels = (
            dict(companion_metric_labels)
            if companion_metric_labels is not None
            else None
        )
        self._configured_companion_metric_value_columns = (
            dict(companion_metric_value_columns)
            if companion_metric_value_columns is not None
            else None
        )
        self._anchor_player_id: Optional[Any] = None
        self._anchor_player_name: Optional[str] = None


    def _resolve_current_metric_config(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Resolve call-ready metric config with precedence:
        method override > constructor config > class defaults.
        """
        return self._resolve_metric_config(
            default_metrics=self.DEFAULT_METRICS,
            default_metric_labels=self.DEFAULT_METRIC_LABELS,
            default_metric_value_columns=self.DEFAULT_METRIC_VALUE_COLUMNS,
            configured_metrics=self._configured_companion_plot_metric_cols,
            configured_metric_labels=self._configured_companion_metric_labels,
            configured_metric_value_columns=self._configured_companion_metric_value_columns,
            override_metrics=metrics,
            override_metric_labels=metric_labels,
            override_metric_value_columns=metric_value_columns,
        )

    def _anchor_candidates(self) -> pd.DataFrame:
        """
        Build candidate anchor table for anchor selection resolution.

        Returns:
            Dataframe with `anchor_player_id` and `anchor_player_name`.
            If IDs are unavailable in source data, synthetic IDs are generated.
        """
        self._ensure_columns(
            self.df_ball_passing_companion_fit,
            ["anchor_player_name"],
            "companion-fit anchor resolution",
        )
        if "anchor_player_id" in self.df_ball_passing_companion_fit.columns:
            return self.df_ball_passing_companion_fit[
                ["anchor_player_id", "anchor_player_name"]
            ].drop_duplicates()

        candidates = self.df_ball_passing_companion_fit[["anchor_player_name"]].drop_duplicates().copy()
        candidates["anchor_player_id"] = np.arange(len(candidates))
        return candidates

    def _resolve_anchor(self, *, Anchor_CB: Any = None, Anchor_CB_ID: Any = None) -> pd.Series:
        """
        Resolve the active anchor CB from name/surname and/or ID input.

        Behavior:
        - if no input is provided, reuses previously initialized anchor when possible; otherwise falls back to first alphabetical anchor.
        - if ID is provided but the dataset has no anchor ID column, raises.
        """
        candidates = self._anchor_candidates()

        if Anchor_CB is None and Anchor_CB_ID is None:
            if self._anchor_player_name is not None:
                cached_by_name = candidates[
                    candidates["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(self._anchor_player_name)
                ]
                if not cached_by_name.empty:
                    return cached_by_name.iloc[0]
            return candidates.sort_values("anchor_player_name").iloc[0]

        if (
            Anchor_CB_ID is not None
            and "anchor_player_id" not in self.df_ball_passing_companion_fit.columns
        ):
            raise ValueError(
                "Anchor_CB_ID was provided, but `anchor_player_id` is not available in the companion dataframe."
            )

        return self._resolve_entity(
            candidates,
            entity_value=Anchor_CB,
            entity_id=Anchor_CB_ID,
            id_col="anchor_player_id",
            name_col="anchor_player_name",
            entity_label="anchor CB",
        )

    def Initialize_Desired_Anchor_CB(self, *, Anchor_CB: Any = None, Anchor_CB_ID: Any = None) -> pd.Series:
        """
        Resolve and cache the anchor CB used by subsequent companion plots.
        """
        resolved_anchor = self._resolve_anchor(Anchor_CB=Anchor_CB, Anchor_CB_ID=Anchor_CB_ID)
        self._anchor_player_id = resolved_anchor.get("anchor_player_id", None)
        self._anchor_player_name = str(resolved_anchor["anchor_player_name"])
        return resolved_anchor

    def _build_anchor_plot_df(
        self,
        *,
        anchor_name: str,
        anchor_id: Any = None,
        top_n: Optional[int] = None,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """
        Build anchor-filtered companion-fit dataframe and attach hover payloads.
        """
        plot_df = self.df_ball_passing_companion_fit.copy()
        self._ensure_columns(
            plot_df,
            ["anchor_player_name", "partner_player_name", "companion_fit_score"],
            "companion-fit plotting dataframe",
        )

        if anchor_id is not None and "anchor_player_id" in plot_df.columns:
            filtered = plot_df[plot_df["anchor_player_id"] == anchor_id].copy()
            if filtered.empty:
                filtered = plot_df[
                    plot_df["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(anchor_name)
                ].copy()
        else:
            filtered = plot_df[
                plot_df["anchor_player_name"].map(self._normalize_text)
                == self._normalize_text(anchor_name)
            ].copy()

        if filtered.empty:
            raise ValueError(f"No companion-fit rows found for anchor CB '{anchor_name}'.")

        if "coverage_gain_z_within_anchor" not in filtered.columns and "coverage_gain_raw" in filtered.columns:
            filtered["coverage_gain_z_within_anchor"] = self._z_standardize(filtered["coverage_gain_raw"])

        if "companion_fit_score_z_within_anchor" not in filtered.columns:
            filtered["companion_fit_score_z_within_anchor"] = self._z_standardize(
                filtered["companion_fit_score"]
            )

        if "companion_rank_for_anchor_z_score" not in filtered.columns:
            if "companion_rank_for_anchor" in filtered.columns:
                filtered["companion_rank_for_anchor_z_score"] = self._z_standardize(
                    filtered["companion_rank_for_anchor"],
                    invert=True,
                    scale=2.0,
                )
            else:
                filtered["companion_rank_for_anchor_z_score"] = float("nan")

        filtered = filtered.sort_values(
            ["companion_fit_score", "partner_player_name"],
            ascending=[False, True],
        ).reset_index(drop=True)

        if top_n is not None:
            if not isinstance(top_n, (int, np.integer)) or int(top_n) <= 0:
                raise ValueError("top_n must be a positive integer or None.")
            filtered = filtered.head(int(top_n)).copy().reset_index(drop=True)

        return self._attach_hover_payload(
            filtered,
            metric_cols=metric_cols,
            metric_labels=dict(metric_labels),
            metric_value_columns=dict(metric_value_columns),
            hover_payload_suffix="_hover_payload",
            fill_missing_rank_z=True,
            rank_z_scale=2.0,
        )

    def _resolve_companion_row(
        self,
        plot_df: pd.DataFrame,
        *,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
    ) -> Optional[pd.Series]:
        """
        Resolve one companion row by either:
        1) companion name/surname,
        2) companion ID, or
        3) companion rank within the current anchor sample.
        """
        if Companion_CB is None and Companion_CB_ID is None and Companion_Rank is None:
            return None

        if Companion_Rank is not None:
            if Companion_CB is not None or Companion_CB_ID is not None:
                raise ValueError(
                    "Please specify either Companion_Rank or Companion_CB/Companion_CB_ID, not both."
                )
            if not isinstance(Companion_Rank, (int, np.integer)) or int(Companion_Rank) <= 0:
                raise ValueError("Companion_Rank must be a positive integer.")

            requested_rank = int(Companion_Rank)
            if "companion_rank_for_anchor" in plot_df.columns:
                rank_values = pd.to_numeric(
                    plot_df["companion_rank_for_anchor"], errors="coerce"
                )
                matches = plot_df[rank_values == float(requested_rank)]
            else:
                # Fallback when rank column is missing: use current deterministic order
                # (already sorted by companion_fit_score desc, partner name asc).
                if requested_rank > len(plot_df):
                    matches = plot_df.iloc[0:0]
                else:
                    matches = plot_df.iloc[[requested_rank - 1]]

            if matches.empty:
                top_n_hint = (
                    " The requested rank may be outside the currently filtered set (e.g., due to top_n)."
                    if len(plot_df) > 0
                    else ""
                )
                raise ValueError(
                    f"No companion found at rank #{requested_rank} for the current anchor CB's sample."
                    f"{top_n_hint}"
                )
            if len(matches) > 1:
                candidates = matches["partner_player_name"].astype(str).tolist()
                raise ValueError(
                    f"Multiple companions found for rank #{requested_rank}: {', '.join(candidates)}"
                )
            return matches.iloc[0]

        self._ensure_columns(
            plot_df,
            ["partner_player_name"],
            "companion selection",
        )

        has_partner_id = "partner_player_id" in plot_df.columns
        if Companion_CB_ID is not None and not has_partner_id:
            raise ValueError(
                "Companion_CB_ID was provided, but `partner_player_id` is not available in the companion dataframe."
            )

        if has_partner_id:
            candidates = plot_df[["partner_player_id", "partner_player_name"]].drop_duplicates().rename(
                columns={"partner_player_id": "player.id", "partner_player_name": "player.name"}
            )
            resolved = self._resolve_entity(
                candidates,
                entity_value=Companion_CB,
                entity_id=Companion_CB_ID,
                id_col="player.id",
                name_col="player.name",
                entity_label="companion CB",
            )
            matches = plot_df[plot_df["partner_player_id"] == resolved["player.id"]]
        else:
            candidates = plot_df[["partner_player_name"]].drop_duplicates().copy()
            candidates["player.id"] = np.arange(len(candidates))
            candidates = candidates.rename(columns={"partner_player_name": "player.name"})
            resolved = self._resolve_entity(
                candidates,
                entity_value=Companion_CB,
                entity_id=None,
                id_col="player.id",
                name_col="player.name",
                entity_label="companion CB",
            )
            matches = plot_df[
                plot_df["partner_player_name"].map(self._normalize_text)
                == self._normalize_text(resolved["player.name"])
            ]

        if matches.empty:
            raise ValueError(f"No companion fit found for '{Companion_CB}'.")
        return matches.iloc[0]

    def _companion_key(self, row: pd.Series) -> str:
        """
        Build a stable deduplication key for companion rows.
        """
        if "partner_player_id" in row.index and pd.notna(row["partner_player_id"]):
            return str(row["partner_player_id"])
        return self._normalize_text(row.get("partner_player_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
        """
        Add one highlighted companion row to the anchor-companion distribution.

        Args mirror the pair-level `_add_row_point` helper and are intentionally
        aligned for maintenance consistency across plot suites.
        """
        if multi_annotations:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_single_annotations=False,
                add_multi_annotations=True,
            )
        else:
            dist_plot.add_data_point(
                ser_plot=row,
                plots="",
                name=display_name,
                hover=hover_payload_suffix,
                hover_string=hover_string,
                add_annotations=add_single_annotations,
            )

    def _create_plot(
        self,
        *,
        anchor_name: str,
        num_candidates: int,
        metric_cols: Sequence[str],
        metric_labels: Mapping[str, str],
        metric_value_columns: Mapping[str, Optional[str]],
    ) -> DistributionPlot:
        """
        Instantiate and title one anchor-companion distribution figure.
        """
        dist_plot = DistributionPlot(
            columns=list(metric_cols),
            labels=["←   Worse", "Average", "Better   →"],
            quality_metric_labels=dict(metric_labels),
            quality_metric_value_columns=dict(metric_value_columns),
        )
        dist_plot.add_title(
            title=f"CB-Companion Ball Passing Potential Fits Distribution   →   {anchor_name} acting as the Anchor CB",
            subtitle=(
                f"All {num_candidates} potential companions for {anchor_name} (directional A → B)"
                "<br>"
                "Metrics are at the CB-Companion-level (i.e. within the Anchor CB's sample),"
                "<br>"
                "showing the expected contribution of the companion to the CB-Pair's overall fit with the specified Anchor CB."
            ),
        )
        return dist_plot

    def Plot_Companion_of_Anchor_CB(
        self,
        *,
        Anchor_CB: Any = None,
        Anchor_CB_ID: Any = None,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
        Companion_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_anchor_CB_pool_average: bool = False,
        top_n: Optional[int] = None,
        multi_annotations: bool = False,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot potential companions for one anchor CB, with optional highlighted rows.

        The highlighted rows can include:
        - an explicitly selected companion
        - an explicitly selected companion rank within the anchor CB's sample
        - multiple explicitly selected companion ranks within the anchor CB's sample
        - best/worst companion for the anchor
        - the anchor's companion-pool average
        """
        resolved_metrics, resolved_labels, resolved_value_columns = (
            self._resolve_current_metric_config(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_value_columns=metric_value_columns,
            )
        )

        resolved_anchor = self.Initialize_Desired_Anchor_CB(
            Anchor_CB=Anchor_CB,
            Anchor_CB_ID=Anchor_CB_ID,
        )
        anchor_name = str(resolved_anchor["anchor_player_name"])
        anchor_id = resolved_anchor.get("anchor_player_id", None)

        plot_df = self._build_anchor_plot_df(
            anchor_name=anchor_name,
            anchor_id=anchor_id,
            top_n=top_n,
            metric_cols=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )
        dist_plot = self._create_plot(
            anchor_name=anchor_name,
            num_candidates=len(plot_df),
            metric_cols=resolved_metrics,
            metric_labels=resolved_labels,
            metric_value_columns=resolved_value_columns,
        )

        hover_payload_suffix = "_hover_payload"
        hover_string = "%{customdata[3]}"

        dist_plot.add_group_data(
            df_plot=plot_df,
            plots="",
            names=plot_df["partner_player_name"],
            legend="All companions",
            hover=hover_payload_suffix,
            hover_string=hover_string,
        )

        used_keys = set()

        selected_companions: List[pd.Series] = []

        # Selection mode A: multiple explicit ranks provided by the caller.
        # Each rank is resolved independently in current anchor/top_n context.
        if Companion_Ranks is not None:
            if Companion_Rank is not None:
                raise ValueError(
                    "Please provide either Companion_Rank or Companion_Ranks, not both."
                )
            if isinstance(Companion_Ranks, (str, bytes)):
                raise ValueError(
                    "Companion_Ranks must be a sequence of positive integers, not a string."
                )

            unique_ranks: List[int] = []
            for rank_value in Companion_Ranks:
                if not isinstance(rank_value, (int, np.integer)) or int(rank_value) <= 0:
                    raise ValueError(
                        f"Invalid rank '{rank_value}' in Companion_Ranks. All ranks must be positive integers."
                    )
                rank_int = int(rank_value)
                if rank_int not in unique_ranks:
                    unique_ranks.append(rank_int)

            for rank_int in unique_ranks:
                rank_selected_companion = self._resolve_companion_row(
                    plot_df,
                    Companion_Rank=rank_int,
                )
                if rank_selected_companion is not None:
                    selected_companions.append(rank_selected_companion)

        # Selection mode B: one explicit companion selector (name/ID/rank).
        single_selected_companion = self._resolve_companion_row(
            plot_df,
            Companion_CB=Companion_CB,
            Companion_CB_ID=Companion_CB_ID,
            Companion_Rank=Companion_Rank,
        )
        if single_selected_companion is not None:
            selected_companions.append(single_selected_companion)

        # Deduplicate selected rows so the same companion is not plotted twice
        # when selection criteria overlap (e.g., explicit rank equals best row).
        for selected_companion in selected_companions:
            selected_key = self._companion_key(selected_companion)
            if selected_key in used_keys:
                continue
            used_keys.add(selected_key)
            selected_label = f"{anchor_name} + {selected_companion['partner_player_name']}"
            self._add_row_point(
                dist_plot,
                row=selected_companion,
                display_name=selected_label,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=True,
            )

        if include_best:
            best_companion = plot_df.sort_values("companion_fit_score", ascending=False).iloc[0]
            best_key = self._companion_key(best_companion)
            if best_key not in used_keys:
                used_keys.add(best_key)
                best_label = f"{anchor_name} + {best_companion['partner_player_name']}"
                self._add_row_point(
                    dist_plot,
                    row=best_companion,
                    display_name=best_label,
                    hover_payload_suffix=hover_payload_suffix,
                    hover_string=hover_string,
                    multi_annotations=multi_annotations,
                    add_single_annotations=True,
                )

        if include_worst:
            worst_companion = plot_df.sort_values("companion_fit_score", ascending=True).iloc[0]
            worst_key = self._companion_key(worst_companion)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                worst_label = f"{anchor_name} + {worst_companion['partner_player_name']}"
                self._add_row_point(
                    dist_plot,
                    row=worst_companion,
                    display_name=worst_label,
                    hover_payload_suffix=hover_payload_suffix,
                    hover_string=hover_string,
                    multi_annotations=multi_annotations,
                    add_single_annotations=True,
                )

        if include_anchor_CB_pool_average:
            average_point = self._build_average_point(
                plot_df=plot_df,
                metric_cols=resolved_metrics,
                metric_labels=resolved_labels,
                metric_value_columns=resolved_value_columns,
                display_name_col="partner_player_name",
                display_name_value="Companion Pool Average",
                hover_payload_suffix=hover_payload_suffix,
            )
            average_point["partner_player_id"] = -1
            average_label = f"{anchor_name}'s Companion Pool Average"
            self._add_row_point(
                dist_plot,
                row=average_point,
                display_name=average_label,
                hover_payload_suffix=hover_payload_suffix,
                hover_string=hover_string,
                multi_annotations=multi_annotations,
                add_single_annotations=False,
            )

        if show:
            dist_plot.fig.show()
        return dist_plot





# --------------------------------------------------------------------------|
# "Global" Quality Radar Plotting Infrastructure                            |
# --------------------------------------------------------------------------|
# The classes below form a layered architecture:                            |
# - `_Global_Qualities_Radar_Resolver`: shared data/selection utilities     |
# - `Single_CB_*`, `CB_Pair_*`, `Anchor_CB_Companion_*`: public plot APIs   |
# --------------------------------------------------------------------------|


class _Global_Qualities_Radar_Resolver(_Ball_Passing_Distribution_Resolver):
    """
    Shared resolver and rendering utilities for global quality radar plots.

    This class intentionally inherits the robust entity-resolution and validation behaviors already used in the quality-specific distribution plot suites.
    """

    AXIS_ORDER = ("Ground Duels", "Aerial Duels", "Ball Passing")
    TRACE_COLORS = (
        Visual.bright_orange,
        Visual.magenta,
        Visual.bright_yellow,
        Visual.bright_blue,
        Visual.pink,
        Visual.gold,
        Visual.silver,
        Visual.ruby,
        Visual.aqua,
        Visual.emerald,
        Visual.lime,
        Visual.white,
    )
    TRACE_DASH_STYLES = (
        "solid",
        "dash",
        "dot",
        "dashdot",
        "longdash",
        "longdashdot",
    )

    @staticmethod
    def _format_float(value: Any, *, decimals: int = 2) -> str:
        """
        Format a numeric-like value for labels and hover text.
        """
        numeric_value = _Global_Qualities_Radar_Resolver._to_number_or_none(value)
        if numeric_value is None:
            return "N/A"
        return f"{numeric_value:.{decimals}f}"

    @staticmethod
    def _format_rank(value: Any) -> str:
        """
        Format a ranking value as `# <rank>` while handling nulls safely.
        """
        numeric_value = _Global_Qualities_Radar_Resolver._to_number_or_none(value)
        if numeric_value is None:
            return "# N/A"
        return f"# {int(round(numeric_value))}"

    @classmethod
    def _ensure_axis_mappings(
        cls,
        *,
        axis_value_columns: Mapping[str, str],
        axis_rank_columns: Optional[Mapping[str, str]] = None,
        axis_rank_z_columns: Optional[Mapping[str, str]] = None,
        context: str,
    ) -> Tuple[Dict[str, str], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
        """
        Validate and normalize axis configuration mappings.

        Returns:
            Tuple of dictionaries keyed by `AXIS_ORDER`:
            - value-column map,
            - raw-rank-column map (nullable),
            - rank-z-column map (nullable).
        """
        value_map: Dict[str, str] = {}
        rank_map: Dict[str, Optional[str]] = {}
        rank_z_map: Dict[str, Optional[str]] = {}

        for axis in cls.AXIS_ORDER:
            value_col = axis_value_columns.get(axis)
            if not value_col:
                raise ValueError(
                    f"Missing axis value-column mapping for '{axis}' in {context}."
                )
            value_map[axis] = str(value_col)

            rank_col = None
            if axis_rank_columns is not None:
                rank_col = axis_rank_columns.get(axis)
            rank_map[axis] = str(rank_col) if rank_col else None

            rank_z_col = None
            if axis_rank_z_columns is not None:
                rank_z_col = axis_rank_z_columns.get(axis)
            rank_z_map[axis] = str(rank_z_col) if rank_z_col else None

        return value_map, rank_map, rank_z_map

    @classmethod
    def _close_loop(cls, values: Sequence[Any]) -> List[Any]:
        """
        Close a radar polygon by repeating the first vertex at the end.
        """
        if not values:
            return []
        return [*values, values[0]]

    @classmethod
    def _resolve_radial_range(
        cls,
        values: Sequence[Any],
        *,
        radial_range: Optional[Sequence[float]] = None,
        min_abs_span: float = 0.6,
        padding_ratio: float = 0.15,
    ) -> Tuple[float, float]:
        """
        Build a symmetric radar radial range with optional user override.
        """
        if radial_range is not None:
            if len(radial_range) != 2:
                raise ValueError("radial_range must contain exactly 2 numeric values.")
            lower = cls._to_number_or_none(radial_range[0])
            upper = cls._to_number_or_none(radial_range[1])
            if lower is None or upper is None:
                raise ValueError("radial_range values must be numeric.")
            if lower >= upper:
                raise ValueError("radial_range must satisfy min < max.")
            return float(lower), float(upper)

        clean_values: List[float] = []
        for raw_value in values:
            numeric_value = cls._to_number_or_none(raw_value)
            if numeric_value is not None:
                clean_values.append(float(numeric_value))

        if not clean_values:
            return (-1.0, 1.0)

        max_abs = max(abs(min(clean_values)), abs(max(clean_values)))
        bound = max(max_abs * (1.0 + padding_ratio), min_abs_span)
        bound = float(np.round(bound, 3))
        return (-bound, bound)

    @classmethod
    def _trace_label(
        cls,
        *,
        entity_name: str,
        global_score: Any,
        global_rank: Any,
    ) -> str:
        """
        Build a concise, information-rich trace label.
        """
        return (
            f"{entity_name}   |   Global Quality Score   →   {cls._format_float(global_score)}   |   Global Quality Rank   →   {cls._format_rank(global_rank)}"
        )

    @classmethod
    def _build_axis_hover_lines(
        cls,
        *,
        row: pd.Series,
        axis_value_columns: Mapping[str, str],
        axis_rank_columns: Mapping[str, Optional[str]],
        axis_rank_z_columns: Mapping[str, Optional[str]],
        global_score_column: str,
        global_rank_column: str,
        global_rank_z_column: Optional[str] = None,
    ) -> Tuple[List[float], List[float], List[str], List[str]]:
        """
        Build axis values and hover payloads for both score and raw-rank radars.
        """
        axis_values: List[float] = []
        axis_rank_values: List[float] = []
        axis_hover_score_lines: List[str] = []
        axis_hover_rank_lines: List[str] = []

        global_score = row.get(global_score_column, float("nan"))
        global_rank = row.get(global_rank_column, float("nan"))
        global_rank_z = (
            row.get(global_rank_z_column, float("nan"))
            if global_rank_z_column
            else float("nan")
        )

        for axis in cls.AXIS_ORDER:
            value_col = axis_value_columns[axis]
            rank_col = axis_rank_columns.get(axis)
            rank_z_col = axis_rank_z_columns.get(axis)

            axis_value = row.get(value_col, float("nan"))
            axis_values.append(float(axis_value) if pd.notna(axis_value) else float("nan"))
            axis_rank = row.get(rank_col, float("nan")) if rank_col is not None else float("nan")
            axis_rank_values.append(float(axis_rank) if pd.notna(axis_rank) else float("nan"))

            score_hover_lines = [
                axis,
                f"Axis Quality Score (Z) = {cls._format_float(axis_value)}",
            ]
            if rank_col is not None:
                score_hover_lines.append(
                    f"Axis Quality Rank = {cls._format_rank(row.get(rank_col, float('nan')))}"
                )
            if rank_z_col is not None:
                score_hover_lines.append(
                    f"Axis Quality Rank Z-score = {cls._format_float(row.get(rank_z_col, float('nan')))}"
                )

            score_hover_lines.extend(
                [
                    f"Global Quality Score = {cls._format_float(global_score)}",
                    f"Global Quality Rank = {cls._format_rank(global_rank)}",
                ]
            )
            if global_rank_z_column is not None:
                score_hover_lines.append(
                    f"Global Quality Rank Z-score = {cls._format_float(global_rank_z)}"
                )

            rank_hover_lines = [
                axis,
                f"Axis Quality Rank = {cls._format_rank(axis_rank)}",
                f"Axis Quality Score (Z) = {cls._format_float(axis_value)}",
            ]
            if rank_z_col is not None:
                rank_hover_lines.append(
                    f"Axis Quality Rank Z-score = {cls._format_float(row.get(rank_z_col, float('nan')))}"
                )
            rank_hover_lines.extend(
                [
                    f"Global Quality Rank = {cls._format_rank(global_rank)}",
                    f"Global Quality Score = {cls._format_float(global_score)}",
                ]
            )
            if global_rank_z_column is not None:
                rank_hover_lines.append(
                    f"Global Quality Rank Z-score = {cls._format_float(global_rank_z)}"
                )

            axis_hover_score_lines.append("<br>".join(score_hover_lines))
            axis_hover_rank_lines.append("<br>".join(rank_hover_lines))

        return (
            axis_values,
            axis_rank_values,
            axis_hover_score_lines,
            axis_hover_rank_lines,
        )

    @classmethod
    def _resolve_rank_radial_range(
        cls,
        values: Sequence[Any],
    ) -> Tuple[float, float]:
        """
        Build the rank-based radial range for the left-side radar.
        """
        clean_values: List[float] = []
        for raw_value in values:
            numeric_value = cls._to_number_or_none(raw_value)
            if numeric_value is not None and numeric_value > 0:
                clean_values.append(float(numeric_value))

        if not clean_values:
            return (1.0, 5.0)

        upper = float(np.ceil(max(clean_values)))
        upper = max(upper, 5.0)
        return (1.0, upper)

    @classmethod
    def _build_rank_ticks(
        cls,
        *,
        rank_range: Tuple[float, float],
    ) -> Tuple[List[float], List[str]]:
        """
        Create readable raw-rank ticks for the left radar axis.
        """
        rank_min, rank_max = rank_range
        max_rank = int(max(2, np.ceil(rank_max)))

        if max_rank <= 6:
            tick_values = list(range(1, max_rank + 1))
        else:
            tick_values = sorted(
                {
                    1,
                    max_rank,
                    int(round(max_rank * 0.25)),
                    int(round(max_rank * 0.50)),
                    int(round(max_rank * 0.75)),
                }
            )
            tick_values = [max(1, min(max_rank, tick)) for tick in tick_values]
            tick_values = sorted(set(tick_values))

        return [float(tick) for tick in tick_values], [f"#{tick}" for tick in tick_values]

    @classmethod
    def _create_dual_radar_figure(
        cls,
        *,
        title: str,
        subtitle: str,
        score_radial_range: Tuple[float, float],
        rank_radial_range: Tuple[float, float],
    ) -> go.Figure:
        """
        Instantiate a two-panel radar figure:
        - left: raw rank positioning per axis,
        - right: z-score profile with qualitative radial labels.
        """
        rank_tick_values, rank_tick_text = cls._build_rank_ticks(
            rank_range=rank_radial_range,
        )
        score_midpoint = (
            0.0
            if score_radial_range[0] <= 0.0 <= score_radial_range[1]
            else float(np.mean(score_radial_range))
        )
        # Reserve vertical space between subplot titles and each radar panel.
        polar_domain_y = [0.02, 1.0]
        subplot_title_y = 1.075

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "polar"}, {"type": "polar"}]],
            horizontal_spacing=0.00,
            subplot_titles=(
                "Global Quality Ranking",
                "Global Quality Score Profile",
            ),
        )

        fig.update_layout(
            autosize=True,
            height=560,
            margin=dict(l=60, r=60, b=95, t=120, pad=16),
            paper_bgcolor=rgb_to_color(Visual.dark_green),
            plot_bgcolor=rgb_to_color(Visual.dark_green),
            polar=dict(
                bgcolor=rgb_to_color(Visual.dark_green),
                domain=dict(y=polar_domain_y),
                radialaxis=dict(
                    visible=True,
                    range=[rank_radial_range[0], rank_radial_range[1]],
                    autorange="reversed",
                    gridcolor=rgb_to_color(Visual.plot_grid_green, 0.75),
                    linecolor=rgb_to_color(Visual.white, 0.75),
                    angle=90,
                    tickmode="array",
                    tickvals=rank_tick_values,
                    ticktext=rank_tick_text,
                    tickfont={
                        "color": rgb_to_color(Visual.white, 0.75),
                        "family": "Gilroy-Light",
                        "size": 11,
                    },
                    tickcolor=rgb_to_color(Visual.white, 0.75),
                    tickangle=90,
                ),
                angularaxis=dict(
                    gridcolor=rgb_to_color(Visual.plot_grid_green, 0.75),
                    linecolor=rgb_to_color(Visual.plot_grid_green, 0.75),
                    tickfont={
                        "color": rgb_to_color(Visual.white, 0.9),
                        "family": "Gilroy-Medium",
                        "size": 12,
                    },
                    tickcolor=rgb_to_color(Visual.white, 0.75),
                ),
            ),
            polar2=dict(
                bgcolor=rgb_to_color(Visual.dark_green),
                domain=dict(y=polar_domain_y),
                radialaxis=dict(
                    visible=True,
                    range=[score_radial_range[0], score_radial_range[1]],
                    gridcolor=rgb_to_color(Visual.plot_grid_green, 0.75),
                    linecolor=rgb_to_color(Visual.white, 0.75),
                    angle=90,
                    tickmode="array",
                    tickvals=[
                        float(score_radial_range[0]),
                        float(score_midpoint),
                        float(score_radial_range[1]),
                    ],
                    ticktext=["Worse   ↓", "Avg.", "Better   ↑"],
                    tickfont={
                        "color": rgb_to_color(Visual.white, 0.80),
                        "family": "Gilroy-Light",
                        "size": 11,
                    },
                    tickcolor=rgb_to_color(Visual.white, 0.75),
                    tickangle=90,
                ),
                angularaxis=dict(
                    gridcolor=rgb_to_color(Visual.plot_grid_green, 0.75),
                    linecolor=rgb_to_color(Visual.plot_grid_green, 0.75),
                    tickfont={
                        "color": rgb_to_color(Visual.white, 0.9),
                        "family": "Gilroy-Medium",
                        "size": 12,
                    },
                    tickcolor=rgb_to_color(Visual.white, 0.75),
                ),
            ),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(Visual.white),
                    "family": "Gilroy-Light",
                    "size": 11,
                },
                itemclick="toggle",
                itemdoubleclick=False,
                # itemclickside="toggle",
                # itemclicklegend="toggle",
                # itemclickgroup="toggle",
                x=0.5,
                xanchor="center",
                y=-0.20,
                yanchor="bottom",
            ),
            font={"color": rgb_to_color(Visual.white)},
            title={
                "text": (
                    f"<span style='font-size: 16px'>{title}</span>"
                    f"<br><span style='font-size: 12px'>{subtitle}</span>"
                ),
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(Visual.white),
                    "size": 13,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.95,
                "yanchor": "top",
            },
        )
        fig.update_annotations(
            font={
                "color": rgb_to_color(Visual.white),
                "family": "Gilroy-Medium",
                "size": 12,
            }
        )
        # Keep subplot titles clearly separated from the radar circles.
        for annotation in fig.layout.annotations:
            annotation.update(
                y=subplot_title_y,
                yanchor="bottom",
            )
        return fig

    @classmethod
    def _add_radar_trace(
        cls,
        fig: go.Figure,
        *,
        axis_values: Sequence[float],
        axis_hover: Sequence[str],
        trace_name: str,
        color_rgb: Tuple[int, int, int],
        line_dash: str = "solid",
        fill_opacity: float = 0.18,
        line_width: float = 2.25,
        marker_size: int = 8,
        subplot_ref: str = "polar",
        showlegend: bool = True,
        legendgroup: Optional[str] = None,
    ) -> None:
        """
        Add one closed radar trace (line + markers + translucent fill).
        """
        theta = cls._close_loop(list(cls.AXIS_ORDER))
        r_values = cls._close_loop([float(v) if pd.notna(v) else float("nan") for v in axis_values])
        hover_values = cls._close_loop(list(axis_hover))

        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=theta,
                mode="lines+markers",
                name=trace_name,
                line={
                    "color": rgb_to_color(color_rgb, 0.95),
                    "width": line_width,
                    "dash": line_dash,
                },
                marker={
                    "size": marker_size,
                    "color": rgb_to_color(color_rgb, 0.95),
                    "line": {
                        "color": rgb_to_color(color_rgb, 1.0),
                        "width": 1.2,
                    },
                },
                fill="toself",
                fillcolor=rgb_to_color(color_rgb, fill_opacity),
                text=hover_values,
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                subplot=subplot_ref,
                showlegend=showlegend,
                legendgroup=legendgroup,
            )
        )

    @classmethod
    def _plot_profiles(
        cls,
        *,
        profiles: Sequence[Dict[str, Any]],
        title: str,
        subtitle: str,
        radial_range: Optional[Sequence[float]] = None,
        show: bool = True,
    ) -> Optional[go.Figure]:
        """
        Render profile dictionaries into a two-panel radar figure.

        Each profile dictionary must contain:
        - `axis_values`: length-3 list aligned to `AXIS_ORDER`,
        - `axis_rank_values`: raw-rank values aligned to `AXIS_ORDER`,
        - `axis_hover`: length-3 list aligned to `AXIS_ORDER`,
        - `axis_rank_hover`: length-3 list aligned to `AXIS_ORDER`,
        - `trace_name`: legend label,
        - `color_rgb`: RGB tuple.
        """
        if not profiles:
            raise ValueError("At least one profile is required to render a radar plot.")

        pooled_score_values: List[Any] = []
        pooled_rank_values: List[Any] = []
        for profile in profiles:
            pooled_score_values.extend(profile["axis_values"])
            pooled_rank_values.extend(profile["axis_rank_values"])

        resolved_score_range = cls._resolve_radial_range(
            pooled_score_values,
            radial_range=radial_range,
        )
        resolved_rank_range = cls._resolve_rank_radial_range(pooled_rank_values)

        fig = cls._create_dual_radar_figure(
            title=title,
            subtitle=subtitle,
            score_radial_range=resolved_score_range,
            rank_radial_range=resolved_rank_range,
        )

        for index, profile in enumerate(profiles):
            legend_group = f"profile_{index}"

            cls._add_radar_trace(
                fig,
                axis_values=profile["axis_rank_values"],
                axis_hover=profile["axis_rank_hover"],
                trace_name=profile["trace_name"],
                color_rgb=profile["color_rgb"],
                line_dash=profile.get("line_dash", "solid"),
                fill_opacity=profile.get("fill_opacity", 0.18),
                line_width=profile.get("line_width", 2.4),
                marker_size=profile.get("marker_size", 8),
                subplot_ref="polar",
                showlegend=False,
                legendgroup=legend_group,
            )
            cls._add_radar_trace(
                fig,
                axis_values=profile["axis_values"],
                axis_hover=profile["axis_hover"],
                trace_name=profile["trace_name"],
                color_rgb=profile["color_rgb"],
                line_dash=profile.get("line_dash", "solid"),
                fill_opacity=profile.get("fill_opacity", 0.18),
                line_width=profile.get("line_width", 2.4),
                marker_size=profile.get("marker_size", 8),
                subplot_ref="polar2",
                showlegend=True,
                legendgroup=legend_group,
            )

        if show:
            fig.show()
            return None
        return fig


class Single_CB_Global_Qualities_Radar_Plot(_Global_Qualities_Radar_Resolver):
    """
    Global radar analysis for individual CBs across the 3 quality dimensions.

    This class expects the strict-intersection global single-CB table and supports:
    - one-player radar analysis,
    - multi-player comparison radar analysis,
    - selectors by name, ID, and global rank.
    """

    DEFAULT_AXIS_VALUE_COLUMNS = {
        "Ground Duels": "CB_ground_duels_quality_z_score",
        "Aerial Duels": "CB_aerial_duels_quality_z_score",
        "Ball Passing": "CB_ball_passing_quality_z_score",
    }
    DEFAULT_AXIS_RANK_COLUMNS = {
        "Ground Duels": "CB_rank_for_ground_duels_quality",
        "Aerial Duels": "CB_rank_for_aerial_duels_quality",
        "Ball Passing": "CB_rank_for_ball_passing_quality",
    }
    DEFAULT_AXIS_RANK_Z_COLUMNS = {
        "Ground Duels": "CB_rank_for_ground_duels_quality_z_score",
        "Aerial Duels": "CB_rank_for_aerial_duels_quality_z_score",
        "Ball Passing": "CB_rank_for_ball_passing_quality_z_score",
    }

    GLOBAL_SCORE_COLUMN = "global_quality_z_score"
    GLOBAL_RANK_COLUMN = "global_quality_rank"
    GLOBAL_RANK_Z_COLUMN = "global_quality_rank_z_score"

    def __init__(
        self,
        *,
        df_global_single_cb: pd.DataFrame,
        axis_value_columns: Optional[Mapping[str, str]] = None,
        axis_rank_columns: Optional[Mapping[str, str]] = None,
        axis_rank_z_columns: Optional[Mapping[str, str]] = None,
        global_score_column: str = GLOBAL_SCORE_COLUMN,
        global_rank_column: str = GLOBAL_RANK_COLUMN,
        global_rank_z_column: str = GLOBAL_RANK_Z_COLUMN,
    ) -> None:
        """
        Args:
            df_global_single_cb: Global single-CB dataframe (strict intersection).
            axis_value_columns: Optional override map for radar axis value columns.
            axis_rank_columns: Optional override map for axis rank columns.
            axis_rank_z_columns: Optional override map for axis rank-z columns.
            global_score_column: Column storing global combined quality z-score.
            global_rank_column: Column storing global rank (1 = best).
            global_rank_z_column: Column storing global rank z-score.
        """
        self.df_global_single_cb = df_global_single_cb.copy()
        self.global_score_column = str(global_score_column)
        self.global_rank_column = str(global_rank_column)
        self.global_rank_z_column = str(global_rank_z_column)

        axis_values = dict(self.DEFAULT_AXIS_VALUE_COLUMNS)
        if axis_value_columns is not None:
            axis_values.update({str(k): str(v) for k, v in axis_value_columns.items()})

        axis_ranks = dict(self.DEFAULT_AXIS_RANK_COLUMNS)
        if axis_rank_columns is not None:
            axis_ranks.update({str(k): str(v) for k, v in axis_rank_columns.items()})

        axis_rank_z = dict(self.DEFAULT_AXIS_RANK_Z_COLUMNS)
        if axis_rank_z_columns is not None:
            axis_rank_z.update(
                {str(k): str(v) for k, v in axis_rank_z_columns.items()}
            )

        (
            self.axis_value_columns,
            self.axis_rank_columns,
            self.axis_rank_z_columns,
        ) = self._ensure_axis_mappings(
            axis_value_columns=axis_values,
            axis_rank_columns=axis_ranks,
            axis_rank_z_columns=axis_rank_z,
            context="Single_CB_Global_Qualities_Radar_Plot",
        )

    def _prepare_plot_df(self) -> pd.DataFrame:
        """
        Validate schema and prepare a deterministic plotting dataframe.
        """
        required_columns = [
            "player.id",
            "player.name",
            self.global_score_column,
            self.global_rank_column,
            self.global_rank_z_column,
        ] + list(self.axis_value_columns.values())

        for maybe_col in self.axis_rank_columns.values():
            if maybe_col is not None:
                required_columns.append(maybe_col)
        for maybe_col in self.axis_rank_z_columns.values():
            if maybe_col is not None:
                required_columns.append(maybe_col)

        self._ensure_columns(
            self.df_global_single_cb,
            required_columns,
            "global single-CB radar plotting dataframe",
        )

        plot_df = self.df_global_single_cb.copy()
        plot_df[self.global_rank_column] = pd.to_numeric(
            plot_df[self.global_rank_column],
            errors="coerce",
        )
        plot_df = plot_df.sort_values(
            [self.global_rank_column, "player.name"],
            ascending=[True, True],
        ).reset_index(drop=True)
        return plot_df

    def _resolve_single_cb(
        self,
        plot_df: pd.DataFrame,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Any = None,
    ) -> pd.Series:
        """
        Resolve one CB row from name/ID/global-rank selectors.
        """
        if CB is None and CB_ID is None and CB_Rank is None:
            return plot_df.iloc[0]

        if CB_Rank is not None:
            if CB is not None or CB_ID is not None:
                raise ValueError("Provide either CB_Rank or CB/CB_ID, not both.")
            return self._resolve_row_by_rank(
                plot_df,
                rank_column=self.global_rank_column,
                rank_value=CB_Rank,
                entity_label="CB player",
                display_column="player.name",
            )

        candidates = plot_df[["player.id", "player.name"]].drop_duplicates()
        resolved = self._resolve_entity(
            candidates,
            entity_value=CB,
            entity_id=CB_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        matches = plot_df[plot_df["player.id"] == resolved["player.id"]]
        if matches.empty:
            raise ValueError(
                f"Resolved CB '{resolved['player.name']}' is not available in the global plotting dataframe."
            )
        return matches.iloc[0]

    def _build_profile(
        self,
        row: pd.Series,
        *,
        trace_name: str,
        color_rgb: Tuple[int, int, int],
        line_dash: str = "solid",
        fill_opacity: float = 0.18,
    ) -> Dict[str, Any]:
        """
        Build one plotting profile dictionary for `_plot_profiles`.
        """
        (
            axis_values,
            axis_rank_values,
            axis_hover,
            axis_rank_hover,
        ) = self._build_axis_hover_lines(
            row=row,
            axis_value_columns=self.axis_value_columns,
            axis_rank_columns=self.axis_rank_columns,
            axis_rank_z_columns=self.axis_rank_z_columns,
            global_score_column=self.global_score_column,
            global_rank_column=self.global_rank_column,
            global_rank_z_column=self.global_rank_z_column,
        )
        return {
            "axis_values": axis_values,
            "axis_rank_values": axis_rank_values,
            "axis_hover": axis_hover,
            "axis_rank_hover": axis_rank_hover,
            "trace_name": trace_name,
            "color_rgb": color_rgb,
            "line_dash": line_dash,
            "fill_opacity": fill_opacity,
        }

    def _build_average_profile(self, plot_df: pd.DataFrame) -> pd.Series:
        """
        Build a synthetic league-average row for radar overlays.
        """
        average_data: Dict[str, Any] = {
            "player.id": -1,
            "player.name": "CBs' Global Average",
        }

        mean_columns = (
            list(self.axis_value_columns.values())
            + [
                self.global_score_column,
                self.global_rank_column,
                self.global_rank_z_column,
            ]
            + [col for col in self.axis_rank_columns.values() if col is not None]
            + [col for col in self.axis_rank_z_columns.values() if col is not None]
        )
        for col in dict.fromkeys(mean_columns):
            average_data[col] = pd.to_numeric(plot_df[col], errors="coerce").mean()

        return pd.Series(average_data)

    def Plot_Single_CB(
        self,
        *,
        CB: Any = None,
        CB_ID: Any = None,
        CB_Rank: Any = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_league_average: bool = True,
        radial_range: Optional[Sequence[float]] = None,
        show: bool = True,
    ) -> Optional[go.Figure]:
        """
        Plot one selected CB on the global 3-quality radar.

        Optional references:
        - best global CB profile,
        - worst global CB profile,
        - global league-average profile.
        """
        plot_df = self._prepare_plot_df()
        selected = self._resolve_single_cb(
            plot_df,
            CB=CB,
            CB_ID=CB_ID,
            CB_Rank=CB_Rank,
        )

        used_ids = set()
        profiles: List[Dict[str, Any]] = []

        selected_id = selected["player.id"]
        used_ids.add(selected_id)
        profiles.append(
            self._build_profile(
                selected,
                trace_name=self._trace_label(
                    entity_name=str(selected["player.name"]),
                    global_score=selected[self.global_score_column],
                    global_rank=selected[self.global_rank_column],
                ),
                color_rgb=self.TRACE_COLORS[0],
                line_dash="solid",
            )
        )

        if include_best:
            best_row = plot_df.sort_values(
                [self.global_score_column, "player.name"],
                ascending=[False, True],
            ).iloc[0]
            best_id = best_row["player.id"]
            if best_id not in used_ids:
                used_ids.add(best_id)
                profiles.append(
                    self._build_profile(
                        best_row,
                        trace_name=self._trace_label(
                            entity_name=f"{best_row['player.name']} (Best)",
                            global_score=best_row[self.global_score_column],
                            global_rank=best_row[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[1],
                        line_dash="dash",
                    )
                )

        if include_worst:
            worst_row = plot_df.sort_values(
                [self.global_score_column, "player.name"],
                ascending=[True, True],
            ).iloc[0]
            worst_id = worst_row["player.id"]
            if worst_id not in used_ids:
                used_ids.add(worst_id)
                profiles.append(
                    self._build_profile(
                        worst_row,
                        trace_name=self._trace_label(
                            entity_name=f"{worst_row['player.name']} (Worst)",
                            global_score=worst_row[self.global_score_column],
                            global_rank=worst_row[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[2],
                        line_dash="dot",
                    )
                )

        if include_league_average:
            average_row = self._build_average_profile(plot_df)
            profiles.append(
                self._build_profile(
                    average_row,
                    trace_name=self._trace_label(
                        entity_name="CBs' Global Average",
                        global_score=average_row[self.global_score_column],
                        global_rank=average_row[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[3],
                    line_dash="longdash",
                )
            )

        return self._plot_profiles(
            profiles=profiles,
            title="Global CB Quality Radar Distribution <br> (Ground Duels + Aerial Duels + Ball Passing)",
            subtitle=(
                f"Strict 3-Quality Intersection Cohort: {len(plot_df)} CBs   |   Equal Weighting (⅓ Each Quality)"
                ),
            radial_range=radial_range,
            show=show,
        )

    def Plot_CBs_Comparison(
        self,
        *,
        CBs: Optional[Sequence[Any]] = None,
        CB_IDs: Optional[Sequence[Any]] = None,
        CB_Rank: Optional[Any] = None,
        CB_Ranks: Optional[Sequence[Any]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_league_average: bool = True,
        radial_range: Optional[Sequence[float]] = None,
        show: bool = True,
    ) -> Optional[go.Figure]:
        """
        Compare multiple CBs on a shared global 3-axis radar.

        Supported selectors:
        - explicit names (`CBs`),
        - explicit IDs (`CB_IDs`),
        - global rank selectors (`CB_Rank`, `CB_Ranks`).
        """
        plot_df = self._prepare_plot_df()

        requested_items: List[Tuple[str, Any, Any]] = []
        for cb_id in (CB_IDs or []):
            requested_items.append(("id", None, cb_id))
        for cb_name in (CBs or []):
            requested_items.append(("name", cb_name, None))

        if CB_Ranks is not None:
            if CB_Rank is not None:
                raise ValueError("Please provide either CB_Rank or CB_Ranks, not both.")
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Ranks,
                param_name="CB_Ranks",
            ):
                requested_items.append(("rank", rank_value, None))
        if CB_Rank is not None:
            requested_items.append(
                (
                    "rank",
                    self._coerce_positive_rank(CB_Rank, param_name="CB_Rank"),
                    None,
                )
            )

        if not requested_items and not any(
            [include_best, include_worst, include_league_average]
        ):
            raise ValueError(
                "Please provide at least one selector via CBs, CB_IDs, CB_Rank, or CB_Ranks; or enable one of include_best/include_worst/include_league_average."
            )

        profiles: List[Dict[str, Any]] = []
        used_ids = set()
        color_index = 0

        for mode, name_or_rank, cb_id in requested_items:
            if mode == "rank":
                row = self._resolve_single_cb(plot_df, CB_Rank=name_or_rank)
            else:
                row = self._resolve_single_cb(plot_df, CB=name_or_rank, CB_ID=cb_id)

            cb_unique_id = row["player.id"]
            if cb_unique_id in used_ids:
                continue
            used_ids.add(cb_unique_id)

            profiles.append(
                self._build_profile(
                    row,
                    trace_name=self._trace_label(
                        entity_name=str(row["player.name"]),
                        global_score=row[self.global_score_column],
                        global_rank=row[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    line_dash=self.TRACE_DASH_STYLES[
                        color_index % len(self.TRACE_DASH_STYLES)
                    ],
                )
            )
            color_index += 1

        if include_best:
            best_row = plot_df.sort_values(
                [self.global_score_column, "player.name"],
                ascending=[False, True],
            ).iloc[0]
            best_id = best_row["player.id"]
            if best_id not in used_ids:
                used_ids.add(best_id)
                profiles.append(
                    self._build_profile(
                        best_row,
                        trace_name=self._trace_label(
                            entity_name=f"{best_row['player.name']} (Best)",
                            global_score=best_row[self.global_score_column],
                            global_rank=best_row[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    )
                )
                color_index += 1

        if include_worst:
            worst_row = plot_df.sort_values(
                [self.global_score_column, "player.name"],
                ascending=[True, True],
            ).iloc[0]
            worst_id = worst_row["player.id"]
            if worst_id not in used_ids:
                used_ids.add(worst_id)
                profiles.append(
                    self._build_profile(
                        worst_row,
                        trace_name=self._trace_label(
                            entity_name=f"{worst_row['player.name']} (Worst)",
                            global_score=worst_row[self.global_score_column],
                            global_rank=worst_row[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    )
                )
                color_index += 1

        if include_league_average:
            average_row = self._build_average_profile(plot_df)
            profiles.append(
                self._build_profile(
                    average_row,
                    trace_name=self._trace_label(
                        entity_name="CBs' Global Average",
                        global_score=average_row[self.global_score_column],
                        global_rank=average_row[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                )
            )

        return self._plot_profiles(
            profiles=profiles,
            title="Global CBs' Quality Radar Distributions Comparison <br> (Ground Duels + Aerial Duels + Ball Passing)",
            subtitle=(
                f"Strict 3-Quality Intersection Cohort: {len(plot_df)} CBs   |   Equal Weighting (⅓ Each Quality)"
                ),
            radial_range=radial_range,
            show=show,
        )


class CB_Pair_Global_Qualities_Radar_Plot(_Global_Qualities_Radar_Resolver):
    """
    Global radar analysis for CB pairs across Ground/Aerial/Ball quality fits.
    """

    DEFAULT_AXIS_VALUE_COLUMNS = {
        "Ground Duels": "ground_CB_pair_fit_z_score",
        "Aerial Duels": "aerial_CB_pair_fit_z_score",
        "Ball Passing": "ball_CB_pair_fit_z_score",
    }
    DEFAULT_AXIS_RANK_COLUMNS = {
        "Ground Duels": "ground_CB_pair_fit_rank",
        "Aerial Duels": "aerial_CB_pair_fit_rank",
        "Ball Passing": "ball_CB_pair_fit_rank",
    }
    DEFAULT_AXIS_RANK_Z_COLUMNS = {
        "Ground Duels": "ground_CB_pair_fit_rank_z_score",
        "Aerial Duels": "aerial_CB_pair_fit_rank_z_score",
        "Ball Passing": "ball_CB_pair_fit_rank_z_score",
    }

    GLOBAL_SCORE_COLUMN = "global_CB_pair_fit_z_score"
    GLOBAL_RANK_COLUMN = "global_CB_pair_fit_rank"
    GLOBAL_RANK_Z_COLUMN = "global_CB_pair_fit_rank_z_score"

    def __init__(
        self,
        *,
        df_global_cb_pairs: pd.DataFrame,
        axis_value_columns: Optional[Mapping[str, str]] = None,
        axis_rank_columns: Optional[Mapping[str, str]] = None,
        axis_rank_z_columns: Optional[Mapping[str, str]] = None,
        global_score_column: str = GLOBAL_SCORE_COLUMN,
        global_rank_column: str = GLOBAL_RANK_COLUMN,
        global_rank_z_column: str = GLOBAL_RANK_Z_COLUMN,
    ) -> None:
        self.df_global_cb_pairs = df_global_cb_pairs.copy()
        self.global_score_column = str(global_score_column)
        self.global_rank_column = str(global_rank_column)
        self.global_rank_z_column = str(global_rank_z_column)

        axis_values = dict(self.DEFAULT_AXIS_VALUE_COLUMNS)
        if axis_value_columns is not None:
            axis_values.update({str(k): str(v) for k, v in axis_value_columns.items()})

        axis_ranks = dict(self.DEFAULT_AXIS_RANK_COLUMNS)
        if axis_rank_columns is not None:
            axis_ranks.update({str(k): str(v) for k, v in axis_rank_columns.items()})

        axis_rank_z = dict(self.DEFAULT_AXIS_RANK_Z_COLUMNS)
        if axis_rank_z_columns is not None:
            axis_rank_z.update(
                {str(k): str(v) for k, v in axis_rank_z_columns.items()}
            )

        (
            self.axis_value_columns,
            self.axis_rank_columns,
            self.axis_rank_z_columns,
        ) = self._ensure_axis_mappings(
            axis_value_columns=axis_values,
            axis_rank_columns=axis_ranks,
            axis_rank_z_columns=axis_rank_z,
            context="CB_Pair_Global_Qualities_Radar_Plot",
        )

    def _prepare_plot_df(self) -> pd.DataFrame:
        required_columns = [
            "pair_key",
            "pair_name",
            "player.id_CB1",
            "player.name_CB1",
            "player.id_CB2",
            "player.name_CB2",
            self.global_score_column,
            self.global_rank_column,
            self.global_rank_z_column,
        ] + list(self.axis_value_columns.values())

        for maybe_col in self.axis_rank_columns.values():
            if maybe_col is not None:
                required_columns.append(maybe_col)
        for maybe_col in self.axis_rank_z_columns.values():
            if maybe_col is not None:
                required_columns.append(maybe_col)

        self._ensure_columns(
            self.df_global_cb_pairs,
            required_columns,
            "global CB-pair radar plotting dataframe",
        )

        plot_df = self.df_global_cb_pairs.copy()
        plot_df[self.global_rank_column] = pd.to_numeric(
            plot_df[self.global_rank_column],
            errors="coerce",
        )
        plot_df = plot_df.sort_values(
            [self.global_rank_column, "pair_name"],
            ascending=[True, True],
        ).reset_index(drop=True)
        return plot_df

    def _player_candidates_from_pairs(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_columns(
            plot_df,
            ["player.id_CB1", "player.name_CB1", "player.id_CB2", "player.name_CB2"],
            "global CB-pair player resolution",
        )
        cb1 = plot_df[["player.id_CB1", "player.name_CB1"]].rename(
            columns={"player.id_CB1": "player.id", "player.name_CB1": "player.name"}
        )
        cb2 = plot_df[["player.id_CB2", "player.name_CB2"]].rename(
            columns={"player.id_CB2": "player.id", "player.name_CB2": "player.name"}
        )
        return pd.concat([cb1, cb2], ignore_index=True).drop_duplicates()

    def _resolve_pair_row(
        self,
        plot_df: pd.DataFrame,
        *,
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
    ) -> Optional[pd.Series]:
        if (
            CB_Pair is None
            and CB_1 is None
            and CB_2 is None
            and CB_1_ID is None
            and CB_2_ID is None
        ):
            return None

        if CB_Pair is not None:
            if isinstance(CB_Pair, str):
                CB_1, CB_2 = self._split_pair_string(CB_Pair)
            elif isinstance(CB_Pair, (list, tuple)) and len(CB_Pair) == 2:
                CB_1, CB_2 = CB_Pair[0], CB_Pair[1]
            elif isinstance(CB_Pair, dict):
                CB_1 = CB_Pair.get("CB_1", CB_1)
                CB_2 = CB_Pair.get("CB_2", CB_2)
                CB_1_ID = CB_Pair.get("CB_1_ID", CB_1_ID)
                CB_2_ID = CB_Pair.get("CB_2_ID", CB_2_ID)
                if CB_Pair.get("CB_Pair") is not None:
                    CB_1, CB_2 = self._split_pair_string(CB_Pair["CB_Pair"])
            else:
                raise ValueError(
                    "CB_Pair must be a string ('A + B', 'A and B', 'A & B'), a tuple/list of two CB inputs, or a dict with CB_1/CB_2 keys."
                )

        if (CB_1 is None and CB_1_ID is None) or (CB_2 is None and CB_2_ID is None):
            raise ValueError("Please provide both CB_1 and CB_2 (name/surname and/or ID).")

        candidates = self._player_candidates_from_pairs(plot_df)
        resolved_cb1 = self._resolve_entity(
            candidates,
            entity_value=CB_1,
            entity_id=CB_1_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )
        resolved_cb2 = self._resolve_entity(
            candidates,
            entity_value=CB_2,
            entity_id=CB_2_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="CB player",
        )

        id_1 = resolved_cb1["player.id"]
        id_2 = resolved_cb2["player.id"]
        pair_mask = (
            ((plot_df["player.id_CB1"] == id_1) & (plot_df["player.id_CB2"] == id_2))
            | ((plot_df["player.id_CB1"] == id_2) & (plot_df["player.id_CB2"] == id_1))
        )
        matches = plot_df[pair_mask]
        if matches.empty:
            raise ValueError(
                f"No CB pair found for '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple rows found for pair '{resolved_cb1['player.name']}' + '{resolved_cb2['player.name']}'."
            )
        return matches.iloc[0]

    def _resolve_pair_row_by_rank(self, plot_df: pd.DataFrame, *, CB_Pair_Rank: Any) -> pd.Series:
        return self._resolve_row_by_rank(
            plot_df,
            rank_column=self.global_rank_column,
            rank_value=CB_Pair_Rank,
            entity_label="CB pair",
            display_column="pair_name",
        )

    @staticmethod
    def _pair_key(row: pd.Series) -> str:
        if "pair_key" in row.index and pd.notna(row["pair_key"]):
            return str(row["pair_key"])
        return str(row.get("pair_name", ""))

    def _build_profile(
        self,
        row: pd.Series,
        *,
        trace_name: str,
        color_rgb: Tuple[int, int, int],
        line_dash: str = "solid",
        fill_opacity: float = 0.18,
    ) -> Dict[str, Any]:
        (
            axis_values,
            axis_rank_values,
            axis_hover,
            axis_rank_hover,
        ) = self._build_axis_hover_lines(
            row=row,
            axis_value_columns=self.axis_value_columns,
            axis_rank_columns=self.axis_rank_columns,
            axis_rank_z_columns=self.axis_rank_z_columns,
            global_score_column=self.global_score_column,
            global_rank_column=self.global_rank_column,
            global_rank_z_column=self.global_rank_z_column,
        )
        return {
            "axis_values": axis_values,
            "axis_rank_values": axis_rank_values,
            "axis_hover": axis_hover,
            "axis_rank_hover": axis_rank_hover,
            "trace_name": trace_name,
            "color_rgb": color_rgb,
            "line_dash": line_dash,
            "fill_opacity": fill_opacity,
        }

    def _build_average_profile(self, plot_df: pd.DataFrame) -> pd.Series:
        average_data: Dict[str, Any] = {
            "pair_key": "__pair_average__",
            "pair_name": "CB-Pairs' Global Average",
        }
        mean_columns = (
            list(self.axis_value_columns.values())
            + [
                self.global_score_column,
                self.global_rank_column,
                self.global_rank_z_column,
            ]
            + [col for col in self.axis_rank_columns.values() if col is not None]
            + [col for col in self.axis_rank_z_columns.values() if col is not None]
        )
        for col in dict.fromkeys(mean_columns):
            average_data[col] = pd.to_numeric(plot_df[col], errors="coerce").mean()
        return pd.Series(average_data)

    def Plot_CB_Pair(
        self,
        *,
        CB_Pair: Any = None,
        CB_1: Any = None,
        CB_2: Any = None,
        CB_1_ID: Any = None,
        CB_2_ID: Any = None,
        CB_Pair_Rank: Optional[Any] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
        radial_range: Optional[Sequence[float]] = None,
        show: bool = True,
    ) -> Optional[go.Figure]:
        """
        Plot one selected CB pair on the global 3-quality radar.
        """
        plot_df = self._prepare_plot_df()

        if CB_Pair_Rank is not None and any(
            value is not None for value in [CB_Pair, CB_1, CB_2, CB_1_ID, CB_2_ID]
        ):
            raise ValueError(
                "Please provide either CB_Pair_Rank or CB_Pair/CB_1/CB_2 selectors, not both."
            )

        if CB_Pair_Rank is not None:
            selected_pair = self._resolve_pair_row_by_rank(
                plot_df,
                CB_Pair_Rank=CB_Pair_Rank,
            )
        else:
            selected_pair = self._resolve_pair_row(
                plot_df,
                CB_Pair=CB_Pair,
                CB_1=CB_1,
                CB_2=CB_2,
                CB_1_ID=CB_1_ID,
                CB_2_ID=CB_2_ID,
            )
            if selected_pair is None:
                selected_pair = plot_df.iloc[0]

        profiles: List[Dict[str, Any]] = []
        used_keys = set()

        selected_key = self._pair_key(selected_pair)
        used_keys.add(selected_key)
        profiles.append(
            self._build_profile(
                selected_pair,
                trace_name=self._trace_label(
                    entity_name=str(selected_pair["pair_name"]),
                    global_score=selected_pair[self.global_score_column],
                    global_rank=selected_pair[self.global_rank_column],
                ),
                color_rgb=self.TRACE_COLORS[0],
            )
        )

        if include_best:
            best_pair = plot_df.sort_values(
                [self.global_score_column, "pair_name"],
                ascending=[False, True],
            ).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                profiles.append(
                    self._build_profile(
                        best_pair,
                        trace_name=self._trace_label(
                            entity_name=f"{best_pair['pair_name']} (Best)",
                            global_score=best_pair[self.global_score_column],
                            global_rank=best_pair[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[1],
                    )
                )

        if include_worst:
            worst_pair = plot_df.sort_values(
                [self.global_score_column, "pair_name"],
                ascending=[True, True],
            ).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                profiles.append(
                    self._build_profile(
                        worst_pair,
                        trace_name=self._trace_label(
                            entity_name=f"{worst_pair['pair_name']} (Worst)",
                            global_score=worst_pair[self.global_score_column],
                            global_rank=worst_pair[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[2],
                    )
                )

        if include_average:
            average_pair = self._build_average_profile(plot_df)
            profiles.append(
                self._build_profile(
                    average_pair,
                    trace_name=self._trace_label(
                        entity_name="CB-Pairs' Global Average",
                        global_score=average_pair[self.global_score_column],
                        global_rank=average_pair[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[3],
                )
            )

        return self._plot_profiles(
            profiles=profiles,
            title="Global CB-Pair Fit Radar Distribution <br> (Ground Duels + Aerial Duels + Ball Passing)",
            subtitle=(
                f"Strict 3-Quality Intersection Cohort: {len(plot_df)} unordered CB-Pairs   |   Equal Weighting (⅓ Each Quality)"
                ),
            radial_range=radial_range,
            show=show,
        )

    def Plot_CB_Pairs_Comparison(
        self,
        *,
        CB_Pairs: Optional[Sequence[Union[str, Sequence[Any], Dict[str, Any]]]] = None,
        CB_Pair_Rank: Optional[int] = None,
        CB_Pair_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
        radial_range: Optional[Sequence[float]] = None,
        show: bool = True,
    ) -> Optional[go.Figure]:
        """
        Compare multiple CB pairs on one global 3-axis radar.
        """
        plot_df = self._prepare_plot_df()

        specs: List[Dict[str, Any]] = []
        for item in (CB_Pairs or []):
            if isinstance(item, str):
                specs.append({"CB_Pair": item})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                specs.append({"CB_1": item[0], "CB_2": item[1]})
            elif isinstance(item, dict):
                specs.append(dict(item))
            else:
                raise ValueError(
                    "Each item in CB_Pairs must be a pair string, a tuple/list of length 2, or a dict."
                )

        if CB_Pair_Ranks is not None:
            if CB_Pair_Rank is not None:
                raise ValueError(
                    "Please provide either CB_Pair_Rank or CB_Pair_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                CB_Pair_Ranks,
                param_name="CB_Pair_Ranks",
            ):
                specs.append({"CB_Pair_Rank": rank_value})
        if CB_Pair_Rank is not None:
            specs.append(
                {
                    "CB_Pair_Rank": self._coerce_positive_rank(
                        CB_Pair_Rank,
                        param_name="CB_Pair_Rank",
                    )
                }
            )

        if not specs and not any([include_best, include_worst, include_average]):
            raise ValueError(
                "Please provide at least one pair selector via CB_Pairs/CB_Pair_Rank/CB_Pair_Ranks; or enable include_best/include_worst/include_average."
            )

        profiles: List[Dict[str, Any]] = []
        used_keys = set()
        color_index = 0

        for spec in specs:
            if spec.get("CB_Pair_Rank") is not None:
                row = self._resolve_pair_row_by_rank(
                    plot_df,
                    CB_Pair_Rank=spec["CB_Pair_Rank"],
                )
            else:
                row = self._resolve_pair_row(
                    plot_df,
                    CB_Pair=spec.get("CB_Pair"),
                    CB_1=spec.get("CB_1"),
                    CB_2=spec.get("CB_2"),
                    CB_1_ID=spec.get("CB_1_ID"),
                    CB_2_ID=spec.get("CB_2_ID"),
                )
                if row is None:
                    continue

            pair_key = self._pair_key(row)
            if pair_key in used_keys:
                continue
            used_keys.add(pair_key)

            profiles.append(
                self._build_profile(
                    row,
                    trace_name=self._trace_label(
                        entity_name=str(row["pair_name"]),
                        global_score=row[self.global_score_column],
                        global_rank=row[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    line_dash=self.TRACE_DASH_STYLES[
                        color_index % len(self.TRACE_DASH_STYLES)
                    ],
                )
            )
            color_index += 1

        if include_best:
            best_pair = plot_df.sort_values(
                [self.global_score_column, "pair_name"],
                ascending=[False, True],
            ).iloc[0]
            best_key = self._pair_key(best_pair)
            if best_key not in used_keys:
                used_keys.add(best_key)
                profiles.append(
                    self._build_profile(
                        best_pair,
                        trace_name=self._trace_label(
                            entity_name=f"{best_pair['pair_name']} (Best)",
                            global_score=best_pair[self.global_score_column],
                            global_rank=best_pair[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    )
                )
                color_index += 1

        if include_worst:
            worst_pair = plot_df.sort_values(
                [self.global_score_column, "pair_name"],
                ascending=[True, True],
            ).iloc[0]
            worst_key = self._pair_key(worst_pair)
            if worst_key not in used_keys:
                used_keys.add(worst_key)
                profiles.append(
                    self._build_profile(
                        worst_pair,
                        trace_name=self._trace_label(
                            entity_name=f"{worst_pair['pair_name']} (Worst)",
                            global_score=worst_pair[self.global_score_column],
                            global_rank=worst_pair[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    )
                )
                color_index += 1

        if include_average:
            average_pair = self._build_average_profile(plot_df)
            profiles.append(
                self._build_profile(
                    average_pair,
                    trace_name=self._trace_label(
                        entity_name="CB-Pairs' Global Average",
                        global_score=average_pair[self.global_score_column],
                        global_rank=average_pair[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                )
            )

        return self._plot_profiles(
            profiles=profiles,
            title="Global CB-Pairs' Fit Radar Distributions Comparison <br> (Ground Duels + Aerial Duels + Ball Passing)",
            subtitle=(
                f"Strict 3-Quality Intersection Cohort: {len(plot_df)} unordered CB-Pairs   |   Equal Weighting (⅓ Each Quality)"
                ),
            radial_range=radial_range,
            show=show,
        )


class Anchor_CB_Companion_Fit_Global_Qualities_Radar_Plot(_Global_Qualities_Radar_Resolver):
    """
    Global directional anchor-companion radar analysis across the 3 qualities.
    """

    DEFAULT_AXIS_VALUE_COLUMNS = {
        "Ground Duels": "ground_companion_fit_score_z_within_anchor",
        "Aerial Duels": "aerial_companion_fit_score_z_within_anchor",
        "Ball Passing": "ball_companion_fit_score_z_within_anchor",
    }
    DEFAULT_AXIS_RANK_COLUMNS = {
        "Ground Duels": "ground_companion_rank_for_anchor",
        "Aerial Duels": "aerial_companion_rank_for_anchor",
        "Ball Passing": "ball_companion_rank_for_anchor",
    }
    DEFAULT_AXIS_RANK_Z_COLUMNS = {
        "Ground Duels": None,
        "Aerial Duels": None,
        "Ball Passing": None,
    }

    GLOBAL_SCORE_COLUMN = "global_companion_fit_score_z_within_anchor"
    GLOBAL_RANK_COLUMN = "global_companion_rank_for_anchor"
    GLOBAL_RANK_Z_COLUMN = "global_companion_rank_for_anchor_z_score"

    def __init__(
        self,
        *,
        df_global_companion_fits: pd.DataFrame,
        axis_value_columns: Optional[Mapping[str, str]] = None,
        axis_rank_columns: Optional[Mapping[str, str]] = None,
        axis_rank_z_columns: Optional[Mapping[str, str]] = None,
        global_score_column: str = GLOBAL_SCORE_COLUMN,
        global_rank_column: str = GLOBAL_RANK_COLUMN,
        global_rank_z_column: str = GLOBAL_RANK_Z_COLUMN,
    ) -> None:
        self.df_global_companion_fits = df_global_companion_fits.copy()
        self.global_score_column = str(global_score_column)
        self.global_rank_column = str(global_rank_column)
        self.global_rank_z_column = str(global_rank_z_column)
        self._anchor_player_id: Optional[Any] = None
        self._anchor_player_name: Optional[str] = None

        axis_values = dict(self.DEFAULT_AXIS_VALUE_COLUMNS)
        if axis_value_columns is not None:
            axis_values.update({str(k): str(v) for k, v in axis_value_columns.items()})

        axis_ranks = dict(self.DEFAULT_AXIS_RANK_COLUMNS)
        if axis_rank_columns is not None:
            axis_ranks.update({str(k): str(v) for k, v in axis_rank_columns.items()})

        axis_rank_z = dict(self.DEFAULT_AXIS_RANK_Z_COLUMNS)
        if axis_rank_z_columns is not None:
            axis_rank_z.update(
                {str(k): str(v) for k, v in axis_rank_z_columns.items()}
            )

        (
            self.axis_value_columns,
            self.axis_rank_columns,
            self.axis_rank_z_columns,
        ) = self._ensure_axis_mappings(
            axis_value_columns=axis_values,
            axis_rank_columns=axis_ranks,
            axis_rank_z_columns=axis_rank_z,
            context="Anchor_CB_Companion_Fit_Global_Qualities_Radar_Plot",
        )

    def _prepare_plot_df(self) -> pd.DataFrame:
        required_columns = [
            "anchor_player_id",
            "anchor_player_name",
            "partner_player_id",
            "partner_player_name",
            "pair_key",
            self.global_score_column,
            self.global_rank_column,
            self.global_rank_z_column,
        ] + list(self.axis_value_columns.values())

        for maybe_col in self.axis_rank_columns.values():
            if maybe_col is not None:
                required_columns.append(maybe_col)
        for maybe_col in self.axis_rank_z_columns.values():
            if maybe_col is not None:
                required_columns.append(maybe_col)

        self._ensure_columns(
            self.df_global_companion_fits,
            required_columns,
            "global companion-fit radar plotting dataframe",
        )
        return self.df_global_companion_fits.copy()

    def _anchor_candidates(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_columns(
            plot_df,
            ["anchor_player_id", "anchor_player_name"],
            "global companion-fit anchor resolution",
        )
        return plot_df[["anchor_player_id", "anchor_player_name"]].drop_duplicates()

    def _resolve_anchor(
        self,
        plot_df: pd.DataFrame,
        *,
        Anchor_CB: Any = None,
        Anchor_CB_ID: Any = None,
    ) -> pd.Series:
        candidates = self._anchor_candidates(plot_df)

        if Anchor_CB is None and Anchor_CB_ID is None:
            if self._anchor_player_name is not None:
                cached_by_name = candidates[
                    candidates["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(self._anchor_player_name)
                ]
                if not cached_by_name.empty:
                    return cached_by_name.iloc[0]
            return candidates.sort_values("anchor_player_name").iloc[0]

        return self._resolve_entity(
            candidates,
            entity_value=Anchor_CB,
            entity_id=Anchor_CB_ID,
            id_col="anchor_player_id",
            name_col="anchor_player_name",
            entity_label="anchor CB",
        )

    def Initialize_Desired_Anchor_CB(
        self,
        *,
        Anchor_CB: Any = None,
        Anchor_CB_ID: Any = None,
    ) -> pd.Series:
        """
        Resolve and cache the anchor CB used by subsequent companion plots.
        """
        plot_df = self._prepare_plot_df()
        resolved_anchor = self._resolve_anchor(
            plot_df,
            Anchor_CB=Anchor_CB,
            Anchor_CB_ID=Anchor_CB_ID,
        )
        self._anchor_player_id = resolved_anchor.get("anchor_player_id", None)
        self._anchor_player_name = str(resolved_anchor["anchor_player_name"])
        return resolved_anchor

    def _build_anchor_plot_df(
        self,
        plot_df: pd.DataFrame,
        *,
        anchor_name: str,
        anchor_id: Any = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        if anchor_id is not None:
            filtered = plot_df[plot_df["anchor_player_id"] == anchor_id].copy()
            if filtered.empty:
                filtered = plot_df[
                    plot_df["anchor_player_name"].map(self._normalize_text)
                    == self._normalize_text(anchor_name)
                ].copy()
        else:
            filtered = plot_df[
                plot_df["anchor_player_name"].map(self._normalize_text)
                == self._normalize_text(anchor_name)
            ].copy()

        if filtered.empty:
            raise ValueError(f"No companion-fit rows found for anchor CB '{anchor_name}'.")

        filtered[self.global_rank_column] = pd.to_numeric(
            filtered[self.global_rank_column],
            errors="coerce",
        )
        filtered = filtered.sort_values(
            [self.global_rank_column, "partner_player_name"],
            ascending=[True, True],
        ).reset_index(drop=True)

        if top_n is not None:
            if not isinstance(top_n, (int, np.integer)) or int(top_n) <= 0:
                raise ValueError("top_n must be a positive integer or None.")
            filtered = filtered.head(int(top_n)).copy().reset_index(drop=True)

        return filtered

    def _resolve_companion_row(
        self,
        plot_df: pd.DataFrame,
        *,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
    ) -> Optional[pd.Series]:
        if Companion_CB is None and Companion_CB_ID is None and Companion_Rank is None:
            return None

        if Companion_Rank is not None:
            if Companion_CB is not None or Companion_CB_ID is not None:
                raise ValueError(
                    "Please specify either Companion_Rank or Companion_CB/Companion_CB_ID, not both."
                )
            requested_rank = self._coerce_positive_rank(
                Companion_Rank,
                param_name="Companion_Rank",
            )
            rank_values = pd.to_numeric(
                plot_df[self.global_rank_column],
                errors="coerce",
            )
            matches = plot_df[rank_values == float(requested_rank)]
            if matches.empty:
                raise ValueError(
                    f"No companion found at global rank #{requested_rank} in the current anchor sample."
                )
            if len(matches) > 1:
                candidates = matches["partner_player_name"].astype(str).tolist()
                raise ValueError(
                    f"Multiple companions found for global rank #{requested_rank}: {', '.join(candidates)}"
                )
            return matches.iloc[0]

        candidates = plot_df[["partner_player_id", "partner_player_name"]].drop_duplicates()
        candidates = candidates.rename(
            columns={"partner_player_id": "player.id", "partner_player_name": "player.name"}
        )
        resolved = self._resolve_entity(
            candidates,
            entity_value=Companion_CB,
            entity_id=Companion_CB_ID,
            id_col="player.id",
            name_col="player.name",
            entity_label="companion CB",
        )
        matches = plot_df[plot_df["partner_player_id"] == resolved["player.id"]]
        if matches.empty:
            raise ValueError(f"No companion fit found for '{Companion_CB}'.")
        return matches.iloc[0]

    @staticmethod
    def _companion_key(row: pd.Series) -> str:
        if "partner_player_id" in row.index and pd.notna(row["partner_player_id"]):
            return str(row["partner_player_id"])
        return str(row.get("partner_player_name", ""))

    def _build_profile(
        self,
        row: pd.Series,
        *,
        trace_name: str,
        color_rgb: Tuple[int, int, int],
        line_dash: str = "solid",
        fill_opacity: float = 0.18,
    ) -> Dict[str, Any]:
        (
            axis_values,
            axis_rank_values,
            axis_hover,
            axis_rank_hover,
        ) = self._build_axis_hover_lines(
            row=row,
            axis_value_columns=self.axis_value_columns,
            axis_rank_columns=self.axis_rank_columns,
            axis_rank_z_columns=self.axis_rank_z_columns,
            global_score_column=self.global_score_column,
            global_rank_column=self.global_rank_column,
            global_rank_z_column=self.global_rank_z_column,
        )
        return {
            "axis_values": axis_values,
            "axis_rank_values": axis_rank_values,
            "axis_hover": axis_hover,
            "axis_rank_hover": axis_rank_hover,
            "trace_name": trace_name,
            "color_rgb": color_rgb,
            "line_dash": line_dash,
            "fill_opacity": fill_opacity,
        }

    def _build_average_profile(self, plot_df: pd.DataFrame) -> pd.Series:
        average_data: Dict[str, Any] = {
            "partner_player_id": -1,
            "partner_player_name": "Companion Pool Average",
        }
        mean_columns = (
            list(self.axis_value_columns.values())
            + [
                self.global_score_column,
                self.global_rank_column,
                self.global_rank_z_column,
            ]
            + [col for col in self.axis_rank_columns.values() if col is not None]
            + [col for col in self.axis_rank_z_columns.values() if col is not None]
        )
        for col in dict.fromkeys(mean_columns):
            average_data[col] = pd.to_numeric(plot_df[col], errors="coerce").mean()
        return pd.Series(average_data)

    def Plot_Companion_of_Anchor_CB(
        self,
        *,
        Anchor_CB: Any = None,
        Anchor_CB_ID: Any = None,
        Companion_CB: Any = None,
        Companion_CB_ID: Any = None,
        Companion_Rank: Optional[int] = None,
        Companion_Ranks: Optional[Sequence[int]] = None,
        include_best: bool = True,
        include_worst: bool = False,
        include_anchor_CB_pool_average: bool = False,
        top_n: Optional[int] = None,
        radial_range: Optional[Sequence[float]] = None,
        show: bool = True,
    ) -> Optional[go.Figure]:
        """
        Plot directional global companion fits for one anchor CB.
        """
        plot_df = self._prepare_plot_df()
        resolved_anchor = self._resolve_anchor(
            plot_df,
            Anchor_CB=Anchor_CB,
            Anchor_CB_ID=Anchor_CB_ID,
        )
        self._anchor_player_id = resolved_anchor.get("anchor_player_id", None)
        self._anchor_player_name = str(resolved_anchor["anchor_player_name"])

        anchor_plot_df = self._build_anchor_plot_df(
            plot_df,
            anchor_name=self._anchor_player_name,
            anchor_id=self._anchor_player_id,
            top_n=top_n,
        )

        profiles: List[Dict[str, Any]] = []
        used_keys = set()
        color_index = 0

        selected_rows: List[pd.Series] = []
        if Companion_Ranks is not None:
            if Companion_Rank is not None:
                raise ValueError(
                    "Please provide either Companion_Rank or Companion_Ranks, not both."
                )
            for rank_value in self._coerce_unique_positive_ranks(
                Companion_Ranks,
                param_name="Companion_Ranks",
            ):
                row = self._resolve_companion_row(
                    anchor_plot_df,
                    Companion_Rank=rank_value,
                )
                if row is not None:
                    selected_rows.append(row)

        row_from_single_selector = self._resolve_companion_row(
            anchor_plot_df,
            Companion_CB=Companion_CB,
            Companion_CB_ID=Companion_CB_ID,
            Companion_Rank=Companion_Rank,
        )
        if row_from_single_selector is not None:
            selected_rows.append(row_from_single_selector)

        # If top_n is requested without explicit selectors, plot all top-N companions.
        # This is the most natural analysis flow for "show me the top N for this anchor".
        if (
            top_n is not None
            and Companion_CB is None
            and Companion_CB_ID is None
            and Companion_Rank is None
            and Companion_Ranks is None
        ):
            selected_rows.extend([row for _, row in anchor_plot_df.iterrows()])

        for row in selected_rows:
            key = self._companion_key(row)
            if key in used_keys:
                continue
            used_keys.add(key)
            profiles.append(
                self._build_profile(
                    row,
                    trace_name=self._trace_label(
                        entity_name=f"{self._anchor_player_name} + {row['partner_player_name']}",
                        global_score=row[self.global_score_column],
                        global_rank=row[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    line_dash=self.TRACE_DASH_STYLES[
                        color_index % len(self.TRACE_DASH_STYLES)
                    ],
                )
            )
            color_index += 1

        if include_best:
            best_row = anchor_plot_df.sort_values(
                [self.global_score_column, "partner_player_name"],
                ascending=[False, True],
            ).iloc[0]
            key = self._companion_key(best_row)
            if key not in used_keys:
                used_keys.add(key)
                profiles.append(
                    self._build_profile(
                        best_row,
                        trace_name=self._trace_label(
                            entity_name=f"{self._anchor_player_name} + {best_row['partner_player_name']} (Best)",
                            global_score=best_row[self.global_score_column],
                            global_rank=best_row[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    )
                )
                color_index += 1

        if include_worst:
            worst_row = anchor_plot_df.sort_values(
                [self.global_score_column, "partner_player_name"],
                ascending=[True, True],
            ).iloc[0]
            key = self._companion_key(worst_row)
            if key not in used_keys:
                used_keys.add(key)
                profiles.append(
                    self._build_profile(
                        worst_row,
                        trace_name=self._trace_label(
                            entity_name=f"{self._anchor_player_name} + {worst_row['partner_player_name']} (Worst)",
                            global_score=worst_row[self.global_score_column],
                            global_rank=worst_row[self.global_rank_column],
                        ),
                        color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                    )
                )
                color_index += 1

        if include_anchor_CB_pool_average:
            average_row = self._build_average_profile(anchor_plot_df)
            profiles.append(
                self._build_profile(
                    average_row,
                    trace_name=self._trace_label(
                        entity_name=f"{self._anchor_player_name}'s Companion Pool Average",
                        global_score=average_row[self.global_score_column],
                        global_rank=average_row[self.global_rank_column],
                    ),
                    color_rgb=self.TRACE_COLORS[color_index % len(self.TRACE_COLORS)],
                )
            )

        if not profiles:
            raise ValueError(
                "No companion profiles selected. Please choose a companion selector or enable best/worst/average overlays."
            )

        return self._plot_profiles(
            profiles=profiles,
            title=(
                f"Global Anchor CB-Companion Fits' Radar Distributions   ->   {self._anchor_player_name} acting as the Anchor CB "
                "<br> (Ground Duels + Aerial Duels + Ball Passing)"
            ),
            subtitle=(
                f"Strict 3-Quality Intersection Companion Sample Size: {len(anchor_plot_df)} Anchor -> Companion Pairs (Within {self._anchor_player_name}'s Sample)   |   Equal Weighting (⅓ Each Quality)"
                ),
            radial_range=radial_range,
            show=show,
        )


__all__ = [
    "Single_CB_Global_Qualities_Radar_Plot",
    "CB_Pair_Global_Qualities_Radar_Plot",
    "Anchor_CB_Companion_Fit_Global_Qualities_Radar_Plot",
]



# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------



class DistributionPlotPersonality(Visual):
    """
    Distribution plot specialized for personality trait comparisons.
    """

    def __init__(self, columns, *args, **kwargs):
        """
        Initialize a personality distribution chart.

        Args:
            columns: Base metric column names (without suffixes).
            *args: Forwarded positional args for `Visual`.
            **kwargs: Forwarded keyword args for `Visual`.
        """
        self.empty = True
        self.columns = columns
        # Use unique (color, shape) combinations within a figure.
        marker_colors = [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        marker_shapes = ["square", "hexagon", "diamond"]
        # First assign one-to-one color/shape styles so highlighted entities
        # start with fully distinct colors and shapes.
        primary_count = min(len(marker_colors), len(marker_shapes))
        step = max(primary_count - 1, 1)
        self._marker_styles = [
            (marker_colors[i], marker_shapes[(i * step) % primary_count])
            for i in range(primary_count)
        ]

        # Then append remaining unique (color, shape) combinations.
        used_styles = set(self._marker_styles)
        for color, shape in itertools.product(marker_colors, marker_shapes):
            if (color, shape) not in used_styles:
                self._marker_styles.append((color, shape))
        self._marker_style_index = 0
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def _next_marker_style(self):
        """
        Return the next unique marker style (color + shape) for highlighted points.
        """
        if self._marker_style_index >= len(self._marker_styles):
            raise ValueError(
                "Exceeded available unique marker style combinations in this plot. "
                f"Maximum unique highlighted entities: {len(self._marker_styles)}."
            )
        color, marker = self._marker_styles[self._marker_style_index]
        self._marker_style_index += 1
        return color, marker

    def _setup_axes(self):
        """
        Configure fixed axis settings for personality trait plots.
        """
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=["Worse", "Average", "Better"],
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        """
        Add background comparison points for each personality metric.

        Args:
            df_plot: DataFrame containing metrics and rank-related columns.
            plots: Suffix appended to each metric for x-values (e.g. `"_Z"`).
            names: Hover label values for each row/entity.
            legend: Legend label for background traces.
            hover: Suffix appended for hover-value columns.
            hover_string: Plotly hover template body.
        """
        # First trace carries legend label; subsequent traces hide it to avoid duplicates.
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(
                go.Scatter(
                    # Suffix-based column addressing keeps plotting logic generic.
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=round(df_plot[col + hover]),
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(self, ser_plot, plots, name, hover="", hover_string="", text=None):
        """
        Add one highlighted person across all configured personality metrics.

        Args:
            ser_plot: Series-like metric source for one person.
            plots: Suffix appended to each metric for x-values (e.g. `"_Z"`).
            name: Legend label for this person.
            hover: Suffix appended for hover-value columns.
            hover_string: Plotly hover template body.
            text: Optional hover text override.
        """
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        # We add one trace per metric, but keep only a single legend entry.
        legend = True
        color, marker = self._next_marker_style()

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[round(ser_plot[col + hover])],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                # Annotation labels are anchored near the center reference axis (x=0).
                x=0,
                y=i + 0.4,
                text=f"<span style=''>{metric_name}: {int(ser_plot[col]):.0f}</span>",
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    def add_person(self, person: Person, n_group, metrics):
        """
        Add a single person with rank-based hover details.

        Args:
            person: Person entity whose metrics are plotted.
            n_group: Group size used in rank hover text.
            metrics: Base metric names used by this view.
        """
        # Keep suffix conventions explicit here for readability and future extensions.
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_data_point(
            ser_plot=person.ser_metrics,
            plots="_Z",
            name=person.name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    def add_persons(self, persons: PersonStat, metrics):
        """
        Add all comparison-group persons to the background layer.

        Args:
            persons: Dataset wrapper containing the comparison population.
            metrics: Base metric names used by this view.
        """

        # Keep suffix conventions explicit here for readability and future extensions.
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_group_data(
            df_plot=persons.df,
            plots="_Z",
            names=persons.df["name"],
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(len(persons.df)),
            legend=f"Other persons  ",
        )

    def add_title_from_person(self, person: Person):
        """
        Create and apply chart title/subtitle for the selected person.
        """
        self.person = person
        title = f"Evaluation of {person.name}"
        subtitle = f"Based on Big Five scores"
        self.add_title(title, subtitle)



"""
class ViolinPlot(Visual):
    def violin(data, point_data):
        # Create a figure object
        fig = go.Figure()

        # Labels for the columnshover
        labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

        # Loop through each label to add a violin plot trace
        for label in labels:
            fig.add_trace(go.Violin(
                x=df_plot[label],  # Use x for the data
                name=label,      # Label each violin plot correctly
                box_visible=True,
                meanline_visible=True,
                line_color='black',  # Color of the violin outline
                fillcolor='rgba(0,100,200,0.3)',  # Color of the violin fill
                opacity=0.6,
                orientation='h'  # Set orientation to horizontal
            )
        )
        for label, value in point_data.items():
            fig.add_trace(
                go.Scatter(x=[value], y=[label], mode='markers', marker=dict(color='red', size=8, symbol='cross'), name=f'{label} Candidate Point'))

        # Update layout for better visualization
        fig.update_layout(
            title='Distribution of Personality Traits',
            xaxis_title='Score',  
            yaxis_title='Trait',
            xaxis=dict(range=[0, 40]),
            violinmode='overlay', 
            showlegend=True)

        # Display the plot in Streamlit
        st.plotly_chart(fig)


    def radarPlot(Visual):
        # Data import
        data_r = data_p.to_list()  
        labels = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        df = pd.DataFrame({'data': data_r,'label': labels})
    
        # Create the radar plot
        fig = px.line_polar(df, r='data', theta='label', line_close=True, markers=True)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 40])),showlegend=True, title= 'Candidate profile')
        fig.update_traces(fill='toself', marker=dict(size=5))
        # Display the plot in Streamlit
        st.plotly_chart(fig)
"""
