"""
Plotting utilities used by the Streamlit visual sandbox.

The module provides reusable helpers and chart classes for player, country, and
personality distribution views built with Plotly.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
import re
import unicodedata


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
    dark_green = hex_to_rgb(
        "#002c1c"
    )  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb(
        "#00A938"
    )  # hex_to_rgb(st.get_option("theme.primaryColor"))
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
        
        # Cycled styles ensure multiple highlighted entities remain visually distinct.
        self.marker_color = (
            c for c in [Visual.bright_orange, Visual.magenta, Visual.bright_yellow, Visual.bright_blue]
        # original  -->  [Visual.white, Visual.bright_yellow, Visual.bright_orange, Visual.bright_blue]
        )

        self.marker_shape = (s for s in ["diamond", "square", "triangle-up", "hexagon"])
        # State used for dynamic multi-entity annotations rendered from add_data_point.
        self._multi_annotation_items = []
        self._multi_annotation_indices = []
        super().__init__(*args, **kwargs)
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
        color = next(self.marker_color)
        marker = next(self.marker_shape)

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
            "square": "■",
            "triangle-up": "▲",
            "hexagon": "⬢",
        }
        glyph = marker_symbol_map.get(marker_symbol, "●")
        return f"<span style='color:{rgb_to_color(color)}'>{glyph}</span>"

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


class _Ground_Duels_Distribution_Resolver:
    """
    Shared helper utilities used by Ground-Duels distribution plot wrappers.
    """

    @staticmethod
    def _normalize_text(value: Any) -> str:
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
        normalized = cls._normalize_text(value)
        if not normalized:
            return tuple()
        return tuple(normalized.split(" "))

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required column(s) for {context}: " + ", ".join(missing)
            )

    @staticmethod
    def _to_number_or_none(value: Any) -> Optional[float]:
        try:
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        except Exception:
            return None
        if pd.isna(converted):
            return None
        return float(converted)

    @classmethod
    def _resolve_entity(cls, candidates: pd.DataFrame, *, entity_value: Any = None, entity_id: Any = None, id_col: str, name_col: str, entity_label: str) -> pd.Series:
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
        metric_token = str(metric_name).lower()
        raw_metric_token = str(raw_metric_name).lower() if raw_metric_name is not None else ""
        return ("rank" in metric_token) or ("rank" in raw_metric_token)

    @classmethod
    def _format_hover_line(cls, metric_name: str, raw_metric_name: Optional[str], metric_label: str, raw_value: Any, z_value: Any) -> str:
        if raw_metric_name is None:
            return f"Z-score = {z_value:.2f}" if pd.notna(z_value) else "Z-score = N/A"
        if cls._is_rank_metric(metric_name, raw_metric_name):
            return f"Rank =   # {int(round(raw_value))}" if pd.notna(raw_value) else "Rank =   # N/A"
        raw_text = f"{raw_value:.2f}" if pd.notna(raw_value) else "N/A"
        z_text = f"{z_value:.2f}" if pd.notna(z_value) else "N/A"
        return f"{metric_label} = {raw_text}<br>Respective Z-score = {z_text}"

    @classmethod
    def _attach_hover_payload(cls, plot_df: pd.DataFrame, *, metric_cols: Sequence[str], metric_labels: Dict[str, str], metric_value_columns: Dict[str, Optional[str]], hover_payload_suffix: str = "_hover_payload", fill_missing_rank_z: bool = True, rank_z_scale: float = 2.0) -> pd.DataFrame:
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
        fallback_metric_columns = [
            metric
            for metric in metrics
            if metric not in z_scores_df.columns and metric in duel_summary_df.columns
        ]
        required_duel_summary_columns = list(
            dict.fromkeys(raw_metric_columns + fallback_metric_columns)
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
            title="CB Ground Duel Quality Distribution",
            subtitle=subtitle,
        )
        return dist_plot

    def _resolve_single_cb(self, plot_df: pd.DataFrame, *, CB: Any = None, CB_ID: Any = None) -> pd.Series:
        candidates = plot_df[["player.id", "player.name"]].drop_duplicates()
        if CB is None and CB_ID is None:
            return plot_df.iloc[0]

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
        include_league_average: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one selected CB against the full single-CB Ground-Duels distribution.
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

        selected_cb = self._resolve_single_cb(plot_df, CB=CB, CB_ID=CB_ID)
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
        include_league_average: bool = True,
        multi_annotations: bool = True,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Mapping[str, str]] = None,
        metric_value_columns: Optional[Mapping[str, Optional[str]]] = None,
        show: bool = True,
    ) -> DistributionPlot:
        """
        Plot one or more selected CBs in comparison mode.
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

        requested_items: List[Tuple[Any, Any]] = []
        for cb_id in (CB_IDs or []):
            requested_items.append((None, cb_id))
        for cb_name in (CBs or []):
            requested_items.append((cb_name, None))

        if not requested_items:
            raise ValueError("Please provide at least one CB in `CBs` and/or `CB_IDs`.")

        used_ids = set()
        for cb_name, cb_id in requested_items:
            selected_cb = self._resolve_single_cb(plot_df, CB=cb_name, CB_ID=cb_id)
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
            title = f"{title} ({section_title})"
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

    def _pair_key(self, row: pd.Series) -> str:
        if "pair_key" in row.index and pd.notna(row["pair_key"]):
            return str(row["pair_key"])
        return str(row.get("pair_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
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
        include_best: bool = True,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for a single CB-pair plot scenario.
        """
        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()

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
        include_best: bool = False,
        include_worst: bool = False,
        include_average: bool = False,
    ) -> List[Tuple[pd.Series, str, bool]]:
        """
        Resolve highlighted rows for CB-pair comparison plots.
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

        if not specs and not any([include_best, include_worst, include_average]):
            raise ValueError(
                "Please provide CB_Pairs and/or enable at least one of include_best/include_worst/include_average."
            )

        highlighted_rows: List[Tuple[pd.Series, str, bool]] = []
        used_keys = set()
        for spec in specs:
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

    def _resolve_companion_row(self, plot_df: pd.DataFrame, *, Companion_CB: Any = None, Companion_CB_ID: Any = None) -> Optional[pd.Series]:
        if Companion_CB is None and Companion_CB_ID is None:
            return None

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
        if "partner_player_id" in row.index and pd.notna(row["partner_player_id"]):
            return str(row["partner_player_id"])
        return self._normalize_text(row.get("partner_player_name", ""))

    def _add_row_point(self, dist_plot: DistributionPlot, *, row: pd.Series, display_name: str, hover_payload_suffix: str, hover_string: str, multi_annotations: bool, add_single_annotations: bool) -> None:
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
            title=f"CB-Companion Ground Duels Potential Fits Distribution ({anchor_name} acting as the Anchor CB)",
            subtitle=(
                f"All {num_candidates} potential companions for {anchor_name} (directional A → B) \n"
                "Metrics are at the CB-Companion-level (i.e. within the Anchor CB's sample), \nshowing the expected contribution of the companion to the CB-Pair's overall fit with the specified Anchor CB."
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

        selected_companion = self._resolve_companion_row(
            plot_df,
            Companion_CB=Companion_CB,
            Companion_CB_ID=Companion_CB_ID,
        )
        if selected_companion is not None:
            selected_key = self._companion_key(selected_companion)
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
        # Cycled styles ensure multiple highlighted entities remain visually distinct.
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        self._setup_axes()

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
        color = next(self.marker_color)
        marker = next(self.marker_shape)

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
