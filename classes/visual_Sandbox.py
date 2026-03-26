"""Plotting utilities used by the Streamlit visual sandbox.

The module provides reusable helpers and chart classes for player, country, and
personality distribution views built with Plotly.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


from utils.sentences import format_metric
from classes.data_point import Player, Country, Person
from classes.data_source import PlayerStats, CountryStats, PersonStat
from typing import Union


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert a hex color string (e.g. `#aabbcc`) to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity=1):
    """Build a CSS rgba color string from an RGB tuple and opacity."""
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color, text, alpha=1.0):
    """Wrap text in an HTML span using the given hex color and alpha."""
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
    """Base visual wrapper that applies shared styling to Plotly figures."""

    # Can't use streamlit options due to report generation
    dark_green = hex_to_rgb(
        "#002c1c"
    )  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb(
        "#00A938"
    )  # hex_to_rgb(st.get_option("theme.primaryColor"))
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")  # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb("#009940")
    table_red = hex_to_rgb("#FF4B00")

    def __init__(self, pdf=False, plot_type="scout"):
        """Initialize the base figure and shared style configuration.

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
        """Render the figure in Streamlit."""
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def _setup_styles(self):
        """Apply common layout, legend, and axis styling."""
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
        """Add the main chart title and subtitle."""
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
        """Add a low, centered annotation below the plotting area."""
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
        """Render the figure in Streamlit."""
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def close(self):
        """No-op placeholder for API symmetry with other visual components."""
        pass


class DistributionPlot(Visual):
    """Distribution chart for player/country metrics on a shared x-axis."""
    def __init__(self, columns, labels = None, *args, quality_metric_labels = None, quality_metric_value_columns = None, **kwargs):
        """Initialize metric columns, marker cycles, and axis labels.

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
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_orange, Visual.bright_blue]
        )

        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _normalize_metric_labels(self, quality_metric_labels, value_name = "quality_metric_labels"):
        """Normalize label input into a dictionary keyed by `self.columns`.

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
        """Resolve which value to show in metric annotations.

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

    def _setup_axes(self, labels=["←   Worse", "Average", "Better   →"]):
        """Set axis range, ticks, grid style, and center reference line."""
        self.fig.update_xaxes(
            range=[-4, 4],
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
        """Add background comparison points for each configured metric.

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
            

    def add_data_point(
        self,
        ser_plot,
        plots,
        name,
        hover="",
        hover_string="",
        text=None,
        add_annotations=True,
    ):
        """Add one highlighted entity (player/country) across all metrics.

        Args:
            ser_plot: Series-like metric source for a single entity.
            plots: Suffix appended to each metric for x-values (e.g. `"_Z"`).
            name: Legend label for this entity.
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

            if add_annotations:
                self.fig.add_annotation(
                    # Annotation labels are anchored near the center reference axis (x=0).
                    x=0,
                    y=i + 0.4,
                    text=self.annotation_text.format(
                        metric_name=metric_name,
                        data=self._resolve_annotation_value(ser_plot, col),
                    ),
                    showarrow=True,
                    font={
                        "color": rgb_to_color(self.white),
                        "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier,
                    },
                )


    def add_player(self, player: Union[Player, Country], n_group, metrics):
        """Add a single `Player` or `Country` to the distribution chart.

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
        """Add comparison-group points for player or country datasets.

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
        """Build and apply an entity-aware title and subtitle."""
        self.player = player

        title = f"Evaluation of {player.name}?"
        if isinstance(player, Player):
            subtitle = f"Based on {player.minutes_played} minutes played"
        elif isinstance(player, Country):
            subtitle = f"Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_title(title, subtitle)


# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


class DistributionPlotPersonality(Visual):
    """Distribution plot specialized for personality trait comparisons."""

    def __init__(self, columns, *args, **kwargs):
        """Initialize a personality distribution chart.

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
        """Configure fixed axis settings for personality trait plots."""
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
        """Add background comparison points for each personality metric.

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

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        """Add one highlighted person across all configured personality metrics.

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
        """Add a single person with rank-based hover details.

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
        """Add all comparison-group persons to the background layer.

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
        """Create and apply chart title/subtitle for the selected person."""
        self.person = person
        title = f"Evaluation of {person.name}"
        subtitle = f"Based on Big Five scores"
        self.add_title(title, subtitle)


"""class ViolinPlot(Visual):
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
        st.plotly_chart(fig)"""
