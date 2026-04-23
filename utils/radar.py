"""
radar.py
--------
Reusable radar (spider) chart for CB quality profiles.

Combines Ground Duel, Aerial Duel, and Ball Playing quality scores
into a single visualisation.  Values are converted to percentile ranks
(0-100) so the chart is always positive and intuitively readable.

Usage
-----
from utils.radar import plot_cb_quality_radar, compare_cb_quality_radar

# Single player - radar only
fig = plot_cb_quality_radar(player_id=370, ...)

# Single player - radar + sub-metric strip
fig = plot_cb_quality_radar(player_id=370, ..., show_detail=True)

# Two-player comparison - radar only
fig = compare_cb_quality_radar(player_id_1=370, player_id_2=415917, ...)

# Two-player comparison + sub-metric strip
fig = compare_cb_quality_radar(player_id_1=370, player_id_2=415917, ..., show_detail=True)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- Palette (matches Visual class in classes/visual.py) ---------------------
_DARK_GREEN   = "#002c1c"
_MEDIUM_GREEN = "#003821"
_BRIGHT_GREEN = "#00A938"
_WHITE        = "#ffffff"
_YELLOW       = "#ffcc00"
_BLUE         = "#0095FF"

# -- Player marker colours ----------------------------------------------------
# Single player -> white.  Two-player comparison -> white / yellow.
_PLAYER_COLORS = [
    ("rgba(255,255,255,0.4)", _WHITE),    # player 1 - white
    ("rgba(255,204,0,0.4)",   _YELLOW),  # player 2 - yellow
]

# -- Sub-metric definitions ---------------------------------------------------
# Each entry: (group_label, source_key, group_header_color, [(display_label, z_col), ...])
# source_key maps to the normalised DataFrame: "gd", "ad", "bp"
_SUBMETRIC_GROUPS = [
    ("Ground Duel", "gd", _BRIGHT_GREEN, [
        ("Duel Success Rate",    "z_duel_success_rate"),
        ("Possession Win Rate",  "z_possession_win_rate"),
        ("Discipline",           "z_discipline"),
        ("Duels per 90",         "z_duels_per90"),
        ("Interceptions p90",    "z_interceptions_per90"),
    ]),
    ("Aerial Duel", "ad", _BLUE, [
        ("Aerial Success Rate",  "z_aerial_duel_success_rate"),
        ("Aerial Duels per 90",  "z_aerial_duels_per90"),
        ("Aerial Won per 90",    "z_aerial_won_duel_per90"),
    ]),
    ("Ball Playing", "bp", _YELLOW, [
        ("xT Risk Pass",          "z_Accuracy_Adjusted_Risk_per_Pass"),
        ("xT per Pass",          "z_xT_per_Pass"),
        ("FT Passes per 90",    "z_FT_Entry_Passes_per_90"),
        ("xT via Carries p90",   "z_xT_via_Carries_per_90"),
    ]),
]


def _normalise(df):
    """Reset MultiIndex -> flat DataFrame with player.id as a regular column."""
    out = df.reset_index()
    if "player.id" not in out.columns:
        cols = out.columns.tolist()
        out = out.rename(columns={cols[0]: "player.id", cols[1]: "player.name"})
    return out


def _add_submetric_strip(fig, sources, players_info):
    """
    Append a sub-metric strip to row 2 of fig.

    Parameters
    ----------
    fig          : go.Figure  - must have been created with make_subplots (row 2 is xy)
    sources      : dict       - {"gd": df, "ad": df, "bp": df}  (normalised DataFrames)
    players_info : list of (player_id, player_name, fill_color, line_color)
                   Single player -> one tuple.  Comparison -> two tuples.
    """
    y_ticks, y_labels = [], []
    group_annotations = []
    separator_ys = []

    y = 0
    for group_name, src_key, group_color, metrics in reversed(_SUBMETRIC_GROUPS):
        group_start = y
        for label, z_col in metrics:
            src_df = sources[src_key]
            if z_col not in src_df.columns:
                y += 1
                continue

            # Background: all CBs as faint green dots
            fig.add_trace(go.Scatter(
                x=src_df[z_col].tolist(),
                y=[y] * len(src_df),
                mode="markers",
                marker=dict(
                    color="rgba(0,44,28,0.2)",
                    size=9,
                    line=dict(width=1.5, color="rgba(0,169,56,0.6)"),
                ),
                hoverinfo="skip",
                showlegend=False,
            ), row=2, col=1)

            # Player highlights - one square per player
            for pid, pname, fill_color, line_color in players_info:
                player_val = src_df.loc[src_df["player.id"] == pid, z_col]
                if not player_val.empty:
                    fig.add_trace(go.Scatter(
                        x=[player_val.iloc[0]],
                        y=[y],
                        mode="markers",
                        marker=dict(
                            color=fill_color,
                            size=10,
                            symbol="square",
                            line=dict(width=1.5, color=line_color),
                        ),
                        name=pname,
                        showlegend=False,
                        hovertemplate=f"{label}: %{{x:.2f}}<extra></extra>",
                    ), row=2, col=1)

            y_ticks.append(y)
            y_labels.append(label)
            y += 1

        # Group header annotation at vertical centre of this group
        group_mid = (group_start + y - 1) / 2
        group_annotations.append(dict(
            xref="x", yref="y",
            x=-3.4, y=group_mid,
            text=f"<b>{group_name}</b>",
            showarrow=False,
            font=dict(color=group_color, family="Gilroy-Medium", size=11),
            xanchor="left",
        ))

        if y > 0:
            separator_ys.append(y - 0.5)
        y += 1.8  # gap between groups

    # Separator lines between groups (skip after the last group)
    for sep_y in separator_ys[:-1]:
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=-3.5, x1=3.5, y0=sep_y, y1=sep_y,
            line=dict(color="rgba(255,255,255,0.08)", width=1),
            row=2, col=1,
        )

    # Centre dotted line at z = 0
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=0, x1=0, y0=0, y1=1,
        line=dict(color="rgba(128,128,128,0.4)", width=1, dash="dot"),
        row=2, col=1,
    )

    fig.update_xaxes(
        range=[-3.5, 3.5],
        tickmode="array",
        tickvals=[-3, 0, 3],
        ticktext=["Worse", "Average", "Better"],
        tickfont=dict(color="rgba(255,255,255,0.5)", family="Gilroy-Light", size=11),
        gridcolor=_MEDIUM_GREEN,
        zerolinecolor=_MEDIUM_GREEN,
        row=2, col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_ticks,
        ticktext=y_labels,
        tickfont=dict(color="rgba(255,255,255,0.7)", family="Gilroy-Light", size=11),
        showgrid=False,
        zeroline=False,
        row=2, col=1,
    )

    for ann in group_annotations:
        fig.add_annotation(**ann, row=2, col=1)


def plot_cb_quality_radar(
    player_id,
    duel_summary,
    aerial_duel_summary,
    ball_playing_summary,
    show_detail=False,
):
    """
    Build a radar chart for a single CB, optionally extended with a
    sub-metric breakdown strip below the radar.

    Parameters
    ----------
    player_id            : int   - Wyscout player ID
    duel_summary         : pd.DataFrame  - must contain 'duel_quality' and
                           z-score columns (z_duel_success_rate, etc.)
    aerial_duel_summary  : pd.DataFrame  - must contain 'aerial_duel_quality'
                           and z-score columns
    ball_playing_summary : pd.DataFrame  - must contain 'ball_playing_quality'
                           and z-score columns; 'player.id' as a flat column
    show_detail          : bool  - if True, appends a sub-metric strip below
                           the radar (default False for backward compatibility)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # -- Normalise DataFrames -------------------------------------------------
    gd    = _normalise(duel_summary)
    ad    = _normalise(aerial_duel_summary)
    bp_df = ball_playing_summary.copy()

    sources = {"gd": gd, "ad": ad, "bp": bp_df}

    # -- Radar data: merge quality scores + percentile ranks ------------------
    radar_df = (
        gd[["player.id", "duel_quality"]]
        .merge(ad[["player.id", "aerial_duel_quality"]], on="player.id", how="inner")
        .merge(bp_df[["player.id", "ball_playing_quality"]], on="player.id", how="inner")
    )
    for col in ["duel_quality", "aerial_duel_quality", "ball_playing_quality"]:
        radar_df[col + "_pct"] = radar_df[col].rank(pct=True) * 100

    match = radar_df[radar_df["player.id"] == player_id]
    if match.empty:
        raise ValueError(
            f"player_id {player_id} not found. "
            "Check that the player appears in all three quality DataFrames."
        )
    row         = match.iloc[0]
    player_name = gd.loc[gd["player.id"] == player_id, "player.name"].iloc[0] \
                  if "player.name" in gd.columns else str(player_id)

    # -- Build figure ---------------------------------------------------------
    if show_detail:
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{"type": "polar"}], [{"type": "xy"}]],
            row_heights=[0.42, 0.58],
            vertical_spacing=0.04,
        )
    else:
        fig = go.Figure()

    # -- Radar trace ----------------------------------------------------------
    categories = ["Ground\nDuel", "Aerial\nDuel", "Ball\nPlaying"]
    values     = [
        row["duel_quality_pct"],
        row["aerial_duel_quality_pct"],
        row["ball_playing_quality_pct"],
    ]
    radar_trace = go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255,255,255,0.08)",
        line=dict(color=_WHITE, width=2),
        mode="lines+markers",
        marker=dict(size=8, color=_WHITE),
        name=player_name,
    )
    if show_detail:
        fig.add_trace(radar_trace, row=1, col=1)
    else:
        fig.add_trace(radar_trace)

    # -- Sub-metric strip -----------------------------------------------------
    if show_detail:
        players_info = [
            (player_id, player_name, _PLAYER_COLORS[0][0], _PLAYER_COLORS[0][1]),
        ]
        _add_submetric_strip(fig, sources, players_info)

    # -- Layout ---------------------------------------------------------------
    fig.update_layout(
        paper_bgcolor=_DARK_GREEN,
        plot_bgcolor=_DARK_GREEN,
        polar=dict(
            bgcolor=_MEDIUM_GREEN,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75],
                ticktext=["25", "50", "75"],
                tickfont=dict(color="rgba(255,255,255,0.4)", size=10, family="Gilroy-Light"),
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                tickfont=dict(color=_WHITE, size=13, family="Gilroy-Medium"),
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
            ),
        ),
        title=dict(
            text=(
                f"<span style='font-size:15px'>{player_name}</span><br>"
                f"<span style='font-size:12px'>CB Quality Profile · 2024-25 Premier League</span>"
            ),
            font=dict(family="Gilroy-Medium", color=_WHITE, size=12),
            x=0.05, xanchor="left", y=0.97, yanchor="top",
        ),
        legend=dict(
            font=dict(color=_WHITE, family="Gilroy-Light", size=11),
            x=0.5, xanchor="center", y=-0.05, yanchor="bottom",
        ),
        margin=dict(l=160, r=60, t=80, b=60),
        height=900 if show_detail else 500,
    )

    return fig


def compare_cb_quality_radar(
    player_id_1,
    player_id_2,
    duel_summary,
    aerial_duel_summary,
    ball_playing_summary,
    show_detail=False,
):
    """
    Compare two CBs on a single radar chart, optionally with a sub-metric strip.

    Player 1 is shown in white, Player 2 in yellow - matching the
    DistributionPlot marker colour sequence used elsewhere in this project.

    Parameters
    ----------
    player_id_1 / player_id_2 : int  - Wyscout player IDs
    duel_summary              : pd.DataFrame  (must have 'duel_quality')
    aerial_duel_summary       : pd.DataFrame  (must have 'aerial_duel_quality')
    ball_playing_summary      : pd.DataFrame  (must have 'ball_playing_quality')
    show_detail               : bool  - if True, appends a sub-metric strip
                                below the radar (default False)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    gd    = _normalise(duel_summary)
    ad    = _normalise(aerial_duel_summary)
    bp_df = ball_playing_summary.copy()

    sources = {"gd": gd, "ad": ad, "bp": bp_df}

    radar_df = (
        gd[["player.id", "duel_quality"]]
        .merge(ad[["player.id", "aerial_duel_quality"]], on="player.id", how="inner")
        .merge(bp_df[["player.id", "ball_playing_quality"]], on="player.id", how="inner")
    )
    for col in ["duel_quality", "aerial_duel_quality", "ball_playing_quality"]:
        radar_df[col + "_pct"] = radar_df[col].rank(pct=True) * 100

    categories   = ["Ground\nDuel", "Aerial\nDuel", "Ball\nPlaying"]
    player_names = []
    players_info = []

    if show_detail:
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{"type": "polar"}], [{"type": "xy"}]],
            row_heights=[0.42, 0.58],
            vertical_spacing=0.04,
        )
    else:
        fig = go.Figure()

    for pid, (fill_color, line_color) in zip(
        [player_id_1, player_id_2], _PLAYER_COLORS
    ):
        match = radar_df[radar_df["player.id"] == pid]
        if match.empty:
            raise ValueError(f"player_id {pid} not found in the merged quality table.")
        row  = match.iloc[0]
        name = gd.loc[gd["player.id"] == pid, "player.name"].iloc[0] \
               if "player.name" in gd.columns else str(pid)
        player_names.append(name)
        players_info.append((pid, name, fill_color, line_color))

        values = [
            row["duel_quality_pct"],
            row["aerial_duel_quality_pct"],
            row["ball_playing_quality_pct"],
        ]
        radar_trace = go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            mode="lines+markers",
            marker=dict(size=8, color=line_color),
            name=name,
        )
        if show_detail:
            fig.add_trace(radar_trace, row=1, col=1)
        else:
            fig.add_trace(radar_trace)

    # -- Sub-metric strip -----------------------------------------------------
    if show_detail:
        _add_submetric_strip(fig, sources, players_info)

    # -- Layout ---------------------------------------------------------------
    fig.update_layout(
        paper_bgcolor=_DARK_GREEN,
        plot_bgcolor=_DARK_GREEN,
        polar=dict(
            bgcolor=_MEDIUM_GREEN,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75],
                ticktext=["25", "50", "75"],
                tickfont=dict(color="rgba(255,255,255,0.4)", size=10, family="Gilroy-Light"),
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                tickfont=dict(color=_WHITE, size=13, family="Gilroy-Medium"),
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
            ),
        ),
        title=dict(
            text=(
                f"<span style='font-size:15px'>{player_names[0]} & {player_names[1]}</span><br>"
                f"<span style='font-size:12px'>CB Quality Comparison · 2024-25 Premier League</span>"
            ),
            font=dict(family="Gilroy-Medium", color=_WHITE, size=12),
            x=0.05, xanchor="left", y=0.97, yanchor="top",
        ),
        legend=dict(
            font=dict(color=_WHITE, family="Gilroy-Light", size=11),
            x=0.5, xanchor="center", y=-0.1, yanchor="bottom",
        ),
        margin=dict(l=160 if show_detail else 60, r=60, t=80, b=60),
        height=900 if show_detail else 500,
    )

    return fig
