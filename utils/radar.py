"""
radar.py
--------
Reusable radar (spider) chart for CB quality profiles.

Combines Ground Duel, Aerial Duel, and Ball Playing quality scores
into a single visualisation.  Values are converted to percentile ranks
(0–100) so the chart is always positive and intuitively readable.

Usage
-----
from utils.radar import plot_cb_quality_radar

fig = plot_cb_quality_radar(
    player_id            = 370,          # Virgil van Dijk
    duel_summary         = duel_summary,
    aerial_duel_summary  = aerial_duel_summary_filt,
    ball_playing_summary = bp,
)
fig.show()
"""

import plotly.graph_objects as go

# ── Palette (matches Visual class in classes/visual.py) ─────────────────────
_DARK_GREEN   = "#002c1c"
_MEDIUM_GREEN = "#003821"
_BRIGHT_GREEN = "#00A938"
_WHITE        = "#ffffff"


def plot_cb_quality_radar(
    player_id,
    duel_summary,
    aerial_duel_summary,
    ball_playing_summary,
):
    """
    Build a radar chart showing Ground Duel, Aerial Duel, and Ball Playing
    quality for a single CB.

    Parameters
    ----------
    player_id            : int   — Wyscout player ID
    duel_summary         : pd.DataFrame
        Must contain a 'duel_quality' column.
        Index can be MultiIndex (player.id, player.name) or flat with a
        'player.id' column — both are handled automatically.
    aerial_duel_summary  : pd.DataFrame
        Must contain 'aerial_duel_quality'.  Same index rules apply.
    ball_playing_summary : pd.DataFrame
        Must contain 'ball_playing_quality' and 'player.id' as a column.

    Returns
    -------
    plotly.graph_objects.Figure
        Call .show() in a notebook or pass to st.plotly_chart() in Streamlit.
    """
    # ── Normalise DataFrames to flat player.id column ────────────────────────
    gd = duel_summary[["duel_quality"]].reset_index()
    if "player.id" not in gd.columns:
        gd.columns = ["player.id", "player.name", "duel_quality"]

    ad = aerial_duel_summary[["aerial_duel_quality"]].reset_index()
    if "player.id" not in ad.columns:
        ad.columns = ["player.id", "player.name", "aerial_duel_quality"]

    bpq = ball_playing_summary[["player.id", "ball_playing_quality"]].copy()

    # ── Merge all three on player.id ─────────────────────────────────────────
    radar_df = (
        gd
        .merge(ad[["player.id", "aerial_duel_quality"]], on="player.id", how="inner")
        .merge(bpq, on="player.id", how="inner")
    )

    # ── Percentile ranks (0–100) ─────────────────────────────────────────────
    for col in ["duel_quality", "aerial_duel_quality", "ball_playing_quality"]:
        radar_df[col + "_pct"] = radar_df[col].rank(pct=True) * 100

    # ── Select player ────────────────────────────────────────────────────────
    match = radar_df[radar_df["player.id"] == player_id]
    if match.empty:
        raise ValueError(
            f"player_id {player_id} not found in the merged quality table. "
            "Check that the player appears in all three quality DataFrames."
        )
    row         = match.iloc[0]
    player_name = row.get("player.name", str(player_id))

    categories = ["Ground\nDuel", "Aerial\nDuel", "Ball\nPlaying"]
    values     = [
        row["duel_quality_pct"],
        row["aerial_duel_quality_pct"],
        row["ball_playing_quality_pct"],
    ]
    # Close the polygon
    values_closed     = values + [values[0]]
    categories_closed = categories + [categories[0]]

    # ── Build chart ──────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(0,169,56,0.15)",
        line=dict(color=_BRIGHT_GREEN, width=2),
        mode="lines+markers",
        marker=dict(size=8, color=_BRIGHT_GREEN),
        name=player_name,
    ))

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
            x=0.5, xanchor="center", y=-0.1, yanchor="bottom",
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        height=500,
    )

    return fig


def compare_cb_quality_radar(
    player_id_1,
    player_id_2,
    duel_summary,
    aerial_duel_summary,
    ball_playing_summary,
):
    """
    Compare two CBs on a single radar chart.

    Player 1 is shown in white, Player 2 in yellow — matching the
    DistributionPlot marker colour sequence used elsewhere in this project.

    Parameters
    ----------
    player_id_1 / player_id_2 : int  — Wyscout player IDs
    duel_summary              : pd.DataFrame  (must have 'duel_quality')
    aerial_duel_summary       : pd.DataFrame  (must have 'aerial_duel_quality')
    ball_playing_summary      : pd.DataFrame  (must have 'ball_playing_quality')

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _PLAYER_COLORS = [
        ("rgba(255,255,255,0.15)", "#ffffff"),   # player 1 — white
        ("rgba(255,204,0,0.15)",   "#ffcc00"),   # player 2 — bright yellow
    ]

    # ── Normalise DataFrames ─────────────────────────────────────────────────
    gd = duel_summary[["duel_quality"]].reset_index()
    if "player.id" not in gd.columns:
        gd.columns = ["player.id", "player.name", "duel_quality"]

    ad = aerial_duel_summary[["aerial_duel_quality"]].reset_index()
    if "player.id" not in ad.columns:
        ad.columns = ["player.id", "player.name", "aerial_duel_quality"]

    bpq = ball_playing_summary[["player.id", "ball_playing_quality"]].copy()

    radar_df = (
        gd
        .merge(ad[["player.id", "aerial_duel_quality"]], on="player.id", how="inner")
        .merge(bpq, on="player.id", how="inner")
    )

    for col in ["duel_quality", "aerial_duel_quality", "ball_playing_quality"]:
        radar_df[col + "_pct"] = radar_df[col].rank(pct=True) * 100

    categories = ["Ground\nDuel", "Aerial\nDuel", "Ball\nPlaying"]

    # ── Build chart ──────────────────────────────────────────────────────────
    fig = go.Figure()

    player_names = []
    for pid, (fill_color, line_color) in zip([player_id_1, player_id_2], _PLAYER_COLORS):
        match = radar_df[radar_df["player.id"] == pid]
        if match.empty:
            raise ValueError(
                f"player_id {pid} not found in the merged quality table."
            )
        row  = match.iloc[0]
        name = row.get("player.name", str(pid))
        player_names.append(name)

        values = [
            row["duel_quality_pct"],
            row["aerial_duel_quality_pct"],
            row["ball_playing_quality_pct"],
        ]
        values_closed     = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            mode="lines+markers",
            marker=dict(size=8, color=line_color),
            name=name,
        ))

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
                f"<span style='font-size:15px'>{player_names[0]} vs {player_names[1]}</span><br>"
                f"<span style='font-size:12px'>CB Quality Comparison · 2024-25 Premier League</span>"
            ),
            font=dict(family="Gilroy-Medium", color=_WHITE, size=12),
            x=0.05, xanchor="left", y=0.97, yanchor="top",
        ),
        legend=dict(
            font=dict(color=_WHITE, family="Gilroy-Light", size=11),
            x=0.5, xanchor="center", y=-0.1, yanchor="bottom",
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        height=500,
    )

    return fig
