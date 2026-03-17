"""
CB Pairing Analyst — Ground Duel Quality Distribution

Compares two center-backs on every duel-quality z-score metric relative to
all other qualifying Premier League center-backs.  Metric weights are
adjustable via sliders and a custom quality score is recalculated live.
"""

import os
import streamlit as st
import pandas as pd

from classes.visual import DistributionPlot
from utils.page_components import add_common_page_elements

# ── Page chrome ───────────────────────────────────────────────────────────────
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

# ── Metric configuration ──────────────────────────────────────────────────────
# Maps each z-score column → (readable label, default weight %)
METRIC_CONFIG = {
    "z_possession_win_rate":  ("Possession Win Rate",  35),
    "z_duel_success_rate":    ("Duel Success Rate",    20),
    "z_interceptions_per90":  ("Interceptions per 90", 15),
    "z_duels_per90":          ("Duels per 90",         10),
    "z_discipline":           ("Discipline",           10),
    "z_card_discipline":      ("Card Discipline",      10),
}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_cb_data():
    """
    Load the pre-computed CB duel quality CSV.
    Returns the raw df and a view df where columns are renamed to readable
    labels and hold z-score values directly (used by DistributionPlot).
    """
    path = os.path.join("data", "PL", "Qualities", "ground_duels.csv")
    df = pd.read_csv(path)

    rename_map = {z_col: label for z_col, (label, _) in METRIC_CONFIG.items()}
    view_df = df[["player.name"] + list(METRIC_CONFIG.keys())].rename(
        columns=rename_map
    )

    return df, view_df


df, view_df = load_cb_data()
player_names = sorted(df["player.name"].dropna().unique())

# ── Sidebar: two player selectors with same-player guard ─────────────────────
sidebar_container.markdown("**Player A**")
name_a = sidebar_container.selectbox(
    "Select first Center Back",
    player_names,
    key="cb_a",
)

# Exclude Player A so the second dropdown can never pick the same player
names_for_b = [n for n in player_names if n != name_a]
sidebar_container.markdown("**Player B**")
name_b = sidebar_container.selectbox(
    "Select second Center Back",
    names_for_b,
    key="cb_b",
)

# ── Weight sliders (above the plot) ───────────────────────────────────────────
st.subheader("Metric Weights")
st.caption(
    "Adjust how much each metric contributes to the quality score. "
    "Values are normalised automatically — relative importance is what matters."
)

cols = st.columns(3)
raw_weights = {}
for idx, (z_col, (label, default)) in enumerate(METRIC_CONFIG.items()):
    with cols[idx % 3]:
        raw_weights[label] = st.slider(label, min_value=0, max_value=100, value=default)

# Normalise to sum=1 (guard against all-zero edge case)
total = sum(raw_weights.values()) or 1
norm_weights = {label: w / total for label, w in raw_weights.items()}

# ── Distribution plot ─────────────────────────────────────────────────────────
display_cols = list(norm_weights.keys())

row_a = view_df[view_df["player.name"] == name_a].iloc[0]
row_b = view_df[view_df["player.name"] == name_b].iloc[0]

idx_a = df[df["player.name"] == name_a].index[0]
idx_b = df[df["player.name"] == name_b].index[0]

visual = DistributionPlot(display_cols[::-1])

visual.add_title(
    title=f"{name_a}  vs  {name_b}",
    subtitle="Ground Duel Quality  |  Premier League 2024",
)

# Background: all qualifying CBs as faint dots
visual.add_group_data(
    df_plot=view_df,
    plots="",           # view_df columns already hold z-score values
    names=view_df["player.name"],
    legend="All Center Backs",
    hover="",
    hover_string="",
)

# Player A highlight (white square — first colour/shape from generator)
visual.add_data_point(
    ser_plot=row_a,
    plots="",
    name=name_a,
    hover="",
    hover_string="",
)

# Player B highlight (yellow hexagon — second colour/shape from generator)
visual.add_data_point(
    ser_plot=row_b,
    plots="",
    name=name_b,
    hover="",
    hover_string="",
)

visual.show()

# ── Custom quality scores ─────────────────────────────────────────────────────
# Recompute a weighted composite for every CB using the current slider weights,
# then re-standardise so the scale stays consistent across weight configurations.

view_metrics = view_df[display_cols]
raw_quality   = sum(view_metrics[lbl] * w for lbl, w in norm_weights.items())
std           = raw_quality.std() or 1
custom_quality = (raw_quality - raw_quality.mean()) / std

score_a  = custom_quality.loc[idx_a]
score_b  = custom_quality.loc[idx_b]
rank_a   = int((custom_quality > score_a).sum()) + 1
rank_b   = int((custom_quality > score_b).sum()) + 1
n        = len(df)

# ── Comparison table ──────────────────────────────────────────────────────────
st.divider()

col_label, col_a, col_b = st.columns([2, 1, 1])
col_label.markdown("&nbsp;")          # spacer
col_a.markdown(f"**{name_a}**")
col_b.markdown(f"**{name_b}**")

rows = [
    ("Custom Quality Score", f"{score_a:.2f}",                              f"{score_b:.2f}"),
    ("Rank",                 f"{rank_a} / {n}",                             f"{rank_b} / {n}"),
    ("Duel Success Rate",    f"{df.loc[idx_a,'duel_success_rate']:.1%}",    f"{df.loc[idx_b,'duel_success_rate']:.1%}"),
    ("Possession Win Rate",  f"{df.loc[idx_a,'possession_win_rate']:.1%}",  f"{df.loc[idx_b,'possession_win_rate']:.1%}"),
    ("Discipline",           f"{df.loc[idx_a,'discipline']:.1%}",           f"{df.loc[idx_b,'discipline']:.1%}"),
    ("Duels per 90",         f"{df.loc[idx_a,'duels_per90']:.1f}",          f"{df.loc[idx_b,'duels_per90']:.1f}"),
    ("Interceptions per 90", f"{df.loc[idx_a,'interceptions_per90']:.1f}",  f"{df.loc[idx_b,'interceptions_per90']:.1f}"),
    ("Minutes played",       f"{df.loc[idx_a,'minutes']:.0f}",              f"{df.loc[idx_b,'minutes']:.0f}"),
]

for label, val_a, val_b in rows:
    c_lbl, c_a, c_b = st.columns([2, 1, 1])
    c_lbl.caption(label)
    c_a.write(val_a)
    c_b.write(val_b)
