"""
CB Pairing Analyst — Ground Duel Quality Distribution

Two tabs:
  1. Individual Player — explore a single CB's metrics vs the league
  2. CB Pair          — compare two CBs and chat with the AI analyst
"""

import os
import streamlit as st
import pandas as pd

from classes.visual import DistributionPlot
from classes.description import DefenderDescription, CBPairingDescription
from classes.data_point import Player
from classes.chat import CBPairingChat, SingleCBChat
from utils.page_components import add_common_page_elements
from utils.utils import create_chat

# ── Page chrome ───────────────────────────────────────────────────────────────
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

# ── Metric configuration ──────────────────────────────────────────────────────
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
    path = os.path.join("data", "PL", "Qualities", "ground_duels.csv")
    df = pd.read_csv(path)

    rename_map = {z_col: label for z_col, (label, _) in METRIC_CONFIG.items()}
    view_df = df[["player.name"] + list(METRIC_CONFIG.keys())].rename(
        columns=rename_map
    )

    return df, view_df


df, view_df = load_cb_data()
player_names = sorted(df["player.name"].dropna().unique())

# ── Sidebar selectors ─────────────────────────────────────────────────────────
sidebar_container.markdown("### Individual Player")
name_single = sidebar_container.selectbox(
    "Select a Centre Back",
    player_names,
    key="cb_single",
)

sidebar_container.markdown("### CB Pair")
sidebar_container.markdown("**Player A**")
name_a = sidebar_container.selectbox(
    "Select first Center Back",
    player_names,
    key="cb_a",
)

names_for_b = [n for n in player_names if n != name_a]
sidebar_container.markdown("**Player B**")
name_b = sidebar_container.selectbox(
    "Select second Center Back",
    names_for_b,
    key="cb_b",
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Individual Player", "CB Pair"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Individual Player
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Weight sliders ────────────────────────────────────────────────────────
    st.subheader("Metric Weights")
    st.caption(
        "Adjust how much each metric contributes to the quality score. "
        "Values are normalised automatically — relative importance is what matters."
    )

    cols_s = st.columns(3)
    raw_weights_s = {}
    for idx, (z_col, (label, default)) in enumerate(METRIC_CONFIG.items()):
        with cols_s[idx % 3]:
            raw_weights_s[label] = st.slider(
                label, min_value=0, max_value=100, value=default, key=f"single_{z_col}"
            )

    total_s = sum(raw_weights_s.values()) or 1
    norm_weights_s = {label: w / total_s for label, w in raw_weights_s.items()}

    display_cols_s = list(norm_weights_s.keys())

    idx_s = df[df["player.name"] == name_single].index[0]
    row_s = view_df[view_df["player.name"] == name_single].iloc[0]

    # ── Distribution plot ─────────────────────────────────────────────────────
    visual = DistributionPlot(display_cols_s[::-1])
    visual.add_title(
        title=name_single,
        subtitle="Ground Duel Quality  |  Premier League 2024",
    )
    visual.add_group_data(
        df_plot=view_df,
        plots="",
        names=view_df["player.name"],
        legend="All Center Backs",
        hover="",
        hover_string="",
    )
    visual.add_data_point(
        ser_plot=row_s,
        plots="",
        name=name_single,
        hover="",
        hover_string="",
    )
    visual.show()

    # ── Custom quality score ──────────────────────────────────────────────────
    view_metrics_s = view_df[display_cols_s]
    raw_quality_s  = sum(view_metrics_s[lbl] * w for lbl, w in norm_weights_s.items())
    std_s          = raw_quality_s.std() or 1
    custom_quality_s = (raw_quality_s - raw_quality_s.mean()) / std_s

    score_s = custom_quality_s.loc[idx_s]
    rank_s  = int((custom_quality_s > score_s).sum()) + 1
    n_s     = len(df)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.divider()

    col_label_s, col_val_s = st.columns([2, 1])
    col_label_s.markdown("&nbsp;")
    col_val_s.markdown(f"**{name_single}**")

    rows_s = [
        ("Custom Quality Score", f"{score_s:.2f}"),
        ("Rank",                 f"{rank_s} / {n_s}"),
        ("Duel Success Rate",    f"{df.loc[idx_s, 'duel_success_rate']:.1%}"),
        ("Possession Win Rate",  f"{df.loc[idx_s, 'possession_win_rate']:.1%}"),
        ("Discipline",           f"{df.loc[idx_s, 'discipline']:.1%}"),
        ("Duels per 90",         f"{df.loc[idx_s, 'duels_per90']:.1f}"),
        ("Interceptions per 90", f"{df.loc[idx_s, 'interceptions_per90']:.1f}"),
        ("Minutes played",       f"{df.loc[idx_s, 'minutes']:.0f}"),
    ]

    for label, val in rows_s:
        c_lbl, c_val = st.columns([2, 1])
        c_lbl.caption(label)
        c_val.write(val)

    # ── AI-Generated Summary ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Ground Duel Quality Summary")

    player_single = Player(
        id=df.loc[idx_s, "player.id"],
        name=name_single,
        minutes_played=df.loc[idx_s, "minutes"],
        gender="Male",
        position="CB",
        ser_metrics=df.loc[idx_s],
        relevant_metrics=list(METRIC_CONFIG.keys()),
    )

    try:
        desc = DefenderDescription(player_single)
        summary = desc.stream_gpt()
        st.write(summary)
    except Exception as e:
        st.error(f"Could not generate summary: {e}")

    # ── Interactive Chat ──────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Ask Questions About {name_single}")

    to_hash_s = (name_single, "cb_single_analyst")
    chat_s = create_chat(to_hash_s, SingleCBChat, player_single, df)

    if chat_s.state == "empty":
        chat_s.add_message(
            f"Tell me about {name_single}'s defensive profile.",
            role="user",
            user_only=False,
            visible=False,
        )
        welcome_msg_s = (
            f"I can help you analyze {name_single}'s defensive profile. "
            f"Ask me about their ground duel quality metrics, strengths, weaknesses, "
            f"or how they compare to other centre-backs in the league."
        )
        chat_s.add_message(welcome_msg_s)
        chat_s.state = "default"

    # ── Predefined questions ──────────────────────────────────────────────────
    st.caption("Quick questions to try:")
    preset_questions = [
        f"What are {name_single}'s main defensive strengths?",
        f"Where does {name_single} rank compared to other centre-backs?",
        f"How active is {name_single} in defensive duels?",
        f"What areas could {name_single} improve?",
        f"How disciplined is {name_single} in duels?",
        f"What defensive style does {name_single} have?",
    ]
    q_cols = st.columns(2)
    for i, question in enumerate(preset_questions):
        with q_cols[i % 2]:
            if st.button(question, key=f"preset_s_{i}"):
                chat_s.handle_input(question, stream=True)

    chat_s.get_input()
    chat_s.display_messages()
    chat_s.save_state()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CB Pair
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # ── Weight sliders ────────────────────────────────────────────────────────
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

    total = sum(raw_weights.values()) or 1
    norm_weights = {label: w / total for label, w in raw_weights.items()}

    display_cols = list(norm_weights.keys())

    row_a = view_df[view_df["player.name"] == name_a].iloc[0]
    row_b = view_df[view_df["player.name"] == name_b].iloc[0]

    idx_a = df[df["player.name"] == name_a].index[0]
    idx_b = df[df["player.name"] == name_b].index[0]

    # ── Distribution plot ─────────────────────────────────────────────────────
    visual = DistributionPlot(display_cols[::-1])
    visual.add_title(
        title=f"{name_a}  vs  {name_b}",
        subtitle="Ground Duel Quality  |  Premier League 2024",
    )
    visual.add_group_data(
        df_plot=view_df,
        plots="",
        names=view_df["player.name"],
        legend="All Center Backs",
        hover="",
        hover_string="",
    )
    visual.add_data_point(
        ser_plot=row_a,
        plots="",
        name=name_a,
        hover="",
        hover_string="",
        show_annotation=False,
    )
    visual.add_data_point(
        ser_plot=row_b,
        plots="",
        name=name_b,
        hover="",
        hover_string="",
        show_annotation=False,
    )
    visual.add_pair_annotations(row_a, row_b)
    visual.show()

    # ── Custom quality scores ─────────────────────────────────────────────────
    view_metrics  = view_df[display_cols]
    raw_quality   = sum(view_metrics[lbl] * w for lbl, w in norm_weights.items())
    std           = raw_quality.std() or 1
    custom_quality = (raw_quality - raw_quality.mean()) / std

    score_a = custom_quality.loc[idx_a]
    score_b = custom_quality.loc[idx_b]
    rank_a  = int((custom_quality > score_a).sum()) + 1
    rank_b  = int((custom_quality > score_b).sum()) + 1
    n       = len(df)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.divider()

    col_label, col_a, col_b = st.columns([2, 1, 1])
    col_label.markdown("&nbsp;")
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

    # ── Player objects ────────────────────────────────────────────────────────
    player_a = Player(
        id=df.loc[idx_a, "player.id"],
        name=name_a,
        minutes_played=df.loc[idx_a, "minutes"],
        gender="Male",
        position="CB",
        ser_metrics=df.loc[idx_a],
        relevant_metrics=list(METRIC_CONFIG.keys()),
    )

    player_b = Player(
        id=df.loc[idx_b, "player.id"],
        name=name_b,
        minutes_played=df.loc[idx_b, "minutes"],
        gender="Male",
        position="CB",
        ser_metrics=df.loc[idx_b],
        relevant_metrics=list(METRIC_CONFIG.keys()),
    )

    # ── AI-Generated Pairing Summary ──────────────────────────────────────────
    st.divider()
    st.subheader(f"Partnership Analysis: {name_a} & {name_b}")

    use_wordalisation = not st.checkbox(
        "Run without wordalisation (no domain Q&A or few-shot examples)",
        value=False,
        key="no_wordalisation",
    )

    try:
        z_cols = list(METRIC_CONFIG.keys())
        df_metrics = df[["player.name"] + z_cols]
        pairing_desc = CBPairingDescription(player_a, player_b, df=df_metrics, use_wordalisation=use_wordalisation)
        pairing_summary = pairing_desc.stream_gpt()
        st.write(pairing_summary)
    except Exception as e:
        st.error(f"Could not generate pairing summary: {e}")

    # ── Interactive Chat ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Ask Questions About the Pairing")

    to_hash = (name_a, name_b, "cb_pairing_analyst")
    chat = create_chat(to_hash, CBPairingChat, player_a, player_b, df)

    if chat.state == "empty":
        chat.add_message(
            f"Tell me about the partnership between {name_a} and {name_b}.",
            role="user",
            user_only=False,
            visible=False,
        )
        welcome_msg = (
            f"I can help you analyze the defensive partnership between {name_a} and {name_b}. "
            f"Ask me about how their ground duel quality metrics complement each other, "
            f"which player is stronger in specific areas, or how they might work together as a pairing."
        )
        chat.add_message(welcome_msg)
        chat.state = "default"

    # ── Predefined questions ──────────────────────────────────────────────────
    st.caption("Quick questions to try:")
    preset_pair_questions = [
        f"How do {name_a} and {name_b} complement each other?",
        f"Which player is stronger in ground duels?",
        f"How do they compare in defensive activity?",
        f"Which player is more disciplined?",
        f"What are the weaknesses of this pairing?",
        f"What defensive system would suit this pair?",
    ]
    q_cols_pair = st.columns(2)
    for i, question in enumerate(preset_pair_questions):
        with q_cols_pair[i % 2]:
            if st.button(question, key=f"preset_pair_{i}"):
                chat.handle_input(question, stream=True)

    chat.get_input()
    chat.display_messages()
    chat.save_state()
