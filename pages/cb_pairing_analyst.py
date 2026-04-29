"""
CB Pairing Analyst
==================
Unified page: sidebar selects Player A (required) and Player B (optional).
A single CBAnalystChat routes across all three modules based on the user's question:
  - get_player_summary    → individual CB profile
  - get_pair_evaluation   → how two CBs work together
  - get_better_partner    → suggest a better partner for Player A
"""

import os
import streamlit as st
import pandas as pd

from classes.data_point import Player
from classes.chat import CBAnalystChat
from utils.page_components import add_common_page_elements
from utils.utils import create_chat
from utils.radar import plot_cb_quality_radar, compare_cb_quality_radar

# ── Page chrome ───────────────────────────────────────────────────────────────
sidebar_container = add_common_page_elements()
page_container    = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_all_cb_data():
    base = os.path.join("data", "PL", "Qualities")

    gd = pd.read_csv(os.path.join(base, "ground_duels.csv"), index_col=0)
    bp = pd.read_csv(os.path.join(base, "ball_playing_quality.csv"))

    ad_path = os.path.join(base, "aerial_duels.csv")
    if os.path.exists(ad_path):
        ad = pd.read_csv(ad_path)
    else:
        st.warning("aerial_duels.csv not found — run the qualities notebook first.")
        ad = pd.DataFrame(columns=["player.id", "aerial_duel_quality",
                                    "z_aerial_duel_success_rate",
                                    "z_aerial_duels_per90",
                                    "z_aerial_won_duel_per90"])

    merged = (
        gd[["player.id", "player.name", "duel_quality",
            "z_duel_success_rate", "z_possession_win_rate",
            "z_discipline", "z_card_discipline",
            "z_duels_per90", "z_interceptions_per90"]]
        .merge(
            ad[["player.id", "aerial_duel_quality",
                "z_aerial_duel_success_rate", "z_aerial_duels_per90",
                "z_aerial_won_duel_per90"]],
            on="player.id", how="inner",
        )
        .merge(
            bp[["player.id", "ball_playing_quality",
                "z_Accuracy_Adjusted_Risk_per_Pass", "z_xT_per_Pass",
                "z_FT_Entry_Passes_per_90", "z_xT_via_Carries_per_90"]],
            on="player.id", how="inner",
        )
    )
    return gd, ad, bp, merged


gd_df, ad_df, bp_df, merged_df = load_all_cb_data()
player_names = sorted(merged_df["player.name"].dropna().unique())


def _make_player(name: str) -> Player:
    """Build a Player object with all merged metrics for the given name."""
    row = merged_df[merged_df["player.name"] == name].iloc[0]
    return Player(
        id=row["player.id"],
        name=name,
        minutes_played=gd_df.loc[gd_df["player.name"] == name, "minutes"].iloc[0]
                       if "minutes" in gd_df.columns else 0,
        gender="Male",
        position="CB",
        ser_metrics=row,
        relevant_metrics=list(merged_df.columns),
    )


# ── Sidebar selectors ─────────────────────────────────────────────────────────
sidebar_container.markdown("### Player A")
name_a = sidebar_container.selectbox(
    "Select Player A", player_names, key="cb_a"
)

sidebar_container.markdown("### Player B (optional)")
name_b = sidebar_container.selectbox(
    "Select Player B", ["None"] + [n for n in player_names if n != name_a], key="cb_b"
)

player_a = _make_player(name_a)
player_b = _make_player(name_b) if name_b != "None" else None

# ── Radar ─────────────────────────────────────────────────────────────────────
try:
    if player_b is not None:
        fig = compare_cb_quality_radar(
            player_id_1          = int(player_a.id),
            player_id_2          = int(player_b.id),
            duel_summary         = gd_df,
            aerial_duel_summary  = ad_df,
            ball_playing_summary = bp_df,
            show_detail          = False,
        )
    else:
        fig = plot_cb_quality_radar(
            player_id            = int(player_a.id),
            duel_summary         = gd_df,
            aerial_duel_summary  = ad_df,
            ball_playing_summary = bp_df,
            show_detail          = True,
        )
    st.plotly_chart(fig, use_container_width=True, theme=None)
except Exception as e:
    st.warning(f"Radar chart unavailable: {e}")

# ── AI Analyst Chat ───────────────────────────────────────────────────────────
st.divider()
title = f"CB Analyst — {name_a} & {name_b}" if player_b else f"CB Analyst — {name_a}"
st.subheader(title)

to_hash = (name_a, name_b, "cb_analyst_unified_v1")
chat = create_chat(
    to_hash, CBAnalystChat,
    player_a, player_b,
    gd_df, ad_df, bp_df,
)

if chat.state == "empty":
    if player_b:
        intro = (
            f"Hi! I'm your CB analyst. Ask me how {name_a} and {name_b} work together, "
            f"whether they complement each other, what their combined weaknesses are, "
            f"or who would make a better partner."
        )
    else:
        intro = (
            f"Hi! I'm your CB analyst. Ask me anything about {name_a} — "
            f"his strengths, weaknesses, aerial ability, ball playing, or how he ranks overall. "
            f"You can also select a second player in the sidebar to compare pairings."
        )
    chat.add_message(intro)
    chat.state = "default"

# ── Quick question chips ──────────────────────────────────────────────────────
st.caption("Quick questions:")

if player_b:
    preset_questions = [
        f"How do {name_a} and {name_b} complement each other?",
        f"Are {name_a} and {name_b} too similar?",
        f"What is the main weakness of this pairing?",
        f"Who would be a better partner for {name_a}?",
        f"Who could improve the aerial quality of this pair?",
        f"Which player improves ball playing next to {name_a}?",
    ]
else:
    preset_questions = [
        f"What are {name_a}'s main strengths?",
        f"What are {name_a}'s weaknesses?",
        f"How good is {name_a} aerially?",
        f"How does {name_a} contribute on the ball?",
        f"How does {name_a} rank overall?",
        f"What kind of partner would suit {name_a}?",
    ]

cols = st.columns(2)
for i, q in enumerate(preset_questions):
    with cols[i % 2]:
        if st.button(q, key=f"preset_{i}"):
            chat.handle_input(q, stream=True)

chat.get_input()
chat.display_messages()
chat.save_state()
