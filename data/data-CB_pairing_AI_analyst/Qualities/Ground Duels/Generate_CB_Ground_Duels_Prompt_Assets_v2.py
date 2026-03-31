from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (
            (candidate / "settings.py").exists()
            and (candidate / "utils").exists()
            and (candidate / "data").exists()
        ):
            return candidate
    raise RuntimeError("Could not locate repository root from script location.")


SCRIPT_PATH = Path(__file__).resolve()
GROUND_DIR = SCRIPT_PATH.parent
REPO_ROOT = _find_repo_root(GROUND_DIR)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from utils import sentences as sentence_utils
except Exception:
    sentence_utils = None


CB_CSV = GROUND_DIR / "CB_ground_duels.csv"
PAIR_CSV = GROUND_DIR / "CB_pairs_ground_duels.csv"
COMP_CSV = GROUND_DIR / "CB_companion_fit_ground_duels.csv"

DESCRIBE_DIR = REPO_ROOT / "data" / "describe"
GPT_EXAMPLES_DIR = REPO_ROOT / "data" / "gpt_examples"

JSON_OUT = GROUND_DIR / "prompts_v2_CBs_and_CB_Pairs.json"
DESCRIBE_XLSX_OUT = DESCRIBE_DIR / "CBs_and_CB_Pairs.xlsx"
GPT_EXAMPLES_XLSX_OUT = GPT_EXAMPLES_DIR / "CBs_and_CB_Pairs.xlsx"


KNOWLEDGE_PAIR_COUNT = 60
FEW_SHOT_PER_CONTEXT = 20
FEW_SHOT_BUCKET_TOP = 6
FEW_SHOT_BUCKET_MID = 6
FEW_SHOT_BUCKET_LOW = 6
FEW_SHOT_EDGE = 2


THRESHOLDS = [1.5, 1, 0.5, -0.5, -1]
WORDS = ["outstanding", "excellent", "good", "average", "below average", "poor"]

Z_METRIC_COLS = [
    "z_duel_success_rate",
    "z_possession_win_rate",
    "z_discipline",
    "z_card_discipline",
    "z_duels_per90",
    "z_interceptions_per90",
]

METRIC_FOOTBALL_MEANING = {
    "z_duel_success_rate": "stopping attackers and winning the duel itself",
    "z_possession_win_rate": "turning defensive duels into direct ball recoveries",
    "z_discipline": "defending cleanly without fouling",
    "z_card_discipline": "avoiding bookings and dismissals in those actions",
    "z_duels_per90": "engaging consistently in defensive ground actions",
    "z_interceptions_per90": "stepping in to break up attacks and regain control",
}


@dataclass(frozen=True)
class FewShotExample:
    context: str
    key: str
    user: str
    assistant: str
    assistant_sentence_count: int


def describe_level(value: float) -> str:
    if sentence_utils is not None:
        return sentence_utils.describe_level(value, thresholds=THRESHOLDS, words=WORDS)
    i = 0
    while i < len(THRESHOLDS) and value < THRESHOLDS[i]:
        i += 1
    return WORDS[i]


def ordinal(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def with_indef_article(word: str) -> str:
    article = "an" if str(word).strip().lower()[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {word}"


def bool_to_word(v: object) -> str:
    return "yes" if bool(v) else "no"


def join_phrases(items: Sequence[str]) -> str:
    values = [str(i).strip() for i in items if str(i).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _dedupe_preserve_order(values: Sequence[object]) -> List[object]:
    out: List[object] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _evenly_spaced(values: Sequence[object], n: int) -> List[object]:
    values = list(values)
    if n <= 0 or not values:
        return []
    if len(values) <= n:
        return values.copy()
    picks = np.linspace(0, len(values) - 1, n)
    out: List[object] = []
    seen = set()
    for pick in picks:
        value = values[int(round(pick))]
        if value not in seen:
            seen.add(value)
            out.append(value)
    if len(out) < n:
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)
            if len(out) == n:
                break
    return out[:n]


def _stable_sort(df: pd.DataFrame, score_col: str, key_cols: Sequence[str]) -> pd.DataFrame:
    sort_cols = [score_col, *key_cols]
    ascending = [False, *([True] * len(key_cols))]
    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _build_text_from_sentences(sentences: Sequence[str]) -> Tuple[str, int]:
    cleaned = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    if len(cleaned) != 4:
        raise ValueError(f"Expected 4 sentences, received {len(cleaned)}.")
    text = " ".join(cleaned)
    return text, len(cleaned)


def _top_metric_phrases(row: pd.Series, n: int = 2) -> List[str]:
    ranked = sorted(Z_METRIC_COLS, key=lambda c: row[c], reverse=True)
    selected = [metric for metric in ranked if row[metric] >= 0.5][:n]
    if not selected:
        selected = ranked[:n]
    return [METRIC_FOOTBALL_MEANING[metric] for metric in selected]


def _low_metric_phrases(row: pd.Series, n: int = 2) -> List[str]:
    ranked = sorted(Z_METRIC_COLS, key=lambda c: row[c])
    selected = [metric for metric in ranked if row[metric] <= -0.5][:n]
    if not selected:
        selected = ranked[:n]
    return [METRIC_FOOTBALL_MEANING[metric] for metric in selected]


def _build_cb_user(row: pd.Series) -> str:
    lines = [
        f"Here is a statistical description of {row['player.name']}, who played for {int(round(row['minutes_played']))} minutes as a centre-back.",
        f"He was {describe_level(row['z_duel_success_rate'])} in duel_success_rate ({row['duel_success_rate']:.3f}; z={row['z_duel_success_rate']:.2f}) compared to other centre-backs in the same sample.",
        f"He was {describe_level(row['z_possession_win_rate'])} in possession_win_rate ({row['possession_win_rate']:.3f}; z={row['z_possession_win_rate']:.2f}).",
        f"He was {describe_level(row['z_discipline'])} in discipline ({row['discipline']:.3f}; z={row['z_discipline']:.2f}) and {describe_level(row['z_card_discipline'])} in card_discipline ({row['card_discipline']:.3f}; z={row['z_card_discipline']:.2f}).",
        f"He was {describe_level(row['z_duels_per90'])} in duels_per90 ({row['duels_per90']:.3f}; z={row['z_duels_per90']:.2f}) and {describe_level(row['z_interceptions_per90'])} in interceptions_per90 ({row['interceptions_per90']:.3f}; z={row['z_interceptions_per90']:.2f}).",
        f"He recorded {int(row['yellow_cards'])} yellow cards and {int(row['red_cards'])} red cards across {int(row['total_duels'])} defensive ground duels.",
        f"His ground_duel_quality_score was {row['ground_duel_quality_score']:.3f} (z-scale).",
    ]
    return "\n".join(lines)


def _build_cb_assistant(row: pd.Series, n_players: int) -> Tuple[str, int]:
    rank = int(row["ground_duel_quality_score_rank_desc"])
    percentile = int(round((1 - (rank - 1) / n_players) * 100))

    strengths = _top_metric_phrases(row, n=2)
    weaknesses = _low_metric_phrases(row, n=2)

    risk_flags: List[str] = []
    if row["z_discipline"] <= -0.5:
        risk_flags.append("foul control is a risk")
    if row["z_card_discipline"] <= -0.5:
        risk_flags.append("card exposure is a risk")
    risk_clause = f"; {join_phrases(risk_flags)}" if risk_flags else ""

    sentences = [
        f"{row['player.name']} profiles as {with_indef_article(describe_level(row['ground_duel_quality_score']))} ground-duel centre-back in this Premier League sample.",
        f"His strongest ground-duel traits are {join_phrases(strengths)}, which drives his defensive value in one-v-one phases.",
        f"His weaker areas are {join_phrases(weaknesses)} against this CB cohort{risk_clause}.",
        f"Overall, he ranks {ordinal(rank)} out of {n_players} centre-backs by ground_duel_quality_score, which is around the {ordinal(percentile)} percentile.",
    ]
    return _build_text_from_sentences(sentences)


def _build_pair_user(row: pd.Series) -> str:
    lines = [
        f"Here is a statistical description of the centre-back pair {row['pair_name']} in the Ground Duels model.",
        f"The pair was {describe_level(row['z_duel_success_rate'])} in duel_success_rate ({row['duel_success_rate']:.3f}; z={row['z_duel_success_rate']:.2f}) and {describe_level(row['z_possession_win_rate'])} in possession_win_rate ({row['possession_win_rate']:.3f}; z={row['z_possession_win_rate']:.2f}).",
        f"They were {describe_level(row['z_discipline'])} in discipline ({row['discipline']:.3f}; z={row['z_discipline']:.2f}) and {describe_level(row['z_card_discipline'])} in card_discipline ({row['card_discipline']:.3f}; z={row['z_card_discipline']:.2f}).",
        f"They were {describe_level(row['z_duels_per90'])} in duels_per90 ({row['duels_per90']:.3f}; z={row['z_duels_per90']:.2f}) and {describe_level(row['z_interceptions_per90'])} in interceptions_per90 ({row['interceptions_per90']:.3f}; z={row['z_interceptions_per90']:.2f}).",
        f"Component z-scores: quality_z={row['quality_z']:.3f}, complement_z={row['complement_z']:.3f}, floor_z={row['floor_z']:.3f}.",
        f"Final CB_pair_fit_z_score={row['CB_pair_fit_z_score']:.3f}; rank={int(row['CB_pair_fit_rank'])}.",
        f"Metadata: same_dominant_team={bool_to_word(row['same_dominant_team'])}, opposite_side_pair={bool_to_word(row['opposite_side_pair'])}, shared_minutes_overlap={row['shared_minutes_overlap']:.1f}, evidence_band={row['evidence_band']}.",
    ]
    return "\n".join(lines)


def _build_pair_assistant(row: pd.Series, n_pairs: int) -> Tuple[str, int]:
    rank = int(row["CB_pair_fit_rank"])
    percentile = int(round((1 - (rank - 1) / n_pairs) * 100))

    strengths = _top_metric_phrases(row, n=2)
    weaknesses = _low_metric_phrases(row, n=2)

    floor_clause = ""
    if row["floor_z"] < -0.5:
        floor_clause = f"; the floor_z value ({row['floor_z']:.2f}) flags weak-link risk in difficult matchups"

    sentences = [
        f"{row['pair_name']} projects as {with_indef_article(describe_level(row['CB_pair_fit_z_score']))} ground-duel centre-back pairing in this model.",
        f"The pair's strongest shared outcomes are {join_phrases(strengths)}, supported by quality_z {row['quality_z']:.2f} and complement_z {row['complement_z']:.2f}.",
        f"The weaker areas are {join_phrases(weaknesses)} versus stronger attacks{floor_clause}.",
        f"Overall, they rank {ordinal(rank)} out of {n_pairs} pairs by CB_pair_fit_z_score, around the {ordinal(percentile)} percentile, with evidence_band {row['evidence_band']} and {row['shared_minutes_overlap']:.0f} shared minutes.",
    ]
    return _build_text_from_sentences(sentences)


def _build_comp_user(row: pd.Series, anchor_total_options: int) -> str:
    lines = [
        f"Here is a directional companion-fit description for anchor {row['anchor_player_name']} and partner {row['partner_player_name']} (A -> B).",
        f"CB_pair_fit_z_score={row['CB_pair_fit_z_score']:.3f}; coverage_gain_raw={row['coverage_gain_raw']:.3f}; coverage_gain_z_within_anchor={row['coverage_gain_z_within_anchor']:.3f}.",
        f"companion_fit_score={row['companion_fit_score']:.3f}; companion_rank_for_anchor={int(row['companion_rank_for_anchor'])} out of {anchor_total_options}.",
        "All values come from the Ground Duels companion model for this exact anchor context.",
    ]
    return "\n".join(lines)


def _build_comp_assistant(row: pd.Series, anchor_total_options: int) -> Tuple[str, int]:
    rank = int(row["companion_rank_for_anchor"])
    percentile = int(round((1 - (rank - 1) / anchor_total_options) * 100))

    pair_word = describe_level(row["CB_pair_fit_z_score"])
    coverage_word = describe_level(row["coverage_gain_z_within_anchor"])
    companion_word = describe_level(row["companion_fit_score"])

    caution = ""
    if row["coverage_gain_z_within_anchor"] < -0.5:
        caution = "; this partner does not cover many of the anchor's weaker duel dimensions"
    elif row["CB_pair_fit_z_score"] < -0.5:
        caution = "; baseline pair quality is weak before directional adjustment"

    sentences = [
        f"For anchor {row['anchor_player_name']}, {row['partner_player_name']} grades as {with_indef_article(companion_word)} directional companion in ground-duel terms.",
        f"The global pair platform is {pair_word} at CB_pair_fit_z_score {row['CB_pair_fit_z_score']:.2f}, while deficit coverage is {coverage_word} at coverage_gain_z_within_anchor {row['coverage_gain_z_within_anchor']:.2f}.",
        f"That blend yields companion_fit_score {row['companion_fit_score']:.2f}, combining overall pair quality and anchor-specific coverage{caution}.",
        f"This partner ranks {ordinal(rank)} out of {anchor_total_options} options for this anchor, around the {ordinal(percentile)} percentile within-anchor.",
    ]
    return _build_text_from_sentences(sentences)


def _select_top_mid_low_ids(sorted_df: pd.DataFrame, row_id_col: str) -> List[int]:
    top_ids = sorted_df[row_id_col].head(FEW_SHOT_BUCKET_TOP).tolist()
    low_ids = sorted_df[row_id_col].tail(FEW_SHOT_BUCKET_LOW).tolist()

    blocked = set(top_ids + low_ids)
    remaining = sorted_df[~sorted_df[row_id_col].isin(blocked)].copy()
    if remaining.empty:
        mid_ids: List[int] = []
    else:
        quarter = max(1, len(remaining) // 4)
        mid_pool = remaining.iloc[quarter : len(remaining) - quarter]
        if mid_pool.empty:
            mid_pool = remaining
        mid_ids = _evenly_spaced(mid_pool[row_id_col].tolist(), FEW_SHOT_BUCKET_MID)

    return _dedupe_preserve_order([*top_ids, *mid_ids, *low_ids])


def _pick_edge_ids(
    edge_groups: Sequence[Sequence[int]], selected_ids: Sequence[int], fallback_ids: Sequence[int]
) -> List[int]:
    selected = set(selected_ids)
    out: List[int] = []

    for group in edge_groups:
        for candidate in group:
            if candidate in selected or candidate in out:
                continue
            out.append(candidate)
            break
        if len(out) == FEW_SHOT_EDGE:
            return out

    for candidate in fallback_ids:
        if candidate in selected or candidate in out:
            continue
        out.append(candidate)
        if len(out) == FEW_SHOT_EDGE:
            break

    return out


def _selected_rows_in_order(sorted_df: pd.DataFrame, row_id_col: str, selected_ids: Sequence[int]) -> pd.DataFrame:
    if not selected_ids:
        return sorted_df.head(0).copy()
    order = pd.Series(range(len(selected_ids)), index=list(selected_ids), name="_order")
    selected = sorted_df[sorted_df[row_id_col].isin(selected_ids)].copy()
    selected = selected.join(order, on=row_id_col)
    selected = selected.sort_values("_order", kind="mergesort").drop(columns=["_order"])
    return selected.reset_index(drop=True)


def _ensure_pattern_coverage(
    sorted_df: pd.DataFrame,
    row_id_col: str,
    selected_ids: Sequence[int],
    pattern_functions: Sequence[Callable[[pd.DataFrame], pd.Series]],
    target_n: int,
) -> List[int]:
    ids = _dedupe_preserve_order(list(selected_ids))
    forced: set[int] = set()

    for pattern_fn in pattern_functions:
        selected_df = _selected_rows_in_order(sorted_df, row_id_col, ids)
        if not pattern_fn(selected_df).any():
            candidate_pool = sorted_df[pattern_fn(sorted_df) & ~sorted_df[row_id_col].isin(ids)]
            if not candidate_pool.empty:
                candidate_id = int(candidate_pool.iloc[0][row_id_col])
                ids.append(candidate_id)
                forced.add(candidate_id)

    ids = _dedupe_preserve_order(ids)

    if len(ids) < target_n:
        for candidate in sorted_df[row_id_col].tolist():
            if candidate not in ids:
                ids.append(int(candidate))
            if len(ids) == target_n:
                break

    while len(ids) > target_n:
        removed = False
        for idx in range(len(ids) - 1, -1, -1):
            candidate = ids[idx]
            if candidate in forced:
                continue
            trial_ids = ids[:idx] + ids[idx + 1 :]
            trial_df = _selected_rows_in_order(sorted_df, row_id_col, trial_ids)
            if all(pattern_fn(trial_df).any() for pattern_fn in pattern_functions):
                ids = trial_ids
                removed = True
                break
        if not removed:
            ids = ids[:-1]

    return ids[:target_n]


def _build_cb_edge_groups(sorted_df: pd.DataFrame, row_id_col: str) -> List[List[int]]:
    red_card_group = (
        sorted_df[sorted_df["red_cards"] > 0]
        .sort_values(
            ["red_cards", "yellow_cards", "ground_duel_quality_score", "player.name"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    activity_weak_mask = (
        ((sorted_df["z_duels_per90"] >= 1.0) | (sorted_df["z_interceptions_per90"] >= 1.0))
        & ((sorted_df["z_duel_success_rate"] <= -0.5) | (sorted_df["z_possession_win_rate"] <= -0.5))
    )
    activity_weak_group = (
        sorted_df[activity_weak_mask]
        .assign(
            activity_edge_score=(
                sorted_df["z_duels_per90"]
                + sorted_df["z_interceptions_per90"]
                - sorted_df["z_duel_success_rate"]
                - sorted_df["z_possession_win_rate"]
            )
        )
        .sort_values(
            ["activity_edge_score", "ground_duel_quality_score", "player.name"],
            ascending=[False, True, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    contradiction_mask = (
        ((sorted_df["z_discipline"] <= -0.5) | (sorted_df["z_card_discipline"] <= -0.5))
        & (
            (sorted_df["z_duel_success_rate"] >= 0.5)
            | (sorted_df["z_possession_win_rate"] >= 0.5)
            | (sorted_df["ground_duel_quality_score"] >= 0.5)
        )
    )
    contradiction_group = (
        sorted_df[contradiction_mask]
        .assign(
            contradiction_edge_score=(
                sorted_df["z_duel_success_rate"]
                + sorted_df["z_possession_win_rate"]
                + sorted_df["ground_duel_quality_score"]
                + np.abs(sorted_df["z_discipline"])
                + np.abs(sorted_df["z_card_discipline"])
            )
        )
        .sort_values(
            ["contradiction_edge_score", "player.name"],
            ascending=[False, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    return [red_card_group, activity_weak_group, contradiction_group]


def _build_pair_edge_groups(sorted_df: pd.DataFrame, row_id_col: str) -> List[List[int]]:
    score_q90 = float(sorted_df["CB_pair_fit_z_score"].quantile(0.90))
    score_q35 = float(sorted_df["CB_pair_fit_z_score"].quantile(0.35))
    score_q65 = float(sorted_df["CB_pair_fit_z_score"].quantile(0.65))
    overlap_q90 = float(sorted_df["shared_minutes_overlap"].quantile(0.90))

    high_fit_no_evidence_group = (
        sorted_df[
            (sorted_df["evidence_band"].astype(str) == "none")
            & (sorted_df["CB_pair_fit_z_score"] >= score_q90)
        ]
        .sort_values(
            ["CB_pair_fit_z_score", "pair_name"],
            ascending=[False, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    high_overlap_modest_group = (
        sorted_df[
            (sorted_df["shared_minutes_overlap"] >= overlap_q90)
            & (sorted_df["CB_pair_fit_z_score"] >= score_q35)
            & (sorted_df["CB_pair_fit_z_score"] <= score_q65)
        ]
        .sort_values(
            ["shared_minutes_overlap", "pair_name"],
            ascending=[False, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    weak_floor_group = (
        sorted_df[sorted_df["floor_z"] <= -0.5]
        .sort_values(["floor_z", "pair_name"], ascending=[True, True], kind="mergesort")[row_id_col]
        .astype(int)
        .tolist()
    )

    return [high_fit_no_evidence_group, high_overlap_modest_group, weak_floor_group]


def _build_comp_edge_groups(sorted_df: pd.DataFrame, row_id_col: str) -> List[List[int]]:
    score_q25 = float(sorted_df["companion_fit_score"].quantile(0.25))

    high_coverage_modest_pair_group = (
        sorted_df[
            (sorted_df["coverage_gain_z_within_anchor"] >= 1.0)
            & (sorted_df["CB_pair_fit_z_score"].abs() <= 0.5)
        ]
        .sort_values(
            ["coverage_gain_z_within_anchor", "anchor_player_name", "partner_player_name"],
            ascending=[False, True, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    high_pair_low_coverage_group = (
        sorted_df[
            (sorted_df["CB_pair_fit_z_score"] >= 1.0)
            & (sorted_df["coverage_gain_z_within_anchor"] <= -0.5)
        ]
        .sort_values(
            ["CB_pair_fit_z_score", "coverage_gain_z_within_anchor", "anchor_player_name", "partner_player_name"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    low_coverage_negative_group = (
        sorted_df[
            (sorted_df["coverage_gain_z_within_anchor"] <= -1.0)
            & (sorted_df["companion_fit_score"] <= score_q25)
        ]
        .sort_values(
            ["coverage_gain_z_within_anchor", "companion_fit_score", "anchor_player_name", "partner_player_name"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )[row_id_col]
        .astype(int)
        .tolist()
    )

    return [high_coverage_modest_pair_group, high_pair_low_coverage_group, low_coverage_negative_group]


def _cb_pattern_functions() -> List[Callable[[pd.DataFrame], pd.Series]]:
    return [
        lambda d: d["red_cards"] > 0,
        lambda d: (
            ((d["z_duels_per90"] >= 1.0) | (d["z_interceptions_per90"] >= 1.0))
            & ((d["z_duel_success_rate"] <= -0.5) | (d["z_possession_win_rate"] <= -0.5))
        ),
        lambda d: (
            ((d["z_discipline"] <= -0.5) | (d["z_card_discipline"] <= -0.5))
            & (
                (d["z_duel_success_rate"] >= 0.5)
                | (d["z_possession_win_rate"] >= 0.5)
                | (d["ground_duel_quality_score"] >= 0.5)
            )
        ),
    ]


def _pair_pattern_functions(sorted_df: pd.DataFrame) -> List[Callable[[pd.DataFrame], pd.Series]]:
    score_q90 = float(sorted_df["CB_pair_fit_z_score"].quantile(0.90))
    score_q35 = float(sorted_df["CB_pair_fit_z_score"].quantile(0.35))
    score_q65 = float(sorted_df["CB_pair_fit_z_score"].quantile(0.65))
    overlap_q90 = float(sorted_df["shared_minutes_overlap"].quantile(0.90))

    return [
        lambda d: (d["evidence_band"].astype(str) == "none") & (d["CB_pair_fit_z_score"] >= score_q90),
        lambda d: (
            (d["shared_minutes_overlap"] >= overlap_q90)
            & (d["CB_pair_fit_z_score"] >= score_q35)
            & (d["CB_pair_fit_z_score"] <= score_q65)
        ),
        lambda d: d["floor_z"] <= -0.5,
    ]


def _comp_pattern_functions(sorted_df: pd.DataFrame) -> List[Callable[[pd.DataFrame], pd.Series]]:
    score_q25 = float(sorted_df["companion_fit_score"].quantile(0.25))
    return [
        lambda d: (d["coverage_gain_z_within_anchor"] >= 1.0) & (d["CB_pair_fit_z_score"].abs() <= 0.5),
        lambda d: (d["CB_pair_fit_z_score"] >= 1.0) & (d["coverage_gain_z_within_anchor"] <= -0.5),
        lambda d: (d["coverage_gain_z_within_anchor"] <= -1.0) & (d["companion_fit_score"] <= score_q25),
    ]


def _assert_pattern_coverage(
    selected_df: pd.DataFrame,
    pattern_functions: Sequence[Callable[[pd.DataFrame], pd.Series]],
    context_name: str,
) -> None:
    for i, pattern_fn in enumerate(pattern_functions, start=1):
        if not pattern_fn(selected_df).any():
            raise AssertionError(f"{context_name}: missing required edge pattern #{i}.")


def _select_context_rows(
    df: pd.DataFrame,
    score_col: str,
    key_cols: Sequence[str],
    row_id_col: str,
    edge_group_builder: Callable[[pd.DataFrame, str], List[List[int]]],
    pattern_builder: Callable[[pd.DataFrame], List[Callable[[pd.DataFrame], pd.Series]]],
    context_name: str,
) -> pd.DataFrame:
    sorted_df = _stable_sort(df, score_col=score_col, key_cols=key_cols)
    top_mid_low_ids = _select_top_mid_low_ids(sorted_df, row_id_col=row_id_col)
    edge_groups = edge_group_builder(sorted_df, row_id_col=row_id_col)
    edge_ids = _pick_edge_ids(
        edge_groups=edge_groups,
        selected_ids=top_mid_low_ids,
        fallback_ids=sorted_df[row_id_col].astype(int).tolist(),
    )

    selected_ids = _dedupe_preserve_order([*top_mid_low_ids, *edge_ids])
    pattern_functions = pattern_builder(sorted_df)
    selected_ids = _ensure_pattern_coverage(
        sorted_df=sorted_df,
        row_id_col=row_id_col,
        selected_ids=selected_ids,
        pattern_functions=pattern_functions,
        target_n=FEW_SHOT_PER_CONTEXT,
    )
    selected_df = _selected_rows_in_order(sorted_df, row_id_col, selected_ids)

    if len(selected_df) != FEW_SHOT_PER_CONTEXT:
        raise AssertionError(f"{context_name}: expected {FEW_SHOT_PER_CONTEXT} rows, got {len(selected_df)}.")

    if selected_df[list(key_cols)].duplicated().any():
        raise AssertionError(f"{context_name}: selected rows contain duplicate keys.")

    _assert_pattern_coverage(selected_df, pattern_functions=pattern_functions, context_name=context_name)
    return selected_df


def _prepare_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    cb = pd.read_csv(CB_CSV).copy()
    pairs = pd.read_csv(PAIR_CSV).copy()
    comp = pd.read_csv(COMP_CSV).copy()

    if "Unnamed: 0" in cb.columns:
        cb = cb.drop(columns=["Unnamed: 0"])

    cb = cb.reset_index(drop=True)
    pairs = pairs.reset_index(drop=True)
    comp = comp.reset_index(drop=True)

    cb["_row_id"] = cb.index.astype(int)
    pairs["_row_id"] = pairs.index.astype(int)
    comp["_row_id"] = comp.index.astype(int)

    cb["ground_duel_quality_score_rank_desc"] = (
        cb["ground_duel_quality_score"].rank(ascending=False, method="min").astype(int)
    )

    if "CB_pair_fit_rank" not in pairs.columns:
        pairs["CB_pair_fit_rank"] = (
            pairs["CB_pair_fit_z_score"].rank(ascending=False, method="min").astype(int)
        )
    else:
        pairs["CB_pair_fit_rank"] = pairs["CB_pair_fit_rank"].astype(int)

    if "companion_rank_for_anchor" not in comp.columns:
        comp = comp.sort_values(
            ["anchor_player_id", "companion_fit_score", "partner_player_name"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        comp["companion_rank_for_anchor"] = comp.groupby("anchor_player_id").cumcount() + 1
    comp["companion_rank_for_anchor"] = comp["companion_rank_for_anchor"].astype(int)

    anchor_counts = (
        comp.groupby("anchor_player_id")["partner_player_id"].nunique().astype(int).to_dict()
    )
    return cb, pairs, comp, anchor_counts


def _build_few_shot_examples(
    cb: pd.DataFrame,
    pairs: pd.DataFrame,
    comp: pd.DataFrame,
    anchor_counts: Dict[int, int],
) -> List[FewShotExample]:
    cb_selected = _select_context_rows(
        df=cb,
        score_col="ground_duel_quality_score",
        key_cols=["player.name"],
        row_id_col="_row_id",
        edge_group_builder=_build_cb_edge_groups,
        pattern_builder=lambda sorted_df: _cb_pattern_functions(),
        context_name="cb",
    )
    pair_selected = _select_context_rows(
        df=pairs,
        score_col="CB_pair_fit_z_score",
        key_cols=["pair_name"],
        row_id_col="_row_id",
        edge_group_builder=_build_pair_edge_groups,
        pattern_builder=_pair_pattern_functions,
        context_name="pair",
    )
    comp_selected = _select_context_rows(
        df=comp,
        score_col="companion_fit_score",
        key_cols=["anchor_player_name", "partner_player_name"],
        row_id_col="_row_id",
        edge_group_builder=_build_comp_edge_groups,
        pattern_builder=_comp_pattern_functions,
        context_name="companion",
    )

    few_shots: List[FewShotExample] = []

    for _, row in cb_selected.iterrows():
        assistant_text, sentence_count = _build_cb_assistant(row, n_players=len(cb))
        few_shots.append(
            FewShotExample(
                context="cb",
                key=str(row["player.name"]),
                user=_build_cb_user(row),
                assistant=assistant_text,
                assistant_sentence_count=sentence_count,
            )
        )

    for _, row in pair_selected.iterrows():
        assistant_text, sentence_count = _build_pair_assistant(row, n_pairs=len(pairs))
        few_shots.append(
            FewShotExample(
                context="pair",
                key=str(row["pair_name"]),
                user=_build_pair_user(row),
                assistant=assistant_text,
                assistant_sentence_count=sentence_count,
            )
        )

    for _, row in comp_selected.iterrows():
        anchor_total_options = int(anchor_counts.get(int(row["anchor_player_id"]), len(cb) - 1))
        assistant_text, sentence_count = _build_comp_assistant(row, anchor_total_options=anchor_total_options)
        few_shots.append(
            FewShotExample(
                context="companion",
                key=f"{row['anchor_player_name']}->{row['partner_player_name']}",
                user=_build_comp_user(row, anchor_total_options=anchor_total_options),
                assistant=assistant_text,
                assistant_sentence_count=sentence_count,
            )
        )

    if len(few_shots) != FEW_SHOT_PER_CONTEXT * 3:
        raise AssertionError(
            f"Expected {FEW_SHOT_PER_CONTEXT * 3} few-shot examples, got {len(few_shots)}."
        )

    context_counts = pd.Series([example.context for example in few_shots]).value_counts().to_dict()
    for context in ["cb", "pair", "companion"]:
        if int(context_counts.get(context, 0)) != FEW_SHOT_PER_CONTEXT:
            raise AssertionError(
                f"Expected {FEW_SHOT_PER_CONTEXT} few-shots for {context}, got {context_counts.get(context, 0)}."
            )

    if any(example.assistant_sentence_count != 4 for example in few_shots):
        raise AssertionError("Every few-shot assistant example must contain exactly 4 sentences.")

    return few_shots


def _build_knowledge_pairs() -> List[Dict[str, str]]:
    pairs: List[Tuple[str, str]] = [
        ("What is a centre-back in this project?", "A centre-back is a defensive player assessed on one-v-one defensive ground duels and related duel-impact metrics."),
        ("What are the three main assessment contexts?", "The three contexts are individual CB assessment, CB-pair assessment, and directional anchor-to-partner companion-fit assessment."),
        ("Why is Ground Duels isolated as a dedicated quality block?", "Ground duels are a core defensive phase for centre-backs and this block isolates duel impact, discipline, and activity before mixing in other qualities."),
        ("Which data source files must be used for this Ground Duels prompt asset?", "Use only CB_ground_duels.csv, CB_pairs_ground_duels.csv, and CB_companion_fit_ground_duels.csv from the Ground Duels folder."),
        ("Which file is used for individual CB analysis?", "Individual CB analysis uses CB_ground_duels.csv."),
        ("Which file is used for CB-pair analysis?", "CB-pair analysis uses CB_pairs_ground_duels.csv."),
        ("Which file is used for anchor-to-partner companion-fit analysis?", "Directional companion-fit analysis uses CB_companion_fit_ground_duels.csv."),
        ("What competition context is represented in these Ground Duels tables?", "These exports are from the Premier League 2024 workflow built in the Ground Duels notebook."),
        ("Which six metrics define the individual Ground Duels model?", "The individual model uses duel_success_rate, possession_win_rate, discipline, card_discipline, duels_per90, and interceptions_per90."),
        ("What does duel_success_rate measure?", "duel_success_rate is the share of defensive ground duels where the CB wins possession or stops progression."),
        ("What does possession_win_rate measure?", "possession_win_rate is the share of defensive ground duels where the CB directly recovers the ball."),
        ("What is foul_rate in this pipeline?", "foul_rate is the share of defensive ground duels that end with a foul committed by the player."),
        ("How is discipline defined?", "discipline is 1 minus foul_rate, so higher values mean cleaner defensive duel behavior."),
        ("How is card_discipline defined?", "card_discipline is 1 minus ((yellow_cards + 3*red_cards) / total_duels), so higher values mean fewer and less severe cards."),
        ("Why are red cards weighted by 3 in card_discipline?", "The model applies a higher red-card penalty because a dismissal has much larger match impact than a yellow card."),
        ("What are the volume metrics in this Ground Duels model?", "duels_per90 and interceptions_per90 are the activity metrics used to complement duel outcome rates."),
        ("How should duels_per90 be interpreted for a CB?", "duels_per90 reflects how frequently the CB engages in defensive ground duel actions per 90 minutes."),
        ("How should interceptions_per90 be interpreted for a CB?", "interceptions_per90 reflects how often the CB steps in to break up attacks and regain control per 90 minutes."),
        ("Why combine rate metrics with volume metrics?", "Rate metrics capture duel efficiency while volume metrics capture involvement, so both are needed for a balanced profile."),
        ("How are z-scores interpreted in this prompt stack?", "Use z-score buckets to describe relative standing against the same cohort, not absolute universal quality."),
        ("What exact z-score thresholds must be used?", "Use thresholds [1.5, 1, 0.5, -0.5, -1] mapped to words [outstanding, excellent, good, average, below average, poor]."),
        ("What does outstanding mean in this mapping?", "Outstanding means the metric is in the highest band and materially above cohort average."),
        ("What does poor mean in this mapping?", "Poor means the metric is in the lowest band and materially below cohort average."),
        ("How should average be interpreted?", "Average means performance is around cohort baseline and not a clear differentiator in that metric."),
        ("How should yellow_cards and red_cards be treated semantically?", "Higher raw card counts are negative defensive control signals and should never be framed as a strength."),
        ("How should high card_discipline with low raw cards be described?", "High card_discipline should be described as clean duel management with low booking and dismissal exposure."),
        ("What is the weighted formula for Zq_raw?", "Zq_raw is 0.10*z_duels_per90 + 0.15*z_interceptions_per90 + 0.20*z_duel_success_rate + 0.35*z_possession_win_rate + 0.10*z_discipline + 0.10*z_card_discipline."),
        ("What is required about Ground Duels metric weights?", "All metric importance weights must sum to 1.0."),
        ("Why does possession_win_rate have the largest weight?", "The notebook assigns the largest weight to possession_win_rate because directly regaining the ball is the most valuable duel outcome."),
        ("Why are duels_per90 and interceptions_per90 lower-weight contributors?", "They are supporting activity dimensions and should not dominate quality and recovery effectiveness."),
        ("How is ground_duel_quality_score produced from Zq_raw?", "ground_duel_quality_score is the standardized z-score version of Zq_raw so results are centered and comparable."),
        ("How should individual rank by ground_duel_quality_score be used?", "Use rank and percentile as explicit cohort-relative summary statements in the final sentence."),
        ("Why include minutes_played in the statistical description text?", "minutes_played gives context for sample exposure and avoids over-framing tiny samples as stable outcomes."),
        ("How are candidate CB pairs formed for the pair model?", "Pairs are built as unordered CB combinations, then evaluated with pair-level means plus complement and floor components."),
        ("How are pair-level raw metrics computed?", "Each pair raw metric is the equal-weight average of the two players' raw values."),
        ("How are pair z-metrics represented?", "For each z metric, the pair uses pair_mean, pair_best, and pair_floor views derived from the two player values."),
        ("What is quality_raw in the pair model?", "quality_raw is the weighted sum of pair_mean z metrics and captures average overall pair strength."),
        ("What is complement_raw in the pair model?", "complement_raw is the weighted sum of pair_best minus pair_mean and captures complementarity upside."),
        ("What is floor_raw in the pair model?", "floor_raw is the weighted sum of pair_floor z metrics and captures weak-link protection."),
        ("How are quality_raw, complement_raw, and floor_raw put on a common scale?", "Each component is standardized to quality_z, complement_z, and floor_z before final blending."),
        ("What is the final CB_pair_fit_z_score formula?", "CB_pair_fit_z_score = 0.50*quality_z + 0.30*complement_z + 0.20*floor_z."),
        ("How should a pair with high quality_z but low complement_z be interpreted?", "It indicates strong average level but weaker mutual coverage across dimensions."),
        ("How should low floor_z be interpreted?", "Low floor_z indicates weak-link risk where one player can pull down the pair in difficult duel contexts."),
        ("Which pair metadata is descriptive but not part of the fit score?", "same_dominant_team, opposite_side_pair, shared_matches, shared_minutes_overlap, and evidence_band are contextual metadata fields."),
        ("What does opposite_side_pair indicate?", "opposite_side_pair is yes when one CB is dominant on the left and the other is dominant on the right."),
        ("What does shared_minutes_overlap represent?", "shared_minutes_overlap estimates real on-pitch overlap minutes for the two CBs in the same team and match contexts."),
        ("How is evidence_band defined from shared_minutes_overlap?", "evidence_band uses bins [-0.1, 0, 180, 450, 900, inf] mapped to [none, low, medium, high, very_high]."),
        ("What is directional companion fit in this project?", "Directional companion fit evaluates partner B specifically for anchor A, so A->B and B->A are separate contexts."),
        ("Why is companion fit directional while pair fit is symmetric?", "Pair fit is one shared global profile, but deficit coverage depends on which player is treated as the anchor."),
        ("How are anchor deficits computed in the companion model?", "For each weighted z metric, anchor deficit is max(0, -anchor_z_metric)."),
        ("How is partner coverage computed for each anchor deficit?", "Coverage per metric is min(anchor_deficit_metric, max(0, partner_z_metric))."),
        ("How is coverage_gain_raw computed?", "coverage_gain_raw is the weighted sum of metric-level coverage values across all weighted z metrics."),
        ("What is coverage_gain_z_within_anchor?", "It is the within-anchor standardization of coverage_gain_raw so partner options are compared on the same anchor baseline."),
        ("What is the final companion_fit_score formula?", "companion_fit_score = 0.65*CB_pair_fit_z_score + 0.35*coverage_gain_z_within_anchor."),
        ("How is companion_rank_for_anchor interpreted?", "It is the rank of each partner inside one anchor's candidate partner set only."),
        ("How should high coverage_gain with modest pair score be interpreted?", "It signals strong deficit coverage for that anchor even if the global pair platform is only moderate."),
        ("How should high pair score with weak coverage_gain be interpreted?", "It means strong global pair quality but limited anchor-specific deficit solving."),
        ("Which utility mappings should be respected for metric wording and z-label buckets?", "Respect the utils.sentences mappings and the describe_level bucket thresholds so metric wording and qualitative labels stay consistent."),
        ("What should the model do if no data row is provided by the user?", "State that no data was provided and request a valid statistical description row, without inventing numbers."),
        ("What is the final non-negotiable rule on evidence use?", "Use only the data explicitly provided in the prompt context and do not rely on priors or outside assumptions."),
    ]

    if len(pairs) != KNOWLEDGE_PAIR_COUNT:
        raise AssertionError(f"Expected {KNOWLEDGE_PAIR_COUNT} knowledge Q&A pairs, got {len(pairs)}.")

    return [{"user": user, "assistant": assistant} for user, assistant in pairs]


def _build_style_pairs() -> List[Dict[str, str]]:
    return [
        {
            "user": "How should you answer user requests about Ground Duels outputs?",
            "assistant": "I must answer in the same language as the user, keep UK football vocabulary, and stay warm, clear, concise, and professional.",
        },
        {
            "user": "What response format must you follow for Ground Duels scouting summaries?",
            "assistant": "I must return exactly four sentences: overall profile, key strengths, weaker or average areas, and exact cohort comparison.",
        },
        {
            "user": "What is the rule on data usage when generating the final text?",
            "assistant": "I must only use the supplied statistical description and relevant prior user-assistant Ground Duels instructions in context.",
        },
        {
            "user": "How should rankings be reported in the final sentence?",
            "assistant": "The last sentence should state exact rank and percentile for the relevant cohort in the active context.",
        },
        {
            "user": "Please use the statistical description enclosed with ``` to produce a concise 4 sentence summary.",
            "assistant": "Understood. I will use only the provided data, keep exactly four sentences, and follow the required sentence-by-sentence structure.",
        },
        {
            "user": "If no data is provided for a CB, pair, or companion-fit request, how should you respond?",
            "assistant": "I must not speculate. I should state that no data row was provided and ask the user to provide the relevant statistical description.",
        },
    ]


def _build_system_instruction() -> str:
    return (
        "You are a UK-based CB and CB-Pairing AI Analyst focused on centre-back Ground Duels in elite professional football. "
        "You provide concise, data-grounded scouting explanations using only the statistics supplied in context. "
        "You must handle three contexts: individual CB assessment, CB-pair assessment, and directional anchor-to-partner companion-fit assessment. "
        "You must not hallucinate metrics, assumptions, or background facts when data is missing."
    )


def _build_history(
    knowledge_pairs: List[Dict[str, str]],
    style_pairs: List[Dict[str, str]],
    few_shot_examples: List[FewShotExample],
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = [
        {"role": "user", "parts": "Do you refer to the game as soccer or football?"},
        {"role": "model", "parts": "I refer to it as football, in the UK sense of the term."},
        {"role": "user", "parts": "First, could you answer some questions about Ground Duels for me?"},
        {"role": "model", "parts": "Sure."},
    ]

    for pair in knowledge_pairs:
        history.append({"role": "user", "parts": pair["user"]})
        history.append({"role": "model", "parts": pair["assistant"]})

    for pair in style_pairs:
        history.append({"role": "user", "parts": pair["user"]})
        history.append({"role": "model", "parts": pair["assistant"]})

    for example in few_shot_examples:
        history.append({"role": "user", "parts": example.user})
        history.append({"role": "model", "parts": example.assistant})

    return history


def _write_outputs(
    payload: Dict[str, object],
    knowledge_pairs: List[Dict[str, str]],
    few_shot_examples: List[FewShotExample],
) -> None:
    DESCRIBE_DIR.mkdir(parents=True, exist_ok=True)
    GPT_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")

    describe_df = pd.DataFrame(knowledge_pairs)[["user", "assistant"]]
    gpt_examples_df = pd.DataFrame(
        [{"user": example.user, "assistant": example.assistant} for example in few_shot_examples]
    )[["user", "assistant"]]

    with pd.ExcelWriter(DESCRIBE_XLSX_OUT, engine="openpyxl") as writer:
        describe_df.to_excel(writer, index=False, sheet_name="Sheet1")

    with pd.ExcelWriter(GPT_EXAMPLES_XLSX_OUT, engine="openpyxl") as writer:
        gpt_examples_df.to_excel(writer, index=False, sheet_name="Sheet1")


def _validate_outputs(
    payload: Dict[str, object],
    knowledge_pairs: List[Dict[str, str]],
    few_shot_examples: List[FewShotExample],
    cb: pd.DataFrame,
    pairs: pd.DataFrame,
    comp: pd.DataFrame,
    anchor_counts: Dict[int, int],
) -> None:
    if not JSON_OUT.exists():
        raise AssertionError(f"JSON output file was not created: {JSON_OUT}")
    if not DESCRIBE_XLSX_OUT.exists():
        raise AssertionError(f"Describe XLSX output file was not created: {DESCRIBE_XLSX_OUT}")
    if not GPT_EXAMPLES_XLSX_OUT.exists():
        raise AssertionError(f"GPT examples XLSX output file was not created: {GPT_EXAMPLES_XLSX_OUT}")

    loaded_json = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    if set(loaded_json.keys()) != {"system_instruction", "history", "content"}:
        raise AssertionError("JSON schema mismatch.")
    if loaded_json["content"] != {"role": "user", "parts": ""}:
        raise AssertionError("JSON content payload mismatch.")

    loaded_describe = pd.read_excel(DESCRIBE_XLSX_OUT, sheet_name="Sheet1")
    if list(loaded_describe.columns) != ["user", "assistant"]:
        raise AssertionError("Describe workbook schema mismatch.")
    if len(loaded_describe) != KNOWLEDGE_PAIR_COUNT:
        raise AssertionError(f"Describe workbook row count mismatch: {len(loaded_describe)}")
    if loaded_describe[["user", "assistant"]].isna().any().any():
        raise AssertionError("Describe workbook contains empty cells.")
    if (loaded_describe["user"].astype(str).str.strip() == "").any():
        raise AssertionError("Describe workbook contains blank user prompts.")
    if (loaded_describe["assistant"].astype(str).str.strip() == "").any():
        raise AssertionError("Describe workbook contains blank assistant answers.")

    loaded_gpt_examples = pd.read_excel(GPT_EXAMPLES_XLSX_OUT, sheet_name="Sheet1")
    if list(loaded_gpt_examples.columns) != ["user", "assistant"]:
        raise AssertionError("GPT examples workbook schema mismatch.")
    if len(loaded_gpt_examples) != FEW_SHOT_PER_CONTEXT * 3:
        raise AssertionError(f"GPT examples workbook row count mismatch: {len(loaded_gpt_examples)}")

    context_counts = pd.Series([example.context for example in few_shot_examples]).value_counts().to_dict()
    for context in ["cb", "pair", "companion"]:
        if int(context_counts.get(context, 0)) != FEW_SHOT_PER_CONTEXT:
            raise AssertionError(f"Context balance mismatch for {context}: {context_counts.get(context, 0)}")

    if any(example.assistant_sentence_count != 4 for example in few_shot_examples):
        raise AssertionError("Few-shot assistant sentence-count validation failed.")

    expected_tail_pairs = [(example.user, example.assistant) for example in few_shot_examples]
    history = loaded_json["history"]
    if len(history) < FEW_SHOT_PER_CONTEXT * 3 * 2:
        raise AssertionError("JSON history too short for few-shot tail alignment check.")
    json_tail_pairs: List[Tuple[str, str]] = []
    tail_start = len(history) - (FEW_SHOT_PER_CONTEXT * 3 * 2)
    for i in range(tail_start, len(history), 2):
        user_row = history[i]
        model_row = history[i + 1]
        json_tail_pairs.append((user_row["parts"], model_row["parts"]))
    if json_tail_pairs != expected_tail_pairs:
        raise AssertionError("JSON tail does not match gpt_examples workbook ordering/content.")

    second_run_examples = _build_few_shot_examples(cb=cb, pairs=pairs, comp=comp, anchor_counts=anchor_counts)
    sig1 = [(e.context, e.key, e.user, e.assistant) for e in few_shot_examples]
    sig2 = [(e.context, e.key, e.user, e.assistant) for e in second_run_examples]
    h1 = hashlib.sha256(json.dumps(sig1, ensure_ascii=False).encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(json.dumps(sig2, ensure_ascii=False).encode("utf-8")).hexdigest()
    if h1 != h2:
        raise AssertionError("Determinism check failed: repeated selection/build produced different output.")

    if payload != loaded_json:
        raise AssertionError("Serialized JSON does not match in-memory payload.")
    if len(knowledge_pairs) != KNOWLEDGE_PAIR_COUNT:
        raise AssertionError("Knowledge pair count mismatch.")


def main() -> None:
    cb, pairs, comp, anchor_counts = _prepare_frames()
    knowledge_pairs = _build_knowledge_pairs()
    style_pairs = _build_style_pairs()
    few_shot_examples = _build_few_shot_examples(cb=cb, pairs=pairs, comp=comp, anchor_counts=anchor_counts)

    history = _build_history(
        knowledge_pairs=knowledge_pairs,
        style_pairs=style_pairs,
        few_shot_examples=few_shot_examples,
    )

    payload = {
        "system_instruction": _build_system_instruction(),
        "history": history,
        "content": {"role": "user", "parts": ""},
    }

    _write_outputs(
        payload=payload,
        knowledge_pairs=knowledge_pairs,
        few_shot_examples=few_shot_examples,
    )

    _validate_outputs(
        payload=payload,
        knowledge_pairs=knowledge_pairs,
        few_shot_examples=few_shot_examples,
        cb=cb,
        pairs=pairs,
        comp=comp,
        anchor_counts=anchor_counts,
    )

    print(f"Created JSON: {JSON_OUT}")
    print(f"Created describe workbook: {DESCRIBE_XLSX_OUT}")
    print(f"Created gpt_examples workbook: {GPT_EXAMPLES_XLSX_OUT}")
    print("Validated: schema, counts, 20/20/20 context balance, JSON tail alignment, and deterministic selection.")


if __name__ == "__main__":
    main()
