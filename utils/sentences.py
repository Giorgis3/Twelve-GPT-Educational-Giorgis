"""
Text helpers for converting raw metric values into readable sentences.

This module is intentionally lightweight and is used by description builders to:
1) map numeric z-scores to qualitative labels (for example, "excellent"), and
2) normalize metric names into human-readable phrases.
"""

# Explicit labels for all columns currently used in Ground Duels CSV exports.
# These overrides keep display text clear and domain-specific, while fallback
# normalization still supports all other datasets.
GROUND_DUELS_FORMAT_METRIC_MAP = {
    "": "Row index",
    "anchor_player_id": "Anchor player ID",
    "anchor_player_name": "Anchor player name",
    "card_discipline": "Card discipline",
    "CB_dominant_playing_side_CB1": "CB1 dominant playing side",
    "CB_dominant_playing_side_CB2": "CB2 dominant playing side",
    "CB_pair_fit_rank": "CB pair fit rank",
    "CB_pair_fit_z_score": "CB pair fit z-score",
    "companion_fit_score": "Companion fit score",
    "companion_rank_for_anchor": "Companion rank for anchor",
    "complement_raw": "Complementarity (raw)",
    "complement_z": "Complementarity z-score",
    "coverage_gain_raw": "Deficit-coverage gain (raw)",
    "coverage_gain_z_within_anchor": "Deficit-coverage gain z-score within anchor",
    "discipline": "Discipline",
    "duel_success_rate": "Duel success rate",
    "duels_per90": "Duels per 90",
    "evidence_band": "Evidence band",
    "floor_raw": "Floor score (raw)",
    "floor_z": "Floor score z-score",
    "ground_duel_quality_score": "Ground duel quality score",
    "interceptions_per90": "Interceptions per 90",
    "minutes_played": "Minutes played",
    "opposite_side_pair": "Opposite-side pair",
    "pair_key": "Pair key",
    "pair_name": "Pair name",
    "partner_player_id": "Partner player ID",
    "partner_player_name": "Partner player name",
    "player.id": "Player ID",
    "player.id_CB1": "CB1 player ID",
    "player.id_CB2": "CB2 player ID",
    "player.name": "Player name",
    "player.name_CB1": "CB1 player name",
    "player.name_CB2": "CB2 player name",
    "possession_win_rate": "Possession win rate",
    "quality_raw": "Quality score (raw)",
    "quality_z": "Quality score z-score",
    "red_cards": "Red cards",
    "same_dominant_team": "Same dominant team",
    "shared_matches": "Shared matches",
    "shared_minutes_overlap": "Shared minutes overlap",
    "team_CB1_id": "CB1 team ID",
    "team_CB2_id": "CB2 team ID",
    "total_duels": "Total duels",
    "yellow_cards": "Yellow cards",
    "z_card_discipline": "Card discipline z-score",
    "z_discipline": "Discipline z-score",
    "z_duel_success_rate": "Duel success rate z-score",
    "z_duels_per90": "Duels per 90 z-score",
    "z_interceptions_per90": "Interceptions per 90 z-score",
    "z_possession_win_rate": "Possession win rate z-score",
}

GROUND_DUELS_WRITE_OUT_METRIC_MAP = {
    "": "row index",
    "anchor_player_id": "anchor player identifier",
    "anchor_player_name": "anchor player name",
    "card_discipline": "card discipline",
    "CB_dominant_playing_side_CB1": "dominant playing side for CB1",
    "CB_dominant_playing_side_CB2": "dominant playing side for CB2",
    "CB_pair_fit_rank": "CB pair fit rank",
    "CB_pair_fit_z_score": "CB pair fit z-score",
    "companion_fit_score": "companion fit score",
    "companion_rank_for_anchor": "companion rank for anchor",
    "complement_raw": "raw complementarity score",
    "complement_z": "complementarity z-score",
    "coverage_gain_raw": "raw deficit-coverage gain",
    "coverage_gain_z_within_anchor": "deficit-coverage gain z-score within anchor",
    "discipline": "discipline score",
    "duel_success_rate": "duel success rate",
    "duels_per90": "duels per 90 minutes",
    "evidence_band": "evidence band",
    "floor_raw": "raw floor score",
    "floor_z": "floor score z-score",
    "ground_duel_quality_score": "ground duel quality score",
    "interceptions_per90": "interceptions per 90 minutes",
    "minutes_played": "minutes played",
    "opposite_side_pair": "opposite-side pair indicator",
    "pair_key": "pair key",
    "pair_name": "pair name",
    "partner_player_id": "partner player identifier",
    "partner_player_name": "partner player name",
    "player.id": "player identifier",
    "player.id_CB1": "CB1 player identifier",
    "player.id_CB2": "CB2 player identifier",
    "player.name": "player name",
    "player.name_CB1": "CB1 player name",
    "player.name_CB2": "CB2 player name",
    "possession_win_rate": "possession win rate",
    "quality_raw": "raw quality score",
    "quality_z": "quality z-score",
    "red_cards": "red cards",
    "same_dominant_team": "same dominant team indicator",
    "shared_matches": "shared matches",
    "shared_minutes_overlap": "shared minutes overlap",
    "team_CB1_id": "CB1 team identifier",
    "team_CB2_id": "CB2 team identifier",
    "total_duels": "total duels",
    "yellow_cards": "yellow cards",
    "z_card_discipline": "card discipline z-score",
    "z_discipline": "discipline z-score",
    "z_duel_success_rate": "duel success rate z-score",
    "z_duels_per90": "duels per 90 z-score",
    "z_interceptions_per90": "interceptions per 90 z-score",
    "z_possession_win_rate": "possession win rate z-score",
}


def pronouns(gender):
    """
    Return subject, object, and possessive pronouns from a gender label.

    Parameters:
        gender: Any value that can be lowercased (typically "male" or "female").

    Returns:
        tuple[str, str, str]: Pronouns in the order
            (subject, object, possessive), e.g. ("he", "him", "his").

    Notes:
        Historical behavior is preserved: any value other than "male" maps to
        ("she", "her", "her").
    """
    if gender.lower() == "male":
        subject_p, object_p, possessive_p = "he", "him", "his"
    else:
        subject_p, object_p, possessive_p = "she", "her", "her"

    return subject_p, object_p, possessive_p


def describe_level(value, thresholds=[1.5, 1, 0.5, -0.5, -1], words=["outstanding", "excellent", "good", "average", "below average", "poor"]):
    """
    Describe a z-score-like value using default football-scouting buckets.

    Parameters:
        value: Numeric score to classify (typically a z-score).
        thresholds: Descending lower bounds for each bucket transition.
        words: Ordered labels with exactly one more item than thresholds.

    Returns:
        str: Qualitative label corresponding to ``value``.

    Example:
        With defaults, 1.2 -> "excellent", 0.2 -> "average", -1.3 -> "poor".
    """
    return describe(thresholds, words, value)


def describe(thresholds, words, value):
    """
    Map a numeric value to a label using descending thresholds.

    Parameters:
        thresholds: Sequence of lower bounds in descending order.
            Example: [1.5, 1.0, 0.5, -0.5, -1.0]
        words: Labels for each interval. Must satisfy:
            ``len(words) == len(thresholds) + 1``.
        value: Numeric value to map.

    Returns:
        str: Label selected from ``words``.

    How the intervals are interpreted:
        For thresholds ``[t0, t1, t2]`` and words ``[w0, w1, w2, w3]``:
        - value >= t0 -> w0
        - t1 <= value < t0 -> w1
        - t2 <= value < t1 -> w2
        - value < t2 -> w3
    """
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    i = 0

    # Move down buckets until we find the first threshold the value meets.
    while i < len(thresholds) and value < thresholds[i]:
        i += 1

    return words[i]


def format_metric(metric):
    """
    Format a metric key into a short display label.

    Intended for compact UI or sentence fragments where we want readable text
    without extra qualifiers such as "adjusted per90".
    Ground Duels CSV columns are handled with explicit label overrides.

    Parameters:
        metric: Raw metric key (for example, ``"npxG_adjusted_per90"``).

    Returns:
        str: Cleaned, capitalized metric label.
    """
    if metric in GROUND_DUELS_FORMAT_METRIC_MAP:
        return GROUND_DUELS_FORMAT_METRIC_MAP[metric]

    return (
        metric.replace("_", " ")
        .replace(" adjusted per90", "")
        .replace(".id", " ID")
        .replace(".name", " name")
        .replace("npxG", "non-penalty expected goals")
        .capitalize()
    )


def write_out_metric(metric):
    """
    Expand a metric key into a more explicit narrative phrase.

    This version is designed for full sentences, so it keeps contextual wording
    (for example "adjusted for possession" and "per 90"). Ground Duels CSV
    columns are handled with explicit phrase overrides.

    Parameters:
        metric: Raw metric key (for example, ``"passes_adjusted_per90"``).

    Returns:
        str: Verbose metric phrase ready to embed in generated prose.
    """
    if metric in GROUND_DUELS_WRITE_OUT_METRIC_MAP:
        return GROUND_DUELS_WRITE_OUT_METRIC_MAP[metric]

    return (
        metric.replace("_", " ")
        .replace("adjusted", "adjusted for possession")
        .replace("per90", "per 90")
        .replace(".id", " identifier")
        .replace(".name", " name")
        .replace("npxG", "non-penalty expected goals")
        + " minutes"
    )
