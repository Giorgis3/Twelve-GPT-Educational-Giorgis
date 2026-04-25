"""
Text helpers for converting raw metric values into readable sentences.

This module is intentionally lightweight and is used by description builders to:
1) map numeric z-scores to qualitative labels (for example, "excellent"), and
2) normalize metric names into human-readable phrases.
"""

# Explicit labels for all columns currently used in CB quality CSV exports.
# These overrides keep display text clear and domain-specific, while fallback
# normalization still supports all other datasets.
CB_QUALITY_FORMAT_METRIC_MAP = {
    "": "Row index",

    # Shared player, pair, and evidence metadata.
    "anchor_player_id": "Anchor CB player's ID",
    "anchor_player_name": "Anchor CB player's name",
    "CB_dominant_playing_side_CB1": "CB1's dominant playing side",
    "CB_dominant_playing_side_CB2": "CB2's dominant playing side",
    "evidence_band": "Evidence band",
    "minutes_played": "Minutes played",
    "opposite_side_pair": "Opposite-side pair",
    "pair_key": "Pair's key",
    "pair_name": "Pair's name",
    "partner_player_id": "Partner player's ID",
    "partner_player_name": "Partner player's name",
    "player.id": "Player's ID",
    "player.id_CB1": "CB1 player's ID",
    "player.id_CB2": "CB2 player's ID",
    "player.name": "Player's name",
    "player.name_CB1": "CB1 player's name",
    "player.name_CB2": "CB2 player's name",
    "same_dominant_team": "Playing on the same team indicator",
    "shared_matches": "Shared matches",
    "shared_minutes_overlap": "Shared minutes overlap",
    "team_CB1_id": "CB1 team's ID",
    "team_CB2_id": "CB2 team's ID",

    # Ground and aerial duel quality metrics.
    "aerial_duel_quality_score": "Aerial duel quality score",
    "aerial_wins_per90": "Aerial wins per 90",
    "card_discipline": "Card discipline",
    "discipline": "Discipline",
    "duel_success_rate": "Duel success rate",
    "duels_per90": "Duels per 90",
    "ground_duel_quality_score": "Ground duel quality score",
    "interceptions_per90": "Interceptions per 90",
    "possession_win_rate": "Possession win rate",
    "red_cards": "Red cards",
    "total_duels": "Total duels",
    "yellow_cards": "Yellow cards",
    "z_aerial_wins_per90": "Aerial wins per 90 z-score",
    "z_card_discipline": "Card discipline z-score",
    "z_discipline": "Discipline z-score",
    "z_duel_success_rate": "Duel success rate z-score",
    "z_duels_per90": "Duels per 90 z-score",
    "z_interceptions_per90": "Interceptions per 90 z-score",
    "z_possession_win_rate": "Possession win rate z-score",

    # Ball passing quality metrics.
    "accurate_long_passes": "Accurate long passes",
    "accurate_passes": "Accurate passes",
    "accurate_progressive_passes": "Accurate progressive passes",
    "CB_ball_passing_quality_z_score": "CB's ball passing quality z-score",
    "CB_rank_for_ball_passing_quality": "CB's rank for ball passing quality",
    "CB_rank_for_ball_passing_quality_z_score": "CB's rank for ball passing quality z-score",
    "final_third_passes": "Final third passes",
    "final_third_passes_per90": "Final third passes per 90",
    "long_pass_accuracy": "Long pass accuracy",
    "long_passes": "Long passes",
    "pass_accuracy_rate": "Pass accuracy rate",
    "passes_per90": "Passes per 90",
    "progressive_pass_accuracy": "Progressive pass accuracy",
    "progressive_passes": "Progressive passes",
    "progressive_passes_per90": "Progressive passes per 90",
    "total_passes": "Total passes",
    "z_final_third_passes_per90": "Final third passes per 90 z-score",
    "z_long_pass_accuracy": "Long pass accuracy z-score",
    "z_pass_accuracy_rate": "Pass accuracy rate z-score",
    "z_passes_per90": "Passes per 90 z-score",
    "z_progressive_pass_accuracy": "Progressive pass accuracy z-score",
    "z_progressive_passes_per90": "Progressive passes per 90 z-score",

    # Generic pair-quality components used by individual quality folders.
    "CB_pair_fit_rank": "CB's pair fit rank",
    "CB_pair_fit_rank_z_score": "CB's pair fit rank z-score",
    "CB_pair_fit_z_score": "CB's pair fit z-score",
    "companion_fit_score": "(CB) Companion's fit score",
    "companion_rank_for_anchor": "(CB) Companion's rank for anchor",
    "complement_raw": "Complementarity (raw)",
    "complement_z": "Complementarity z-score",
    "coverage_gain_raw": "Deficit-coverage gain (raw)",
    "coverage_gain_z_within_anchor": "Deficit-coverage gain z-score within anchor CB",
    "floor_raw": "Floor (weak-link protection) score (raw)",
    "floor_z": "Floor (weak-link protection) score z-score",
    "quality_raw": "Quality score (raw)",
    "quality_z": "Quality score z-score",

    # Global single-CB quality summaries.
    "CB_aerial_duels_quality_z_score": "CB's aerial duels quality z-score",
    "CB_ground_duels_quality_z_score": "CB's ground duels quality z-score",
    "CB_rank_for_aerial_duels_quality": "CB's rank for aerial duels quality",
    "CB_rank_for_aerial_duels_quality_z_score": "CB's rank for aerial duels quality z-score",
    "CB_rank_for_ground_duels_quality": "CB's rank for ground duels quality",
    "CB_rank_for_ground_duels_quality_z_score": "CB's rank for ground duels quality z-score",
    "global_quality_rank": "Global quality rank",
    "global_quality_rank_z_score": "Global quality rank z-score",
    "global_quality_z_score": "Global quality z-score",

    # Global companion-fit summaries by quality area.
    "aerial_companion_fit_score": "Aerial companion's fit score",
    "aerial_companion_fit_score_z_within_anchor": "Aerial companion's fit score z-score within anchor CB",
    "aerial_companion_rank_for_anchor": "Aerial companion's rank for anchor",
    "aerial_coverage_gain_raw": "Aerial deficit-coverage gain (raw)",
    "aerial_coverage_gain_z_within_anchor": "Aerial deficit-coverage gain z-score within anchor CB",
    "ball_companion_fit_score": "Ball passing companion's fit score",
    "ball_companion_fit_score_z_within_anchor": "Ball passing companion's fit score z-score within anchor CB",
    "ball_companion_rank_for_anchor": "Ball passing companion's rank for anchor",
    "ball_coverage_gain_raw": "Ball passing deficit-coverage gain (raw)",
    "ball_coverage_gain_z_within_anchor": "Ball passing deficit-coverage gain z-score within anchor CB",
    "global_companion_fit_score_z_within_anchor": "Global companion's fit score z-score within anchor CB",
    "global_companion_rank_for_anchor": "Global companion's rank for anchor",
    "global_companion_rank_for_anchor_z_score": "Global companion's rank for anchor z-score",
    "ground_companion_fit_score": "Ground duel companion's fit score",
    "ground_companion_fit_score_z_within_anchor": "Ground duel companion's fit score z-score within anchor CB",
    "ground_companion_rank_for_anchor": "Ground duel companion's rank for anchor",
    "ground_coverage_gain_raw": "Ground duel deficit-coverage gain (raw)",
    "ground_coverage_gain_z_within_anchor": "Ground duel deficit-coverage gain z-score within anchor CB",
    
    # Global pair-fit components by quality area.
    "aerial_CB_pair_fit_rank": "Aerial CB pair's fit rank",
    "aerial_CB_pair_fit_rank_z_score": "Aerial CB pair's fit rank z-score",
    "aerial_CB_pair_fit_z_score": "Aerial CB pair's fit z-score",
    "aerial_complement_raw": "Aerial complementarity (raw)",
    "aerial_complement_z": "Aerial complementarity z-score",
    "aerial_floor_raw": "Aerial floor score (raw)",
    "aerial_floor_z": "Aerial floor score z-score",
    "aerial_quality_raw": "Aerial quality score (raw)",
    "aerial_quality_z": "Aerial quality score z-score",
    "ball_CB_pair_fit_rank": "Ball passing CB pair's fit rank",
    "ball_CB_pair_fit_rank_z_score": "Ball passing CB pair's fit rank z-score",
    "ball_CB_pair_fit_z_score": "Ball passing CB pair's fit z-score",
    "ball_complement_raw": "Ball passing complementarity (raw)",
    "ball_complement_z": "Ball passing complementarity z-score",
    "ball_floor_raw": "Ball passing floor score (raw)",
    "ball_floor_z": "Ball passing floor score z-score",
    "ball_quality_raw": "Ball passing quality score (raw)",
    "ball_quality_z": "Ball passing quality score z-score",
    "global_CB_pair_fit_rank": "Global CB pair's fit rank",
    "global_CB_pair_fit_rank_z_score": "Global CB pair's fit rank z-score",
    "global_CB_pair_fit_z_score": "Global CB pair's fit z-score",
    "ground_CB_pair_fit_rank": "Ground duel CB pair's fit rank",
    "ground_CB_pair_fit_rank_z_score": "Ground duel CB pair's fit rank z-score",
    "ground_CB_pair_fit_z_score": "Ground duel CB pair's fit z-score",
    "ground_complement_raw": "Ground duel complementarity (raw)",
    "ground_complement_z": "Ground duel complementarity z-score",
    "ground_floor_raw": "Ground duel floor score (raw)",
    "ground_floor_z": "Ground duel floor score z-score",
    "ground_quality_raw": "Ground duel quality score (raw)",
    "ground_quality_z": "Ground duel quality score z-score",
}


CB_QUALITY_WRITE_OUT_METRIC_MAP = {
    "": "row index",

    # Shared player, pair, and evidence metadata.
    "anchor_player_id": "anchor CB player's ID",
    "anchor_player_name": "anchor CB player's name",
    "CB_dominant_playing_side_CB1": "dominant playing side for CB1",
    "CB_dominant_playing_side_CB2": "dominant playing side for CB2",
    "evidence_band": "evidence band",
    "minutes_played": "minutes played",
    "opposite_side_pair": "opposite-side pair indicator",
    "pair_key": "pair's key",
    "pair_name": "pair's name",
    "partner_player_id": "partner player's ID",
    "partner_player_name": "partner player's name",
    "player.id": "player's ID",
    "player.id_CB1": "CB1 player's ID",
    "player.id_CB2": "CB2 player's ID",
    "player.name": "player's name",
    "player.name_CB1": "CB1 player's name",
    "player.name_CB2": "CB2 player's name",
    "same_dominant_team": "playing on the same team indicator",
    "shared_matches": "shared matches",
    "shared_minutes_overlap": "shared minutes overlap",
    "team_CB1_id": "CB1 team's ID",
    "team_CB2_id": "CB2 team's ID",

    # Ground and aerial duel quality metrics.
    "aerial_duel_quality_score": "aerial duel quality score",
    "aerial_wins_per90": "aerial wins per 90 minutes",
    "card_discipline": "card discipline",
    "discipline": "discipline score",
    "duel_success_rate": "duel success rate",
    "duels_per90": "duels per 90 minutes",
    "ground_duel_quality_score": "ground duel quality score",
    "interceptions_per90": "interceptions per 90 minutes",
    "possession_win_rate": "possession win rate",
    "red_cards": "red cards",
    "total_duels": "total duels",
    "yellow_cards": "yellow cards",
    "z_aerial_wins_per90": "aerial wins per 90 z-score",
    "z_card_discipline": "card discipline z-score",
    "z_discipline": "discipline z-score",
    "z_duel_success_rate": "duel success rate z-score",
    "z_duels_per90": "duels per 90 z-score",
    "z_interceptions_per90": "interceptions per 90 z-score",
    "z_possession_win_rate": "possession win rate z-score",

    # Ball passing quality metrics.
    "accurate_long_passes": "accurate long passes",
    "accurate_passes": "accurate passes",
    "accurate_progressive_passes": "accurate progressive passes",
    "CB_ball_passing_quality_z_score": "CB's ball passing quality z-score",
    "CB_rank_for_ball_passing_quality": "CB's rank for ball passing quality",
    "CB_rank_for_ball_passing_quality_z_score": "CB's rank for ball passing quality z-score",
    "final_third_passes": "final third passes",
    "final_third_passes_per90": "final third passes per 90 minutes",
    "long_pass_accuracy": "long pass accuracy",
    "long_passes": "long passes",
    "pass_accuracy_rate": "pass accuracy rate",
    "passes_per90": "passes per 90 minutes",
    "progressive_pass_accuracy": "progressive pass accuracy",
    "progressive_passes": "progressive passes",
    "progressive_passes_per90": "progressive passes per 90 minutes",
    "total_passes": "total passes",
    "z_final_third_passes_per90": "final third passes per 90 z-score",
    "z_long_pass_accuracy": "long pass accuracy z-score",
    "z_pass_accuracy_rate": "pass accuracy rate z-score",
    "z_passes_per90": "passes per 90 z-score",
    "z_progressive_pass_accuracy": "progressive pass accuracy z-score",
    "z_progressive_passes_per90": "progressive passes per 90 z-score",

    # Generic pair-quality components used by individual quality folders.
    "CB_pair_fit_rank": "CB pair's fit rank",
    "CB_pair_fit_rank_z_score": "CB pair's fit rank z-score",
    "CB_pair_fit_z_score": "CB pair's fit z-score",
    "companion_fit_score": "(CB) companion's fit score",
    "companion_rank_for_anchor": "(CB) companion's rank for anchor",
    "complement_raw": "raw complementarity score",
    "complement_z": "complementarity z-score",
    "coverage_gain_raw": "raw deficit-coverage gain",
    "coverage_gain_z_within_anchor": "deficit-coverage gain z-score within anchor CB",
    "floor_raw": "raw floor score",
    "floor_z": "floor score z-score",
    "quality_raw": "raw quality score",
    "quality_z": "quality z-score",

    # Global single-CB quality summaries.
    "CB_aerial_duels_quality_z_score": "CB's aerial duels quality z-score",
    "CB_ground_duels_quality_z_score": "CB's ground duels quality z-score",
    "CB_rank_for_aerial_duels_quality": "CB's rank for aerial duels quality",
    "CB_rank_for_aerial_duels_quality_z_score": "CB's rank for aerial duels quality z-score",
    "CB_rank_for_ground_duels_quality": "CB's rank for ground duels quality",
    "CB_rank_for_ground_duels_quality_z_score": "CB's rank for ground duels quality z-score",
    "global_quality_rank": "global quality rank",
    "global_quality_rank_z_score": "global quality rank z-score",
    "global_quality_z_score": "global quality z-score",

    # Global companion-fit summaries by quality area.
    "aerial_companion_fit_score": "aerial companion's fit score",
    "aerial_companion_fit_score_z_within_anchor": "aerial companion's fit score z-score within anchor CB",
    "aerial_companion_rank_for_anchor": "aerial companion's rank for anchor",
    "aerial_coverage_gain_raw": "raw aerial deficit-coverage gain",
    "aerial_coverage_gain_z_within_anchor": "aerial deficit-coverage gain z-score within anchor CB",
    "ball_companion_fit_score": "ball passing companion's fit score",
    "ball_companion_fit_score_z_within_anchor": "ball passing companion's fit score z-score within anchor CB",
    "ball_companion_rank_for_anchor": "ball passing companion's rank for anchor",
    "ball_coverage_gain_raw": "raw ball passing deficit-coverage gain",
    "ball_coverage_gain_z_within_anchor": "ball passing deficit-coverage gain z-score within anchor CB",
    "global_companion_fit_score_z_within_anchor": "global companion's fit score z-score within anchor CB",
    "global_companion_rank_for_anchor": "global companion's rank for anchor",
    "global_companion_rank_for_anchor_z_score": "global companion's rank for anchor z-score",
    "ground_companion_fit_score": "ground duel companion's fit score",
    "ground_companion_fit_score_z_within_anchor": "ground duel companion's fit score z-score within anchor CB",
    "ground_companion_rank_for_anchor": "ground duel companion's rank for anchor",
    "ground_coverage_gain_raw": "raw ground duel deficit-coverage gain",
    "ground_coverage_gain_z_within_anchor": "ground duel deficit-coverage gain z-score within anchor CB",

    # Global pair-fit components by quality area.
    "aerial_CB_pair_fit_rank": "aerial CB pair's fit rank",
    "aerial_CB_pair_fit_rank_z_score": "aerial CB pair's fit rank z-score",
    "aerial_CB_pair_fit_z_score": "aerial CB pair's fit z-score",
    "aerial_complement_raw": "raw aerial complementarity score",
    "aerial_complement_z": "aerial complementarity z-score",
    "aerial_floor_raw": "raw aerial floor score",
    "aerial_floor_z": "aerial floor score z-score",
    "aerial_quality_raw": "raw aerial quality score",
    "aerial_quality_z": "aerial quality z-score",
    "ball_CB_pair_fit_rank": "ball passing CB pair's fit rank",
    "ball_CB_pair_fit_rank_z_score": "ball passing CB pair's fit rank z-score",
    "ball_CB_pair_fit_z_score": "ball passing CB pair's fit z-score",
    "ball_complement_raw": "raw ball passing complementarity score",
    "ball_complement_z": "ball passing complementarity z-score",
    "ball_floor_raw": "raw ball passing floor score",
    "ball_floor_z": "ball passing floor score z-score",
    "ball_quality_raw": "raw ball passing quality score",
    "ball_quality_z": "ball passing quality z-score",
    "global_CB_pair_fit_rank": "global CB pair's fit rank",
    "global_CB_pair_fit_rank_z_score": "global CB pair's fit rank z-score",
    "global_CB_pair_fit_z_score": "global CB pair's fit z-score",
    "ground_CB_pair_fit_rank": "ground duel CB pair's fit rank",
    "ground_CB_pair_fit_rank_z_score": "ground duel CB pair's fit rank z-score",
    "ground_CB_pair_fit_z_score": "ground duel CB pair's fit z-score",
    "ground_complement_raw": "raw ground duel complementarity score",
    "ground_complement_z": "ground duel complementarity z-score",
    "ground_floor_raw": "raw ground duel floor score",
    "ground_floor_z": "ground duel floor score z-score",
    "ground_quality_raw": "raw ground duel quality score",
    "ground_quality_z": "ground duel quality z-score",
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
    without extra qualifiers such as "adjusted per90". CB quality CSV columns
    are handled with explicit label overrides.

    Parameters:
        metric: Raw metric key (for example, ``"npxG_adjusted_per90"``).

    Returns:
        str: Cleaned, capitalized metric label.
    """
    if metric in CB_QUALITY_FORMAT_METRIC_MAP:
        return CB_QUALITY_FORMAT_METRIC_MAP[metric]

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
    (for example "adjusted for possession" and "per 90"). CB quality CSV
    columns are handled with explicit phrase overrides.

    Parameters:
        metric: Raw metric key (for example, ``"passes_adjusted_per90"``).

    Returns:
        str: Verbose metric phrase ready to embed in generated prose.
    """
    if metric in CB_QUALITY_WRITE_OUT_METRIC_MAP:
        return CB_QUALITY_WRITE_OUT_METRIC_MAP[metric]

    return (
        metric.replace("_", " ")
        .replace("adjusted", "adjusted for possession")
        .replace("per90", "per 90")
        .replace(".id", " ID")
        .replace(".name", " name")
        .replace("npxG", "non-penalty expected goals")
        + " minutes"
    )
