"""
pass_risk.py
------------
Reusable functions for building a passing-risk / danger-zone metric.

The core idea: a pass in the team's own half that fails hands the opponent
possession in a threatening area.  We quantify that threat using Expected
Threat (xT), weighted by the probability that the pass will fail.

The pipeline:
  1. Filter own-half passes from a Wyscout 2024-format events DataFrame.
  2. Engineer geometric and contextual pass features.
  3. Train a GradientBoosting model to predict pass-success probability.
  4. Combine (1 - p_success) with a reversed xT grid → Expected_xT_Risk.
  5. Aggregate to player level (per-90 and per-pass summaries).

All functions accept explicit arguments (no hidden globals) so they work
for any league/season dataset that follows the Wyscout 2024 event format.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# xT grid (12 × 16)
# Source: https://github.com/ML-KULeuven/socceraction
# Rows = y-axis (pitch width, 68 m / 12 ≈ 5.67 m per row)
# Cols = x-axis (pitch length, 105 m / 16 ≈ 6.56 m per col)
# Values increase toward the opponent's goal (right side, high x).
# ---------------------------------------------------------------------------
XT_GRID = np.array([
    [0.00672549, 0.00805248, 0.0103721,  0.01200744, 0.01438679, 0.01707883,
     0.02032915, 0.02133409, 0.02593793, 0.03136372, 0.04022231, 0.05204074,
     0.0730182,  0.09069239, 0.11539901, 0.13910926],
    [0.00631415, 0.00689012, 0.00829943, 0.01011124, 0.01205301, 0.01567271,
     0.01871454, 0.02201926, 0.02664821, 0.03236505, 0.04160811, 0.05835067,
     0.07902805, 0.10499214, 0.13341578, 0.15453183],
    [0.00550273, 0.00674261, 0.00787796, 0.00807278, 0.01072891, 0.01430179,
     0.01897018, 0.0220314,  0.02649758, 0.03256084, 0.04392719, 0.06302237,
     0.08255304, 0.10077727, 0.14307735, 0.17966437],
    [0.00578281, 0.00626537, 0.00673733, 0.00758239, 0.01063904, 0.01404244,
     0.01855326, 0.02059162, 0.0268419,  0.0331566,  0.04461219, 0.05957225,
     0.0754661,  0.0921076,  0.11903919, 0.17885769],
    [0.0053499,  0.0067439,  0.00804624, 0.00756041, 0.0102514,  0.01289887,
     0.01684039, 0.02101277, 0.02793688, 0.03400337, 0.04512778, 0.05891237,
     0.07150569, 0.07677173, 0.1367841,  0.1769704 ],
    [0.00503522, 0.00679683, 0.00922705, 0.00726949, 0.00972899, 0.01303878,
     0.01657317, 0.02059088, 0.02567442, 0.03455939, 0.04510978, 0.05514248,
     0.06929877, 0.10003309, 0.18075745, 0.42059394],
    [0.00498648, 0.00617258, 0.00820134, 0.00683712, 0.01005172, 0.01295557,
     0.01631337, 0.01932426, 0.02471068, 0.03304044, 0.04399589, 0.05607948,
     0.06133564, 0.1321362,  0.18045622, 0.38047852],
    [0.00565008, 0.00658425, 0.00698444, 0.00651967, 0.00960199, 0.01317199,
     0.01591153, 0.01922817, 0.02612512, 0.03181268, 0.04300717, 0.05641633,
     0.06663851, 0.09165561, 0.13649669, 0.17445016],
    [0.00455451, 0.00548285, 0.00657182, 0.00723207, 0.00899293, 0.01247379,
     0.01565568, 0.01840417, 0.02468003, 0.03012028, 0.04243792, 0.05870114,
     0.0709952,  0.084004,   0.11758409, 0.16782486],
    [0.00481693, 0.00585722, 0.00619891, 0.00704794, 0.00933818, 0.01261871,
     0.01592967, 0.01923327, 0.02372912, 0.03023846, 0.04100453, 0.0584064,
     0.07673507, 0.09555109, 0.13139283, 0.1683184 ],
    [0.00442462, 0.0059867,  0.00681912, 0.0077604,  0.01056856, 0.01276884,
     0.01543724, 0.01781678, 0.02227673, 0.02840159, 0.03768963, 0.05346466,
     0.07533083, 0.09490246, 0.12717713, 0.1473329 ],
    [0.00572744, 0.00660803, 0.00768298, 0.00986248, 0.01110282, 0.01345644,
     0.01642725, 0.01767348, 0.02281427, 0.02620101, 0.03369801, 0.04612249,
     0.06679846, 0.08679778, 0.1090803,  0.12502026],
])


# ---------------------------------------------------------------------------
# Step 1 – Filter own-half passes
# ---------------------------------------------------------------------------

def get_own_half_passes(events_df, x_threshold=50):
    """
    Filter pass events that originate in the team's own half.

    Parameters
    ----------
    events_df : pd.DataFrame
        Full Wyscout 2024 event DataFrame (one row per event).
    x_threshold : float
        x-coordinate boundary (0–100 scale).  Passes with location.x <
        x_threshold are considered to start in the danger zone / own half.
        Default 50 = own half.  Use a stricter value (e.g. 35) for the
        deepest defensive third only.

    Returns
    -------
    pd.DataFrame
        Own-half passes with an integer `Successful` column (1 = accurate).
    """
    passes = events_df[events_df["type.primary"] == "pass"].copy()
    passes = passes[passes["location.x"] < x_threshold].copy()

    # Pass success comes directly from pass.accurate — no circular target.
    passes["Successful"] = passes["pass.accurate"].astype(int)

    return passes.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 – Geometric pass features
# ---------------------------------------------------------------------------

def calculate_pass_features(passes_df):
    """
    Compute geometric features for each pass in metres.

    Converts Wyscout 0-100 coordinates to metres (105 × 68 pitch), then adds:
      - X, Y          : pass origin (metres)
      - X_end, Y_end  : pass destination (metres)
      - Pass_Distance  : Euclidean length of the pass
      - Pass_Angle     : direction in radians (arctan2)

    Parameters
    ----------
    passes_df : pd.DataFrame  (own-half passes from get_own_half_passes)

    Returns
    -------
    pd.DataFrame  (original columns + new feature columns)
    """
    df = passes_df.copy()

    # Convert 0-100 Wyscout coords → metres
    df["X"]     = df["location.x"]           * 105 / 100
    df["Y"]     = df["location.y"]           * 68  / 100
    df["X_end"] = df["pass.endLocation.x"]   * 105 / 100
    df["Y_end"] = df["pass.endLocation.y"]   * 68  / 100

    return df


# ---------------------------------------------------------------------------
# Step 3 – Subtype dummy variables
# ---------------------------------------------------------------------------

# Pass subtypes we care about.  These come from the type.secondary list column.
PASS_SUBTYPES = [
    "head_pass",
    "long_pass",
]


def prepare_pass_features(passes_df, subtypes=None):
    """
    Add binary dummy variables for pass subtypes drawn from `type.secondary`.

    Parameters
    ----------
    passes_df : pd.DataFrame  (output of calculate_pass_features)
    subtypes  : list[str] | None
        Subtype strings to encode.  Defaults to PASS_SUBTYPES.

    Returns
    -------
    pd.DataFrame  (original columns + one boolean column per subtype)
    """
    if subtypes is None:
        subtypes = PASS_SUBTYPES

    df = passes_df.copy()

    for subtype in subtypes:
        col = f"is_{subtype}"
        df[col] = df["type.secondary"].apply(
            lambda lst: 1 if isinstance(lst, list) and subtype in lst else 0
        )

    df["is_pass_high"] = (df["pass.height"] == "high").astype(int)

    return df


# ---------------------------------------------------------------------------
# Step 4 – Train pass-success model
# ---------------------------------------------------------------------------

# Default features for the classifier.
DEFAULT_FEATURE_COLS = [
    "X", "Y", "X_end", "Y_end",
    "pass.length", "pass.angle", "is_pass_high",
    "is_head_pass", "is_long_pass"
]


def train_pass_success_model(
    passes_df,
    feature_cols=None,
    target_col="Successful",
    test_size=0.2,
    random_state=42,
    print_report=True,
):
    """
    Train a GradientBoostingClassifier to predict whether a pass will succeed.

    SMOTE over-sampling is applied to the training set to handle the class
    imbalance inherent in high-pass-accuracy datasets.

    Parameters
    ----------
    passes_df    : pd.DataFrame  (must contain feature_cols + target_col)
    feature_cols : list[str] | None  (defaults to DEFAULT_FEATURE_COLS)
    target_col   : str               (binary 0/1 column)
    test_size    : float
    random_state : int
    print_report : bool  – print classification report + ROC-AUC after training

    Returns
    -------
    model  : fitted GradientBoostingClassifier
    scaler : fitted StandardScaler  (apply to any data before predicting)
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    data = passes_df[feature_cols + [target_col]].dropna()
    X = data[feature_cols].values
    y = data[target_col].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # Oversample the minority class (failed passes) so the model learns
    # to recognise risky passes, not just rubber-stamp every pass as safe.
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        max_features="sqrt",
        learning_rate=0.1,
        subsample=1.0,
        random_state=random_state,
    )
    model.fit(X_resampled, y_resampled)

    if print_report:
        y_pred  = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        print(classification_report(y_val, y_pred, target_names=["Failed", "Successful"]))
        print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")

    return model, scaler


# ---------------------------------------------------------------------------
# Step 5 – Predict pass-success probability
# ---------------------------------------------------------------------------

def predict_pass_probability(model, scaler, passes_df, feature_cols=None):
    """
    Add `prob_successful_pass` to passes_df using a trained model.

    Parameters
    ----------
    model        : fitted GradientBoostingClassifier
    scaler       : fitted StandardScaler (must match the one used in training)
    passes_df    : pd.DataFrame
    feature_cols : list[str] | None  (defaults to DEFAULT_FEATURE_COLS)

    Returns
    -------
    pd.DataFrame  (original + prob_successful_pass column)
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    df = passes_df.copy()
    X  = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)
    df["prob_successful_pass"] = model.predict_proba(X_scaled)[:, 1]

    return df


# ---------------------------------------------------------------------------
# Step 6 – xT-based risk calculation
# ---------------------------------------------------------------------------

def  _map_to_xt(x_arr, y_arr, grid, pitch_length=105, pitch_width=68):
    """Vectorised lookup of xT grid values given arrays of metre coordinates."""
    cell_w = pitch_length / grid.shape[1]
    cell_h = pitch_width  / grid.shape[0]

    x_idx = np.minimum((x_arr // cell_w).astype(int), grid.shape[1] - 1)
    y_idx = np.minimum((y_arr // cell_h).astype(int), grid.shape[0] - 1)

    return grid[y_idx, x_idx]


def calculate_xt_risk(passes_df, xt_grid=None):
    """
    Compute xT-based risk metrics for each pass.

    For a pass starting at position (X, Y):
      - xT_start  : threat at the origin
      - xT_end    : threat at the destination (if pass succeeds)
      - xT_reversed : threat at the mirror position (i.e. how dangerous it
                      is for the opponent to receive the ball where you are)

    Derived columns:
      - Expected_xT_Gain  = p_success  × (xT_end − xT_start)
      - Expected_xT_Risk  = (1 − p_success) × xT_reversed
      - Net_xT_Value      = Gain − Risk

    A lower Expected_xT_Risk means the player's passing keeps danger away
    even when passes occasionally go astray.

    Parameters
    ----------
    passes_df : pd.DataFrame  (must have X, Y, X_end, Y_end, prob_successful_pass)
    xt_grid   : np.ndarray (12 × 16) | None  (defaults to XT_GRID)

    Returns
    -------
    pd.DataFrame  (original + xT columns)
    """
    if xt_grid is None:
        xt_grid = XT_GRID

    df = passes_df.copy()

    # Flip left-right so that cells near your own goal map to high-xT cells
    # from the opponent's perspective — capturing how dangerous it is to
    # surrender possession there.
    xt_reversed = np.fliplr(xt_grid)

    df["xT_start"]    = _map_to_xt(df["X"].values, df["Y"].values, xt_grid)
    df["xT_end"]      = _map_to_xt(df["X_end"].values, df["Y_end"].values, xt_grid)
    df["xT_reversed"] = _map_to_xt(df["X"].values, df["Y"].values, xt_reversed)

    df["Change_in_xT"]      = df["xT_end"] - df["xT_start"]
    df["Expected_xT_Gain"]  = df["prob_successful_pass"] * df["Change_in_xT"]
    df["Expected_xT_Risk"]  = (1 - df["prob_successful_pass"]) * df["xT_reversed"]
    df["Net_xT_Value"]      = df["Expected_xT_Gain"] - df["Expected_xT_Risk"]

    return df


# ---------------------------------------------------------------------------
# Step 7 – Per-player pass completion rate
# ---------------------------------------------------------------------------

def calculate_pass_completion_rate(passes_df, player_col="player.id"):
    """
    Compute each player's overall pass completion rate within passes_df.

    Parameters
    ----------
    passes_df  : pd.DataFrame  (must have `Successful` column)
    player_col : str

    Returns
    -------
    pd.DataFrame  with columns [player_col, pass_completion_rate, total_passes]
    """
    stats = (
        passes_df
        .groupby(player_col)
        .agg(
            successful_passes=("Successful", "sum"),
            total_passes=("Successful", "count"),
        )
        .reset_index()
    )
    stats["pass_completion_rate"] = (
        stats["successful_passes"] / stats["total_passes"]
    )
    return stats[[player_col, "pass_completion_rate", "total_passes"]]


# ---------------------------------------------------------------------------
# Step 8 – Aggregate to player level
# ---------------------------------------------------------------------------

def aggregate_player_risk_metrics(
    passes_df,
    players_df,
    minutes_df,
    xt_grid=None,
    filter_minutes=300,
    filter_pass_count=50,
    pass_completion_weight=10.0,
    player_col="player.id",
):
    """
    Roll up pass-level xT risk to per-player metrics.

    Metrics returned:
      - Expected_xT_Risk_per_90        : total risk normalised to 90 minutes
      - Average_xT_Risk_per_Pass       : average risk per own-half pass
      - Accuracy_Adjusted_Risk_per_Pass: risk discounted for accurate passers
            (lower = safer passer even when adjusting for accuracy)

    Parameters
    ----------
    passes_df              : pd.DataFrame  (output of calculate_xt_risk)
    players_df             : pd.DataFrame  (player metadata, must have
                                            player_id, short_name, role)
    minutes_df             : pd.DataFrame  (must have player_id, minutes_played)
    xt_grid                : np.ndarray | None  (defaults to XT_GRID)
    filter_minutes         : int  – drop players with fewer minutes (noise filter)
    filter_pass_count      : int  – drop players with too few own-half passes
    pass_completion_weight : float – scaling factor for accuracy adjustment
    player_col             : str  – player ID column name in passes_df

    Returns
    -------
    pd.DataFrame  sorted by Accuracy_Adjusted_Risk_per_Pass (ascending = safer)
    """
    if xt_grid is None:
        xt_grid = XT_GRID

    df = passes_df.copy()

    # Per-player completion rate (computed on own-half passes)
    completion = calculate_pass_completion_rate(df, player_col=player_col)
    df = df.merge(completion, on=player_col, how="left")

    # Adjusted net xT value (rewards accurate passers)
    df["Adjusted_Net_xT_Value"] = (
        df["Net_xT_Value"]
        * (1 + pass_completion_weight * df["pass_completion_rate"])
    )

    # Aggregate per player
    agg = (
        df.groupby(player_col)
        .agg(
            Expected_xT_Risk          =("Expected_xT_Risk",       "sum"),
            Expected_xT_Gain          =("Expected_xT_Gain",       "sum"),
            Adjusted_Net_xT_Value     =("Adjusted_Net_xT_Value",  "sum"),
            pass_count                =("Expected_xT_Risk",       "count"),
            pass_completion_rate      =("pass_completion_rate",   "mean"),
        )
        .reset_index()
    )

    # Join player metadata (short_name, role)
    players_slim = players_df[["player_id", "short_name", "role"]].drop_duplicates("player_id")
    agg = agg.merge(
        players_slim,
        left_on=player_col,
        right_on="player_id",
        how="inner",
    )

    # Total minutes per player across the season
    mins = (
        minutes_df
        .groupby("player_id")["minutes_played"]
        .sum()
        .reset_index()
    )
    agg = agg.merge(mins, on="player_id", how="inner")

    # Per-90 normalisation
    agg["Expected_xT_Risk_per_90"] = (
        agg["Expected_xT_Risk"] / agg["minutes_played"]
    ) * 90

    agg["Adjusted_Net_xT_Value_per_90"] = (
        agg["Adjusted_Net_xT_Value"] / agg["minutes_played"]
    ) * 90

    # Per-pass metrics
    agg["Average_xT_Risk_per_Pass"] = (
        agg["Expected_xT_Risk"] / agg["pass_count"]
    )

    # Accuracy-adjusted risk: players with high completion rates get a
    # discount because their risky passes rarely materialise into danger.
    agg["Accuracy_Adjusted_Risk_per_Pass"] = (
        agg["Average_xT_Risk_per_Pass"]
        / (1 + (pass_completion_weight ** 2) * agg["pass_completion_rate"])
    )

    # Apply volume/quality filters
    agg = agg[agg["minutes_played"] >= filter_minutes]
    agg = agg[agg["pass_count"]     >= filter_pass_count]

    output_cols = [
        player_col, "short_name", "role",
        "minutes_played", "pass_count", "pass_completion_rate",
        "Expected_xT_Risk_per_90",
        "Average_xT_Risk_per_Pass",
        "Accuracy_Adjusted_Risk_per_Pass",
    ]

    return (
        agg[output_cols]
        .sort_values("Accuracy_Adjusted_Risk_per_Pass", ascending=True)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Angle 1 – xT Generated via Passing
# ---------------------------------------------------------------------------

def calculate_xt_generated(
    events_df,
    players_df,
    minutes_df,
    xt_grid=None,
    filter_minutes=300,
    filter_pass_count=50,
    player_col="player.id",
):
    """
    Compute how much Expected Threat each player generates through passing.

    For every **completed** pass we compute:

        xT_gain = xT_end − xT_start

    where xT_end is the threat value at the pass destination and xT_start is
    the threat value at the pass origin.  Negative values (backward/sideways
    passes) are kept so the aggregate reflects a player's **net** forward
    contribution, not just their best passes.

    Only completed passes are counted: a failed pass doesn't advance your
    team's threat position (the ball is lost), so it contributes nothing here.
    The downside of failed passes is captured separately by the risk metric.

    Metrics returned
    ----------------
    xT_Generated_per_90  : sum of xT_gain normalised to 90 minutes — volume
    xT_per_Pass          : mean xT_gain per completed pass — efficiency
    progressive_passes   : count of completed passes where xT_gain > 0
    progressive_pass_rate: share of completed passes that move the ball forward

    Parameters
    ----------
    events_df      : pd.DataFrame  (full Wyscout 2024 event DataFrame)
    players_df     : pd.DataFrame  (player metadata: player_id, short_name, role)
    minutes_df     : pd.DataFrame  (minutes_played per player per match)
    xt_grid        : np.ndarray (12 × 16) | None  — defaults to XT_GRID
    filter_minutes : int  — minimum minutes played (noise filter)
    filter_pass_count : int  — minimum completed passes
    player_col     : str  — player ID column name in events_df

    Returns
    -------
    pd.DataFrame sorted by xT_Generated_per_90 descending (best creators first)
    """
    if xt_grid is None:
        xt_grid = XT_GRID

    # All passes — not restricted to own half; we want full passing contribution
    passes = events_df[events_df["type.primary"] == "pass"].copy()

    # Convert Wyscout 0-100 coords to metres
    passes = calculate_pass_features(passes)

    # xT at origin and destination
    passes["xT_start"] = _map_to_xt(passes["X"].values, passes["Y"].values, xt_grid)
    passes["xT_end"]   = _map_to_xt(passes["X_end"].values, passes["Y_end"].values, xt_grid)

    # Net change in threat per pass (negative = backward / lateral pass)
    passes["xT_gain"] = passes["xT_end"] - passes["xT_start"]

    # Only completed passes realise the threat at the destination
    completed = passes[passes["pass.accurate"] == True].copy()

    # Per-player aggregation
    agg = (
        completed.groupby(player_col)
        .agg(
            xT_Generated      =("xT_gain", "sum"),
            xT_per_Pass       =("xT_gain", "mean"),
            pass_count        =("xT_gain", "count"),
            progressive_passes=("xT_gain", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    agg["progressive_pass_rate"] = agg["progressive_passes"] / agg["pass_count"]

    # Join player metadata
    players_slim = (
        players_df[["player_id", "short_name", "role"]]
        .drop_duplicates("player_id")
    )
    agg = agg.merge(
        players_slim,
        left_on=player_col,
        right_on="player_id",
        how="inner",
    )

    # Total minutes across the season
    mins = (
        minutes_df
        .groupby("player_id")["minutes_played"]
        .sum()
        .reset_index()
    )
    agg = agg.merge(mins, on="player_id", how="inner")

    # Per-90 normalisation
    agg["xT_Generated_per_90"] = agg["xT_Generated"] / agg["minutes_played"] * 90

    # Apply volume / quality filters
    agg = agg[agg["minutes_played"] >= filter_minutes]
    agg = agg[agg["pass_count"]     >= filter_pass_count]

    output_cols = [
        player_col, "short_name", "role",
        "minutes_played", "pass_count",
        "progressive_passes", "progressive_pass_rate",
        "xT_Generated", "xT_Generated_per_90", "xT_per_Pass",
    ]

    return (
        agg[output_cols]
        .sort_values("xT_Generated_per_90", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Angle 2 – Final Third Entry Passes
# ---------------------------------------------------------------------------

FT_THRESHOLD = 66.7  # Wyscout 0-100 scale; 100 × (2/3) ≈ 66.7


def calculate_ft_entry_passes(
    events_df,
    players_df,
    minutes_df,
    ft_threshold=FT_THRESHOLD,
    filter_minutes=300,
    filter_attempt_count=10,
    player_col="player.id",
):
    """
    Compute Final Third Entry Pass metrics for each player.

    A Final Third Entry Pass is a pass that:
      - starts OUTSIDE the final third  (location.x   <= ft_threshold)
      - ends   INSIDE  the final third  (pass.endLocation.x > ft_threshold)

    Passes within the final third (start AND end > ft_threshold) are excluded
    — they don't represent ball progression into a new zone.

    Metrics returned
    ----------------
    FT_Entry_Attempts_per_90   : attempted FT entries per 90 minutes (volume)
    FT_Entry_Passes_per_90     : completed FT entries per 90 minutes
    FT_Entry_Completion_pct    : share of FT entry attempts that arrive
                                 (style-agnostic quality signal)

    Parameters
    ----------
    events_df          : pd.DataFrame  (full Wyscout 2024 event DataFrame)
    players_df         : pd.DataFrame  (player metadata: player_id, short_name, role)
    minutes_df         : pd.DataFrame  (minutes_played per player per match)
    ft_threshold       : float  — x-coordinate boundary (0-100 scale).
                         Default 66.7 = exact final third (100 × 2/3).
    filter_minutes     : int  — minimum minutes played (noise filter)
    filter_attempt_count : int  — minimum FT entry attempts (ensures completion
                           rate is meaningful; default 10)
    player_col         : str  — player ID column name in events_df

    Returns
    -------
    pd.DataFrame sorted by FT_Entry_Passes_per_90 descending
    """
    passes = events_df[events_df["type.primary"] == "pass"].copy()

    # Entry passes: cross the final-third boundary from outside
    ft_entries = passes[
        (passes["location.x"] <= ft_threshold) &
        (passes["pass.endLocation.x"] > ft_threshold)
    ].copy()

    # Cast to int so .sum() produces a numeric (not object) column even when
    # pass.accurate contains booleans mixed with NaN values
    ft_entries["accurate_int"] = ft_entries["pass.accurate"].fillna(False).astype(int)

    # Per-player aggregation: attempts and completions
    agg = (
        ft_entries.groupby(player_col)
        .agg(
            FT_Entry_Attempts  =("accurate_int", "count"),
            FT_Entry_Completed =("accurate_int", "sum"),
        )
        .reset_index()
    )

    agg["FT_Entry_Completion_pct"] = (
        agg["FT_Entry_Completed"].astype(float) / agg["FT_Entry_Attempts"]
    )

    # Join player metadata
    players_slim = (
        players_df[["player_id", "short_name", "role"]]
        .drop_duplicates("player_id")
    )
    agg = agg.merge(
        players_slim,
        left_on=player_col,
        right_on="player_id",
        how="inner",
    )

    # Total minutes across the season
    mins = (
        minutes_df
        .groupby("player_id")["minutes_played"]
        .sum()
        .reset_index()
    )
    agg = agg.merge(mins, on="player_id", how="inner")

    # Per-90 normalisation
    agg["FT_Entry_Attempts_per_90"] = (
        agg["FT_Entry_Attempts"] / agg["minutes_played"] * 90
    )
    agg["FT_Entry_Passes_per_90"] = (
        agg["FT_Entry_Completed"] / agg["minutes_played"] * 90
    )

    # Apply noise filters
    agg = agg[agg["minutes_played"]      >= filter_minutes]
    agg = agg[agg["FT_Entry_Attempts"]   >= filter_attempt_count]

    output_cols = [
        player_col, "short_name", "role",
        "minutes_played",
        "FT_Entry_Attempts", "FT_Entry_Completed",
        "FT_Entry_Attempts_per_90", "FT_Entry_Passes_per_90",
        "FT_Entry_Completion_pct",
    ]

    return (
        agg[output_cols]
        .sort_values("FT_Entry_Passes_per_90", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Angle 4 – Carries / Dribbles into Advanced Areas
# ---------------------------------------------------------------------------

def calculate_carries(
    events_df,
    players_df,
    minutes_df,
    xt_grid=None,
    ft_threshold=FT_THRESHOLD,
    progressive_distance_m=10.0,
    filter_minutes=300,
    filter_carry_count=5,
    player_col="player.id",
):
    """
    Compute carry / ball-progression metrics for each player.

    A **carry** is any event where type.primary == 'carry'.  Wyscout provides
    start coordinates (location.x/y) and end coordinates (carry.endLocation.x/y)
    for each carry, plus a carry.progression boolean flag.

    Progressive carry definition (either condition satisfies):
      - Wyscout's carry.progression flag is True, OR
      - The carry moves the ball ≥ progressive_distance_m metres forward
        (fallback if the flag is missing / NaN)

    All carries contribute to xT metrics — there is no "failed carry" because
    carry.endLocation always reflects where the ball actually ended up.
    Negative xT gains (backward/lateral carries) are included so the aggregate
    reflects net forward contribution.

    Metrics returned
    ----------------
    Carries_per_90               : all carries per 90 minutes (volume baseline)
    Progressive_Carries_per_90   : carries that advance the ball significantly
    Carries_into_FT_per_90       : carries ending inside the final third
    Carry_Distance_per_90        : total metres carried forward per 90
    xT_via_Carries_per_90        : net xT gained via carrying per 90 minutes
    xT_per_Carry                 : mean xT gain per carry (efficiency)

    Parameters
    ----------
    events_df              : pd.DataFrame  (full Wyscout 2024 event DataFrame)
    players_df             : pd.DataFrame  (player metadata: player_id, short_name, role)
    minutes_df             : pd.DataFrame  (minutes_played per player per match)
    xt_grid                : np.ndarray (12 × 16) | None  — defaults to XT_GRID
    ft_threshold           : float  — final-third boundary on 0-100 scale (default 66.7)
    progressive_distance_m : float  — minimum forward metres to count as progressive
                             (used only when carry.progression flag is NaN)
    filter_minutes         : int  — minimum minutes played
    filter_carry_count     : int  — minimum total carries (noise filter)
    player_col             : str  — player ID column name in events_df

    Returns
    -------
    pd.DataFrame sorted by Progressive_Carries_per_90 descending
    """
    if xt_grid is None:
        xt_grid = XT_GRID

    carry_types = ["touch", "acceleration"]
    carries = events_df[
        events_df["type.primary"].isin(carry_types) &
        events_df["carry.endLocation.x"].notna()
    ].copy()

    # Convert coordinates to metres
    carries["X"]     = carries["location.x"]          * 105 / 100
    carries["Y"]     = carries["location.y"]           * 68  / 100
    carries["X_end"] = carries["carry.endLocation.x"]  * 105 / 100
    carries["Y_end"] = carries["carry.endLocation.y"]  * 68  / 100

    # Forward distance (negative = carried backward)
    carries["forward_distance_m"] = carries["X_end"] - carries["X"]

    # Progressive carry: use Wyscout flag where available, else geometric fallback
    wyscout_flag = carries["carry.progression"].fillna(False).astype(bool)
    geometric     = carries["forward_distance_m"] >= progressive_distance_m
    carries["is_progressive"] = (wyscout_flag | geometric).astype(int)

    # Carry into final third: end location crosses final-third boundary
    carries["into_final_third"] = (
        carries["carry.endLocation.x"] > ft_threshold
    ).astype(int)

    # xT gain per carry — all carries included, negatives kept
    carries["xT_start"] = _map_to_xt(carries["X"].values,     carries["Y"].values,     xt_grid)
    carries["xT_end"]   = _map_to_xt(carries["X_end"].values, carries["Y_end"].values, xt_grid)
    carries["xT_gain"]  = carries["xT_end"] - carries["xT_start"]

    # Aggregate per player
    agg = (
        carries.groupby(player_col)
        .agg(
            total_carries        =("is_progressive",    "count"),
            progressive_carries  =("is_progressive",    "sum"),
            carries_into_ft      =("into_final_third",  "sum"),
            total_forward_dist_m =("forward_distance_m","sum"),
            xT_via_Carries       =("xT_gain",           "sum"),
            xT_per_Carry         =("xT_gain",           "mean"),
        )
        .reset_index()
    )

    # Join player metadata
    players_slim = (
        players_df[["player_id", "short_name", "role"]]
        .drop_duplicates("player_id")
    )
    agg = agg.merge(
        players_slim,
        left_on=player_col,
        right_on="player_id",
        how="inner",
    )

    # Total minutes across the season
    mins = (
        minutes_df
        .groupby("player_id")["minutes_played"]
        .sum()
        .reset_index()
    )
    agg = agg.merge(mins, on="player_id", how="inner")

    # Per-90 normalisation
    agg["Carries_per_90"] = (
        agg["total_carries"] / agg["minutes_played"] * 90
    )
    agg["Progressive_Carries_per_90"] = (
        agg["progressive_carries"] / agg["minutes_played"] * 90
    )
    agg["Carries_into_FT_per_90"] = (
        agg["carries_into_ft"] / agg["minutes_played"] * 90
    )
    agg["Carry_Distance_per_90"] = (
        agg["total_forward_dist_m"] / agg["minutes_played"] * 90
    )
    agg["xT_via_Carries_per_90"] = (
        agg["xT_via_Carries"] / agg["minutes_played"] * 90
    )

    # Apply noise filters
    agg = agg[agg["minutes_played"] >= filter_minutes]
    agg = agg[agg["total_carries"]  >= filter_carry_count]

    output_cols = [
        player_col, "short_name", "role", "minutes_played",
        "total_carries", "progressive_carries", "carries_into_ft",
        "Carries_per_90", "Progressive_Carries_per_90",
        "Carries_into_FT_per_90", "Carry_Distance_per_90",
        "xT_via_Carries", "xT_via_Carries_per_90", "xT_per_Carry",
    ]

    return (
        agg[output_cols]
        .sort_values("Progressive_Carries_per_90", ascending=False)
        .reset_index(drop=True)
    )
