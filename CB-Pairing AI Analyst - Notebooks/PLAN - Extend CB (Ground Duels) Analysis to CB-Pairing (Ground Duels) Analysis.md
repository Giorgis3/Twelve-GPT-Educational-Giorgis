## Ground-Duel CB Pairing Extension (Robust Fit + Companion Ranking)

### Summary
- **Decision**: partially agree with your approach. Generating all CB combinations and using `importance_metric_weights` is the right base, but **plain averaging alone is not robust enough** for fit.
- **Why**: with your current linear setup, pair averaging collapses to mean individual quality (in this dataset, pair-average quality and weighted pair mean are effectively equivalent), so it does not truly capture complementarity.
- We will extend [CB_Ground_Duels_Quality+Metrics_Sandbox.ipynb](C:/Users/Pedro/Desktop/Portfolio/12%20Football%20CE%20-%20CB%20Pairing%20AI%20Analyst/Twelve-GPT-Educational-Giorgis/CB-Pairing%20AI%20Analyst%20-%20Notebooks/CB_Ground_Duels_Quality+Metrics_Sandbox.ipynb) by keeping `df_ground_duels` intact and adding robust pair/companion outputs.

### Implementation Changes
- Add a new notebook section: **“CB Pairings and Companion Fit”** after `df_ground_duels` creation/export.
- Build all unordered pairs from qualified CBs in `df_ground_duels` (`nC2`, currently 2,701 pairs).
- For each metric in `importance_metric_weights`, compute pair-level components from player z-scores:
  - `pair_mean_m = (A_m + B_m)/2`
  - `pair_best_m = max(A_m, B_m)`
  - `pair_floor_m = min(A_m, B_m)`
- Compute three weighted pair components:
  - `quality_raw = Σ w_m * pair_mean_m`   <==>   weighted CB-pair mean: overall quality of the pair based on the average of both players' z-scores for each metric, weighted by importance.
  - `complement_raw = Σ w_m * (pair_best_m - pair_mean_m)`   <==>   coverage/complement bonus: how well the two players' strengths complement each other (e.g. one is strong in metric A, the other in metric B, so they cover more quality dimensions as a pair).
  - `floor_raw = Σ w_m * pair_floor_m`   <==>   weak-link protection: floor quality of the pair based on the weaker player's z-score for each metric, weighted by importance. This captures the idea that a pair is only as strong as its weaker link in each dimension.
- Standardize components across all pairs (`quality_z`, `complement_z`, `floor_z`) and compute final balanced score:
  - `pair_fit_score = 0.50*quality_z + 0.30*complement_z + 0.20*floor_z`
- Create pair-level metric columns compatible with `DistributionPlot` usage:
  - Raw pair metrics: `duel_success_rate`, `possession_win_rate`, `discipline`, `card_discipline`, `duels_per90`, `interceptions_per90`
  - Pair z-metrics (same naming convention): `z_duel_success_rate`, `z_possession_win_rate`, `z_discipline`, `z_card_discipline`, `z_duels_per90`, `z_interceptions_per90`
- Add metadata/realism columns (not used in score, per your choice):
  - `same_dominant_team`, `team_a_id`, `team_b_id`
  - `dominant_side_a`, `dominant_side_b`, `opposite_side_pair`
  - `shared_matches`, `shared_minutes_overlap`, `evidence_band`
- Build **directional companion-fit** table (ordered pairs, A->B):
  - Anchor deficits: `def_i_m = max(0, -z_i_m)`
  - Coverage gain from partner B on A’s deficits
  - `companion_fit_score = 0.65*pair_fit_score + 0.35*coverage_gain_z_within_anchor`
  - Rank partners per anchor (`companion_rank_for_anchor`)
- Export new artifacts:
  - `CB_ground_duel_pairs.csv`
  - `CB_ground_duel_companion_fit.csv`
  - Keep existing `CB_ground_duels.csv` unchanged.

### Test Plan
- Schema checks:
  - Pair count equals `n*(n-1)/2`
  - Companion table count equals `n*(n-1)`
  - No missing values in score/component columns
- Correctness checks:
  - Pair uniqueness (no permutations duplicated)
  - Symmetry of pair score (`A-B == B-A`)
  - Partner ranking uniqueness per anchor (1..n-1, no gaps)
- Metric/weight checks:
  - `importance_metric_weights` keys exist in z-metric columns
  - Weights sum to 1.0
- Compatibility check with `visual_Sandbox.py`:
  - Create one `DistributionPlot` example using pair-level z columns and raw-value mappings
  - Confirm annotations render pair raw values correctly.
- Sanity outputs in notebook:
  - Top 15 global pairs by `pair_fit_score`
  - Top 10 companions for a chosen anchor CB
  - Top same-team pairs filtered from all-pairs table.

### Assumptions and Defaults
- Use all pair combinations as primary universe (selected).
- Use balanced fit blend (selected): quality + complementarity + weak-link protection.
- Do **not** apply shared-minute evidence in scoring (selected), but keep evidence columns for filtering/interpretation.
- Equal contribution of both players to pair metrics (not minute-weighted at pair-score stage).
- Existing `df_ground_duels` pipeline and downstream functionality remain backward compatible.
