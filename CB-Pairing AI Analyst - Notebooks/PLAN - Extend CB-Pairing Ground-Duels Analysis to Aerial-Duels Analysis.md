### Replicate Ground-Duel Notebook into Robust Aerial-Duel Notebook (CB, PL 2024)

**Summary**
- Rebuild [CB_Aerial_Duels_Quality+Metrics.ipynb](C:/Users/Pedro/Desktop/Portfolio/12%20Football%20CE%20-%20CB%20Pairing%20AI%20Analyst/Twelve-GPT-Educational-Giorgis/CB-Pairing%20AI%20Analyst%20-%20Notebooks/CB_Aerial_Duels_Quality+Metrics.ipynb) using the same structure/story/code style as [CB_Ground_Duels_Quality+Metrics.ipynb](C:/Users/Pedro/Desktop/Portfolio/12%20Football%20CE%20-%20CB%20Pairing%20AI%20Analyst/Twelve-GPT-Educational-Giorgis/CB-Pairing%20AI%20Analyst%20-%20Notebooks/CB_Ground_Duels_Quality+Metrics.ipynb), but with aerial-duel semantics.
- Keep the full pipeline: player quality table, CB pair-fit table, directional companion-fit table, validation checks, visuals, and CSV exports.
- Use chosen defaults: `First touch + recovery` outcome model and thresholds `>=30 duels` + `>=350 minutes`.

**Implementation Changes**
- Clone the ground notebook flow (headings, section order, documentation style, checks, pairing logic), then replace ground-specific logic/labels with aerial-specific logic.
- Keep data loading identical (`event_data/*.json`, `players.parquet`, `teams.parquet`, `minutes.parquet`, `matches.parquet`) and CB position filter unchanged (`CB`, `LCB`, `RCB`, `LCB3`, `RCB3`).
- Aerial event extraction:
  - Base duel rows: `aerialDuel.firstTouch` non-null.
  - Include `regular_foul` rows for temporal foul linkage.
  - Sort by `matchId`, `matchPeriod`, `id` before shift-based foul linking.
  - Link duel->foul if same player, same match/period, and time delta `<=1.3s`; shift yellow/red card flags from the foul row.
- Duel outcome model (priority order):
  - `foul`
  - `won_possession` if `type.secondary` contains `recovery`
  - `won_first_touch` if `aerialDuel.firstTouch == True` and not `recovery`
  - `lost_duel` if `aerialDuel.firstTouch == False`
- Player metrics:
  - `duel_success_rate = (won_possession + won_first_touch) / total_duels`
  - `possession_win_rate = won_possession / total_duels`
  - `foul_rate`, `discipline`, `card_discipline`
  - `duels_per90`
  - `aerial_wins_per90 = (won_possession + won_first_touch) / minutes_played * 90`
- Z-score/composite quality:
  - Metrics list: `card_discipline`, `discipline`, `possession_win_rate`, `duel_success_rate`, `duels_per90`, `aerial_wins_per90`
  - Keep same weighting framework and coefficient values as ground notebook.
  - Final individual score: `aerial_duel_quality_score` (z-standardized composite).
- Pairing + companion extension:
  - Reuse same pair-generation, weighted components (`quality_raw`, `complement_raw`, `floor_raw`), z-standardization, and `CB_pair_fit_z_score`.
  - Reuse same directional companion logic (`coverage_gain_raw`, `coverage_gain_z_within_anchor`, `companion_fit_score`) and metadata enrichment (dominant team/side, shared overlap, evidence band).
- Update narrative text, plot titles, and metric labels to aerial wording throughout (including DistributionPlot hover labels).

**Public Interfaces / Outputs**
- Player-level export: `data/data-CB_pairing_AI_analyst/Qualities/CB_aerial_duels.csv`
- Pair-level export: `data/data-CB_pairing_AI_analyst/Qualities/CB_aerial_duel_pairs.csv`
- Companion export: `data/data-CB_pairing_AI_analyst/Qualities/CB_aerial_duel_companion_fit.csv`
- Core schema updates vs ground:
  - Replace ground-specific success subcomponent with `won_first_touch`
  - Replace `interceptions_per90`/`z_interceptions_per90` with `aerial_wins_per90`/`z_aerial_wins_per90`
  - Individual quality column name becomes `aerial_duel_quality_score`

**Test Plan**
- Run notebook top-to-bottom in a clean kernel.
- Validate extraction sanity:
  - Non-empty CB aerial duel set.
  - Outcome buckets present and exhaustive under priority logic.
- Validate thresholding:
  - Filters apply exactly (`total_duels >= 30`, `minutes_played >= 350`).
- Re-run and pass adapted assertions:
  - Pair count `n*(n-1)/2`
  - Companion count `n*(n-1)`
  - No NaNs in score/component columns
  - Pair uniqueness and symmetry checks
  - Companion rank completeness (`1..n-1` per anchor)
  - Metric-weight consistency (keys exist; weights sum to 1)
- Confirm CSVs are written and row counts match in-memory DataFrames.
- Verify player and pair DistributionPlot cells render with aerial labels and hover payloads.

**Assumptions**
- Keep pair/companion scoring coefficients unchanged from ground notebook for comparability.
- Keep card-discipline metric included even if low-variance in aerial context.
- Keep same highlighted-player pattern in visuals unless IDs are missing, in which case fall back to top-ranked players.
- Fully replace the currently empty aerial notebook with the complete replicated aerial workflow.
