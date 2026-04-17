# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Baseball analytics project analyzing batter swing decisions, plate discipline, and clustering players into behavioral archetypes. The original analysis lives in `swing_analysis.ipynb`; the production interface is a Streamlit dashboard (`app.py`).

## Running the Dashboard

Live: https://2025-mlb-swing-decision-analysis.streamlit.app/

```bash
pip install -r requirements.txt
streamlit run app.py
```

CSVs are looked up in `data/` first, then the project root. Placing files in either location works.

## Running the Notebook

```bash
jupyter notebook swing_analysis.ipynb
```

Execute cells sequentially — later cells depend on earlier ones.

## Dependencies

`requirements.txt` covers all runtime deps: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`.

## Data Files

| File | Records | Purpose |
|------|---------|---------|
| `bat-tracking.csv` | 226 | Swing mechanics (bat speed, swing length, contact, whiff rates) |
| `exit_velocity.csv` | 251 | Hit quality (exit velo, barrel rate, distances) |
| `stats.csv` | 144 | Plate discipline (zone/chase/meatball swing rates) |
| `2-strike-bat-tracking.csv` | — | Same schema as bat-tracking, filtered to 2-strike counts |

Datasets are merged on `player_id`. The merged dataset is smaller than any individual source (~144 rows after inner join with stats).

## Architecture / Analysis Pipeline

**1. Data Loading & Merging** — Reads all 3 CSVs and joins on `player_id`.

**2. Feature Engineering** — Three derived metrics are central to the analysis:
- `discipline_score` = `z_swing_percent - oz_swing_percent` (zone selectivity)
- `power_efficiency` = `avg_hit_speed / avg_bat_speed` (bat speed conversion)
- `chase_efficiency` = `ev95percent / (oz_swing_percent / 100)` (value of out-of-zone swings)

**3. Clustering** — StandardScaler normalizes 13 features, then KMeans (k=5, chosen via elbow method) segments players into behavioral archetypes. Features include bat speed, swing length, contact quality metrics, discipline, and power metrics.

**4. Visualization** — Outputs 7 PNGs to `image/`:
- PCA cluster plot, radar charts (cluster profiles), chase vs. power efficiency scatter, 2-strike contact vs. quality, swing length vs. whiff

**5. 2-Strike Analysis** — Separate section using `2-strike-bat-tracking.csv` to examine whiff/contact trade-offs specifically in 2-strike counts.

## Dashboard Architecture (`app.py`)

All data loading and ML run inside Streamlit (no precomputed results). Key caching:
- `@st.cache_data load_data()` — reads CSVs, merges, engineers features, builds `final` (144 rows) and `comp_2s` (2-strike delta frame)
- `@st.cache_data run_clustering()` — StandardScaler → KMeans(k=5) → PCA(2D); returns enriched dataframe, cluster profiles, archetype names, PCA centroids

Cluster names are assigned heuristically in `_assign_cluster_names()` by ranking cluster mean `ev95percent`, `oz_swing_percent`, `whiff_percent`, and `avg_bat_speed`. Each chart function accepts `active_clusters` (list of archetype name strings) and `highlight_player` (search substring) — passing `[]`/`""` shows all data.

Chart functions return `go.Figure` objects; layout is unified through `_base_layout()` and `_axis()` helpers with a shared dark palette. Cluster colors are indexed by cluster integer ID (`CLUSTER_COLORS[cid]`).

The `data/` directory exists but is empty by default — the app falls back to root-level CSVs automatically via `_find_csv()`.
