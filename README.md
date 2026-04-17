# 2025 MLB Swing Decision Analysis

Analyzes MLB batter swing decisions using Statcast bat-tracking, exit velocity, and plate discipline data. Players are segmented into five behavioral archetypes via K-Means clustering.

## Data Sources

| File | Source | Records |
|------|--------|---------|
| `bat-tracking.csv` | Baseball Savant (Statcast) | 226 |
| `exit_velocity.csv` | Baseball Savant (Statcast) | 251 |
| `stats.csv` | Baseball Reference / Statcast | 144 |
| `2-strike-bat-tracking.csv` | Baseball Savant (2-strike filter) | ~144 |

All datasets merge on `player_id`. The inner join on all three season files yields 144 players.

## Key Metrics

- **Bat Speed / Swing Length** — Statcast bat-tracking
- **Squared-Up%, Blast%** — contact quality from bat-tracking
- **Zone Swing% / Chase%** — plate discipline from stats
- **Avg EV, EV95+%, Barrel%** — hit quality from exit velocity
- **Discipline Score** = Zone Swing% − Chase%
- **Power Efficiency** = Avg EV / Bat Speed
- **Chase Efficiency** = Hard Hit% / Chase Rate

## Dashboard

**Live app:** https://2025-mlb-swing-decision-analysis.streamlit.app/

Interactive Streamlit dashboard with Plotly charts, cluster filtering, and player search.

### Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place CSV files in the project root or a `data/` subdirectory. The app detects both automatically.

### Features

- **KPI row** — league avg bat speed, chase rate, hard hit rate, player count; all update with cluster filter
- **PCA scatter** — 2D projection of all 13 features, color-coded by archetype with centroid markers
- **Radar chart** — normalized metric profiles per cluster
- **Cost of Chase map** — chase rate vs hard hit rate, quadrant-labeled with annotated outliers
- **2-Strike scatter** — swing length and whiff rate deltas vs season average, quadrant-labeled
- **Cluster summary table** — mean metrics and representative stars per archetype
- **Player spotlight** — search any player to see their metrics vs cluster and league average as a bar chart; defaults to a league-wide violin distribution

## Notebook

The original exploratory analysis is in `swing_analysis.ipynb`. Run with:

```bash
jupyter notebook swing_analysis.ipynb
```
