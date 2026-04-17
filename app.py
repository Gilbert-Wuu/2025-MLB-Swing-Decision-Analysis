import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from streamlit_searchbox import st_searchbox
import os, warnings
warnings.filterwarnings("ignore")


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2025 MLB Swing Decision Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#FF6B35", "#4ECDC4", "#FFE66D", "#A78BFA", "#6EE7B7"]

FEATURES = [
    "avg_bat_speed", "swing_length", "squared_up_per_swing", "blast_per_swing",
    "z_swing_percent", "oz_swing_percent", "whiff_percent", "meatball_swing_percent",
    "avg_hit_speed", "ev95percent", "brl_percent", "discipline_score", "power_efficiency",
]

RADAR_FEATURES = [
    "avg_bat_speed", "discipline_score", "ev95percent", "brl_percent", "whiff_percent",
]
RADAR_LABELS = [
    "Bat Speed", "Discipline", "Hard Hit%", "Barrel%", "Whiff%",
]

KNOWN_STARS = [
    "Judge, Aaron", "Ohtani, Shohei", "Soto, Juan", "Guerrero Jr., Vladimir",
    "Schwarber, Kyle", "Betts, Mookie", "Harper, Bryce", "Stanton, Giancarlo",
    "Arraez, Luis", "Kwan, Steven", "Cruz, Oneil", "Bichette, Bo",
    "Springer, George", "Perez, Salvador", "Correa, Carlos", "Reynolds, Bryan",
    "Alvarez, Yordan", "Acuna Jr., Ronald", "Freeman, Freddie",
]

CHASE_ANNOTATE = ["Soto, Juan", "Judge, Aaron", "Perez, Salvador", "Ohtani, Shohei"]
TWOSTRIKE_ANNOTATE = ["Correa, Carlos", "Soto, Juan", "Judge, Aaron", "Guerrero Jr., Vladimir"]

CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#1E2935"
TEXT_COLOR = "#8B9AB5"
AXIS_COLOR = "#1E2935"

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg: #07090F;
        --bg-card: #0D1117;
        --bg-card2: #111923;
        --border: #1E2935;
        --border-hi: #2A3E52;
        --text: #C9D1D9;
        --text-dim: #8B9AB5;
        --text-faint: #3D4F63;
        --gold: #E6A817;
        --gold-dim: #9A7010;
        --green: #3FB950;
        --red: #F85149;
        --c0: #FF6B35;
        --c1: #4ECDC4;
        --c2: #FFE66D;
        --c3: #A78BFA;
        --c4: #6EE7B7;
    }

    /* ── Base ── */
    .stApp { background: var(--bg) !important; font-family: 'Barlow', sans-serif; color: var(--text); }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    .block-container { padding: 1.5rem 2.5rem 2rem !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none; }

    /* ── Title block ── */
    .dash-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        letter-spacing: 0.06em;
        color: var(--text);
        line-height: 0.95;
        margin: 0;
    }
    .dash-title em { font-style: normal; color: var(--gold); }
    .dash-sub {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-top: 0.35rem;
    }

    /* ── Horizontal rule ── */
    .dash-rule { border: none; border-top: 1px solid var(--border); margin: 1rem 0 0.75rem; }

    /* ── KPI cards ── */
    .kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.75rem; margin-bottom: 1rem; }
    .kpi {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.1rem 1.3rem 1rem;
        position: relative;
        overflow: hidden;
        min-height: 108px;
    }
    .kpi::after {
        content: '';
        position: absolute;
        inset: 0 0 auto 0;
        height: 2px;
        background: linear-gradient(90deg, var(--gold) 0%, transparent 100%);
    }
    .kpi-label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.45rem;
    }
    .kpi-val {
        font-family: 'DM Mono', monospace;
        font-size: 2.2rem;
        font-weight: 500;
        color: var(--text);
        line-height: 1;
    }
    .kpi-trend {
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }
    .kpi-trend.pos { color: var(--green); }
    .kpi-trend.neg { color: var(--red); }
    .kpi-trend.neu { color: var(--text-dim); }

    /* ── Section labels ── */
    .section-label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.5rem;
    }

    /* ── Chart wrapper ── */
    .chart-wrap {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0 0 0.25rem;
        overflow: hidden;
    }
    .chart-header {
        padding: 0.85rem 1.1rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.25rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .chart-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-dim);
        padding-bottom: 0.7rem;
    }
    .chart-expand-btn {
        background: none;
        border: none;
        color: var(--text-dim);
        cursor: pointer;
        padding: 2px;
        opacity: 0.35;
        transition: opacity 0.18s, color 0.18s;
        flex-shrink: 0;
        line-height: 0;
        border-radius: 3px;
        margin-top: 1px;
    }
    .chart-expand-btn:hover { opacity: 1; color: var(--text); }

    /* Chart modal backdrop */
    .chart-backdrop {
        position: fixed !important;
        inset: 0 !important;
        z-index: 9997 !important;
        background: rgba(7, 9, 15, 0.88) !important;
        backdrop-filter: blur(3px) !important;
        cursor: pointer !important;
    }
    body.chart-open { overflow: hidden; }

    /* Expanded chart panel */
    [data-testid="stVerticalBlock"].chart-expanded {
        position: fixed !important;
        inset: 28px !important;
        z-index: 9998 !important;
        background: #0D1117 !important;
        border: 1px solid #1E2935 !important;
        border-radius: 10px !important;
        padding: 0 8px 8px !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stVerticalBlock"].chart-expanded [data-testid="stElementContainer"]:nth-child(2) {
        flex: 1 !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    /* ── Cluster badge ── */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 12px;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* ── Bottom cards ── */
    .bottom-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.1rem 1.2rem 1rem;
        min-height: 320px;
    }

    /* ── Streamlit widget overrides ── */
    /* Outer baseweb wrappers */
    div[data-testid="stMultiSelect"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"]   div[data-baseweb="select"] {
        background: #111923 !important;
    }
    /* The actual visible control box (one level inside baseweb select) */
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"]   div[data-baseweb="select"] > div {
        background: #111923 !important;
        border-color: #1E2935 !important;
        border-radius: 4px !important;
        min-height: 42px !important;
    }
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:hover,
    div[data-testid="stSelectbox"]   div[data-baseweb="select"] > div:hover {
        border-color: #2A3E52 !important;
    }
    /* Dropdown menu (baseweb popover) */
    div[data-baseweb="popover"] {
        background: #0D1117 !important;
        border: 1px solid #1E2935 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    div[data-baseweb="popover"] ul,
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background: #0D1117 !important;
    }
    div[data-baseweb="popover"] li {
        background: #0D1117 !important;
        color: #C9D1D9 !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background: #1E2935 !important;
    }
    /* Text/placeholder inside selects */
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] span,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] div,
    div[data-testid="stSelectbox"]   div[data-baseweb="select"] span,
    div[data-testid="stSelectbox"]   div[data-baseweb="select"] div { color: #C9D1D9 !important; }
    /* Multi-select tags */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: #2A3E52 !important;
        color: #C9D1D9 !important;
    }
    /* Widget labels */
    label[data-testid="stWidgetLabel"] p {
        color: #8B9AB5 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-family: 'Barlow Condensed', sans-serif !important;
    }

    /* plotly charts background */
    .js-plotly-plot .plotly { background: transparent !important; }
    .plot-container { background: transparent !important; }

    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _find_csv(name):
    for base in ("data", "."):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in data/ or project root")


@st.cache_data(show_spinner=False)
def load_data():
    df_bat   = pd.read_csv(_find_csv("bat-tracking.csv"), encoding="utf-8-sig")
    df_ev    = pd.read_csv(_find_csv("exit_velocity.csv"), encoding="utf-8-sig")
    df_stats = pd.read_csv(_find_csv("stats.csv"), encoding="utf-8-sig")
    df_2s    = pd.read_csv(_find_csv("2-strike-bat-tracking.csv"), encoding="utf-8-sig")

    # Normalise ID column
    df_bat.rename(columns={"id": "player_id"}, inplace=True)
    df_2s.rename(columns={"id": "player_id"}, inplace=True)
    for d in (df_bat, df_ev, df_stats, df_2s):
        d["player_id"] = d["player_id"].astype(str).str.strip()

    # Merge season data
    merged = pd.merge(df_stats, df_ev, on="player_id", how="inner", suffixes=("", "_ev"))
    final  = pd.merge(merged, df_bat, on="player_id", how="inner", suffixes=("", "_bt"))

    keep = [
        "player_id", "last_name, first_name",
        "avg_bat_speed", "swing_length", "squared_up_per_swing", "blast_per_swing",
        "z_swing_percent", "oz_swing_percent", "whiff_percent", "meatball_swing_percent",
        "avg_hit_speed", "ev95percent", "brl_percent",
        "hard_swing_rate",
    ]
    final = final[[c for c in keep if c in final.columns]].dropna()

    # Feature engineering
    final["avg_bat_speed"]   = pd.to_numeric(final["avg_bat_speed"],   errors="coerce")
    final["avg_hit_speed"]   = pd.to_numeric(final["avg_hit_speed"],   errors="coerce")
    final["discipline_score"]  = final["z_swing_percent"]  - final["oz_swing_percent"]
    final["power_efficiency"]  = final["avg_hit_speed"]    / final["avg_bat_speed"]
    final["chase_efficiency"]  = final["ev95percent"]      / (final["oz_swing_percent"] / 100)
    final.dropna(inplace=True)
    final.reset_index(drop=True, inplace=True)

    # 2-strike merge
    comp_2s = pd.merge(
        final,
        df_2s[["player_id", "avg_bat_speed", "swing_length", "whiff_per_swing", "squared_up_per_swing"]],
        on="player_id", suffixes=("_avg", "_2s"), how="inner",
    )
    comp_2s["delta_length_pct"] = (
        (comp_2s["swing_length_2s"] - comp_2s["swing_length_avg"]) / comp_2s["swing_length_avg"] * 100
    )
    comp_2s["delta_speed_pct"] = (
        (comp_2s["avg_bat_speed_2s"] - comp_2s["avg_bat_speed_avg"]) / comp_2s["avg_bat_speed_avg"] * 100
    )
    comp_2s["delta_whiff"] = (
        comp_2s["whiff_per_swing"] - (comp_2s["whiff_percent"] / 100)
    )
    comp_2s["delta_squared_up"] = (
        comp_2s["squared_up_per_swing_2s"] - comp_2s["squared_up_per_swing_avg"]
    )
    comp_2s.dropna(inplace=True)

    return final, comp_2s


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_clustering(final_hash: str, _final: pd.DataFrame):
    feat_cols = [f for f in FEATURES if f in _final.columns]
    X = _final[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    df = _final.copy()
    df["cluster"]  = labels
    df["pca_x"]    = X_pca[:, 0]
    df["pca_y"]    = X_pca[:, 1]

    profiles = df.groupby("cluster")[feat_cols].mean()
    cluster_names = _assign_cluster_names(profiles)
    df["archetype"] = df["cluster"].map(cluster_names)

    return df, profiles, cluster_names, centers_pca, scaler, feat_cols


def _assign_cluster_names(profiles: pd.DataFrame) -> dict:
    p = profiles.copy()
    assigned, names = set(), {}
    remaining = list(p.index)

    def pick(col, maximize=True, label=""):
        nonlocal remaining
        sub = p.loc[remaining, col]
        idx = sub.idxmax() if maximize else sub.idxmin()
        names[idx] = label
        assigned.add(idx)
        remaining.remove(idx)

    pick("ev95percent",      True,  "Elite Power")
    pick("oz_swing_percent", True,  "Aggressive Swingers")
    pick("whiff_percent",    False, "Contact Specialists")
    pick("avg_bat_speed",    True,  "Power Strikeout")
    for idx in remaining:
        names[idx] = "Balanced Hitters"

    return names


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _base_layout(**kwargs):
    base = dict(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Barlow, sans-serif", color=TEXT_COLOR, size=13),
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(size=12, color=TEXT_COLOR),
        ),
        margin=dict(l=48, r=20, t=28, b=44),
        hoverlabel=dict(
            bgcolor="#111923",
            bordercolor="#2A3E52",
            font=dict(color="#C9D1D9", family="Barlow, sans-serif", size=12),
        ),
    )
    base.update(kwargs)
    return base


def _axis(title="", **kw):
    return dict(
        title=dict(text=title, font=dict(size=13, color=TEXT_COLOR)),
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        linecolor=GRID_COLOR,
        tickfont=dict(size=12, color=TEXT_COLOR),
        **kw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 – PCA SCATTER
# ─────────────────────────────────────────────────────────────────────────────
def chart_pca(df: pd.DataFrame, cluster_names: dict, centers_pca, active_clusters, highlight_player):
    fig = go.Figure()

    for cid, color in enumerate(CLUSTER_COLORS):
        if active_clusters and cluster_names.get(cid) not in active_clusters:
            continue
        mask = df["cluster"] == cid
        sub  = df[mask]

        # Dim non-highlighted players when a search is active
        opacity = 0.72 if not highlight_player else 0.35

        hover = (
            "<b>%{customdata[0]}</b><br>"
            + cluster_names.get(cid, f"Cluster {cid}") + "<br>"
            + "Bat Speed: %{customdata[1]:.1f} mph<br>"
            + "Hard Hit%: %{customdata[2]:.1f}%<br>"
            + "Chase%: %{customdata[3]:.1f}%"
            + "<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=sub["pca_x"], y=sub["pca_y"],
            mode="markers",
            marker=dict(color=color, size=7, opacity=opacity,
                        line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
            name=cluster_names.get(cid, f"Cluster {cid}"),
            customdata=sub[["last_name, first_name", "avg_bat_speed", "ev95percent", "oz_swing_percent"]].values,
            hovertemplate=hover,
        ))

    # Highlighted player
    if highlight_player:
        hi = df[df["last_name, first_name"].str.contains(highlight_player, case=False, na=False)]
        if not hi.empty:
            cid = int(hi.iloc[0]["cluster"])
            fig.add_trace(go.Scatter(
                x=hi["pca_x"], y=hi["pca_y"],
                mode="markers",
                marker=dict(color=CLUSTER_COLORS[cid], size=14, opacity=1.0,
                            line=dict(color="white", width=2)),
                name=hi.iloc[0]["last_name, first_name"],
                hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
                customdata=hi[["last_name, first_name"]].values,
            ))

    # Centroids — label with the player closest to each centroid
    for cid, (cx, cy) in enumerate(centers_pca):
        if active_clusters and cluster_names.get(cid) not in active_clusters:
            continue
        sub = df[df["cluster"] == cid]
        dists = np.sqrt((sub["pca_x"] - cx) ** 2 + (sub["pca_y"] - cy) ** 2)
        closest = sub.iloc[dists.values.argmin()]["last_name, first_name"]
        # Display as "First Last"
        parts = closest.split(",")
        label = (parts[1].strip() + " " + parts[0].strip()) if len(parts) == 2 else closest
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(symbol="x", color=CLUSTER_COLORS[cid], size=14,
                        line=dict(color="white", width=1.5)),
            text=[label],
            textposition="top center",
            textfont=dict(size=9, color=CLUSTER_COLORS[cid],
                          family="Barlow Condensed, sans-serif"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        **_base_layout(
            xaxis=_axis("PC1"),
            yaxis=_axis("PC2"),
            height=390,
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 – RADAR
# ─────────────────────────────────────────────────────────────────────────────
def chart_radar(profiles: pd.DataFrame, cluster_names: dict, active_clusters,
                df: pd.DataFrame = None):
    available = [f for f in RADAR_FEATURES if f in profiles.columns]
    labels_used = [RADAR_LABELS[i] for i, f in enumerate(RADAR_FEATURES) if f in profiles.columns]

    # Normalise against all-player range so cluster means map meaningfully:
    # 0 = player-level min, 1 = player-level max, 0.5 ≈ league average.
    # Using only cluster-level min/max would collapse Contact Specialists
    # (which is genuinely the lowest cluster on power metrics) to r=0 everywhere.
    if df is not None and all(f in df.columns for f in available):
        norm_min = df[available].min()
        norm_max = df[available].max()
    else:
        norm_min = profiles[available].min()
        norm_max = profiles[available].max()
    norm = (profiles[available] - norm_min) / (norm_max - norm_min + 1e-9)
    norm = norm.clip(0, 1)

    fig = go.Figure()
    theta = labels_used + [labels_used[0]]

    for cid in profiles.index:
        name = cluster_names.get(cid, f"Cluster {cid}")
        if active_clusters and name not in active_clusters:
            continue
        vals = norm.loc[cid, available].tolist()
        vals_closed = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=theta,
            fill="toself",
            fillcolor=_hex_rgba(CLUSTER_COLORS[cid], 0.16),
            line=dict(color=CLUSTER_COLORS[cid], width=2),
            name=name,
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                    tickfont=dict(size=8, color=TEXT_COLOR),
                    tickvals=[0.25, 0.5, 0.75],
                ),
                angularaxis=dict(
                    gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                    tickfont=dict(size=13, color=TEXT_COLOR),
                ),
            ),
            height=390,
            margin=dict(l=48, r=48, t=28, b=28),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 – COST OF CHASE
# ─────────────────────────────────────────────────────────────────────────────
def chart_chase(df: pd.DataFrame, cluster_names: dict, active_clusters, highlight_player):
    med_chase = df["oz_swing_percent"].median()
    med_power = df["ev95percent"].median()

    fig = go.Figure()

    for cid, color in enumerate(CLUSTER_COLORS):
        name = cluster_names.get(cid, f"Cluster {cid}")
        if active_clusters and name not in active_clusters:
            continue
        sub     = df[df["cluster"] == cid]
        opacity = 0.72 if not highlight_player else 0.35
        fig.add_trace(go.Scatter(
            x=sub["oz_swing_percent"], y=sub["ev95percent"],
            mode="markers",
            marker=dict(color=color, size=7, opacity=opacity,
                        line=dict(color="rgba(0,0,0,0.25)", width=0.5)),
            name=name,
            customdata=sub[["last_name, first_name", "oz_swing_percent", "ev95percent"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Chase%: %{customdata[1]:.1f}%<br>"
                "Hard Hit%: %{customdata[2]:.1f}%<br>"
                + name + "<extra></extra>"
            ),
        ))

    # Highlighted player
    if highlight_player:
        hi = df[df["last_name, first_name"].str.contains(highlight_player, case=False, na=False)]
        if not hi.empty:
            cid = int(hi.iloc[0]["cluster"])
            fig.add_trace(go.Scatter(
                x=hi["oz_swing_percent"], y=hi["ev95percent"],
                mode="markers",
                marker=dict(color=CLUSTER_COLORS[cid], size=14,
                            line=dict(color="white", width=2)),
                showlegend=False, hoverinfo="skip",
            ))

    # Reference lines
    fig.add_hline(y=med_power, line=dict(color="#3D4F63", dash="dash", width=1))
    fig.add_vline(x=med_chase, line=dict(color="#3D4F63", dash="dash", width=1))

    x_min = df["oz_swing_percent"].min()
    x_max = df["oz_swing_percent"].max()
    y_min = df["ev95percent"].min()
    y_max = df["ev95percent"].max()

    # Quadrant labels
    quad_labels = [
        (x_min + 0.3, y_max - 1.5, "PATIENT & POWERFUL", "top left"),
        (x_max - 0.3, y_max - 1.5, "BAD BALL HITTERS",   "top right"),
        (x_min + 0.3, y_min + 1.0, "PATIENT BUT WEAK",   "bottom left"),
        (x_max - 0.3, y_min + 1.0, "STRUGGLING",         "bottom right"),
    ]
    for qx, qy, ql, align in quad_labels:
        ha = "right" if "right" in align else "left"
        fig.add_annotation(
            x=qx, y=qy, text=ql,
            showarrow=False,
            font=dict(size=8, color="#3D4F63", family="Barlow Condensed, sans-serif"),
            xanchor=ha,
        )

    # Named annotations
    for player in CHASE_ANNOTATE:
        row = df[df["last_name, first_name"] == player]
        if row.empty:
            continue
        r = row.iloc[0]
        display = r["last_name, first_name"].split(",")[1].strip() + " " + r["last_name, first_name"].split(",")[0].strip()
        fig.add_annotation(
            x=r["oz_swing_percent"], y=r["ev95percent"],
            text=display,
            showarrow=True,
            arrowhead=0, arrowcolor="#3D4F63", arrowwidth=1,
            ax=20, ay=-18,
            font=dict(size=9, color="#C9D1D9", family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="#2A3E52", borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        **_base_layout(
            xaxis=_axis("Chase Rate (Out-of-Zone Swing %)", range=[x_min - 1, x_max + 1]),
            yaxis=_axis("Hard Hit Rate (EV95+%)", range=[y_min - 1.5, y_max + 2]),
            height=390,
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 – 2-STRIKE SCATTER
# ─────────────────────────────────────────────────────────────────────────────
def chart_2strike(comp_2s: pd.DataFrame, cluster_names: dict, active_clusters, highlight_player):
    fig = go.Figure()

    name_col = "last_name, first_name"

    for cid, color in enumerate(CLUSTER_COLORS):
        name = cluster_names.get(cid, f"Cluster {cid}")
        if active_clusters and name not in active_clusters:
            continue
        sub     = comp_2s[comp_2s["cluster"] == cid]
        opacity = 0.72 if not highlight_player else 0.35
        fig.add_trace(go.Scatter(
            x=sub["delta_length_pct"], y=sub["delta_whiff"],
            mode="markers",
            marker=dict(color=color, size=7, opacity=opacity,
                        line=dict(color="rgba(0,0,0,0.25)", width=0.5)),
            name=name,
            customdata=sub[[name_col, "delta_length_pct", "delta_whiff"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Δ Swing Length: %{customdata[1]:.2f}%<br>"
                "Δ Whiff%: %{customdata[2]:.3f}<br>"
                + name + "<extra></extra>"
            ),
        ))

    # Highlighted player
    if highlight_player:
        hi = comp_2s[comp_2s[name_col].str.contains(highlight_player, case=False, na=False)]
        if not hi.empty:
            cid = int(hi.iloc[0]["cluster"])
            fig.add_trace(go.Scatter(
                x=hi["delta_length_pct"], y=hi["delta_whiff"],
                mode="markers",
                marker=dict(color=CLUSTER_COLORS[cid], size=14,
                            line=dict(color="white", width=2)),
                showlegend=False, hoverinfo="skip",
            ))

    # Quadrant reference lines
    fig.add_hline(y=0, line=dict(color="#3D4F63", dash="dash", width=1))
    fig.add_vline(x=0, line=dict(color="#3D4F63", dash="dash", width=1))

    xl = comp_2s["delta_length_pct"].quantile(0.02)
    xh = comp_2s["delta_length_pct"].quantile(0.98)
    yl = comp_2s["delta_whiff"].quantile(0.02)
    yh = comp_2s["delta_whiff"].quantile(0.98)

    quad_labels = [
        (xl + 0.2, yh - 0.003, "TACTICAL ADJUSTERS",  "top left"),
        (xh - 0.2, yh - 0.003, "STUBBORN AGGRESSORS", "top right"),
        (xl + 0.2, yl + 0.003, "ELITE ADJUSTERS",     "bottom left"),
        (xh - 0.2, yl + 0.003, "WORSENED",            "bottom right"),
    ]
    for qx, qy, ql, align in quad_labels:
        ha = "right" if "right" in align else "left"
        fig.add_annotation(
            x=qx, y=qy, text=ql,
            showarrow=False,
            font=dict(size=8, color="#3D4F63", family="Barlow Condensed, sans-serif"),
            xanchor=ha,
        )

    # Named annotations
    for player in TWOSTRIKE_ANNOTATE:
        row = comp_2s[comp_2s[name_col] == player]
        if row.empty:
            continue
        r = row.iloc[0]
        parts = r[name_col].split(",")
        display = parts[1].strip() + " " + parts[0].strip() if len(parts) == 2 else r[name_col]
        fig.add_annotation(
            x=r["delta_length_pct"], y=r["delta_whiff"],
            text=display,
            showarrow=True,
            arrowhead=0, arrowcolor="#3D4F63", arrowwidth=1,
            ax=22, ay=-18,
            font=dict(size=9, color="#C9D1D9", family="Barlow Condensed, sans-serif"),
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="#2A3E52", borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        **_base_layout(
            xaxis=_axis("Δ Swing Length (%)"),
            yaxis=_axis("Δ Whiff Rate"),
            height=390,
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def chart_cluster_table(df: pd.DataFrame, profiles: pd.DataFrame, cluster_names: dict, active_clusters):
    rows = []
    for cid in sorted(cluster_names):
        name = cluster_names[cid]
        sub  = df[df["cluster"] == cid]
        n    = len(sub)
        spd  = profiles.loc[cid, "avg_bat_speed"] if "avg_bat_speed" in profiles.columns else 0
        chs  = profiles.loc[cid, "oz_swing_percent"] if "oz_swing_percent" in profiles.columns else 0
        hhit = profiles.loc[cid, "ev95percent"] if "ev95percent" in profiles.columns else 0

        stars_in = sub[sub["last_name, first_name"].isin(KNOWN_STARS)]["last_name, first_name"].tolist()
        star_str = ", ".join(
            s.split(",")[1].strip() + " " + s.split(",")[0].strip()
            for s in stars_in[:3]
        ) if stars_in else "—"

        rows.append({
            "cid": cid, "Archetype": name, "N": n,
            "Bat Spd": f"{spd:.1f}", "Chase%": f"{chs:.1f}",
            "Hard Hit%": f"{hhit:.1f}", "Stars": star_str,
        })

    df_tbl = pd.DataFrame(rows)

    highlight_mask = [
        (active_clusters and r["Archetype"] in active_clusters)
        for _, r in df_tbl.iterrows()
    ]
    fill_colors = []
    for i, row in df_tbl.iterrows():
        cid   = row["cid"]
        color = CLUSTER_COLORS[cid]
        if highlight_mask[i]:
            fill_colors.append(_hex_rgba(color, 0.19))
        else:
            fill_colors.append("#0D1117")

    header_vals = ["Archetype", "N", "Bat Spd", "Chase%", "Hard Hit%", "Stars"]
    cell_vals   = [
        df_tbl["Archetype"].tolist(),
        df_tbl["N"].tolist(),
        df_tbl["Bat Spd"].tolist(),
        df_tbl["Chase%"].tolist(),
        df_tbl["Hard Hit%"].tolist(),
        df_tbl["Stars"].tolist(),
    ]

    fig = go.Figure(go.Table(
        columnwidth=[150, 40, 70, 70, 80, 200],
        header=dict(
            values=[f"<b>{v}</b>" for v in header_vals],
            fill_color="#0D1117",
            align="left",
            font=dict(color=TEXT_COLOR, family="Barlow Condensed, sans-serif",
                      size=13),
            line_color=GRID_COLOR,
            height=32,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[fill_colors] * len(cell_vals),
            align="left",
            font=dict(color=TEXT_COLOR, family="Barlow, sans-serif", size=13),
            line_color=GRID_COLOR,
            height=34,
        ),
    ))

    fig.update_layout(
        paper_bgcolor=CHART_BG,
        margin=dict(l=0, r=0, t=8, b=0),
        height=240,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER SPOTLIGHT
# ─────────────────────────────────────────────────────────────────────────────
def chart_spotlight(df: pd.DataFrame, profiles: pd.DataFrame,
                    cluster_names: dict, player_query: str):
    display_cols = [
        "avg_bat_speed", "swing_length", "squared_up_per_swing", "blast_per_swing",
        "z_swing_percent", "oz_swing_percent", "whiff_percent", "meatball_swing_percent",
        "avg_hit_speed", "ev95percent", "brl_percent", "discipline_score", "power_efficiency",
    ]
    available = [c for c in display_cols if c in df.columns]
    col_labels = {
        "avg_bat_speed": "Bat Speed", "swing_length": "Swing Length",
        "squared_up_per_swing": "Squared-Up%", "blast_per_swing": "Blast%",
        "z_swing_percent": "Zone Swing%", "oz_swing_percent": "Chase%",
        "whiff_percent": "Whiff%", "meatball_swing_percent": "Meatball Swing%",
        "avg_hit_speed": "Avg EV", "ev95percent": "Hard Hit%",
        "brl_percent": "Barrel%", "discipline_score": "Discipline Score",
        "power_efficiency": "Power Efficiency",
    }
    labels = [col_labels.get(c, c) for c in available]

    if not player_query:
        # Default: violin chart — z-score each metric so mixed-scale columns
        # (e.g. Bat Speed ~70 vs Blast% ~0.07) are all visually comparable
        # without distorting distribution shapes (min-max would; percentile-rank
        # would flatten everything to uniform). The hover shows original values.
        SCALE_100 = {"squared_up_per_swing", "blast_per_swing"}
        fig = go.Figure()
        cols8  = available[:8]
        labels8 = labels[:8]
        for i, (col, lbl) in enumerate(zip(cols8, labels8)):
            raw = df[col].dropna().values
            # Scale fractions to % for display in tooltip
            disp = raw * 100 if col in SCALE_100 else raw
            std = raw.std()
            z = (raw - raw.mean()) / std if std > 0 else raw - raw.mean()

            # Build clean hover text using original (unscaled) values
            unit = "%" if (col.endswith("percent") or col.endswith("_percent")
                           or col in SCALE_100 or "swing" in col or "brl" in col
                           or col == "ev95percent" or col == "whiff_percent") else ""
            q1, med, q3 = np.percentile(disp, [25, 50, 75])
            hover_txt = (
                f"<b>{lbl}</b><br>"
                f"Mean:   {disp.mean():.1f}{unit}<br>"
                f"Median: {med:.1f}{unit}<br>"
                f"IQR:    {q1:.1f} – {q3:.1f}{unit}<br>"
                f"Range:  {disp.min():.1f} – {disp.max():.1f}{unit}"
            )

            # Explicit categorical x so the scatter overlay aligns correctly
            fig.add_trace(go.Violin(
                x=[lbl] * len(z),
                y=z,
                name=lbl,
                fillcolor=_hex_rgba(CLUSTER_COLORS[i % len(CLUSTER_COLORS)], 0.25),
                line_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                box_visible=True,
                meanline_visible=True,
                points=False,
                hoverinfo="skip",
                showlegend=False,
            ))
            # Dense near-invisible markers spanning the violin's y range —
            # opacity=0.001 keeps them visually invisible but hoverable by Plotly
            n_pts = 30
            y_pts = list(np.linspace(float(z.min()) - 0.1, float(z.max()) + 0.1, n_pts))
            fig.add_trace(go.Scatter(
                x=[lbl] * n_pts,
                y=y_pts,
                mode="markers",
                marker=dict(size=18, color="#ffffff", opacity=0.001),
                hoverinfo="text",
                hovertext=hover_txt,
                showlegend=False,
                name="",
            ))

        # ±1σ reference band
        fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(255,255,255,0.03)",
                      line_width=0, layer="below")
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.18)",
                      line_width=1)

        fig.update_layout(
            **_base_layout(
                showlegend=False,
                xaxis=dict(tickfont=dict(size=9, color=TEXT_COLOR),
                           gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
                yaxis=dict(
                    title=dict(text="← Below avg  |  Above avg →",
                               font=dict(size=10, color="#8B9AB5")),
                    tickfont=dict(size=9, color="#8B9AB5"),
                    gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                    zeroline=False,
                ),
                height=270,
                margin=dict(l=40, r=10, t=10, b=60),
            )
        )
        return fig, None, None

    # Player found
    match = df[df["last_name, first_name"].str.contains(player_query, case=False, na=False)]
    if match.empty:
        return None, None, None

    row     = match.iloc[0]
    cid     = int(row["cluster"])
    name    = row["last_name, first_name"]
    archname = cluster_names.get(cid, f"Cluster {cid}")
    color   = CLUSTER_COLORS[cid]

    # squared_up_per_swing and blast_per_swing are stored as fractions (0–1);
    # multiply by 100 so they display on the same 0–100 scale as other % metrics.
    SCALE_100 = {"squared_up_per_swing", "blast_per_swing"}
    def _sv(series_or_scalar, col):
        v = series_or_scalar[col] if hasattr(series_or_scalar, '__getitem__') else series_or_scalar
        return float(v) * 100 if col in SCALE_100 else float(v)

    player_vals  = [_sv(row, c) for c in available]
    cluster_vals = [_sv(profiles.loc[cid], c) if c in profiles.columns else np.nan for c in available]
    league_vals  = [df[c].mean() * 100 if c in SCALE_100 else df[c].mean() for c in available]

    display_name = name.split(",")[1].strip() if "," in name else name

    fig = go.Figure()
    # Bar 1 — player: full cluster color, solid
    fig.add_trace(go.Bar(
        x=player_vals, y=labels,
        orientation="h", name=display_name,
        marker=dict(color=color, line=dict(color="rgba(0,0,0,0)", width=0)),
        opacity=1.0,
    ))
    # Bar 2 — cluster avg: white, semi-transparent (high contrast against dark bg)
    fig.add_trace(go.Bar(
        x=cluster_vals, y=labels,
        orientation="h", name=f"{archname} Avg",
        marker=dict(color="#E8EDF5", line=dict(color="rgba(0,0,0,0)", width=0)),
        opacity=0.45,
    ))
    # Bar 3 — league avg: teal accent, clearly different hue
    fig.add_trace(go.Bar(
        x=league_vals, y=labels,
        orientation="h", name="League Avg",
        marker=dict(color="#4ECDC4", line=dict(color="rgba(0,0,0,0)", width=0)),
        opacity=0.6,
    ))

    fig.update_layout(
        **_base_layout(
            barmode="group",
            xaxis=_axis("Value"),
            yaxis=dict(
                tickfont=dict(size=12, color="#C9D1D9"),
                gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                automargin=True,
            ),
            height=420,
            # top margin gives legend room above the plot area
            margin=dict(l=130, r=20, t=48, b=36),
            legend=dict(
                orientation="h", y=1.08, x=0,
                font=dict(size=12, color="#C9D1D9"), bgcolor="rgba(0,0,0,0)",
                traceorder="normal",
            ),
        )
    )
    return fig, archname, color


# ─────────────────────────────────────────────────────────────────────────────
# KPI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _kpi_html(label, value, trend_val=None, trend_label="", direction="neutral"):
    cls = "pos" if direction == "pos" else ("neg" if direction == "neg" else "neu")
    if trend_val is None:
        trend_row = ""
    else:
        sign = "+" if trend_val >= 0 else ""
        trend_row = f'<div class="kpi-trend {cls}">{sign}{trend_val:.2f} {trend_label}</div>'
    # Single-line HTML avoids blank lines that Streamlit's Markdown parser treats as code blocks
    return (
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'{trend_row}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    # Inject CSS into the searchbox iframe via JS (st.markdown strips <script> tags;
    # st.components.v1.html runs inside its own iframe but can reach sibling iframes
    # via window.parent since all are same-origin on localhost).
    st.components.v1.html("""
    <script>
    (function patch() {
      var CSS = [
        'body{background:#07090F!important;margin:0}',
        'div[class*="-menu"]{background:#0D1117!important;border:1px solid #1E2935!important;border-radius:4px!important;box-shadow:none!important}',
      ].join('');
      function inject(doc) {
        if (!doc || !doc.head || doc.head.querySelector('[data-sb-patch]')) return;
        var s = doc.createElement('style');
        s.setAttribute('data-sb-patch','1');
        s.textContent = CSS;
        doc.head.appendChild(s);
      }
      function run() {
        try {
          var iframes = window.parent.document.querySelectorAll('iframe[src*="searchbox"]');
          iframes.forEach(function(f){ try{ inject(f.contentDocument); }catch(e){} });
        } catch(e){}
      }
      run();
      var obs = new MutationObserver(run);
      try{ obs.observe(window.parent.document.body,{childList:true,subtree:true}); }catch(e){}
    })();

    // ── Chart expand/collapse ──────────────────────────────────────────────
    (function setupChartExpand() {
      var win = window.parent;
      var doc = win.document;

      function collapseChart() {
        var exp = doc.querySelector('[data-testid="stVerticalBlock"].chart-expanded');
        if (exp) {
          exp.classList.remove('chart-expanded');
          exp.style.position = '';
          var plt = exp.querySelector('.js-plotly-plot');
          if (plt && win.Plotly) {
            win.Plotly.relayout(plt, {height: 450, autosize: true});
          }
        }
        doc.body.classList.remove('chart-open');
        var bd = doc.getElementById('chart-bd');
        if (bd) bd.remove();
        var cb = doc.getElementById('chart-close-btn');
        if (cb) cb.remove();
        doc.removeEventListener('keydown', onEsc);
      }

      function onEsc(e) {
        if (e.key === 'Escape') collapseChart();
      }

      function expandChart(btn) {
        // walk up to the nearest stVerticalBlock
        var el = btn;
        while (el && !(el.getAttribute && el.getAttribute('data-testid') === 'stVerticalBlock')) {
          el = el.parentElement;
        }
        if (!el) return;
        el.classList.add('chart-expanded');
        doc.body.classList.add('chart-open');

        // backdrop
        var bd = doc.createElement('div');
        bd.className = 'chart-backdrop';
        bd.id = 'chart-bd';
        bd.onclick = collapseChart;
        doc.body.appendChild(bd);

        // close button (×) in top-right of expanded panel
        var cb = doc.createElement('button');
        cb.id = 'chart-close-btn';
        cb.innerHTML = '&times;';
        cb.onclick = collapseChart;
        cb.style.cssText = (
          'position:absolute;top:10px;right:14px;z-index:9999;'
          + 'background:none;border:none;color:#8B9AB5;font-size:1.4rem;'
          + 'cursor:pointer;line-height:1;padding:4px 6px;'
          + 'font-family:monospace;opacity:0.6;transition:opacity 0.15s;'
        );
        cb.onmouseover = function(){ cb.style.opacity='1'; };
        cb.onmouseout  = function(){ cb.style.opacity='0.6'; };
        el.style.position = 'fixed';
        el.appendChild(cb);

        // resize plotly to fill the panel
        var plt = el.querySelector('.js-plotly-plot');
        if (plt && win.Plotly) {
          win.Plotly.relayout(plt, {
            height: win.innerHeight - 120,
            autosize: true
          });
        }

        doc.addEventListener('keydown', onEsc);
      }

      // Wire up click listeners on expand buttons (Streamlit strips onclick attrs)
      function wireButtons() {
        doc.querySelectorAll('.chart-expand-btn').forEach(function(btn) {
          if (!btn._expandWired) {
            btn._expandWired = true;
            btn.addEventListener('click', function() { expandChart(btn); });
          }
        });
      }

      wireButtons();
      var obs2 = new MutationObserver(wireButtons);
      obs2.observe(doc.body, {childList: true, subtree: true});
    })();
    </script>
    """, height=0)

    with st.spinner("Loading data…"):
        final, comp_2s = load_data()

    _hash = str(len(final))
    df, profiles, cluster_names, centers_pca, scaler, feat_cols = run_clustering(_hash, final)

    # Sync cluster labels into comp_2s
    cid_map = df.set_index("player_id")["cluster"].to_dict()
    arch_map = df.set_index("player_id")["archetype"].to_dict()
    comp_2s = comp_2s.copy()
    comp_2s["cluster"]   = comp_2s["player_id"].map(cid_map)
    comp_2s["archetype"] = comp_2s["player_id"].map(arch_map)
    comp_2s.dropna(subset=["cluster"], inplace=True)
    comp_2s["cluster"] = comp_2s["cluster"].astype(int)

    all_archetypes = [cluster_names[k] for k in sorted(cluster_names)]

    # Build player name lookup: display "First Last" → raw "Last, First"
    raw_names = sorted(df["last_name, first_name"].unique())
    def _to_display(name):
        parts = name.split(",", 1)
        return (parts[1].strip() + " " + parts[0].strip()) if len(parts) == 2 else name
    display_to_raw = {_to_display(n): n for n in raw_names}
    display_names  = sorted(display_to_raw.keys())

    # ── Shared searchbox style (used by both header widgets) ──────────────────
    from streamlit_searchbox import StyleOverrides, SearchboxStyle, DropdownStyle, OptionStyle, ClearStyle

    _LABEL_CSS = (
        "color:#8B9AB5;font-size:0.75rem;letter-spacing:0.15em;"
        "text-transform:uppercase;font-family:'Barlow Condensed',sans-serif;"
        "font-weight:600;display:block;margin-bottom:2px;"
    )
    def _sb_label(text):
        return f'<label style="{_LABEL_CSS}">{text}</label>'

    _dark_overrides = StyleOverrides(
        wrapper={"backgroundColor": "#07090F", "padding": "0"},
        searchbox=SearchboxStyle(
            control={
                "backgroundColor": "#111923",
                "borderColor": "#1E2935",
                "borderWidth": "1px",
                "borderRadius": "4px",
                "boxShadow": "none",
                "minHeight": "42px",
                "cursor": "default",
                "&:hover": {"borderColor": "#2A3E52"},
            },
            placeholder={"color": "#8B9AB5", "fontFamily": "Barlow, sans-serif", "fontSize": "14px"},
            input={"color": "#C9D1D9", "fontFamily": "Barlow, sans-serif"},
            singleValue={"color": "#C9D1D9", "fontFamily": "Barlow, sans-serif"},
            menuList={
                "backgroundColor": "#0D1117",
                "border": "1px solid #1E2935",
                "borderRadius": "4px",
                "padding": "4px 0",
            },
            option=OptionStyle(
                color="#C9D1D9",
                backgroundColor="#0D1117",
                highlightColor="#1E2935",
            ),
        ),
        dropdown=DropdownStyle(rotate=True, width=16, height=16, fill="#8B9AB5"),
        clear=ClearStyle(
            icon="cross",
            clearable="after-submit",
            width=14, height=14,
            fill="#8B9AB5", stroke="#8B9AB5",
            **{"stroke-width": 2},
        ),
    )

    # ── HEADER ────────────────────────────────────────────────────────────────
    hcol1, hcol2, hcol3 = st.columns([3, 1.6, 1.4])
    with hcol1:
        st.markdown(
            '<p class="dash-title">2025 MLB <em>Swing Decision</em> Analysis</p>'
            '<p class="dash-sub">Statcast Bat-Tracking · Exit Velocity · Plate Discipline</p>',
            unsafe_allow_html=True,
        )
    with hcol2:
        def _search_archetypes(q: str):
            # Always show all archetypes; filter by substring if user types
            if not q:
                return all_archetypes
            return [a for a in all_archetypes if q.lower() in a.lower()]

        st.markdown(_sb_label("Filter by Archetype"), unsafe_allow_html=True)
        selected_archetype = st_searchbox(
            _search_archetypes,
            placeholder="All archetypes",
            label=None,
            default_options=all_archetypes,
            key="archetype_searchbox",
            style_overrides=_dark_overrides,
        )
    with hcol3:
        def _search_players(q: str):
            if not q:
                return []
            return [n for n in display_names if q.lower() in n.lower()]

        st.markdown(_sb_label("Player Search"), unsafe_allow_html=True)
        selected_display = st_searchbox(
            _search_players,
            placeholder="Search player…",
            label=None,
            key="player_searchbox",
            style_overrides=_dark_overrides,
        )
    # Convert selections to filter values
    player_search = display_to_raw.get(selected_display, "") if selected_display else ""

    st.markdown('<hr class="dash-rule">', unsafe_allow_html=True)

    # Single archetype selected → filter to that one; nothing → show all
    active = [selected_archetype] if selected_archetype else []
    hi_player = player_search  # already cleaned in display_to_raw lookup

    # Filter dataframe for table/KPI
    df_filtered = df.copy()
    if active:
        df_filtered = df_filtered[df_filtered["archetype"].isin(active)]

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    league_spd = df["avg_bat_speed"].mean()
    league_chs = df["oz_swing_percent"].mean()
    league_hit = df["ev95percent"].mean()

    is_player_selected   = bool(player_search)
    is_archetype_selected = bool(active)
    is_filtered = is_player_selected or is_archetype_selected

    if is_player_selected:
        p_row = df[df["last_name, first_name"] == player_search]
        if not p_row.empty:
            f_spd = float(p_row.iloc[0]["avg_bat_speed"])
            f_chs = float(p_row.iloc[0]["oz_swing_percent"])
            f_hit = float(p_row.iloc[0]["ev95percent"])
        else:
            f_spd, f_chs, f_hit = league_spd, league_chs, league_hit
        n_play = 1
        fourth_trend = selected_display or player_search
    elif is_archetype_selected:
        f_spd  = df_filtered["avg_bat_speed"].mean()
        f_chs  = df_filtered["oz_swing_percent"].mean()
        f_hit  = df_filtered["ev95percent"].mean()
        n_play = len(df_filtered)
        fourth_trend = active[0]
    else:
        f_spd, f_chs, f_hit = league_spd, league_chs, league_hit
        n_play = len(df)
        fourth_trend = "All archetypes"

    kpi_html = '<div class="kpi-row">'
    if is_filtered:
        spd_delta = f_spd - league_spd
        chs_delta = f_chs - league_chs
        hit_delta = f_hit - league_hit
        kpi_html += _kpi_html("Avg Bat Speed", f"{f_spd:.1f} mph",
                              spd_delta, "vs league avg",
                              "pos" if spd_delta > 0 else "neg")
        kpi_html += _kpi_html("Avg Chase Rate", f"{f_chs:.1f}%",
                              chs_delta, "vs league avg",
                              "pos" if chs_delta > 0 else "neg")
        kpi_html += _kpi_html("Avg Hard Hit Rate", f"{f_hit:.1f}%",
                              hit_delta, "vs league avg",
                              "pos" if hit_delta > 0 else "neg")
    else:
        kpi_html += _kpi_html("League Avg Bat Speed", f"{f_spd:.1f} mph")
        kpi_html += _kpi_html("League Avg Chase Rate", f"{f_chs:.1f}%")
        kpi_html += _kpi_html("League Avg Hard Hit Rate", f"{f_hit:.1f}%")
    kpi_html += f"""
    <div class="kpi">
      <div class="kpi-label">Total Players Analyzed</div>
      <div class="kpi-val">{n_play}</div>
      <div class="kpi-trend neu">{fourth_trend}</div>
    </div>"""
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── CHARTS 2×2 ────────────────────────────────────────────────────────────
    _expand_icon = (
        '<button class="chart-expand-btn" title="Expand">'
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        '</svg></button>'
    )

    def _chart_header(title):
        return (
            f'<div class="chart-wrap">'
            f'<div class="chart-header">'
            f'<div class="chart-title">{title}</div>'
            f'{_expand_icon}'
            f'</div>'
        )

    c_top_l, c_top_r = st.columns(2, gap="medium")

    with c_top_l:
        st.markdown(_chart_header("Batter Archetype Scatter — PCA Projection"), unsafe_allow_html=True)
        st.plotly_chart(
            chart_pca(df, cluster_names, centers_pca, active, hi_player),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_top_r:
        st.markdown(_chart_header("Cluster Radar — Normalized Metric Profiles"), unsafe_allow_html=True)
        st.plotly_chart(
            chart_radar(profiles, cluster_names, active, df),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    c_bot_l, c_bot_r = st.columns(2, gap="medium")

    with c_bot_l:
        st.markdown(_chart_header("Cost of Chase — Strategic Map"), unsafe_allow_html=True)
        st.plotly_chart(
            chart_chase(df, cluster_names, active, hi_player),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_bot_r:
        st.markdown(_chart_header("2-Strike Adjustment — Swing vs Whiff Delta"), unsafe_allow_html=True)
        st.plotly_chart(
            chart_2strike(comp_2s, cluster_names, active, hi_player),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── CLUSTER SUMMARY — full width ─────────────────────────────────────────
    st.markdown('<div class="chart-wrap"><div class="chart-header"><div class="chart-title">Cluster Summary</div></div>', unsafe_allow_html=True)
    st.plotly_chart(
        chart_cluster_table(df_filtered, profiles, cluster_names, active),
        use_container_width=True, config={"displayModeBar": False},
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── PLAYER SPOTLIGHT — full width ────────────────────────────────────────
    result = chart_spotlight(df, profiles, cluster_names, hi_player)
    fig_sp, archname, badge_color = result

    if hi_player and fig_sp is None:
        st.markdown(
            '<div class="chart-wrap"><div class="chart-header">'
            '<div class="chart-title">Player Spotlight</div></div>'
            '<div style="padding:1.5rem; color:#3D4F63; font-family:\'Barlow Condensed\', sans-serif; letter-spacing:0.1em; text-transform:uppercase;">No player found</div></div>',
            unsafe_allow_html=True,
        )
    else:
        title_suffix = ""
        if archname and badge_color:
            title_suffix = (
                f'&nbsp;<span class="badge" style="background:rgba(0,0,0,0.3);'
                f'color:{badge_color};border:1px solid {badge_color}">{archname}</span>'
            )
        default_note = "" if hi_player else '<span style="color:#3D4F63; font-size:0.8rem; margin-left:0.5rem;">— league distribution (search a player above)</span>'
        st.markdown(
            f'<div class="chart-wrap"><div class="chart-header">'
            f'<div class="chart-title">Player Spotlight{title_suffix}{default_note}</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_sp, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
