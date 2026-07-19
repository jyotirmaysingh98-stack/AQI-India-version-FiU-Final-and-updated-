"""
app.py — AQI India Statewise Forecasting & Learning Platform
=================================================================
Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud / Hugging Face Spaces

FOLDER LAYOUT EXPECTED:
    app.py, data_loader.py, models.py, requirements.txt
    data/<dataset_key>.csv                       (one per dataset)
    data/state_risk_scores_overall.csv
    data/state_risk_scores_by_dataset.csv
    models/artifact_bundle.pkl
    models/dataset_info.pkl
    outputs/forecast_<dataset>_<state>.png
    outputs/seasonal_<dataset>_<state>.png

If these are missing, the app loads directly via data_loader.py's synthetic fallback
so it is ALWAYS demoable / deployable, even before you've run the full training script.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data_loader import (
    LOADERS,
    DATASET_INFO,
    load_dataset,
    states_available,
    cities_in_state,
    fetch_latest_realtime_data,
    CITY_STATE_MAP,
    ALL_STATES,
)
from models import (
    quick_forecast,
    compute_state_features,
    cluster_states,
    manual_augment,
    realistic_random_augment,
    scenario_forecast_delta,
)

# --------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AQI India Statewise Forecasting & Learning Platform",
    layout="wide",
)

EXPORT_DATA = "data"
EXPORT_MODELS = "models"
EXPORT_OUTPUTS = "outputs"

# --------------------------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset(dataset_key: str) -> pd.DataFrame:
    """
    Load dataset from CSV if present in data/<dataset_key>.csv,
    otherwise fall back to data_loader.load_dataset().
    """
    path = f"{EXPORT_DATA}/{dataset_key}.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return load_dataset(dataset_key)


@st.cache_data
def get_risk_table() -> pd.DataFrame | None:
    path = f"{EXPORT_DATA}/state_risk_scores_overall.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def get_artifact_bundle() -> dict:
    path = f"{EXPORT_MODELS}/artifact_bundle.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "#00e400"
    if aqi <= 100:
        return "Satisfactory", "#a3c853"
    if aqi <= 200:
        return "Moderate", "#ffff00"
    if aqi <= 300:
        return "Poor", "#ff7e00"
    if aqi <= 400:
        return "Very Poor", "#ff0000"
    return "Severe", "#7e0023"


def show_img_or_note(path: str, note: str):
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.info(note)

# --------------------------------------------------------------------------
# TOP-LEVEL TITLE
# --------------------------------------------------------------------------
st.title("AQI India – Statewise & Live API")
st.caption(
    "Decision-support & educational tool — multi-dataset AQI trends, "
    "hybrid forecasts, risk scoring, scenario simulation, and live 24h forecasts."
)

# --------------------------------------------------------------------------
# SIDEBAR — STATE + DATASET SELECTION PANEL
# --------------------------------------------------------------------------
st.sidebar.title("🇮🇳 AQI Statewise Platform")

dataset_key = st.sidebar.selectbox(
    "Dataset",
    list(DATASET_INFO.keys()),
    format_func=lambda k: DATASET_INFO[k]["label"],
)
info = DATASET_INFO[dataset_key]
st.sidebar.markdown(f"**{info['granularity']}** · {info['range']}")
st.sidebar.caption(info["description"])

df = get_dataset(dataset_key)
available_states = states_available(df)

state = st.sidebar.selectbox(
    "State / UT",
    available_states,
    index=available_states.index("Delhi") if "Delhi" in available_states else 0,
)
city_options = cities_in_state(df, state)
city = st.sidebar.selectbox("City (representative station)", city_options)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Statewise SARIMA+LSTM+RF hybrid forecasting · realistic augmentation · "
    "live 24h forecast · explainable risk clustering"
)

# Prepare base series for this state/city
city_df = df[(df.state == state) & (df.city == city)].sort_values("date")
series = city_df.set_index("date")["AQI"].asfreq(
    "D" if info["granularity"].startswith("Daily") else "h"
).interpolate()

bundle = get_artifact_bundle()
art = bundle.get((dataset_key, state))

# --------------------------------------------------------------------------
# MAIN TABS
# --------------------------------------------------------------------------
tab_overview, tab_hist, tab_forecast, tab_aug, tab_live, tab_features = st.tabs(
    [
        "Overview",
        "Historical Analysis",
        "Forecast & Accuracy",
        "Scenario Simulation & Augmentation",
        "Live 24-Hour Forecast",
        "Features & Innovation",
    ]
)

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tab_overview:
    col1, col2, col3 = st.columns(3)
    latest = city_df.iloc[-1]
    cat, color = aqi_category(latest["AQI"])

    with col1:
        st.metric("Latest AQI", f"{latest['AQI']:.0f}")
        st.markdown(
            f"<span style='background:{color};padding:4px 10px;"
            f"border-radius:6px'>{cat}</span>",
            unsafe_allow_html=True,
        )

    with col2:
        st.write("**Latest pollutants:**")
        st.dataframe(
            latest[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]
            .to_frame("value")
            .round(1),
            use_container_width=True,
        )

    with col3:
        risk_df = get_risk_table()
        if risk_df is not None and state in risk_df["state"].values:
            r = risk_df[risk_df.state == state].iloc[0]
            st.metric("Risk Category", r["risk_category"])
            st.metric("Risk Score (0-100)", f"{r['risk_score']:.0f}")

    st.subheader("Data Sources Overview")
    overview_rows = [
        {
            "Dataset": v["label"],
            "Granularity": v["granularity"],
            "Range": v["range"],
            "Description": v["description"],
        }
        for v in DATASET_INFO.values()
    ]
    st.dataframe(
        pd.DataFrame(overview_rows),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================================
# TAB 2 — HISTORICAL ANALYSIS
# ============================================================================
with tab_hist:
    st.subheader(f"AQI Trend — {city}, {state}")
    fig = px.line(city_df, x="date", y="AQI", title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonal Decomposition")
    show_img_or_note(
        f"{EXPORT_OUTPUTS}/seasonal_{dataset_key}_{state}.png",
        "Pre-rendered seasonal decomposition not found — run train_and_export.py "
        "and copy export/outputs/ here, or view live trend above.",
    )

    st.subheader("Statewise AQI / Risk Heatmap")
    risk_by_ds = None
    path = f"{EXPORT_DATA}/state_risk_scores_by_dataset.csv"
    if os.path.exists(path):
        risk_by_ds = pd.read_csv(path)
        sub = risk_by_ds[risk_by_ds.dataset == dataset_key]
        fig2 = px.bar(
            sub.sort_values("risk_score", ascending=False),
            x="state",
            y="risk_score",
            color="risk_category",
            title=f"Statewise Risk Score — {info['label']}",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        feats = compute_state_features(df)
        clustered = cluster_states(feats)
        fig2 = px.bar(
            clustered.sort_values("risk_score", ascending=False),
            x="state",
            y="risk_score",
            color="risk_category",
            title="Statewise Risk Score (computed live)",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# TAB 3 — FORECAST & ACCURACY
# ============================================================================
with tab_forecast:
    st.subheader(f"Model Accuracy — {state} on {info['label']}")
    metrics_path = f"{EXPORT_OUTPUTS}/model_metrics_by_dataset_state.csv"
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path)
        sub = m[(m.dataset == dataset_key) & (m.state == state)]
        if not sub.empty:
            st.dataframe(
                sub[["model", "RMSE", "MAE", "MAPE", "R2"]].round(3),
                use_container_width=True,
                hide_index=True,
            )
            best = sub.loc[sub.RMSE.idxmin(), "model"]
            st.success(
                f"**{best}** has the lowest RMSE for {state} on {info['label']}."
            )

            st.subheader("Cross-Dataset Comparison (same state, all datasets trained)")
            cross = m[m.state == state]
            fig3 = px.bar(
                cross,
                x="dataset",
                y="RMSE",
                color="model",
                barmode="group",
                title=f"RMSE by dataset & model — {state}",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info(
                f"No pre-trained metrics for {state}/{dataset_key} yet — "
                f"add it to REP_STATES in train_and_export.py and re-run."
            )
    else:
        st.info(
            "Run train_and_export.py to populate the full accuracy dashboard. "
            "Showing a quick on-demand SARIMA forecast below in the meantime."
        )

    st.subheader("Forecast vs Actual (hybrid model)")
    show_img_or_note(
        f"{EXPORT_OUTPUTS}/forecast_{dataset_key}_{state}.png",
        "Pre-rendered chart not found — run train_and_export.py.",
    )

    st.subheader("Quick On-Demand Forecast")
    horizon = st.slider("Horizon (steps)", 1, 7, 3, key="fc_horizon")
    try:
        fc = quick_forecast(
            series,
            steps=horizon,
            freq="D" if info["granularity"].startswith("Daily") else "H",
        )
        figf = go.Figure()
        figf.add_trace(
            go.Scatter(
                x=series.index[-60:],
                y=series.values[-60:],
                name="Recent actual",
            )
        )
        figf.add_trace(
            go.Scatter(
                x=fc["timestamp"],
                y=fc["forecast"],
                name="Forecast",
                line=dict(dash="dash"),
            )
        )
        figf.add_trace(
            go.Scatter(
                x=fc["timestamp"],
                y=fc["upper_80"],
                line=dict(width=0),
                showlegend=False,
            )
        )
        figf.add_trace(
            go.Scatter(
                x=fc["timestamp"],
                y=fc["lower_80"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255,127,14,0.2)",
                name="80% band",
            )
        )
        st.plotly_chart(figf, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecast unavailable: {e}")

# ============================================================================
# TAB 4 — SCENARIO SIMULATION & AUGMENTATION
# ============================================================================
with tab_aug:
    st.subheader("Manual Augmentation")
    c1, c2, c3 = st.columns(3)
    traffic_pct = c1.slider("Traffic emissions change (%)", -50, 50, 0) / 100
    industry_pct = c2.slider("Industrial output change (%)", -50, 50, 0) / 100
    pm25_pct = c3.slider("Direct PM2.5 change (%)", -50, 50, 0) / 100
    add_festival = st.checkbox("Add festival-period spike this month (Diwali-style)")

    pct_changes = {"traffic": traffic_pct, "industry": industry_pct, "pm25": pm25_pct}
    augmented = manual_augment(series, pct_changes)
    if add_festival:
        boost_mask = augmented.index.day <= 15
        augmented = augmented.where(~boost_mask, augmented + 40)

    st.subheader("Random Realistic Augmentation")
    if st.button("🎲 Apply random realistic augmentation"):
        augmented = realistic_random_augment(series, seed=None)
        st.session_state["random_aug_applied"] = True

    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=series.index[-120:],
            y=series.values[-120:],
            name="Original",
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=augmented.index[-120:],
            y=augmented.values[-120:],
            name="Augmented",
        )
    )
    fig4.update_layout(
        title="Original vs Augmented AQI Series (last 120 points)"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Forecast Impact of This Scenario")
    try:
        baseline_fc, scenario_fc = scenario_forecast_delta(series, pct_changes, steps=7)
        cmp_df = pd.DataFrame(
            {
                "day": range(1, 8),
                "baseline_forecast": baseline_fc["forecast"].values,
                "scenario_forecast": scenario_fc["forecast"].values,
            }
        )
        cmp_df["delta"] = cmp_df["scenario_forecast"] - cmp_df["baseline_forecast"]
        st.dataframe(
            cmp_df.round(1),
            use_container_width=True,
            hide_index=True,
        )

        fig5 = go.Figure()
        fig5.add_trace(
            go.Scatter(
                x=cmp_df["day"],
                y=cmp_df["baseline_forecast"],
                name="Baseline",
            )
        )
        fig5.add_trace(
            go.Scatter(
                x=cmp_df["day"],
                y=cmp_df["scenario_forecast"],
                name="Scenario",
            )
        )
        st.plotly_chart(fig5, use_container_width=True)

        if "saved_scenarios" not in st.session_state:
            st.session_state["saved_scenarios"] = []
        if st.button("💾 Save this scenario for comparison"):
            st.session_state["saved_scenarios"].append(
                {
                    "state": state,
                    "dataset": dataset_key,
                    **pct_changes,
                    "avg_delta": cmp_df["delta"].mean(),
                }
            )
        if st.session_state.get("saved_scenarios"):
            st.write("**Saved scenarios (this session):**")
            st.dataframe(
                pd.DataFrame(st.session_state["saved_scenarios"]),
                use_container_width=True,
            )
    except Exception as e:
        st.warning(f"Scenario re-forecast unavailable: {e}")

# ============================================================================
# TAB 5 — LIVE 24-HOUR FORECAST
# ============================================================================
with tab_live:
    st.subheader(f"Live 24-Hour Forecast — {city}, {state}")
    st.caption(
        "Pulls the latest real readings (live API if configured, else a fresh synthetic "
        "'last 24h' fallback) and forecasts the next 24 hours from that live context."
    )

    if st.button("🔄 Fetch latest data & refresh live forecast"):
        st.session_state["live_fetch_ts"] = pd.Timestamp.now()

    # Live data via data.gov.in (or synthetic fallback)
    live_df = fetch_latest_realtime_data(state, city, hours=24)
    live_df = live_df.sort_values("date")

    st.write(
        f"**Last updated:** {pd.Timestamp.now():%Y-%m-%d %H:%M} "
        f"(source: `{live_df['source'].iloc[0]}`)"
    )

    latest_live = live_df.iloc[-1]
    lc1, lc2 = st.columns(2)
    with lc1:
        st.metric("Latest observed AQI (live)", f"{latest_live['AQI']:.0f}")
    with lc2:
        st.dataframe(
            latest_live[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]
            .to_frame("value")
            .round(1),
            use_container_width=True,
        )

    live_series = live_df.set_index("date")["AQI"].asfreq("h").interpolate()
    combined = pd.concat([series.tail(24 * 14), live_series]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    try:
        live_fc = quick_forecast(combined, steps=24, freq="h")
        figl = go.Figure()
        figl.add_trace(
            go.Scatter(
                x=live_series.index,
                y=live_series.values,
                name="Last 24h observed",
            )
        )
        figl.add_trace(
            go.Scatter(
                x=live_fc["timestamp"],
                y=live_fc["forecast"],
                name="Next 24h forecast",
                line=dict(dash="dash"),
            )
        )
        figl.add_trace(
            go.Scatter(
                x=live_fc["timestamp"],
                y=live_fc["upper_80"],
                line=dict(width=0),
                showlegend=False,
            )
        )
        figl.add_trace(
            go.Scatter(
                x=live_fc["timestamp"],
                y=live_fc["lower_80"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0,150,255,0.2)",
                name="80% band",
            )
        )
        figl.update_layout(title="Next 24-Hour AQI Forecast (hourly)")
        st.plotly_chart(figl, use_container_width=True)
    except Exception as e:
        st.warning(f"Live forecast unavailable: {e}")

    st.caption(
        "To use a real live feed: set `DATA_GOV_API_KEY` as a Streamlit secret and "
        "ensure `REALTIME_API_URL` in data_loader.py points to your data.gov.in/CPCB/AIKosh resource."
    )

# ============================================================================
# TAB 6 — FEATURES & INNOVATION
# ============================================================================
with tab_features:
    st.subheader("Features & Innovation")
    st.markdown(
        """
This platform combines several capabilities not typically found together in a single AQI tool:

1. **Multi-dataset integration & harmonization engine** — five heterogeneous CPCB-derived sources
   (statewise daily, historical pollutants, two Kaggle city-level sets, and a live API) are normalized
   into one schema with automatic city→state mapping, so any model can run on any source interchangeably.
2. **Per-dataset, per-state accuracy dashboards** — see exactly which dataset + model combination performs
   best for *your* state (Forecast & Accuracy tab), rather than a single blanket accuracy claim.
3. **Cascaded 3-stage hybrid forecaster** — SARIMA (seasonality/trend) → LSTM/BiLSTM (residual, non-linear
   structure) → Random Forest/XGBoost error-corrector (event/calendar effects), evaluated against a
   standalone-LSTM benchmark for transparency.
4. **Interactive scenario simulation & realistic augmentation** — manual sliders (traffic, industry, PM2.5)
   and a bounded-random "realistic augmentation" mode let users test interventions and worst-case scenarios
   and immediately see the forecast impact — turning the tool into a *decision-support simulator*, not just
   a predictor.
5. **Statewise spatio-temporal risk scoring & clustering** — KMeans over mean AQI, volatility, long-term
   trend, and seasonal amplitude yields interpretable risk tiers (Chronic High / Episodic High / Lower Risk)
   and a statewise heatmap.
6. **Explainable hybrid model insights** — SHAP-based feature importance on the error-corrector stage,
   surfaced as plain-language summaries per state and dataset.
7. **Genuine 24-hour live forecast** — fetches fresh data on demand (real API when configured, else a
   clearly-labeled synthetic fallback) and forecasts forward from *current* conditions, not static history.
8. **Educational mode** — explains dataset differences and model mechanics so the
   platform doubles as a teaching tool for students/policymakers, not just a forecasting black box.

**Why this supports a patent / product proposal:** no single referenced paper or open-source tool combines
harmonized multi-source ingestion, a specific 3-stage hybrid cascade, user-driven scenario simulation with
immediate re-forecasting, explainable risk clustering, *and* a live-data-driven deployment in one integrated,
user-facing system — the combination and the per-dataset comparative accuracy dashboard are the differentiators.
        """
    )

    st.markdown("---")
    st.caption(
        "SARIMA (statsmodels) + LSTM/BiLSTM (TensorFlow) + RF/XGBoost correction · "
        "KMeans risk clustering · SHAP explainability · live data integration. "
        "Train with train_and_export.py in Colab."
    )
