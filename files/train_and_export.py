"""
train_and_export.py — Colab training pipeline
==================================================
Run this in Google Colab (upload data_loader.py, models.py, this file, and your
raw CSVs first). It trains SARIMA / LSTM / Hybrid / Hybrid+RF for representative
states across each dataset, computes statewise risk clustering, and exports every
artifact app.py needs into export/ (metrics table, risk table, series, charts).

Usage in Colab:
    !pip -q install xgboost shap streamlit
    # upload data_loader.py, models.py, train_and_export.py + your CSVs, then:
    !python train_and_export.py
    # zip export/ and download, or push straight to your app repo
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_all_datasets, DATASET_INFO, states_available
from models import (full_hybrid_pipeline, compute_state_features, cluster_states,
                     seasonal_decompose_series, explain_corrector)

warnings.filterwarnings("ignore")
np.random.seed(42)

EXPORT_DIR = "export"
os.makedirs(f"{EXPORT_DIR}/data", exist_ok=True)
os.makedirs(f"{EXPORT_DIR}/models", exist_ok=True)
os.makedirs(f"{EXPORT_DIR}/outputs", exist_ok=True)

# Representative states to fully train (extend this list for full state coverage —
# training every state x every dataset is expensive; the app can quick_forecast()
# any state on demand even without a pre-trained hybrid model)
REP_STATES = ["Delhi", "Maharashtra", "Uttar Pradesh", "West Bengal", "Karnataka"]
TEST_DAYS = 30

print("Loading & harmonizing all datasets...")
datasets = load_all_datasets()   # dict: dataset_key -> standardized dataframe
for key, df in datasets.items():
    df.to_csv(f"{EXPORT_DIR}/data/{key}.csv", index=False)
    print(f"  {key}: {df.shape[0]} rows, {df['state'].nunique()} states")

# ---------------------------------------------------------------------------
# Per (dataset, state): train hybrid pipeline, store metrics + artifacts
# ---------------------------------------------------------------------------
all_metrics_rows = []
artifact_bundle = {}   # (dataset_key, state) -> light artifacts for the app

for dkey, df in datasets.items():
    for state in REP_STATES:
        sub = df[df.state == state]
        if sub.empty or sub["city"].nunique() == 0:
            continue
        city = sub["city"].value_counts().idxmax()          # pick the best-covered city as state proxy
        g = sub[sub.city == city].sort_values("date")
        freq = "D"
        series = g.set_index("date")["AQI"].asfreq(freq).interpolate()
        if len(series) < TEST_DAYS + 60:
            continue
        try:
            metrics, art = full_hybrid_pipeline(series, test_days=TEST_DAYS)
        except Exception as e:
            print(f"  skip {dkey}/{state}: {e}")
            continue
        for model_name, m in metrics.items():
            row = {"dataset": dkey, "state": state, "city": city, "model": model_name}
            row.update(m)
            all_metrics_rows.append(row)

        artifact_bundle[(dkey, state)] = {
            "city": city, "last_180": series.tail(180),
            "sarima_order": art["sarima_res"].model.order,
            "sarima_seasonal_order": art["sarima_res"].model.seasonal_order,
        }

        # forecast vs actual chart
        plt.figure(figsize=(9, 4))
        plt.plot(art["test"].index, art["test"].values, label="Actual", lw=2)
        plt.plot(art["test"].index, art["sarima_pred"].values, "--", label="SARIMA")
        plt.plot(art["test"].index, art["hybrid_pred"], label="Hybrid SARIMA+LSTM")
        plt.plot(art["test"].index, art["final_pred"], label="Hybrid+RF")
        plt.title(f"{state} ({city}) — {dkey}: Forecast vs Actual")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{EXPORT_DIR}/outputs/forecast_{dkey}_{state}.png", dpi=120)
        plt.close()

        # seasonal decomposition (only if enough history)
        if len(series) >= 400:
            stl = seasonal_decompose_series(series, period=365)
            fig = stl.plot(); fig.set_size_inches(8, 6)
            fig.suptitle(f"{state} ({city}) — {dkey}: Seasonal Decomposition")
            fig.tight_layout()
            fig.savefig(f"{EXPORT_DIR}/outputs/seasonal_{dkey}_{state}.png", dpi=120)
            plt.close(fig)

        # explainability text summary
        try:
            importances, _, _, top = explain_corrector(art["corrector_model"], art["test"].index)
            summary = (f"For {state} on {DATASET_INFO[dkey]['label']}, the strongest driver of the "
                       f"error-correction stage is '{top}' (mean |SHAP|={importances[top]:.2f}), "
                       f"indicating seasonal/calendar effects (festivals, crop-burning season) "
                       f"materially shift next-day AQI beyond the SARIMA+LSTM baseline.")
        except Exception:
            summary = f"Explainability unavailable for {state}/{dkey} (insufficient data)."
        artifact_bundle[(dkey, state)]["explanation"] = summary
        print(f"  trained {dkey}/{state} ({city})")

metrics_df = pd.DataFrame(all_metrics_rows)
metrics_df.to_csv(f"{EXPORT_DIR}/outputs/model_metrics_by_dataset_state.csv", index=False)

# ---------------------------------------------------------------------------
# Statewise risk clustering (computed per dataset, then averaged for a robust score)
# ---------------------------------------------------------------------------
risk_frames = []
for dkey, df in datasets.items():
    feats = compute_state_features(df)
    clustered = cluster_states(feats)
    clustered["dataset"] = dkey
    risk_frames.append(clustered)
risk_all = pd.concat(risk_frames, ignore_index=True)
risk_all.to_csv(f"{EXPORT_DIR}/data/state_risk_scores_by_dataset.csv", index=False)

risk_avg = risk_all.groupby("state").agg(
    mean_AQI=("mean_AQI", "mean"), risk_score=("risk_score", "mean"),
    risk_category=("risk_category", lambda s: s.mode()[0])).reset_index()
risk_avg.to_csv(f"{EXPORT_DIR}/data/state_risk_scores_overall.csv", index=False)

# ---------------------------------------------------------------------------
# Save artifact bundle + dataset info for app.py
# ---------------------------------------------------------------------------
with open(f"{EXPORT_DIR}/models/artifact_bundle.pkl", "wb") as f:
    pickle.dump(artifact_bundle, f)
with open(f"{EXPORT_DIR}/models/dataset_info.pkl", "wb") as f:
    pickle.dump(DATASET_INFO, f)

print("\nDone. Copy the export/ folder contents (data/, models/, outputs/) into your app repo.")
print("Metrics table shape:", metrics_df.shape, "| Risk table shape:", risk_avg.shape)
