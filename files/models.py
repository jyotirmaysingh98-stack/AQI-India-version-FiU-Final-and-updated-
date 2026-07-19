"""
models.py — Hybrid forecasting, risk clustering, augmentation, explainability
=================================================================================
Reusable, dataset-agnostic model functions. Same functions are used by the
training script (train_and_export.py, run in Colab) and by app.py (for the
lightweight on-demand SARIMA re-forecast + augmentation preview).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LOOKBACK = 14


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Stage 1: SARIMA
# ---------------------------------------------------------------------------
def fit_sarima(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    m = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False)
    return m.fit(disp=False)


def sarima_forecast(res, steps, alpha=0.2):
    fc = res.get_forecast(steps=steps)
    ci = fc.conf_int(alpha=alpha)
    return fc.predicted_mean, ci


# ---------------------------------------------------------------------------
# Stage 2: LSTM on residuals (TensorFlow — only imported when training, keeps
# app.py lightweight since it never needs to import TF for the live/demo path)
# ---------------------------------------------------------------------------
def make_supervised(series, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_lstm(input_shape, bidirectional=True):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    m = Sequential()
    layer = LSTM(48, return_sequences=False)
    m.add(Bidirectional(layer, input_shape=input_shape) if bidirectional else layer)
    m.add(Dropout(0.2))
    m.add(Dense(24, activation="relu"))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse")
    return m


def train_residual_lstm(sarima_res, train_series, epochs=15):
    fitted = sarima_res.fittedvalues.reindex(train_series.index).fillna(method="bfill")
    resid = (train_series - fitted).values
    scaler = MinMaxScaler()
    resid_scaled = scaler.fit_transform(resid.reshape(-1, 1)).flatten()
    X, y = make_supervised(resid_scaled)
    model = build_lstm((LOOKBACK, 1))
    model.fit(X.reshape(-1, LOOKBACK, 1), y, epochs=epochs, batch_size=32, verbose=0)
    return model, scaler, resid_scaled


def roll_residual_forecast(model, scaler, resid_scaled_history, steps):
    hist = list(resid_scaled_history[-LOOKBACK:])
    preds = []
    for _ in range(steps):
        x_in = np.array(hist[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        p = model.predict(x_in, verbose=0)[0][0]
        preds.append(p)
        hist.append(p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Stage 3: RF/XGBoost error-corrector using event + risk features
# ---------------------------------------------------------------------------
def add_event_features(index: pd.DatetimeIndex):
    return pd.DataFrame({
        "is_weekend": (index.dayofweek >= 5).astype(float),
        "is_diwali_window": ((index.month.isin([10, 11])) & (index.day <= 15)).astype(float),
        "is_crop_burning_season": index.month.isin([10, 11]).astype(float),
        "month": index.month.astype(float),
    }, index=index)


def train_error_corrector(residuals: np.ndarray, index: pd.DatetimeIndex, model_type="rf"):
    feats = add_event_features(index).iloc[-len(residuals):]
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
        except ImportError:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(feats.fillna(0), residuals[-len(feats):])
    return model


def full_hybrid_pipeline(series: pd.Series, test_days=30, corrector="rf"):
    """SARIMA -> LSTM residual -> RF/XGBoost correction. Returns predictions + metrics
    for all 3 stages plus a standalone-LSTM benchmark."""
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]

    sarima_res = fit_sarima(train)
    sarima_pred, sarima_ci = sarima_forecast(sarima_res, test_days)
    sarima_pred.index = test.index

    lstm_resid_model, r_scaler, resid_scaled_train = train_residual_lstm(sarima_res, train)
    resid_preds = roll_residual_forecast(lstm_resid_model, r_scaler, resid_scaled_train, test_days)
    hybrid_pred = sarima_pred.values + resid_preds

    fitted = sarima_res.fittedvalues.reindex(train.index).fillna(method="bfill")
    resid_train_actual = (train - fitted).values
    corrector_model = train_error_corrector(resid_train_actual, train.index, corrector)
    correction = corrector_model.predict(add_event_features(test.index).fillna(0))
    final_pred = hybrid_pred + 0.15 * correction

    # standalone LSTM benchmark (trained directly on AQI, not residuals)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    Xs, ys = make_supervised(scaled)
    split = len(Xs) - test_days
    lstm_std = build_lstm((LOOKBACK, 1))
    lstm_std.fit(Xs[:split].reshape(-1, LOOKBACK, 1), ys[:split], epochs=15, batch_size=32, verbose=0)
    lstm_std_pred = scaler.inverse_transform(
        lstm_std.predict(Xs[split:].reshape(-1, LOOKBACK, 1), verbose=0)).flatten()

    y_true = test.values
    metrics = {
        "SARIMA": evaluate(y_true, sarima_pred.values),
        "LSTM_standalone": evaluate(y_true[-len(lstm_std_pred):], lstm_std_pred),
        "Hybrid_SARIMA_LSTM": evaluate(y_true, hybrid_pred),
        "Hybrid_SARIMA_LSTM_RF": evaluate(y_true, final_pred),
    }
    artifacts = dict(sarima_res=sarima_res, lstm_resid_model=lstm_resid_model, r_scaler=r_scaler,
                      corrector_model=corrector_model, sarima_pred=sarima_pred, hybrid_pred=hybrid_pred,
                      final_pred=final_pred, test=test, sarima_ci=sarima_ci)
    return metrics, artifacts


# ---------------------------------------------------------------------------
# Lightweight on-demand forecast for the deployed app (SARIMA only — fast, no TF)
# ---------------------------------------------------------------------------
def quick_forecast(series: pd.Series, steps=24, freq="D"):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7) if freq == "D" else (1, 1, 1, 24)
    res = fit_sarima(series, order, seasonal_order)
    mean, ci = sarima_forecast(res, steps)
    step_unit = "D" if freq == "D" else "h"
    out = pd.DataFrame({
        "timestamp": pd.date_range(series.index[-1] + pd.Timedelta(**{("days" if freq == "D" else "hours"): 1}),
                                    periods=steps, freq=step_unit),
        "forecast": mean.values, "lower_80": ci.iloc[:, 0].values, "upper_80": ci.iloc[:, 1].values,
    })
    return out


# ---------------------------------------------------------------------------
# Seasonal decomposition
# ---------------------------------------------------------------------------
def seasonal_decompose_series(series: pd.Series, period=365):
    return STL(series, period=period, robust=True).fit()


# ---------------------------------------------------------------------------
# Spatio-temporal risk clustering
# ---------------------------------------------------------------------------
def compute_state_features(df: pd.DataFrame):
    rows = []
    for s, g in df.groupby("state"):
        g = g.sort_values("date")
        x = (g["date"] - g["date"].min()).dt.days.values / 365.25
        slope = np.polyfit(x, g["AQI"].values, 1)[0] if len(x) > 30 else 0.0
        m = g["date"].dt.month
        winter = g.loc[m.isin([11, 12, 1, 2]), "AQI"].mean()
        summer = g.loc[m.isin([4, 5, 6]), "AQI"].mean()
        ratio = winter / summer if summer and not np.isnan(summer) else np.nan
        rows.append({"state": s, "mean_AQI": g["AQI"].mean(), "std_AQI": g["AQI"].std(),
                     "trend_slope_per_year": slope, "winter_summer_ratio": ratio})
    return pd.DataFrame(rows)


def cluster_states(state_features: pd.DataFrame, n_clusters=3):
    feats = state_features[["mean_AQI", "std_AQI", "trend_slope_per_year", "winter_summer_ratio"]].fillna(0)
    X = MinMaxScaler().fit_transform(feats)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    out = state_features.copy()
    out["cluster"] = km.labels_
    out["_composite"] = (out["mean_AQI"] * 0.5 + out["std_AQI"] * 0.2 +
                          out["trend_slope_per_year"] * 5 + out["winter_summer_ratio"] * 10)
    summary = out.groupby("cluster").agg(mean_AQI=("mean_AQI", "mean"),
                                          winter_summer_ratio=("winter_summer_ratio", "mean"),
                                          composite=("_composite", "mean"))
    chronic = summary["composite"].idxmax()
    rest = [c for c in summary.index if c != chronic]
    episodic = summary.loc[rest, "winter_summer_ratio"].idxmax()
    lower = [c for c in summary.index if c not in (chronic, episodic)][0]
    label_map = {chronic: "Chronic High Risk", episodic: "Episodic High Risk", lower: "Lower Risk"}
    out["risk_category"] = out["cluster"].map(label_map)
    rs = MinMaxScaler((0, 100))
    out["risk_score"] = rs.fit_transform(out[["_composite"]].values)
    return out.drop(columns=["_composite"])


# ---------------------------------------------------------------------------
# Realistic data augmentation / scenario simulation
# ---------------------------------------------------------------------------
def manual_augment(series: pd.Series, pct_changes: dict, pollutant_weights: dict = None):
    """Apply user-specified % changes (e.g. {'traffic': -0.3, 'industry': -0.15}) to AQI,
    using default sensible weights if not provided."""
    weights = pollutant_weights or {"traffic": 0.45, "industry": 0.35, "pm25": 0.55}
    delta_pct = sum(weights.get(k, 0.3) * v for k, v in pct_changes.items())
    return series * (1 + delta_pct)


def realistic_random_augment(series: pd.Series, seed=None, event_boost=True):
    """Bounded Gaussian noise scaled to the series' own historical std, with optional
    festival-period spikes — keeps augmented data statistically plausible."""
    rng = np.random.default_rng(seed)
    std = series.std()
    noise = rng.normal(0, std * 0.25, len(series))
    aug = series.values + noise
    if event_boost:
        diwali_mask = ((series.index.month.isin([10, 11])) & (series.index.day <= 15))
        aug = np.where(diwali_mask, aug + rng.uniform(20, 60, len(series)), aug)
    return pd.Series(np.clip(aug, 10, 500), index=series.index)


def scenario_forecast_delta(series: pd.Series, pct_changes: dict, steps=7):
    """Re-forecasts with SARIMA on both the original and the (manually) augmented series
    to show how a scenario changes the near-term forecast."""
    baseline = quick_forecast(series, steps)
    aug_series = manual_augment(series, pct_changes)
    scenario = quick_forecast(aug_series, steps)
    return baseline, scenario


# ---------------------------------------------------------------------------
# Explainability (SHAP on the RF/XGBoost corrector)
# ---------------------------------------------------------------------------
def explain_corrector(corrector_model, index: pd.DatetimeIndex):
    import shap
    feats = add_event_features(index).tail(200)
    explainer = shap.TreeExplainer(corrector_model)
    shap_values = explainer.shap_values(feats)
    importances = dict(zip(feats.columns, np.abs(shap_values).mean(axis=0)))
    top = max(importances, key=importances.get)
    return importances, shap_values, feats, top
