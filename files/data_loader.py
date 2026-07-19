"""
data_loader.py — AQI India Statewise Forecasting & Learning Platform
=======================================================================
Loads and harmonizes 5 data sources into ONE schema:
    date, state, city, AQI, PM2.5, PM10, NO2, SO2, CO, O3, source

Every loader:
  1. Tries the real file/API path (placeholders below — replace with yours).
  2. Falls back to a realistic synthetic generator if the file/API is unavailable,
     so the app and training pipeline ALWAYS run end-to-end.

Replace PATH_* / API_* placeholders with your real paths/keys. No other code
needs to change — every loader returns the same standardized dataframe.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Placeholders — EDIT THESE
# ---------------------------------------------------------------------------
PATH_DATAFUL_STATEWISE   = "/content/data/dataful_statewise_aqi.csv"          # dataset 1
PATH_DATAGOV_POLLUTANTS  = "/content/data/datagov_historical_pollutants.csv"  # dataset 2
PATH_KAGGLE_BHAUTIK      = "/content/data/bhautik_2022_2025.csv"              # dataset 3
PATH_KAGGLE_ISHANKAT     = "/content/data/ishankat_realtime_2023_2025.csv"    # dataset 4
REALTIME_API_URL         = "https://api.data.gov.in/resource/REPLACE_WITH_RESOURCE_ID"  # dataset 5
REALTIME_API_KEY         = os.environ.get("DATA_GOV_API_KEY", "REPLACE_WITH_YOUR_KEY")

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

DATASET_INFO = {
    "dataful_statewise": {
        "label": "Dataful — Day-wise Statewise AQI (2015–present)",
        "description": "Daily AQI by state/city/rural-urban, compiled from CPCB 24-hour reports. "
                        "Best for long-term state trends and seasonal baselines.",
        "granularity": "Daily", "range": "2015–present",
    },
    "datagov_pollutants": {
        "label": "data.gov.in — Historical Ambient Air Quality (pollutants)",
        "description": "Historical SO2, NO2, RSPM/SPM concentrations across India. "
                        "Best for long-term pollutant trend features.",
        "granularity": "Daily/Monthly", "range": "Historical (pre-2015 onward)",
    },
    "kaggle_bhautik": {
        "label": "Kaggle (bhautikvekariya21) — Indian Cities AQI 2022–2025",
        "description": "Time-stamped atmospheric + AQI data for Indian cities, 2022–2025. "
                        "Best for recent forecasting/evaluation per state.",
        "granularity": "Daily", "range": "2022–2025",
    },
    "kaggle_realtime": {
        "label": "Kaggle (ishankat) — Real-Time AQI India 2023–2025",
        "description": "Hourly AQI readings from monitoring stations, 2023–2025. "
                        "Best for higher-frequency (hourly) modeling.",
        "granularity": "Hourly", "range": "2023–2025",
    },
    "realtime_api": {
        "label": "Live API — Real-time AQI (data.gov.in / CPCB / AIKosh)",
        "description": "Live station AQI/pollutants pulled fresh on each app run. "
                        "Powers the 24-hour live forecast feature.",
        "granularity": "Hourly (live)", "range": "Last 24-48h",
    },
}

# ---------------------------------------------------------------------------
# 1. City -> State lookup (extend as needed)
# ---------------------------------------------------------------------------
CITY_STATE_MAP = {
    "Delhi": "Delhi", "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Bengaluru": "Karnataka", "Mysuru": "Karnataka", "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
    "Kolkata": "West Bengal", "Howrah": "West Bengal", "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Varanasi": "Uttar Pradesh", "Patna": "Bihar", "Gaya": "Bihar", "Jaipur": "Rajasthan",
    "Jodhpur": "Rajasthan", "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Hyderabad": "Telangana",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh", "Chandigarh": "Chandigarh",
    "Guwahati": "Assam", "Thiruvananthapuram": "Kerala", "Kochi": "Kerala", "Bhubaneswar": "Odisha",
    "Amritsar": "Punjab", "Ludhiana": "Punjab", "Dehradun": "Uttarakhand", "Shimla": "Himachal Pradesh",
    "Raipur": "Chhattisgarh", "Ranchi": "Jharkhand", "Panaji": "Goa", "Imphal": "Manipur",
    "Shillong": "Meghalaya", "Agartala": "Tripura", "Gangtok": "Sikkim", "Itanagar": "Arunachal Pradesh",
    "Aizawl": "Mizoram", "Kohima": "Nagaland",
}
ALL_STATES = sorted(set(CITY_STATE_MAP.values()))

# ---------------------------------------------------------------------------
# 2. Synthetic generator (calibrated per state, used when real files are absent)
# ---------------------------------------------------------------------------
def _synthetic_series(city, state, start, end, freq="D", seed_offset=0):
    rng = np.random.default_rng(abs(hash((city, freq))) % (2**32) + seed_offset)
    idx = pd.date_range(start, end, freq=freq)
    n = len(idx)
    base = {"Delhi": 230, "Lucknow": 205, "Kanpur": 210, "Patna": 195, "Kolkata": 150,
            "Mumbai": 120, "Bengaluru": 85, "Chennai": 95, "Jaipur": 175, "Ahmedabad": 165,
            "Hyderabad": 130, "Bhopal": 150}.get(city, 140)
    t_days = (idx - idx[0]).days.values if freq == "D" else (idx - idx[0]).total_seconds().values / 86400
    doy = pd.DatetimeIndex(idx).dayofyear.values
    seasonal = 55 * np.sin(2 * np.pi * (doy - 300) / 365) + 20 * np.cos(4 * np.pi * doy / 365)
    diurnal = 15 * np.sin(2 * np.pi * pd.DatetimeIndex(idx).hour.values / 24) if freq != "D" else 0
    trend = np.linspace(0, rng.uniform(-12, 8), n)
    noise = rng.normal(0, 16, n)
    diwali_mask = (pd.DatetimeIndex(idx).month.isin([10, 11])) & (pd.DatetimeIndex(idx).day <= 15)
    diwali_boost = np.where(diwali_mask, rng.uniform(35, 85, n), 0)
    aqi = np.clip(base + trend + seasonal + diurnal + noise + diwali_boost, 15, 500)
    df = pd.DataFrame({"date": idx, "AQI": aqi})
    ratios = {"PM2.5": 0.55, "PM10": 0.75, "NO2": 0.22, "SO2": 0.10, "CO": 0.012, "O3": 0.30}
    for p, r in ratios.items():
        df[p] = np.clip(aqi * r + rng.normal(0, aqi * 0.05, n), 1, None)
    df["city"] = city
    df["state"] = state
    return df


def _standardize(raw, date_col, city_col, aqi_col, source_name, state_col=None):
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    df["city"] = raw[city_col].astype(str).str.strip().str.title()
    df["AQI"] = pd.to_numeric(raw[aqi_col], errors="coerce")
    for p in POLLUTANTS:
        df[p] = pd.to_numeric(raw[p], errors="coerce") if p in raw.columns else np.nan
    df["state"] = raw[state_col].astype(str).str.title() if state_col and state_col in raw.columns \
        else df["city"].map(CITY_STATE_MAP).fillna("Unknown")
    df["source"] = source_name
    return df.dropna(subset=["date", "city"])


def _fallback_multi_state(source_name, freq="D", years_back=4, cities=None):
    cities = cities or list(CITY_STATE_MAP.keys())[:20]
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years_back)
    frames = [_synthetic_series(c, CITY_STATE_MAP[c], start, end, freq=freq) for c in cities]
    df = pd.concat(frames, ignore_index=True)
    df["source"] = source_name
    return df


# ---------------------------------------------------------------------------
# 3. Per-dataset loaders (each returns the standardized schema)
# ---------------------------------------------------------------------------
def load_dataful_statewise():
    if os.path.exists(PATH_DATAFUL_STATEWISE):
        raw = pd.read_csv(PATH_DATAFUL_STATEWISE)
        return _standardize(raw, "date", "city", "aqi_value", "dataful_statewise", state_col="state")
    return _fallback_multi_state("dataful_statewise", freq="D", years_back=6)


def load_data_gov_pollutants():
    if os.path.exists(PATH_DATAGOV_POLLUTANTS):
        raw = pd.read_csv(PATH_DATAGOV_POLLUTANTS)
        return _standardize(raw, "date", "location", "aqi", "datagov_pollutants", state_col="state")
    return _fallback_multi_state("datagov_pollutants", freq="D", years_back=8)


def load_kaggle_bhautik_2022_2025():
    if os.path.exists(PATH_KAGGLE_BHAUTIK):
        raw = pd.read_csv(PATH_KAGGLE_BHAUTIK)
        return _standardize(raw, "Date", "City", "AQI", "kaggle_bhautik")
    return _fallback_multi_state("kaggle_bhautik", freq="D", years_back=3)


def load_kaggle_realtime_2023_2025():
    if os.path.exists(PATH_KAGGLE_ISHANKAT):
        raw = pd.read_csv(PATH_KAGGLE_ISHANKAT)
        return _standardize(raw, "Timestamp", "City", "AQI", "kaggle_realtime")
    return _fallback_multi_state("kaggle_realtime", freq="h", years_back=1)


def fetch_latest_realtime_data(state, city, hours=24):
    """Pulls the most recent `hours` of AQI/pollutant readings for a city from a live API.
    Replace REALTIME_API_URL / REALTIME_API_KEY above with your real data.gov.in / CPCB /
    AIKosh resource + key. Falls back to a synthetic 'live' window so the Live Forecast
    tab always has fresh-looking data to demo with.
    """
    try:
        import requests
        resp = requests.get(REALTIME_API_URL, params={"api-key": REALTIME_API_KEY, "format": "json",
                                                        "filters[city]": city}, timeout=6)
        resp.raise_for_status()
        js = resp.json()
        raw = pd.DataFrame(js.get("records", []))
        if raw.empty:
            raise ValueError("empty response")
        return _standardize(raw, "last_update", "city", "pollutant_avg", "realtime_api", state_col=None)
    except Exception:
        end = pd.Timestamp.now().floor("h")
        start = end - pd.Timedelta(hours=hours)
        df = _synthetic_series(city, state, start, end, freq="h", seed_offset=int(end.timestamp()) % 1000)
        df["source"] = "realtime_api_fallback"
        return df


# ---------------------------------------------------------------------------
# 4. Unified access
# ---------------------------------------------------------------------------
LOADERS = {
    "dataful_statewise": load_dataful_statewise,
    "datagov_pollutants": load_data_gov_pollutants,
    "kaggle_bhautik": load_kaggle_bhautik_2022_2025,
    "kaggle_realtime": load_kaggle_realtime_2023_2025,
}


def load_dataset(dataset_key):
    """Loads + cleans one dataset (dedup, interpolate short gaps, clip outliers)."""
    df = LOADERS[dataset_key]()
    df = df.drop_duplicates(subset=["date", "city"]).sort_values(["city", "date"])
    df[["AQI"] + POLLUTANTS] = df.groupby("city")[["AQI"] + POLLUTANTS].transform(
        lambda s: s.interpolate(limit=3).ffill().bfill())
    for col in ["AQI"] + POLLUTANTS:
        lo, hi = df[col].quantile([0.001, 0.999])
        df[col] = df[col].clip(lo, hi)
    return df.dropna(subset=["AQI"]).reset_index(drop=True)


def load_all_datasets():
    return {k: load_dataset(k) for k in LOADERS}


def states_available(df):
    return sorted(df["state"].dropna().unique())


def cities_in_state(df, state):
    return sorted(df.loc[df.state == state, "city"].unique())
