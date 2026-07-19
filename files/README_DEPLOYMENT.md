# Web App & Deployment Guide — AQI India Statewise Platform

## 1. Repo layout to deploy
```
aqi-statewise-platform/
├── app.py
├── data_loader.py
├── models.py
├── requirements.txt
├── data/                       # from train_and_export.py's export/data/
│   ├── dataful_statewise.csv
│   ├── datagov_pollutants.csv
│   ├── kaggle_bhautik.csv
│   ├── kaggle_realtime.csv
│   ├── state_risk_scores_overall.csv
│   └── state_risk_scores_by_dataset.csv
├── models/                     # from export/models/
│   ├── artifact_bundle.pkl
│   └── dataset_info.pkl
└── outputs/                    # from export/outputs/
    ├── forecast_<dataset>_<state>.png
    ├── seasonal_<dataset>_<state>.png
    └── model_metrics_by_dataset_state.csv
```
`app.py` works even with **none of the `data/`, `models/`, `outputs/` files present** — it calls
`data_loader.py` directly (synthetic fallback), so your first deploy is always live.

## 2. Train (Colab) — optional but recommended before deploying
1. Open a new Colab notebook, upload `data_loader.py`, `models.py`, `train_and_export.py`, and any
   real CSVs you have (edit the `PATH_*` placeholders at the top of `data_loader.py` first).
2. Run:
   ```python
   !pip -q install xgboost shap
   !python train_and_export.py
   !zip -r export.zip export
   from google.colab import files
   files.download("export.zip")
   ```
3. Unzip `export/` into your repo (as laid out above).

## 3. Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

## 4. Deploy for free — Option A: Streamlit Community Cloud (recommended)
1. Push the repo to GitHub.
2. https://share.streamlit.io → **New app** → select repo/branch → file path `app.py`.
3. If using the live API (Section 5 of the app), add your key under **App settings → Secrets**:
   ```toml
   DATA_GOV_API_KEY = "your_real_key_here"
   ```
4. Deploy. You'll get a public URL like `https://<app-name>-<id>.streamlit.app` —
   **this is your deployment link.**

## 5. Deploy for free — Option B: Hugging Face Spaces
1. New Space → SDK = Streamlit.
2. Upload all files/folders above (or `git push` to the Space's repo).
3. Add `DATA_GOV_API_KEY` under **Settings → Repository secrets**.
4. URL: `https://huggingface.co/spaces/<username>/<space-name>` — **your deployment link.**

## 6. Live 24-hour forecast — connecting a real API
In `data_loader.py`, replace:
```python
REALTIME_API_URL = "https://api.data.gov.in/resource/REPLACE_WITH_RESOURCE_ID"
REALTIME_API_KEY = os.environ.get("DATA_GOV_API_KEY", "REPLACE_WITH_YOUR_KEY")
```
with your actual data.gov.in / CPCB / AIKosh resource URL, and set `DATA_GOV_API_KEY` as a secret
(Section 4/5). Locally, `export DATA_GOV_API_KEY=your_key` before `streamlit run app.py`.
Without a key, the Live Forecast tab clearly labels its data as `realtime_api_fallback` (synthetic)
so it's always honest about data provenance.

## 7. Optional — shareable landing page
```html
<!DOCTYPE html>
<html><body style="margin:0">
  <iframe src="https://<your-app>.streamlit.app" style="border:none;position:fixed;
          top:0;left:0;width:100%;height:100%;"></iframe>
</body></html>
```
Host via GitHub Pages for a second link, e.g. `https://<username>.github.io/aqi-statewise-platform/`.

## 8. Updating after retraining
Re-run `train_and_export.py`, re-copy `export/` into the repo, commit + push —
Streamlit Community Cloud and Hugging Face Spaces both auto-redeploy on push.
