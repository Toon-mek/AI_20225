# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import gdown

st.set_page_config(page_title="Laptop Recommender (BMCS2009)", layout="wide")

DATA_PATH = "laptop_dataset_expanded_myr_full_clean.csv"
DRIVE_ID = st.secrets.get("DATA_DRIVE_ID", "") or "https://docs.google.com/spreadsheets/d/18QknTkpJ-O_26Aj41aRKoEiN6a34vX5VpcXyAkkObp4/edit?usp=sharing" 

# ─────────────────────────────
# Data loading
# ─────────────────────────────
@st.cache_data(show_spinner=False, ttl=24*3600)
def _download_from_drive(file_id: str, output: str) -> str:
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=True)
    return output

@st.cache_data(show_spinner=False)
def load_df() -> pd.DataFrame:
    p = Path(DATA_PATH)
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        source = "local file"
    else:
        if not DRIVE_ID or DRIVE_ID == "PUT_YOUR_DRIVE_FILE_ID_HERE":
            raise FileNotFoundError(
                f"CSV not found at {DATA_PATH} and DRIVE_ID not set.\n"
                f"Either place the CSV in repo root or set st.secrets['DATA_DRIVE_ID']."
            )
        tmp = "/tmp/laptops.csv"
        downloaded = _download_from_drive(DRIVE_ID, tmp)
        df = pd.read_csv(downloaded, low_memory=False)  # add compression="infer" if using .csv.gz
        source = "Google Drive (cached)"
    return df, source

# ─────────────────────────────
# Baseline rule-based recommender (simple & fast)
# ─────────────────────────────
EXPECTED = [
    "brand","series","model","year",
    "cpu_brand","cpu_family","cpu_model","cpu_cores",
    "gpu_brand","gpu_model","gpu_vram_GB",
    "ram_base_GB","ram_type",
    "storage_primary_type","storage_primary_capacity_GB",
    "display_size_in","display_resolution","display_refresh_Hz","display_panel",
    "battery_capacity_Wh","weight_kg",
    "os","price_myr","intended_use_case"
]
NUMERIC = [
    "year","cpu_cores","gpu_vram_GB","ram_base_GB",
    "storage_primary_capacity_GB","display_size_in","display_refresh_Hz",
    "battery_capacity_Wh","weight_kg","price_myr"
]

def prepare_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # ensure columns exist
    for c in EXPECTED:
        if c not in df.columns:
            df[c] = np.nan if c in NUMERIC else ""

    # normalize
    df["model"] = df["model"].fillna("").astype(str)
    df = df[df["model"].str.len() > 0].copy()

    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["cpu_brand","gpu_brand","ram_type","storage_primary_type","display_resolution"]:
        df[c] = df[c].astype(str).str.upper()
    df["os"] = df["os"].astype(str).str.title()

    return df.reset_index(drop=True)

def _nz(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)

def recommend_rule_based(df: pd.DataFrame, prefs: dict, top_k: int = 10) -> pd.DataFrame:
    view = df.copy()

    # budget filter (keep NaN-priced too so user can still see models without price)
    pmin, pmax = prefs.get("budget_min", 0), prefs.get("budget_max", float("inf"))
    if "price_myr" in view:
        m = view["price_myr"].between(pmin, pmax, inclusive="both") | view["price_myr"].isna()
        view = view[m].copy()

    # signals
    ram = _nz(view.get("ram_base_GB", pd.Series()))
    vram = _nz(view.get("gpu_vram_GB", pd.Series()))
    cores = _nz(view.get("cpu_cores", pd.Series()))
    refresh = _nz(view.get("display_refresh_Hz", pd.Series()), 60)
    battery = _nz(view.get("battery_capacity_Wh", pd.Series()), 40)
    weight = _nz(view.get("weight_kg", pd.Series()), 2.0)
    year = _nz(view.get("year", pd.Series()), 2018)
    storage = _nz(view.get("storage_primary_capacity_GB", pd.Series()), 256)

    score = np.zeros(len(view), dtype=float)
    # general desirability
    score += 0.10 * np.clip((year - 2018) / 7.0, 0, 1)          # newer
    score += 0.10 * np.clip(storage / 1024.0, 0, 1)             # 1TB nice

    use = str(prefs.get("use_case", "")).lower()
    if use in ("gaming", "gamer"):
        score += 0.18 * np.clip((vram - 6) / 6.0, 0, 1)
        score += 0.15 * np.clip((cores - 6) / 6.0, 0, 1)
        score += 0.12 * np.clip((refresh - 120) / 60.0, 0, 1)
    elif use in ("creator", "content creation", "video editing", "designer"):
        score += 0.18 * np.clip((ram - 16) / 16.0, 0, 1)
        score += 0.12 * np.clip((vram - 4) / 8.0, 0, 1)
        score += 0.10 * view["display_resolution"].astype(str).str.contains(
            "2560|2880|3000|3200|3840|4K", case=False, na=False
        ).astype(float)
    elif use in ("student", "office", "productivity"):
        score += 0.12 * np.clip((ram - 8) / 8.0, 0, 1)
        score += 0.18 * np.clip((60 - weight) / 60.0, 0, 1)     # lighter better
        score += 0.10 * np.clip((battery - 50) / 40.0, 0, 1)
    elif use in ("business", "programming", "data"):
        score += 0.16 * np.clip((cores - 8) / 8.0, 0, 1)
        score += 0.14 * np.clip((ram - 16) / 16.0, 0, 1)

    view = view.assign(score=score).sort_values("score", ascending=False)

    cols = [c for c in [
        "brand","series","model","year","price_myr",
        "cpu_brand","cpu_family","cpu_model","cpu_cores",
        "gpu_brand","gpu_model","gpu_vram_GB",
        "ram_base_GB","ram_type",
        "storage_primary_type","storage_primary_capacity_GB",
        "display_size_in","display_resolution","display_refresh_Hz","display_panel",
        "battery_capacity_Wh","weight_kg","os","intended_use_case","score"
    ] if c in view.columns]

    return view[cols].head(top_k).reset_index(drop=True)

# ─────────────────────────────
# UI
# ─────────────────────────────
st.title("💻 Laptop Recommender — Simple Baseline")

# load
try:
    raw_df, src = load_df()
except Exception as e:
    st.error(str(e))
    st.stop()

df = prepare_df(raw_df)
st.caption(f"{len(df)} models loaded from {src}")

# sidebar preferences
with st.sidebar:
    st.subheader("Preferences")

    price_col = pd.to_numeric(df["price_myr"], errors="coerce")
    price_min = int(max(0, np.nanmin(price_col))) if np.isfinite(np.nanmin(price_col)) else 0
    price_max = int(min(20000, np.nanmax(price_col))) if np.isfinite(np.nanmax(price_col)) else 20000

    budget = st.slider(
        "Budget (MYR)",
        min_value=0,
        max_value=max(price_max, 10000),
        value=(max(price_min, 1500), min(price_max, 8000)),
        step=100,
    )

    if "intended_use_case" in df and df["intended_use_case"].dropna().size:
        options = sorted({str(x) for x in df["intended_use_case"].dropna() if str(x).strip()})
    else:
        options = ["Student","Gaming","Creator","Business","Programming"]
    use_case = st.selectbox("Use case", options)

    top_k = st.slider("Top K", 3, 20, 10)

prefs = dict(budget_min=budget[0], budget_max=budget[1], use_case=use_case)

# recommend
with st.spinner("Scoring laptops..."):
    recs = recommend_rule_based(df, prefs, top_k=top_k)

st.subheader("Results (Rule-based)")
st.dataframe(recs, use_container_width=True)

st.download_button(
    "Download results (CSV)",
    recs.to_csv(index=False).encode("utf-8"),
    file_name="laptop_recommendations_baseline.csv",
)
