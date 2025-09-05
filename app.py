# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Config & Page
# =========================
st.set_page_config(page_title="💻 Laptop Recommender (BMCS2009)", layout="wide")

# Local CSV name (if present, we use this)
DATA_PATH = "laptop_dataset_expanded_myr_full_clean.csv"
# Or set your Google Drive file id here or in Streamlit Secrets as DATA_DRIVE_ID
DRIVE_ID = st.secrets.get("DATA_DRIVE_ID", "") or "https://docs.google.com/spreadsheets/d/18QknTkpJ-O_26Aj41aRKoEiN6a34vX5VpcXyAkkObp4/edit?usp=sharing"

# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False, ttl=24*3600)
def download_data_from_drive(file_id: str, output: str) -> str:
    if not file_id or file_id == "PUT_YOUR_DRIVE_FILE_ID_HERE":
        raise FileNotFoundError("CSV not found locally and Google Drive ID not set.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=True)
    return output

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    p = Path(DATA_PATH)
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    tmp = "/tmp/laptops.csv"
    downloaded = download_data_from_drive(DRIVE_ID, tmp)
    return pd.read_csv(downloaded, low_memory=False)

# =========================
# Prep / Utilities
# =========================
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

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # ensure columns exist
    for c in EXPECTED:
        if c not in data.columns:
            data[c] = np.nan if c in NUMERIC else ""
    # normalize
    data["model"] = data["model"].fillna("").astype(str)
    data = data[data["model"].str.len() > 0].copy()
    for c in NUMERIC:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    for c in ["cpu_brand","gpu_brand","ram_type","storage_primary_type","display_resolution"]:
        data[c] = data[c].astype(str).str.upper()
    data["os"] = data["os"].astype(str).str.title()

    # build a simple text field to drive TF-IDF (spec string)
    def tok(name, val):
        s = str(val).strip()
        return s if s else ""
    def numtok(prefix, val, unit=""):
        try:
            if pd.isna(val): return ""
            v = float(val)
            if v.is_integer(): v = int(v)
            return f"{prefix}{v}{unit}"
        except Exception:
            return ""

    text_parts = []
    for _, r in data.iterrows():
        parts = [
            tok("brand", r["brand"]), tok("series", r["series"]), tok("model", r["model"]),
            tok("cpu_brand", r["cpu_brand"]), tok("cpu_family", r["cpu_family"]), tok("cpu_model", r["cpu_model"]),
            tok("gpu_brand", r["gpu_brand"]), tok("gpu_model", r["gpu_model"]),
            numtok("RAM", r["ram_base_GB"], "GB"), tok("RAMTYPE", r["ram_type"]),
            tok("STOR", r["storage_primary_type"]), numtok("SSD", r["storage_primary_capacity_GB"], "GB"),
            numtok("SIZE", r["display_size_in"], "IN"), tok("RES", r["display_resolution"]),
            numtok("HZ", r["display_refresh_Hz"], "HZ"), tok("PANEL", r["display_panel"]),
            numtok("BAT", r["battery_capacity_Wh"], "WH"), numtok("WT", r["weight_kg"], "KG"),
            tok("OS", r["os"]), tok("USE", r["intended_use_case"]), numtok("Y", r["year"])
        ]
        text_parts.append(" ".join(p for p in parts if p))
    data["spec_text"] = text_parts

    # dedupe by model + year if present
    if "year" in data:
        data = data.drop_duplicates(subset=["model","year"], keep="first")
    else:
        data = data.drop_duplicates(subset=["model"], keep="first")

    return data.reset_index(drop=True)

# =========================
# Content features (TF-IDF)
# =========================
@st.cache_resource(show_spinner=False)
def build_tfidf(spec_text: pd.Series):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words=None)
    X = vec.fit_transform(spec_text.fillna(""))
    return vec, X

@st.cache_data(show_spinner=False)
def compute_similarity_to_row(_X, row_index: int) -> np.ndarray:
    sim = cosine_similarity(_X[row_index], _X).ravel()
    return sim

@st.cache_data(show_spinner=False)
def compute_similarity_to_query(_vec: TfidfVectorizer, _X, query_text: str) -> np.ndarray:
    q = _vec.transform([query_text])
    sim = cosine_similarity(q, _X).ravel()
    return sim

# =========================
# Rule-based scoring
# =========================
def rule_based_scores(view: pd.DataFrame, use_case: str) -> np.ndarray:
    def nz(col, default=0.0):
        return pd.to_numeric(view.get(col, default), errors="coerce").fillna(default).astype(float)

    ram = nz("ram_base_GB")
    vram = nz("gpu_vram_GB")
    cores = nz("cpu_cores")
    refresh = nz("display_refresh_Hz", 60)
    battery = nz("battery_capacity_Wh", 40)
    weight = nz("weight_kg", 2.0)
    year = nz("year", 2018)
    storage = nz("storage_primary_capacity_GB", 256)

    s = np.zeros(len(view), dtype=float)
    s += 0.10 * np.clip((year - 2018) / 7.0, 0, 1)
    s += 0.10 * np.clip(storage / 1024.0, 0, 1)

    u = str(use_case or "").lower()
    if u in ("gaming","gamer"):
        s += 0.18 * np.clip((vram - 6) / 6.0, 0, 1)
        s += 0.15 * np.clip((cores - 6) / 6.0, 0, 1)
        s += 0.12 * np.clip((refresh - 120) / 60.0, 0, 1)
    elif u in ("creator","content creation","video editing","designer"):
        s += 0.18 * np.clip((ram - 16) / 16.0, 0, 1)
        s += 0.12 * np.clip((vram - 4) / 8.0, 0, 1)
        s += 0.10 * view["display_resolution"].astype(str).str.contains(
            "2560|2880|3000|3200|3840|4K", case=False, na=False
        ).astype(float)
    elif u in ("student","office","productivity"):
        s += 0.12 * np.clip((ram - 8) / 8.0, 0, 1)
        s += 0.18 * np.clip((60 - weight) / 60.0, 0, 1)
        s += 0.10 * np.clip((battery - 50) / 40.0, 0, 1)
    elif u in ("business","programming","data"):
        s += 0.16 * np.clip((cores - 8) / 8.0, 0, 1)
        s += 0.14 * np.clip((ram - 16) / 16.0, 0, 1)
    return np.clip(s, 0, 1)

# =========================
# Recommenders
# =========================
def recommend_similar(df: pd.DataFrame, vec, X, selected_model: str, top_n: int = 5):
    idxs = df.index[df["model"].astype(str).str.lower() == selected_model.lower()].tolist()
    if not idxs:
        st.write("Model not found.")
        return pd.DataFrame()
    i = idxs[0]
    sim = compute_similarity_to_row(X, i)
    order = np.argsort(-sim)
    order = [j for j in order if j != i][:top_n]
    out = df.iloc[order].copy()
    out["similarity"] = sim[order]
    return out

def build_pref_query(prefs: dict) -> str:
    # build a "fake" spec text from user preferences
    parts = []
    if prefs.get("use_case"): parts += [f"USE {prefs['use_case']}"]
    if prefs.get("min_ram"): parts += [f"RAM{prefs['min_ram']}GB"]
    if prefs.get("min_vram"): parts += [f"VRAM{prefs['min_vram']}GB"]
    if prefs.get("min_cpu_cores"): parts += [f"CORES{prefs['min_cpu_cores']}"]
    if prefs.get("min_refresh"): parts += [f"HZ{prefs['min_refresh']}HZ"]
    if prefs.get("min_storage"): parts += [f"SSD{prefs['min_storage']}GB"]
    if prefs.get("min_year"): parts += [f"Y{prefs['min_year']}"]
    if prefs.get("min_battery_wh"): parts += [f"BAT{prefs['min_battery_wh']}WH"]
    if prefs.get("max_weight"): parts += [f"WT{prefs['max_weight']}KG"]
    return " ".join(parts)

def recommend_by_prefs(df: pd.DataFrame, vec, X, prefs: dict, algo: str, top_n: int = 10):
    # budget prefilter (keep NaN price too)
    pmin, pmax = prefs.get("budget_min", 0), prefs.get("budget_max", float("inf"))
    m = (df["price_myr"].between(pmin, pmax, inclusive="both") | df["price_myr"].isna()) if "price_myr" in df else np.ones(len(df), dtype=bool)
    view = df[m].copy()
    if view.empty:
        return view

    if algo == "Rule-Based":
        s = rule_based_scores(view, prefs.get("use_case"))
    else:
        q = build_pref_query(prefs)
        sim = compute_similarity_to_query(vec, X, q)
        sim = sim[m]
        if algo == "Content-Based":
            s = sim
        else:
            # Hybrid
            if sim.max() > sim.min():
                sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
            else:
                sim_norm = sim
            rb = rule_based_scores(view, prefs.get("use_case"))
            a = float(prefs.get("alpha", 0.6))
            s = a * sim_norm + (1 - a) * rb

    view = view.copy()
    view["score"] = s
    cols = [c for c in [
        "brand","series","model","year","price_myr",
        "cpu_brand","cpu_family","cpu_model","cpu_cores",
        "gpu_brand","gpu_model","gpu_vram_GB",
        "ram_base_GB","ram_type",
        "storage_primary_type","storage_primary_capacity_GB",
        "display_size_in","display_resolution","display_refresh_Hz","display_panel",
        "battery_capacity_Wh","weight_kg","os","intended_use_case","score"
    ] if c in view.columns]
    return view.sort_values("score", ascending=False).head(top_n)[cols].reset_index(drop=True)

# =========================
# UI (similar style to your song app)
# =========================
# Simple CSS accent
st.markdown("""
<style>
h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
.stButton>button { background-color: #0ea5e9; color: white; border-radius: 8px; }
.stTextInput input { border: 1px solid #0ea5e9; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("💻 Laptop Recommender — Search, Similar, and Preferences")

# Load + prep
try:
    raw = load_dataset()
except Exception as e:
    st.error(str(e))
    st.stop()

df = prepare_df(raw)
st.caption(f"{len(df)} laptops loaded from {'local file' if Path(DATA_PATH).exists() else 'Google Drive (cached)'}")

# Build TF-IDF space
vec, X = build_tfidf(df["spec_text"])

# -------- Search block (like your previous project) --------
search_term = st.text_input("Search for a Laptop Model or Brand 🔎").strip()
if search_term:
    found = df[
        df["model"].str.contains(search_term, case=False, na=False) |
        df["brand"].str.contains(search_term, case=False, na=False) |
        df["series"].str.contains(search_term, case=False, na=False)
    ].copy()

    found = found.sort_values(by=["brand","series","model"]).reset_index(drop=True)
    if found.empty:
        st.write("No laptops found for that search.")
    else:
        st.write(f"### Search Results for: {search_term}")
        for idx, row in found.head(30).iterrows():
            with st.container():
                st.markdown(f"<h3><b>{idx+1}. {row['brand']} {row['series']} {row['model']}</b></h3>", unsafe_allow_html=True)
                st.markdown(f"*Price (MYR):* {row['price_myr'] if pd.notna(row['price_myr']) else 'Unknown'}")
                st.markdown(f"*CPU:* {row['cpu_brand']} {row['cpu_family']} {row['cpu_model']}  |  *Cores:* {row['cpu_cores']}")
                st.markdown(f"*GPU:* {row['gpu_brand']} {row['gpu_model']}  |  *VRAM:* {row['gpu_vram_GB']} GB")
                st.markdown(f"*RAM:* {row['ram_base_GB']} GB {row['ram_type']}  |  *Storage:* {row['storage_primary_capacity_GB']} GB {row['storage_primary_type']}")
                st.markdown(f"*Display:* {row['display_size_in']}-inch {row['display_resolution']} {row['display_refresh_Hz']}Hz {row['display_panel']}")
                with st.expander("Show/Hide more specs"):
                    st.write(row.to_frame().T)
                st.markdown("---")

        choice_list = found["model"].head(50).tolist()
        selected_model = st.selectbox("Select a model to get similar laptops 🎧", choice_list)
        if st.button("Recommend Similar Laptops"):
            rec_sim = recommend_similar(df, vec, X, selected_model, top_n=5)
            if rec_sim.empty:
                st.write("No similar items found.")
            else:
                st.write(f"### Similar to **{selected_model}**")
                for k, r in enumerate(rec_sim.itertuples(index=False), 1):
                    st.markdown(f"**{k}. {r.brand} {r.series} {r.model}** — Score: {getattr(r, 'similarity', 0):.2f}")
                    st.markdown(f"*CPU:* {r.cpu_brand} {r.cpu_family} {r.cpu_model} | *GPU:* {r.gpu_brand} {r.gpu_model}")
                    st.markdown(f"*RAM/Storage:* {r.ram_base_GB}GB / {r.storage_primary_capacity_GB}GB {r.storage_primary_type} | *Price:* {r.price_myr}")
                    st.markdown("---")

# -------- Preference block (rule/content/hybrid) --------
st.write("## Recommend by Preferences")
col_a, col_b = st.columns(2)
with col_a:
    price_min = int(max(0, pd.to_numeric(df["price_myr"], errors="coerce").min(skipna=True))) if "price_myr" in df else 0
    price_max = int(min(20000, pd.to_numeric(df["price_myr"], errors="coerce").max(skipna=True))) if "price_myr" in df else 20000
    budget = st.slider("Budget (MYR)", 0, max(price_max, 10000),
                       (max(price_min, 1500), min(price_max, 8000)), 100)
    use_options = sorted({str(x) for x in df["intended_use_case"].dropna() if str(x).strip()}) \
                  if "intended_use_case" in df and df["intended_use_case"].dropna().size else \
                  ["Student","Gaming","Creator","Business","Programming"]
    use_case = st.selectbox("Use case", use_options)
    algo = st.selectbox("Algorithm", ["Hybrid","Content-Based","Rule-Based"])
with col_b:
    min_ram = st.number_input("Min RAM (GB)", 0, 128, 8, 4)
    min_storage = st.number_input("Min Storage (GB)", 0, 4096, 512, 128)
    min_vram = st.number_input("Min GPU VRAM (GB)", 0, 24, 0, 2)
    min_cores = st.number_input("Min CPU Cores", 0, 32, 4, 1)
    min_year = st.number_input("Min Release Year", 2015, 2025, 2019, 1)
    min_refresh = st.number_input("Min Refresh (Hz)", 0, 360, 60, 30)
    min_battery_wh = st.number_input("Min Battery (Wh)", 0, 120, 0, 5)
    max_weight = st.number_input("Max Weight (kg)", 0.0, 6.0, 3.0, 0.1)

alpha = st.slider("Hybrid α (Content weight)", 0.0, 1.0, 0.6, 0.05)
top_n = st.slider("Top N", 3, 30, 10)

prefs = dict(
    budget_min=budget[0], budget_max=budget[1],
    use_case=use_case,
    min_ram=min_ram, min_storage=min_storage, min_vram=min_vram, min_cpu_cores=min_cores,
    min_year=min_year, min_refresh=min_refresh, min_battery_wh=min_battery_wh, max_weight=max_weight,
    alpha=alpha
)

if st.button("Recommend by Preferences"):
    with st.spinner("Scoring..."):
        recs = recommend_by_prefs(df, vec, X, prefs, algo, top_n)
    if recs.empty:
        st.write("No matching laptops found for your preferences.")
    else:
        st.dataframe(recs, use_container_width=True)
        st.download_button(
            "Download results (CSV)",
            recs.to_csv(index=False).encode("utf-8"),
            file_name=f"laptop_recommendations_{algo.lower()}.csv",
        )
