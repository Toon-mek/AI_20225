# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# =========================
# Config & Page
# =========================
st.set_page_config(page_title="💻 Laptop Recommender (BMCS2009)", layout="wide")

DATA_PATH = "laptop_dataset_expanded_myr_full_clean.csv"
DRIVE_ID = "18QknTkpJ-O_26Aj41aRKoEiN6a34vX5VpcXyAkkObp4"
GID = "418897947" 

# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False, ttl=24*3600)
def download_sheet_csv(output="/tmp/laptops.csv"):
    url = f"https://docs.google.com/spreadsheets/d/{DRIVE_ID}/export?format=csv&gid={GID}"
    gdown.download(url, output, quiet=True)
    return output

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    p = Path(DATA_PATH)
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    downloaded = download_sheet_csv("/tmp/laptops.csv")
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
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=None)
    X = vec.fit_transform(spec_text.fillna(""))
    return vec, X

def compute_row_sim(_X, row_index: int) -> np.ndarray:
    return cosine_similarity(_X[row_index], _X).ravel()

def compute_query_sim(_vec: TfidfVectorizer, _X, query_text: str) -> np.ndarray:
    q = _vec.transform([query_text])
    return cosine_similarity(q, _X).ravel()
    
def split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Safer stratified split by 'intended_use_case'.
    - Collapses ultra-rare labels (<2 rows) to 'other'
    - If still infeasible for chosen test_size, falls back to non-stratified.
    """
    if "intended_use_case" not in df.columns:
        tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
        return tr.reset_index(drop=True), te.reset_index(drop=True)

    y = df["intended_use_case"].astype(str).str.strip()
    y = y.replace({"nan": "", "None": ""})
    y = y.mask(y.eq(""), "unknown")

    vc = y.value_counts()
    rare = vc[vc < 2].index
    if len(rare) > 0:
        y = y.where(~y.isin(rare), "other")

    n_test = max(1, int(round(len(df) * test_size)))
    if y.nunique() > n_test:
        order = y.value_counts(ascending=True).index
        keep = set(order[-n_test:])
        y = y.where(y.isin(keep), "other")

    try:
        tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def evaluate_precision_recall_at_k_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    alpha: float = 0.6
) -> pd.DataFrame:
    """
    Fit TF-IDF on TRAIN only; evaluate on TEST.
    For each 'intended_use_case' label, build a preference query, score TEST with Hybrid,
    and compute Precision@K and Recall@K (scenario-level recall).
    """
    # Fit TF-IDF on TRAIN only
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=None)
    X_train = vec.fit_transform(train_df["spec_text"].fillna(""))
    X_test  = vec.transform(test_df["spec_text"].fillna(""))

    # Labels (scenarios) to evaluate
    if "intended_use_case" in train_df and train_df["intended_use_case"].dropna().size:
        labels = sorted({str(x).strip() for x in train_df["intended_use_case"].dropna() if str(x).strip()})
    else:
        labels = ["Gaming", "Student", "Creator", "Business"]

    price_te = pd.to_numeric(test_df.get("price_myr"), errors="coerce") if "price_myr" in test_df else None
    results = []

    for lab in labels:
        # Build a reasonable preference query for this scenario
        prefs = dict(
            use_case=lab,
            budget_min=float(price_te.quantile(0.05)) if price_te is not None else 0,
            budget_max=float(price_te.quantile(0.95)) if price_te is not None else 20000,
            min_ram=8, min_storage=512, min_vram=0, min_cpu_cores=4,
            min_year=2018, min_refresh=60, min_battery_wh=0, max_weight=3.0,
            alpha=alpha,
        )

        # Budget mask on TEST
        if price_te is not None:
            mask = (price_te.between(prefs["budget_min"], prefs["budget_max"], inclusive="both")) | price_te.isna()
        else:
            mask = pd.Series(True, index=test_df.index)

        view = test_df.loc[mask].copy()
        if view.empty:
            results.append({"scenario": lab, "matched@k": 0, "precision@k": np.nan, "recall@k": np.nan})
            continue

        # Content similarity on TEST using TRAIN-fit vectorizer
        query_text = build_pref_query(prefs)  # uses your existing helper
        q = vec.transform([query_text])
        sim_all = cosine_similarity(q, X_test).ravel()
        sim = sim_all[mask.values]

        # Hybrid score
        if sim.max() > sim.min():
            sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
        else:
            sim_norm = sim
        rb = rule_based_scores(view, lab)  # reuses your rule-based function
        scores = alpha * sim_norm + (1 - alpha) * rb

        topk = view.assign(score=scores).sort_values("score", ascending=False).head(k)

        # Scenario-level match: does top-K contain the target label?
        truth_labels = topk.get("intended_use_case", pd.Series(dtype=str)).astype(str).str.lower()
        matched = (truth_labels == lab.lower()).sum()
        precision = matched / max(k, 1)
        recall = 1.0 if matched > 0 else 0.0

        results.append({
            "scenario": lab,
            "matched@k": int(matched),
            "precision@k": round(precision, 3),
            "recall@k": round(recall, 3),
        })

    return pd.DataFrame(results)


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
    sim = compute_row_sim(X, i)
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
        sim = compute_query_sim(vec, X, q)
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

with st.sidebar:
    if st.button("🔄 Clear cache & rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
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
    st.dataframe(recs, width="stretch")
    st.download_button(
        "Download results (CSV)",
        recs.to_csv(index=False).encode("utf-8"),
        file_name=f"laptop_recommendations_{algo.lower()}.csv",
    )
DEV_MODE = True  # set to False to hide this panel from normal users

if DEV_MODE:
    with st.expander("Train/Test evaluation (Precision@K & Recall@K)"):
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="tt_size")
        k_eval    = st.slider("K", 3, 20, 10, key="tt_k")
        alpha_tt  = st.slider("Hybrid α (content weight)", 0.0, 1.0, float(alpha) if "alpha" in locals() else 0.6, 0.05, key="tt_alpha")

        if st.button("Run train/test evaluation"):
            tr_df, te_df = split_df(df, test_size=test_size)
            res = evaluate_precision_recall_at_k_train_test(tr_df, te_df, k=k_eval, alpha=alpha_tt)

            st.write(f"**Train:** {len(tr_df)}  |  **Test:** {len(te_df)}")
            if "intended_use_case" in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Train label counts")
                    st.write(tr_df["intended_use_case"].value_counts(dropna=False))
                with col2:
                    st.write("Test label counts")
                    st.write(te_df["intended_use_case"].value_counts(dropna=False))

            st.dataframe(res, width=True)
            if not res.empty and res["precision@k"].notna().any():
                st.write(f"**Mean Precision@{k_eval}:** {res['precision@k'].mean():.3f}")
                st.write(f"**Mean Recall@{k_eval}:** {res['recall@k'].mean():.3f}")

            st.download_button(
                "Download evaluation (CSV)",
                res.to_csv(index=False).encode("utf-8"),
                file_name=f"eval_precision_recall_at_{k_eval}.csv",
            )
