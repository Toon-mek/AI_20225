import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Config & Page
st.set_page_config(page_title="💻 Laptop Recommender (BMCS2009)", layout="wide")

DATA_PATH = "laptop_dataset_expanded_myr_full_clean.csv"
DRIVE_ID = "18QknTkpJ-O_26Aj41aRKoEiN6a34vX5VpcXyAkkObp4"
GID = "418897947" 

# Data loading
@st.cache_data(show_spinner=False, ttl=24*3600)
def download_sheet_csv(output="/tmp/laptops.csv"):
    url = f"https://docs.google.com/spreadsheets/d/{DRIVE_ID}/export?format=csv&gid={GID}"
    try:
        gdown.download(url, output, quiet=True, fuzzy=True)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download CSV from Google Sheets. "
            f"Check DRIVE_ID='{DRIVE_ID}' and GID='{GID}'. Details: {e}"
        )
    p = Path(output)
    if (not p.exists()) or p.stat().st_size == 0:
        raise FileNotFoundError(
            "Downloaded file is empty or missing. Verify the sheet is public/readable and the GID is correct."
        )
    return output

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    p = Path(DATA_PATH)
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    csv_path = download_sheet_csv("/tmp/laptops.csv")
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV at {csv_path}. Parser/encoding issue? Details: {e}")

# Prep / Utilities
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
# === Step 4A: Data validation helpers (place right after prepare_df) ===
def summarize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    s = df.isna().sum()
    out = pd.DataFrame({"column": s.index, "nulls": s.values})
    out["pct"] = (out["nulls"] / max(len(df), 1)) * 100
    return out.sort_values("nulls", ascending=False).reset_index(drop=True)

def range_checks(df: pd.DataFrame) -> list[str]:
    msgs = []

    # year sanity
    if "year" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce")
        bad = ((y < 2010) | (y > 2026)) & y.notna()
        if bad.any():
            msgs.append(f"{int(bad.sum())} rows with suspicious year (<2010 or >2026).")

    # price sanity
    if "price_myr" in df.columns:
        p = pd.to_numeric(df["price_myr"], errors="coerce")
        if (p < 0).sum() > 0:
            msgs.append(f"{int((p < 0).sum())} rows with negative price.")
        if p.notna().sum() > 0:
            qhi = p.quantile(0.995)
            if (p > qhi).sum() > 0:
                msgs.append(f"{int((p > qhi).sum())} extreme price outliers (>99.5th percentile).")

    # numeric spec sanity
    bounds = [
        ("ram_base_GB", 2, 128),
        ("gpu_vram_GB", 0, 48),
        ("cpu_cores", 1, 64),
        ("storage_primary_capacity_GB", 64, 8192),
        ("display_refresh_Hz", 30, 360),
        ("weight_kg", 0.5, 6.0),
    ]
    for col, lo, hi in bounds:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = ((v < lo) | (v > hi)) & v.notna()
            if bad.any():
                msgs.append(f"{int(bad.sum())} values in '{col}' outside [{lo}, {hi}].")

    return msgs

# Content features (TF-IDF)
@st.cache_resource(show_spinner=False)
def build_tfidf(spec_text: pd.Series):
    s = spec_text.fillna("").astype(str)
    if not s.str.strip().any():
        # All empty -> return empty matrix
        from scipy.sparse import csr_matrix
        vec = TfidfVectorizer()
        X = csr_matrix((len(s), 0))
        return vec, X
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=None)
        X = vec.fit_transform(s)
    except ValueError:
        # Fallback for tiny datasets
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words=None)
        X = vec.fit_transform(s)
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

# Rule-based scoring
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

# Recommenders
def recommend_similar(df: pd.DataFrame, vec, X, selected_model: str, top_n: int = 5):
    idxs = df.index[df["model"].astype(str).str.lower() == selected_model.lower()].tolist()
    if not idxs:
        st.write("Model not found.")
        return pd.DataFrame()

    i = idxs[0]

    # Guard: empty TF-IDF vocabulary
    if getattr(X, "shape", (0, 0))[1] == 0:
        st.warning("Similarity is unavailable (empty TF-IDF vocabulary). Add more text features or rows.")
        return pd.DataFrame()

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

# ---------- Validations & business rules for preferences ----------
def validate_prefs(prefs: dict) -> list[str]:
    """Return a list of error messages; empty list means OK."""
    errs = []
    # numbers must be non-negative
    for k in ["budget_min","budget_max","min_ram","min_storage","min_vram",
              "min_cpu_cores","min_refresh","min_battery_wh","max_weight"]:
        if k in prefs and prefs[k] is not None:
            try:
                v = float(prefs[k])
                if v < 0:
                    errs.append(f"{k} must be ≥ 0.")
            except Exception:
                errs.append(f"{k} must be a number.")

    # budget range
    if "budget_min" in prefs and "budget_max" in prefs:
        try:
            if float(prefs["budget_min"]) > float(prefs["budget_max"]):
                errs.append("budget_min cannot be greater than budget_max.")
        except Exception:
            pass

    # sensible year
    if prefs.get("min_year") is not None:
        try:
            y = int(prefs["min_year"])
            if y < 2010 or y > 2035:
                errs.append("min_year looks out of range (2010–2035).")
        except Exception:
            errs.append("min_year must be an integer.")
    return errs

def enforce_business_rules(prefs: dict) -> tuple[dict, list[str]]:
    """
    Adjust questionable inputs to sensible values; return (adjusted_prefs, notes).
    """
    p = dict(prefs)  # copy
    notes = []

    # Round RAM to nearest 4 GB (up)
    if p.get("min_ram") is not None:
        try:
            r = int(np.ceil(float(p["min_ram"]) / 4.0) * 4)
            if r != p["min_ram"]:
                notes.append(f"Rounded min_ram to {r} GB.")
            p["min_ram"] = r
        except Exception:
            pass

    # Gaming defaults
    u = str(p.get("use_case") or "").lower()
    if u in ("gaming","gamer","budget gaming"):
        if p.get("min_vram") is None or float(p["min_vram"]) < 4:
            p["min_vram"] = 4
            notes.append("Set min_vram to 4 GB for gaming.")
        if p.get("min_refresh") is None or float(p["min_refresh"]) < 120:
            p["min_refresh"] = 120
            notes.append("Set min_refresh to 120 Hz for gaming.")

    # Weight sanity
    if p.get("max_weight") is not None:
        try:
            w = float(p["max_weight"])
            if w < 0.8:
                p["max_weight"] = 0.8
                notes.append("Raised max_weight to 0.8 kg (very light).")
            elif w > 6.0:
                p["max_weight"] = 6.0
                notes.append("Capped max_weight at 6.0 kg.")
        except Exception:
            pass

    # Budget sanity
    if p.get("budget_min") is not None and p.get("budget_max") is not None:
        try:
            lo, hi = float(p["budget_min"]), float(p["budget_max"])
            if lo > hi:
                p["budget_min"], p["budget_max"] = hi, lo
                notes.append("Swapped budget_min and budget_max.")
        except Exception:
            pass

    return p, notes

def make_filter_mask(df: pd.DataFrame, prefs: dict) -> pd.Series:
    """
    Build a boolean mask for hard filters. Missing values in the data do NOT
    exclude rows (i.e., they pass the filter).
    """
    n = len(df)
    mask = pd.Series(True, index=df.index)

    def ge(col, v):
        s = pd.to_numeric(df.get(col), errors="coerce")
        return (s >= float(v)) | s.isna()

    def le(col, v):
        s = pd.to_numeric(df.get(col), errors="coerce")
        return (s <= float(v)) | s.isna()

    # Budget
    if "price_myr" in df.columns:
        price = pd.to_numeric(df["price_myr"], errors="coerce")
        lo = float(prefs.get("budget_min", 0))
        hi = float(prefs.get("budget_max", float("inf")))
        mask &= (price.between(lo, hi, inclusive="both")) | price.isna()

    # Numeric mins
    if prefs.get("min_ram") is not None:
        mask &= ge("ram_base_GB", prefs["min_ram"])
    if prefs.get("min_storage") is not None:
        mask &= ge("storage_primary_capacity_GB", prefs["min_storage"])
    if prefs.get("min_vram") is not None:
        mask &= ge("gpu_vram_GB", prefs["min_vram"])
    if prefs.get("min_cpu_cores") is not None:
        mask &= ge("cpu_cores", prefs["min_cpu_cores"])
    if prefs.get("min_year") is not None:
        mask &= ge("year", prefs["min_year"])
    if prefs.get("min_refresh") is not None:
        mask &= ge("display_refresh_Hz", prefs["min_refresh"])
    if prefs.get("min_battery_wh") is not None:
        mask &= ge("battery_capacity_Wh", prefs["min_battery_wh"])

    # Max weight
    if prefs.get("max_weight") is not None:
        mask &= le("weight_kg", prefs["max_weight"])

    return mask

def recommend_by_prefs(df: pd.DataFrame, vec, X, prefs: dict, algo: str, top_n: int = 10) -> pd.DataFrame:
    """Recommend laptops using rule/content/hybrid with validations + business rules."""
    # 1) Validate inputs
    errs = validate_prefs(prefs)
    if errs:
        st.error(" | ".join(errs))
        return pd.DataFrame()

    # 2) Enforce soft business rules
    prefs2, notes = enforce_business_rules(prefs)
    if notes:
        st.info(" ".join(notes))

    # 3) Filter candidates (hard constraints)
    mask = make_filter_mask(df, prefs2)
    view = df.loc[mask].copy()
    if view.empty:
        return view

    # 4) Score
    if algo == "Rule-Based":
        scores = rule_based_scores(view, prefs2.get("use_case"))
    else:
        # Guard: empty TF-IDF vocabulary
        if getattr(X, "shape", (0, 0))[1] == 0:
            st.warning("Content-based scoring unavailable (empty TF-IDF). Showing rule-based instead.")
            scores = rule_based_scores(view, prefs2.get("use_case"))
        else:
            query = build_pref_query(prefs2)
            sim_all = compute_query_sim(vec, X, query)
            sim = sim_all[mask.values]

            if algo == "Content-Based":
                scores = sim
            else:
                # Hybrid
                rng = np.ptp(sim)
                sim_norm = (sim - np.min(sim)) / (rng if rng else 1.0)
                rb = rule_based_scores(view, prefs2.get("use_case"))
                a = float(prefs2.get("alpha", 0.6))
                scores = a * sim_norm + (1 - a) * rb

    # 5) Rank & return
    view["score"] = scores
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

# UI
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
st.title("💻 Laptop Recommender")

# Load + prep
try:
    raw = load_dataset()
except Exception as e:
    st.error(str(e))
    st.stop()

df = prepare_df(raw)
st.caption(f"{len(df)} laptops loaded from {'local file' if Path(DATA_PATH).exists() else 'Google Drive (cached)'}")

with st.expander("📋 Data validation & data quality checks", expanded=False):
    issues = []

    missing_in_raw = [c for c in EXPECTED if c not in raw.columns]
    if missing_in_raw:
        issues.append(f"Missing in source file: {missing_in_raw}")

    key_cols = [c for c in ["brand", "series", "model", "year"] if c in df.columns]
    if key_cols:
        dup_count = df.duplicated(subset=key_cols, keep=False).sum()
        if dup_count > 0:
            issues.append(f"{int(dup_count)} duplicate rows by {key_cols}.")

    nulls = summarize_nulls(df)
    if (nulls["nulls"] > 0).any():
        st.write("**Null counts (post-clean):**")
        st.dataframe(nulls[nulls["nulls"] > 0], width="stretch")

    issues += range_checks(df)

    if issues:
        st.error("• " + "\n• ".join(issues))
    else:
        st.success("No data quality issues detected.")

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

# --- Action: Recommend by Preferences ---
recs = None
if st.button("Recommend by Preferences"):
    with st.spinner("Scoring..."):
        recs = recommend_by_prefs(df, vec, X, prefs, algo, top_n)

if recs is not None:
    if recs.empty:
        st.warning("No matching laptops found for your preferences.")
    else:
        st.dataframe(recs, width="stretch")  # <- replaces deprecated use_container_width
        st.download_button(
            "Download results (CSV)",
            recs.to_csv(index=False).encode("utf-8"),
            file_name=f"laptop_recommendations_{algo.lower()}.csv",
        )
        
DEV_MODE= True

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

            st.dataframe(recs, width="stretch")
            if not res.empty and res["precision@k"].notna().any():
                st.write(f"**Mean Precision@{k_eval}:** {res['precision@k'].mean():.3f}")
                st.write(f"**Mean Recall@{k_eval}:** {res['recall@k'].mean():.3f}")

            st.download_button(
                "Download evaluation (CSV)",
                res.to_csv(index=False).encode("utf-8"),
                file_name=f"eval_precision_recall_at_{k_eval}.csv",
            )
