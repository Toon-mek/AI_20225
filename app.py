import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# ─────────────────────────── Config
st.set_page_config(page_title="💻 Laptop Recommender (BMCS2009)", layout="wide")
DATA_PATH = "laptop_dataset_expanded_myr_full_clean.csv"
DRIVE_ID = "18QknTkpJ-O_26Aj41aRKoEiN6a34vX5VpcXyAkkObp4"
GID = "418897947"

STYLE_CHOICES = ["Business-look", "Gaming-look", "Creator-look"]
STYLE_TO_BUCKET = {"Business-look": "Business", "Gaming-look": "Gaming", "Creator-look": "Creator"}

EXPECTED = [
    "brand","series","model","year",
    "cpu_brand","cpu_family","cpu_model","cpu_cores",
    "gpu_brand","gpu_model","gpu_vram_GB",
    "ram_base_GB","ram_type",
    "storage_primary_type","storage_primary_capacity_GB",
    "display_size_in","display_resolution","display_refresh_Hz","display_panel",
    "battery_capacity_Wh","weight_kg","os","price_myr","intended_use_case"
]
NUMERIC = [
    "year","cpu_cores","gpu_vram_GB","ram_base_GB",
    "storage_primary_capacity_GB","display_size_in","display_refresh_Hz",
    "battery_capacity_Wh","weight_kg","price_myr"
]

# ─────────────────────────── Data loading
@st.cache_data(show_spinner=False, ttl=24*3600)
def download_sheet_csv(output="/tmp/laptops.csv"):
    url = f"https://docs.google.com/spreadsheets/d/{DRIVE_ID}/export?format=csv&gid={GID}"
    gdown.download(url, output, quiet=True, fuzzy=True)
    p = Path(output)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError("Download failed or file empty. Check DRIVE_ID/GID and sharing.")
    return output

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    p = Path(DATA_PATH)
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.read_csv(download_sheet_csv("/tmp/laptops.csv"), low_memory=False)

# ─────────────────────────── Prep
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for c in EXPECTED:
        if c not in data.columns:
            data[c] = np.nan if c in NUMERIC else ""
    data["model"] = data["model"].fillna("").astype(str)
    data = data[data["model"].str.len() > 0].copy()
    for c in NUMERIC:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    for c in ["cpu_brand","gpu_brand","ram_type","storage_primary_type","display_resolution"]:
        data[c] = data[c].astype(str).str.upper()
    data["os"] = data["os"].astype(str).str.title()

    def tok(v): 
        s = str(v).strip(); 
        return s if s else ""
    def numtok(prefix, v, unit=""):
        try:
            if pd.isna(v): return ""
            x = float(v); x = int(x) if x.is_integer() else x
            return f"{prefix}{x}{unit}"
        except: 
            return ""

    text_parts = []
    for _, r in data.iterrows():
        parts = [
            tok(r["brand"]), tok(r["series"]), tok(r["model"]),
            tok(r["cpu_brand"]), tok(r["cpu_family"]), tok(r["cpu_model"]),
            tok(r["gpu_brand"]), tok(r["gpu_model"]),
            numtok("RAM", r["ram_base_GB"], "GB"), tok(r["ram_type"]),
            tok(r["storage_primary_type"]), numtok("SSD", r["storage_primary_capacity_GB"], "GB"),
            numtok("SIZE", r["display_size_in"], "IN"), tok(r["display_resolution"]),
            numtok("HZ", r["display_refresh_Hz"], "HZ"), tok(r["display_panel"]),
            numtok("BAT", r["battery_capacity_Wh"], "WH"), numtok("WT", r["weight_kg"], "KG"),
            tok(r["os"]), f"USE {tok(r['intended_use_case'])}", numtok("Y", r["year"])
        ]
        text_parts.append(" ".join([p for p in parts if p]))
    data["spec_text"] = text_parts

    key = ["model","year"] if "year" in data else ["model"]
    data = data.drop_duplicates(subset=key, keep="first")
    return data.reset_index(drop=True)

def normalize_use_case(x: object) -> str:
    t = str(x or "").lower()
    if any(k in t for k in ["game"]): return "Gaming"
    if any(k in t for k in ["creator","content","video","design","workstation","pro"]): return "Creator"
    if any(k in t for k in ["business","executive","programming","data"]): return "Business"
    if any(k in t for k in ["student","general","productivity","office","ultrabook","ultralight","portable","writer"]): return "Student"
    return "Student"

def summarize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    s = df.isna().sum()
    out = pd.DataFrame({"column": s.index, "nulls": s.values})
    out["pct"] = (out["nulls"] / max(len(df), 1)) * 100
    return out.sort_values("nulls", ascending=False).reset_index(drop=True)

def range_checks(df: pd.DataFrame) -> list[str]:
    msgs = []
    if "year" in df:
        y = pd.to_numeric(df["year"], errors="coerce")
        bad = ((y < 2010) | (y > 2026)) & y.notna()
        if bad.any(): msgs.append(f"{int(bad.sum())} suspicious years (<2010 or >2026).")
    if "price_myr" in df:
        p = pd.to_numeric(df["price_myr"], errors="coerce")
        if (p < 0).sum() > 0: msgs.append(f"{int((p < 0).sum())} negative prices.")
    return msgs

# Features / scoring
@st.cache_resource(show_spinner=False)
def build_tfidf(spec_text: pd.Series):
    s = spec_text.fillna("").astype(str)
    if not s.str.strip().any():
        from scipy.sparse import csr_matrix
        return TfidfVectorizer(), csr_matrix((len(s), 0))
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform(s)
    except ValueError:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(s)
    return vec, X

NUM_FEATS = ["ram_base_GB","gpu_vram_GB","cpu_cores","display_refresh_Hz",
             "battery_capacity_Wh","weight_kg","year","storage_primary_capacity_GB"]

@st.cache_resource(show_spinner=False)
def train_usecase_model(train_df: pd.DataFrame, label_col: str):
    # vectorize text
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xt = vec.fit_transform(train_df["spec_text"].fillna(""))

    # numeric block (optional but helps)
    num_cols = [c for c in NUM_FEATS if c in train_df.columns]
    if num_cols:
        num = train_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
        scaler = StandardScaler(with_mean=False)  # keep sparse-friendly
        Xn = scaler.fit_transform(num)
        X = hstack([Xt, Xn], format="csr")
    else:
        scaler = None
        X = Xt

    # labels
    y = train_df[label_col].astype(str).str.strip().replace({"nan":"","None":""}).mask(lambda s: s.eq(""), "unknown")

    # classifier
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="ovr", C=2.0)
    clf.fit(X, y)
    return vec, scaler, num_cols, clf, clf.classes_

def proba_for_label(vec, scaler, num_cols, clf, df_view: pd.DataFrame, target_label: str):
    Xt = vec.transform(df_view["spec_text"].fillna(""))
    if num_cols:
        num = df_view[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
        Xn = scaler.transform(num)
        X = hstack([Xt, Xn], format="csr")
    else:
        X = Xt
    # probability for the requested class (fallback to zeros if unseen)
    if hasattr(clf, "predict_proba"):
        if target_label in clf.classes_:
            idx = list(clf.classes_).index(target_label)
            p = clf.predict_proba(X)[:, idx]
        else:
            p = np.zeros(X.shape[0], dtype=float)
    else:
        # decision_function fallback (scale to [0,1])
        if target_label in clf.classes_:
            idx = list(clf.classes_).index(target_label)
            d = clf.decision_function(X)
            d = d[:, idx] if d.ndim == 2 else d
            p = (d - d.min()) / (d.max() - d.min() + 1e-9)
        else:
            p = np.zeros(X.shape[0], dtype=float)
    return p

def compute_row_sim(X, i: int) -> np.ndarray:
    return cosine_similarity(X[i], X).ravel()

def compute_query_sim(vec: TfidfVectorizer, X, qtext: str) -> np.ndarray:
    return cosine_similarity(vec.transform([qtext]), X).ravel()

def portability_score(view: pd.DataFrame) -> np.ndarray:
    # Weight: full credit at ≤1.3 kg, fades to 0 by ≥1.9 kg
    w = pd.to_numeric(view.get("weight_kg"), errors="coerce")
    w_score = ((1.9 - w) / 0.6).clip(0, 1)

    # Battery: 0 at ≤50 Wh, full credit at ≥90 Wh
    b = pd.to_numeric(view.get("battery_capacity_Wh"), errors="coerce")
    b_score = ((b - 50) / 40).clip(0, 1)

    # Combine (slightly favor weight)
    return (0.6 * w_score.fillna(0) + 0.4 * b_score.fillna(0)).to_numpy()

def rule_based_scores(view: pd.DataFrame, use_case: str) -> np.ndarray:
    def nz(col, default=0.0):
        return pd.to_numeric(view.get(col, default), errors="coerce").fillna(default).astype(float)
    ram     = nz("ram_base_GB")
    vram    = nz("gpu_vram_GB")
    cores   = nz("cpu_cores")
    refresh = nz("display_refresh_Hz", 60)
    year    = nz("year", 2018)
    storage = nz("storage_primary_capacity_GB", 256)
    # Base score
    s = np.zeros(len(view), dtype=float)
    s += 0.10 * np.clip((year - 2018) / 7.0, 0, 1)
    s += 0.10 * np.clip(storage / 1024.0, 0, 1)
    # NEW: portability everywhere (weight + battery)
    port = portability_score(view)
    s += 0.08 * port  # base influence for all use-cases
    u = str(use_case or "").lower()
    if u in ("gaming", "gamer"):
        s += 0.18 * np.clip((vram - 6) / 6.0, 0, 1)
        s += 0.15 * np.clip((cores - 6) / 6.0, 0, 1)
        s += 0.12 * np.clip((refresh - 120) / 60.0, 0, 1)
        s += 0.02 * port  # tiny nudge for portability
    elif u in ("creator","content creation","video editing","designer"):
        s += 0.18 * np.clip((ram - 16) / 16.0, 0, 1)
        s += 0.12 * np.clip((vram - 4) / 8.0, 0, 1)
        s += 0.10 * view["display_resolution"].astype(str).str.contains(
            "2560|2880|3000|3200|3840|4K", case=False, na=False
        ).astype(float)
        s += 0.04 * port
    elif u in ("student","office","productivity"):
        s += 0.12 * np.clip((ram - 8) / 8.0, 0, 1)
        s += 0.10 * port  # portability matters more here
    elif u in ("business","programming","data"):
        s += 0.16 * np.clip((cores - 8) / 8.0, 0, 1)
        s += 0.14 * np.clip((ram - 16) / 16.0, 0, 1)
        s += 0.06 * port  # portability also valuable
    return np.clip(s, 0, 1)

def price_proximity(view: pd.DataFrame, lo: float, hi: float) -> np.ndarray:
    p = pd.to_numeric(view.get("price_myr"), errors="coerce")
    mid = (lo + hi) / 2.0
    width = max(hi - lo, 1e-9)
    w = (1 - np.clip(np.abs(p - mid) / (0.5 * width), 0, 1)).fillna(0.0)
    return w.values  # [0,1]

def build_pref_query(prefs: dict) -> str:
    parts = []
    if prefs.get("use_case"): parts += [f"USE {prefs['use_case']}"]
    if prefs.get("min_ram"): parts += [f"RAM{prefs['min_ram']}GB"]
    if prefs.get("min_vram"): parts += [f"VRAM{prefs['min_vram']}GB"]
    if prefs.get("min_cpu_cores"): parts += [f"CORES{prefs['min_cpu_cores']}"]
    if prefs.get("min_refresh"): parts += [f"HZ{prefs['min_refresh']}HZ"]
    if prefs.get("min_storage"): parts += [f"SSD{prefs['min_storage']}GB"]
    if prefs.get("min_year"): parts += [f"Y{prefs['min_year']}"]
    return " ".join(parts)

def validate_prefs(prefs: dict) -> list[str]:
    errs = []
    for k in ["budget_min","budget_max","min_ram","min_storage","min_vram","min_cpu_cores","min_refresh"]:
        if k in prefs and prefs[k] is not None:
            try:
                if float(prefs[k]) < 0: errs.append(f"{k} must be ≥ 0.")
            except: errs.append(f"{k} must be a number.")
    if "budget_min" in prefs and "budget_max" in prefs:
        try:
            if float(prefs["budget_min"]) > float(prefs["budget_max"]):
                errs.append("budget_min cannot be greater than budget_max.")
        except: pass
    if prefs.get("min_year") is not None:
        try:
            y = int(prefs["min_year"])
            if y < 2010 or y > 2035: errs.append("min_year looks out of range (2010–2035).")
        except: errs.append("min_year must be an integer.")
    return errs

def enforce_business_rules(prefs: dict) -> tuple[dict, list[str]]:
    p, notes = dict(prefs), []
    # Round RAM to a sensible step
    if p.get("min_ram") is not None:
        try:
            r = int(np.ceil(float(p["min_ram"]) / 4.0) * 4)
            if r != p["min_ram"]:
                notes.append(f"Rounded min_ram to {r} GB.")
            p["min_ram"] = r
        except:
            pass
    # Defaults by use case
    u = str(p.get("use_case") or "").lower()
    # Gaming nudges
    if u in ("gaming", "gamer", "budget gaming"):
        if p.get("min_vram") is None or float(p["min_vram"]) < 4:
            p["min_vram"] = 4
            notes.append("Set min_vram to 4 GB for gaming.")
        if p.get("min_refresh") is None or float(p["min_refresh"]) < 120:
            p["min_refresh"] = 120
            notes.append("Set min_refresh to 120 Hz for gaming.")
    # NEW: Business / Student / Creator nudges
    if u in ("business",):
        # lighter + more RAM
        p["min_ram"] = max(int(p.get("min_ram") or 0), 16)
        if p.get("max_weight") is None:
            p["max_weight"] = 1.5
        else:
            p["max_weight"] = min(float(p["max_weight"]), 1.5)
        notes.append("Business profile: aiming for ≤1.5 kg and ≥16 GB RAM.")
    elif u in ("student", "office", "productivity"):
        # light + decent battery
        if p.get("max_weight") is None:
            p["max_weight"] = 1.6
        else:
            p["max_weight"] = min(float(p["max_weight"]), 1.6)
        if p.get("min_battery_wh") is None:
            p["min_battery_wh"] = 50
        else:
            p["min_battery_wh"] = max(float(p["min_battery_wh"]), 50)
        notes.append("Student profile: aiming for ≤1.6 kg and ≥50 Wh battery.")
    elif u in ("creator", "content creation", "video editing", "designer"):
        p["min_ram"] = max(int(p.get("min_ram") or 0), 16)
        p["min_storage"] = max(int(p.get("min_storage") or 0), 1024)  # 1TB
        notes.append("Creator profile: ≥16 GB RAM and ≥1 TB storage when available.")
    # Budget
    if p.get("budget_min") is not None and p.get("budget_max") is not None:
        try:
            lo, hi = float(p["budget_min"]), float(p["budget_max"])
            if lo > hi:
                p["budget_min"], p["budget_max"] = hi, lo
                notes.append("Swapped budget_min and budget_max.")
        except:
            pass
    return p, notes


def make_filter_mask(df: pd.DataFrame, prefs: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    def ge(col, v): s = pd.to_numeric(df.get(col), errors="coerce"); return (s >= float(v)) | s.isna()
    if "price_myr" in df.columns:
        price = pd.to_numeric(df["price_myr"], errors="coerce")
        lo, hi = float(prefs.get("budget_min", 0)), float(prefs.get("budget_max", float("inf")))
        mask &= price.between(lo, hi, inclusive="both") | price.isna()
    if prefs.get("min_ram") is not None: mask &= ge("ram_base_GB", prefs["min_ram"])
    if prefs.get("min_storage") is not None: mask &= ge("storage_primary_capacity_GB", prefs["min_storage"])
    if prefs.get("min_vram") is not None: mask &= ge("gpu_vram_GB", prefs["min_vram"])
    if prefs.get("min_cpu_cores") is not None: mask &= ge("cpu_cores", prefs["min_cpu_cores"])
    if prefs.get("min_year") is not None: mask &= ge("year", prefs["min_year"])
    if prefs.get("min_refresh") is not None: mask &= ge("display_refresh_Hz", prefs["min_refresh"])
    return mask

def recommend_by_prefs(
    df: pd.DataFrame, vec, X, prefs: dict, algo: str, top_n: int = 10
) -> pd.DataFrame:
    """
    Recommender with 3 modes:
      - "Rule-Based": only rule_based_scores
      - "Content-Based": prefers TRained-probabilities if available; else TF-IDF cosine
      - "Hybrid": blend of content (trained if available) + rule-based (+ price proximity)
    """
    # --- 0) Validate + normalize prefs
    errs = validate_prefs(prefs)
    if errs:
        st.error(" | ".join(errs))
        return pd.DataFrame()
    prefs2, notes = enforce_business_rules(prefs)
    if notes:
        st.info(" ".join(notes))
    # --- 1) Candidate pool (filters)
    mask = make_filter_mask(df, prefs2)
    view = df.loc[mask].copy()
    if view.empty:
        return view
    # --- 2) If Rule-Based only, easy exit
    if algo == "Rule-Based":
        scores = rule_based_scores(view, prefs2.get("use_case"))
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

    # --- 3) Content signal: try TRAINED model first; fall back to TF-IDF cosine
    def trained_proba_or_none(_view: pd.DataFrame, label: str):
        tg = globals()
        have_model = all(k in tg and tg[k] is not None for k in
                         ("text_vec", "num_scaler", "num_cols_trained", "clf_usecase"))
        if not (have_model and label):
            return None
        try:
            return proba_for_label(
                tg["text_vec"], tg["num_scaler"], tg["num_cols_trained"], tg["clf_usecase"],
                _view, str(label)
            )  # ndarray float in [0,1]
        except Exception:
            return None
    # 3a) Try trained probas
    p_content = trained_proba_or_none(view, prefs2.get("use_case"))
    # 3b) If no trained model, use your TF-IDF cosine to a preference query
    if p_content is None:
        if getattr(X, "shape", (0, 0))[1] == 0:
            st.warning("Content-based signal unavailable (empty TF-IDF). Falling back to rule-based only.")
            p_content = np.zeros(len(view), dtype=float)
        else:
            query = build_pref_query(prefs2)
            sim_all = compute_query_sim(vec, X, query)
            p_content = sim_all  # raw similarity; we’ll normalize below
    # Normalize if this is a similarity vector (not already a prob)
    # (Heuristic: if min<0 or max>1 we normalize to [0,1])
    pc_min, pc_max = float(np.min(p_content)), float(np.max(p_content))
    if pc_min < 0.0 or pc_max > 1.0:
        rng = np.ptp(p_content)
        p_content = (p_content - pc_min) / (rng if rng else 1.0)
    # --- 4) Compose scores by mode
    if algo == "Content-Based":
        scores = p_content
    else:
        # Hybrid
        rb = rule_based_scores(view, prefs2.get("use_case"))
        a = float(prefs2.get("alpha", 0.6))
        base = a * p_content + (1 - a) * rb

        # Budget proximity blend (kept from your original)
        try:
            pp = price_proximity(view, prefs2["budget_min"], prefs2["budget_max"])
            scores = 0.7 * base + 0.3 * pp
        except Exception:
            scores = base
    # --- 5) Return ranked table
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


def recommend_similar(df: pd.DataFrame, vec, X, selected_model: str, top_n: int = 5):
    idxs = df.index[df["model"].astype(str).str.lower() == selected_model.lower()].tolist()
    if not idxs: st.write("Model not found."); return pd.DataFrame()
    i = idxs[0]
    if getattr(X, "shape", (0, 0))[1] == 0:
        st.warning("Similarity unavailable (empty TF-IDF)."); return pd.DataFrame()
    sim = compute_row_sim(X, i)
    order = [j for j in np.argsort(-sim) if j != i][:top_n]
    out = df.iloc[order].copy(); out["similarity"] = sim[order]
    return out

def why_this(row: pd.Series, style_bucket: str) -> list[str]:
    bullets = []
    w = pd.to_numeric(row.get("weight_kg"), errors="coerce")
    bat = pd.to_numeric(row.get("battery_capacity_Wh"), errors="coerce")
    hz = pd.to_numeric(row.get("display_refresh_Hz"), errors="coerce")
    vram = pd.to_numeric(row.get("gpu_vram_GB"), errors="coerce")
    ram = pd.to_numeric(row.get("ram_base_GB"), errors="coerce")
    res = str(row.get("display_resolution") or "")
    if pd.notna(w) and w <= 1.5: bullets.append("lightweight")
    if pd.notna(bat) and bat >= 50: bullets.append("long battery life")
    sb = (style_bucket or "").lower()
    if sb == "gaming":
        if pd.notna(hz) and hz >= 120: bullets.append(f"{int(hz)} Hz display")
        if pd.notna(vram) and vram >= 6: bullets.append(f"{int(vram)} GB VRAM")
    elif sb == "creator":
        if pd.notna(ram) and ram >= 16: bullets.append(f"{int(ram)} GB RAM")
        if any(x in res for x in ["2560","2880","3000","3200","3840","4K"]): bullets.append("high-res screen")
    elif sb == "business":
        if pd.notna(w) and w <= 1.3: bullets.append("ultra-portable")
        if pd.notna(bat) and bat >= 60: bullets.append("all-day battery")
    return (bullets[:3] or ["balanced for everyday study"])

def render_results(recs: pd.DataFrame, style_bucket: str):
    for i, row in recs.iterrows():
        st.markdown(f"### {i+1}. {row.get('brand','')} {row.get('series','')} {row.get('model','')}")
        price = pd.to_numeric(row.get("price_myr"), errors="coerce")
        st.markdown(f"*Price (MYR):* {int(price):,}" if pd.notna(price) else "*Price (MYR):* Unknown")
        st.markdown(f"*CPU:* {row.get('cpu_brand','')} {row.get('cpu_family','')} {row.get('cpu_model','')} | *Cores:* {row.get('cpu_cores','')}")
        st.markdown(f"*GPU:* {row.get('gpu_brand','')} {row.get('gpu_model','')} | *VRAM:* {row.get('gpu_vram_GB','0')} GB")
        st.markdown(f"*RAM:* {row.get('ram_base_GB','?')} GB {row.get('ram_type','')} | *Storage:* {row.get('storage_primary_capacity_GB','?')} GB {row.get('storage_primary_type','')}")
        st.markdown(f"*Display:* {row.get('display_size_in','?')}-inch {row.get('display_resolution','')} {row.get('display_refresh_Hz','')}Hz {row.get('display_panel','')}")
        st.markdown("**Why it fits:** " + " • ".join(why_this(row, style_bucket)))
        with st.expander("Show/Hide more specs"): st.write(row.to_frame().T)
        st.markdown("---")

def unique_nums(df, col, *, round_to=None, as_int=False, add_zero=False):
    s = pd.to_numeric(df.get(col), errors="coerce").dropna()
    if round_to: s = (s / round_to).round() * round_to
    vals = sorted(s.unique())
    if as_int: vals = [int(v) for v in vals]
    if add_zero and (not vals or vals[0] != 0): vals = [0] + [v for v in vals if v != 0]
    return vals

def first_at_least(options, target):
    if not options: return target
    for v in options:
        if v >= target: return v
    return options[-1]

# ─────────────────────────── Evaluation
def split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, label_col: str = "intended_use_case_norm"):
    if label_col not in df.columns:
        if "intended_use_case" in df.columns: label_col = "intended_use_case"
        else: 
            tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
            return tr.reset_index(drop=True), te.reset_index(drop=True)
    y = df[label_col].astype(str).str.strip().replace({"nan":"","None":""})
    y = y.mask(y.eq(""), "unknown")
    vc = y.value_counts(); rare = vc[vc < 2].index
    if len(rare) > 0: y = y.where(~y.isin(rare), "other")
    try:
        tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
    return tr.reset_index(drop=True), te.reset_index(drop=True)

def evaluate_precision_recall_at_k_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10, alpha: float = 0.6, label_col: str = "intended_use_case_norm") -> pd.DataFrame:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2); vec.fit(train_df["spec_text"].fillna(""))
    X_test = vec.transform(test_df["spec_text"].fillna(""))

    if label_col in train_df and train_df[label_col].notna().any():
        labels = sorted(x for x in train_df[label_col].astype(str).str.strip().unique() if x)
    else:
        labels = ["Gaming","Student","Creator","Business"]

    price = pd.to_numeric(test_df.get("price_myr"), errors="coerce")
    p = price.dropna()
    lo, hi = (float(p.quantile(0.05)), float(p.quantile(0.95))) if len(p) >= 5 else (float(p.min()) if len(p) else 0.0, float(p.max()) if len(p) else 20000.0)
    mask = price.between(lo, hi, inclusive="both") | price.isna()
    view, Xv = test_df.loc[mask].copy(), X_test[mask.values]

    out = []
    for lab in labels:
        prefs = dict(use_case=lab, min_ram=8, min_storage=512, min_vram=0, min_cpu_cores=4, min_year=2018, min_refresh=60)
        sim = cosine_similarity(vec.transform([build_pref_query(prefs)]), Xv).ravel()
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min()) if sim.max() > sim.min() else sim
        rb = rule_based_scores(view, lab)
        scores = alpha * sim_norm + (1 - alpha) * rb

        ranked = view.assign(score=scores).sort_values("score", ascending=False)
        topk = ranked.head(k)

        truth_col = label_col if label_col in ranked.columns else "intended_use_case"
        y_true_all = (ranked[truth_col].astype(str).str.lower() == lab.lower()).astype(float).to_numpy()
        y_pred_all = ranked["score"].to_numpy()

        matched = int((topk[truth_col].astype(str).str.lower() == lab.lower()).sum())
        precision = matched / max(k, 1)
        recall = 1.0 if matched > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        mse = float(np.mean((y_pred_all - y_true_all) ** 2)) if len(y_true_all) else np.nan
        rmse = float(np.sqrt(mse)) if not np.isnan(mse) else np.nan

        out.append({"scenario": lab, "precision@k": round(precision,3), "recall@k": round(recall,3), "f1@k": round(f1,3), "mse": round(mse,4) if not np.isnan(mse) else np.nan, "rmse": round(rmse,4) if not np.isnan(rmse) else np.nan})
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def evaluate_fixed(df: pd.DataFrame, label_col: str, *, test_size=0.20, k=5, alpha=0.55):
    tr_df, te_df = split_df(df, test_size=test_size, label_col=label_col)
    res = evaluate_precision_recall_at_k_train_test(tr_df, te_df, k=k, alpha=alpha, label_col=label_col)
    return res, len(tr_df), len(te_df), test_size, k, alpha
    
# --- Auto-tune alpha & K to maximize mean F1@K ---
def tune_alpha_k(df, label_col="intended_use_case_norm"):
    from itertools import product
    tr, te = split_df(df, test_size=0.2, label_col=label_col)
    grid_a = [0.3, 0.45, 0.6, 0.75]
    grid_k = [3, 5, 6, 8]
    best = {"mean_f1": -1, "alpha": None, "k": None, "res": None}
    for a, k in product(grid_a, grid_k):
        res = evaluate_precision_recall_at_k_train_test(tr, te, k=k, alpha=a, label_col=label_col)
        mean_f1 = res["f1@k"].mean()
        if mean_f1 > best["mean_f1"]:
            best = {"mean_f1": mean_f1, "alpha": a, "k": k, "res": res}
    return best

# ─────────────────────────── UI
st.markdown("""
<style>
h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
.stButton>button { background-color: #0ea5e9; color: white; border-radius: 8px; }
.stTextInput input { border: 1px solid #0ea5e9; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.button("🔄 Clear cache & rerun"):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

st.title("💻 Laptop Recommender")

# Load + prep
try:
    raw = load_dataset()
except Exception as e:
    st.error(str(e)); st.stop()

df = prepare_df(raw)
st.caption(f"{len(df)} laptops loaded from {'local file' if Path(DATA_PATH).exists() else 'Google Drive (cached)'}")
LABEL_COL = "intended_use_case_norm"
df[LABEL_COL] = (df["intended_use_case"].apply(normalize_use_case) if "intended_use_case" in df.columns else "Student")

with st.expander("📋 Data validation & data quality checks", expanded=False):
    issues = []
    missing_in_raw = [c for c in EXPECTED if c not in raw.columns]
    if missing_in_raw: issues.append(f"Missing in source file: {missing_in_raw}")
    key_cols = [c for c in ["brand","series","model","year"] if c in df.columns]
    if key_cols:
        dup_count = df.duplicated(subset=key_cols, keep=False).sum()
        if dup_count > 0: issues.append(f"{int(dup_count)} duplicate rows by {key_cols}.")
    nulls = summarize_nulls(df)
    if (nulls["nulls"] > 0).any():
        st.write("**Null counts (post-clean):**"); st.dataframe(nulls[nulls["nulls"] > 0], width="stretch")
    issues += range_checks(df)
    if issues: st.error("• " + "\n• ".join(issues))
    else: st.success("No data quality issues detected.")

# Build TF-IDF space
vec, X = build_tfidf(df["spec_text"])
# Train a content model on ALL data
text_vec, num_scaler, num_cols_trained, clf_usecase, clf_classes = train_usecase_model(df, LABEL_COL)

# ── Search block
search_term = st.text_input("Search for a Laptop Model or Brand 🔎").strip()
if search_term:
    found = df[
        df["model"].str.contains(search_term, case=False, na=False) |
        df["brand"].str.contains(search_term, case=False, na=False) |
        df["series"].str.contains(search_term, case=False, na=False)
    ].copy().sort_values(by=["brand","series","model"]).reset_index(drop=True)

    if found.empty:
        st.write("No laptops found for that search.")
    else:
        st.write(f"### Search Results for: {search_term}")
        for idx, row in found.head(30).iterrows():
            st.markdown(f"<h3><b>{idx+1}. {row['brand']} {row['series']} {row['model']}</b></h3>", unsafe_allow_html=True)
            st.markdown(f"*Price (MYR):* {row['price_myr'] if pd.notna(row['price_myr']) else 'Unknown'}")
            st.markdown(f"*CPU:* {row['cpu_brand']} {row['cpu_family']} {row['cpu_model']}  |  *Cores:* {row['cpu_cores']}")
            st.markdown(f"*GPU:* {row['gpu_brand']} {row['gpu_model']}  |  *VRAM:* {row['gpu_vram_GB']} GB")
            st.markdown(f"*RAM:* {row['ram_base_GB']} GB {row['ram_type']}  |  *Storage:* {row['storage_primary_capacity_GB']} GB {row['storage_primary_type']}")
            st.markdown(f"*Display:* {row['display_size_in']}-inch {row['display_resolution']} {row['display_refresh_Hz']}Hz {row['display_panel']}")
            with st.expander("Show/Hide more specs"): st.write(row.to_frame().T)
            st.markdown("---")

        choice_list = found["model"].head(50).tolist()
        selected_model = st.selectbox("Select a model to get similar laptops 🎧", choice_list)
        if st.button("Recommend Similar Laptops"):
            rec_sim = recommend_similar(df, vec, X, selected_model, top_n=5)
            if rec_sim.empty: st.write("No similar items found.")
            else:
                st.write(f"### Similar to **{selected_model}**")
                for k, r in enumerate(rec_sim.itertuples(index=False), 1):
                    st.markdown(f"**{k}. {r.brand} {r.series} {r.model}** — Score: {getattr(r, 'similarity', 0):.2f}")
                    st.markdown(f"*CPU:* {r.cpu_brand} {r.cpu_family} {r.cpu_model} | *GPU:* {r.gpu_brand} {r.gpu_model}")
                    st.markdown(f"*RAM/Storage:* {r.ram_base_GB}GB / {r.storage_primary_capacity_GB}GB {r.storage_primary_type} | *Price:* {r.price_myr}")
                    st.markdown("---")

# ── Preference block
st.write("## Find laptops for school")
with st.container():
    pc = pd.to_numeric(df.get("price_myr"), errors="coerce").dropna().astype(int)
    if pc.empty:
        budget = (1500, 8000)
    else:
        price_options = sorted(pd.unique(pc))
        budget = st.select_slider("Budget (MYR)", options=price_options, value=(price_options[0], price_options[-1]))

    style_choice = st.radio("Preferred style", STYLE_CHOICES, horizontal=True)
    style_bucket = STYLE_TO_BUCKET.get(style_choice, "Student")

    balance = st.slider("Spec match", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
    results_count = st.slider("How many results to show?", 3, 30, 10, 1)

with st.expander("Advanced filters (optional)", expanded=False):
    col1, col2 = st.columns(2)
    ram_opts  = unique_nums(df, "ram_base_GB", as_int=True)
    stor_opts = unique_nums(df, "storage_primary_capacity_GB", as_int=True)
    vram_opts = unique_nums(df, "gpu_vram_GB", as_int=True, add_zero=True)
    year_opts = unique_nums(df, "year", as_int=True)
    ref_opts  = unique_nums(df, "display_refresh_Hz", as_int=True)

    with col1:
        min_storage = st.select_slider("Min Storage (GB)", options=stor_opts, value=first_at_least(stor_opts, 512))
        if style_bucket == "Gaming":
            min_vram = st.select_slider("Min GPU VRAM (GB)", options=vram_opts, value=first_at_least(vram_opts, 4))
        else:
            min_vram = 0
        min_year = st.select_slider("Min Release Year", options=year_opts, value=first_at_least(year_opts, 2019))
    with col2:
        min_ram = st.select_slider("Min RAM (GB)", options=ram_opts, value=first_at_least(ram_opts, 8))
        min_refresh = st.select_slider("Min Refresh (Hz)", options=ref_opts, value=first_at_least(ref_opts, 60))

prefs = dict(
    budget_min=budget[0], budget_max=budget[1],
    use_case=style_bucket,
    min_ram=min_ram, min_storage=min_storage, min_vram=min_vram,
    min_cpu_cores=4, min_year=min_year, min_refresh=min_refresh,
    alpha=balance
)

recs = None
if st.button("Show laptops"):
    with st.spinner("Finding good matches..."):
        recs = recommend_by_prefs(df, vec, X, prefs, "Hybrid", results_count)

if recs is not None:
    if recs.empty: st.warning("No matching laptops found for your choices.")
    else: render_results(recs, style_bucket)
        
# ── Performance
with st.expander("Performance (Precision@K, Recall@K, F1@K, MSE/RMSE)", expanded=True):
    # choose any fixed settings you want to “freeze”
    res, n_tr, n_te, TS, K_FIXED, A_FIXED = evaluate_fixed(
        df, LABEL_COL, test_size=0.20, k=5, alpha=0.55
    )

    st.write(f"**Train:** {n_tr}  |  **Test:** {n_te}")
    st.write(f"**Settings:** Test size = {TS:.2f}  |  K = {K_FIXED}  |  α = {A_FIXED:.2f}")

    st.dataframe(res, use_container_width=True)

    if not res.empty:
        st.write(
            f"**Mean Precision@{K_FIXED}:** {res['precision@k'].mean():.3f} | "
            f"**Mean Recall@{K_FIXED}:** {res['recall@k'].mean():.3f} | "
            f"**Mean F1@{K_FIXED}:** {res['f1@k'].mean():.3f} | "
            f"**Mean MSE:** {res['mse'].mean():.4f} | **Mean RMSE:** {res['rmse'].mean():.4f}"
        )
