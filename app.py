
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

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
ALLOWED_LABELS = {"Business","Gaming","Creator"}  # train/eval focus

ANY = "Any"

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
        s = str(v).strip()
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

        gpu = str(r.get("gpu_model") or "").upper()
        hz  = pd.to_numeric(r.get("display_refresh_Hz"), errors="coerce")
        res = str(r.get("display_resolution") or "").upper()
        is_discrete = any(x in gpu for x in ["RTX","GTX","RX","ARC"])
        creator_res = any(x in res for x in ["2560","2880","3000","3200","3840","4K"])
        parts += [
            "DISCRETE_GPU" if is_discrete else "IGPU",
            "HZ120PLUS" if (pd.notna(hz) and hz >= 120) else "",
            "CREATOR_RES" if creator_res else "",
            numtok("VRAM", r.get("gpu_vram_GB"), "GB"),
            numtok("CORES", r.get("cpu_cores"))
        ]
        text_parts.append(" ".join([p for p in parts if p]))

    data["spec_text"] = text_parts

    key = ["model","year"] if "year" in data else ["model"]
    data = data.drop_duplicates(subset=key, keep="first")
    return data.reset_index(drop=True)

def normalize_use_case(x: object) -> str:
    t = str(x or "").lower()
    if any(k in t for k in ["gaming","gamer","game"]): return "Gaming"
    if any(k in t for k in ["creator","content","video","design","workstation","edit","render"]): return "Creator"
    if any(k in t for k in ["business","executive","programming","data","engineer","developer","coding"]): return "Business"
    if any(k in t for k in ["student","general","productivity","office","ultrabook","ultralight","portable","writer"]): return "Student"
    return "Student"

def map_to_three(lab: str) -> str:
    lab = str(lab or "").strip()
    return lab if lab in {"Business", "Gaming", "Creator"} else "Student"

# ─────────────────────────── Features / scoring
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

def compute_row_sim(X, i: int) -> np.ndarray:
    return cosine_similarity(X[i], X).ravel()

def compute_query_sim(vec: TfidfVectorizer, X, qtext: str) -> np.ndarray:
    return cosine_similarity(vec.transform([qtext]), X).ravel()

def portability_score(view: pd.DataFrame) -> np.ndarray:
    w = pd.to_numeric(view.get("weight_kg"), errors="coerce")
    w_score = ((1.9 - w) / 0.6).clip(0, 1)
    b = pd.to_numeric(view.get("battery_capacity_Wh"), errors="coerce")
    b_score = ((b - 50) / 40).clip(0, 1)
    return (0.6 * w_score.fillna(0) + 0.4 * b_score.fillna(0)).to_numpy()

def rule_based_scores(view: pd.DataFrame, use_case: str) -> np.ndarray:
    def nz(col, default=0.0):
        return pd.to_numeric(view.get(col, default), errors="coerce").fillna(default).astype(float)
    ram, vram, cores = nz("ram_base_GB"), nz("gpu_vram_GB"), nz("cpu_cores")
    refresh, year = nz("display_refresh_Hz", 60), nz("year", 2018)
    storage = nz("storage_primary_capacity_GB", 256)

    s = np.zeros(len(view), dtype=float)
    s += 0.10 * np.clip((year - 2018) / 7.0, 0, 1)
    s += 0.10 * np.clip(storage / 1024.0, 0, 1)

    port = portability_score(view)
    s += 0.08 * port

    u = str(use_case or "").lower()
    if u in ("gaming","gamer"):
        s += 0.18 * np.clip((vram - 6) / 6.0, 0, 1)
        s += 0.15 * np.clip((cores - 6) / 6.0, 0, 1)
        s += 0.12 * np.clip((refresh - 120) / 60.0, 0, 1)
        s += 0.02 * port
    elif u in ("creator","content creation","video editing","designer"):
        s += 0.18 * np.clip((ram - 16) / 16.0, 0, 1)
        s += 0.12 * np.clip((vram - 4) / 8.0, 0, 1)
        s += 0.10 * view["display_resolution"].astype(str).str.contains(
            "2560|2880|3000|3200|3840|4K", case=False, na=False
        ).astype(float)
        s += 0.04 * port
    elif u in ("business","programming","data"):
        s += 0.16 * np.clip((cores - 8) / 8.0, 0, 1)
        s += 0.14 * np.clip((ram - 16) / 16.0, 0, 1)
        s += 0.06 * port
    return np.clip(s, 0, 1)

def price_proximity(view: pd.DataFrame, lo: float, hi: float) -> np.ndarray:
    p = pd.to_numeric(view.get("price_myr"), errors="coerce")
    mid = (lo + hi) / 2.0
    width = max(hi - lo, 1e-9)
    w = (1 - np.clip(np.abs(p - mid) / (0.5 * width), 0, 1)).fillna(0.0)
    return w.values

def build_pref_query(prefs: dict) -> str:
    parts = []
    u = (prefs.get("use_case") or "").lower()

    if prefs.get("use_case"): parts += [f"USE {prefs['use_case']}"]
    if u == "gaming": parts += ["DISCRETE_GPU", "HZ120PLUS"]
    elif u == "creator": parts += ["CREATOR_RES", "DISCRETE_GPU"]
    elif u == "business": parts += ["IGPU"]

    if prefs.get("min_ram"):        parts += [f"RAM{prefs['min_ram']}GB"]
    if prefs.get("min_vram"):       parts += [f"VRAM{prefs['min_vram']}GB"]
    if prefs.get("min_cpu_cores"):  parts += [f"CORES{prefs['min_cpu_cores']}"]
    if prefs.get("min_refresh"):    parts += [f"HZ{prefs['min_refresh']}HZ"]
    if prefs.get("min_storage"):    parts += [f"SSD{prefs['min_storage']}GB"]
    if prefs.get("year_exact"):     parts += [f"Y{int(prefs['year_exact'])}"]
    return " ".join([p for p in parts if p])

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
    if prefs.get("year_exact") is not None:
        try:
            y = int(prefs["year_exact"])
            if y < 2010 or y > 2035: errs.append("year looks out of range (2010–2035).")
        except: errs.append("year must be an integer.")
    return errs

# ─────────────────────────── HARD filter (style + exact year + mins)
def make_filter_mask(df: pd.DataFrame, prefs: dict, *, label_col: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    # 1) Style is a hard requirement
    if label_col in df.columns and prefs.get("use_case"):
        mask &= df[label_col].astype(str).str.lower().eq(str(prefs["use_case"]).lower())

    # 2) Budget is always applied
    if "price_myr" in df.columns:
        price = pd.to_numeric(df["price_myr"], errors="coerce")
        lo, hi = float(prefs.get("budget_min", 0)), float(prefs.get("budget_max", float("inf")))
        mask &= price.notna() & price.between(lo, hi, inclusive="both")

    def ge(col, v):
        s = pd.to_numeric(df.get(col), errors="coerce")
        return s.notna() & (s >= float(v))

    # 3) Minimum constraints (only if user set a value)
    if prefs.get("min_storage")   is not None: mask &= ge("storage_primary_capacity_GB", prefs["min_storage"])
    if prefs.get("min_ram")       is not None: mask &= ge("ram_base_GB", prefs["min_ram"])
    if prefs.get("min_vram")      is not None: mask &= ge("gpu_vram_GB", prefs["min_vram"])
    if prefs.get("min_refresh")   is not None: mask &= ge("display_refresh_Hz", prefs["min_refresh"])
    if prefs.get("min_cpu_cores") is not None: mask &= ge("cpu_cores", prefs["min_cpu_cores"])

    # 4) Exact year (not >=)
    if prefs.get("year_exact") is not None:
        y = pd.to_numeric(df.get("year"), errors="coerce")
        mask &= y.notna() & (y.astype(int) == int(prefs["year_exact"]))

    return mask

# ─────────────────────────── Recommender
def recommend_by_prefs(df: pd.DataFrame, vec, X, prefs: dict, algo: str, top_n: int = 10) -> pd.DataFrame:
    errs = validate_prefs(prefs)
    if errs:
        st.error(" | ".join(errs)); return pd.DataFrame()

    prefs2 = dict(prefs)  # use exactly what user chose (no auto-enforcement)
    mask = make_filter_mask(df, prefs2, label_col=LABEL_COL)
    view = df.loc[mask].copy()
    if view.empty: return view

    # ranking within the hard-filtered candidate set
    if algo == "Rule-Based":
        scores = rule_based_scores(view, prefs2.get("use_case"))
    else:
        if getattr(X, "shape", (0, 0))[1] == 0:
            st.warning("Content-based scoring unavailable (empty TF-IDF). Showing rule-based instead.")
            scores = rule_based_scores(view, prefs2.get("use_case"))
        else:
            sim_all = compute_query_sim(vec, X, build_pref_query(prefs2))
            sim = sim_all[mask.values]
            if algo == "Content-Based":
                scores = sim
            else:
                rng = np.ptp(sim); sim_norm = (sim - np.min(sim)) / (rng if rng else 1.0)
                rb = rule_based_scores(view, prefs2.get("use_case"))
                a = float(prefs2.get("alpha", 0.6))
                base = a * sim_norm + (1 - a) * rb
                pp = price_proximity(view, prefs2["budget_min"], prefs2["budget_max"])
                scores = 0.7 * base + 0.3 * pp

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
    return (bullets[:3] or ["balanced for everyday use"])

# ─────────────────────────── Utils for data-driven options
def opts_from_subset(sub: pd.DataFrame, col: str, *, any_label: str = ANY, as_int=True):
    s = pd.to_numeric(sub.get(col), errors="coerce").dropna()
    if s.empty: return [any_label]
    if as_int: s = s.round().astype(int)
    vals = sorted(s.unique())
    return [any_label] + list(vals)

def style_subset(df: pd.DataFrame, style: str, *, label_col: str) -> pd.DataFrame:
    if label_col in df.columns:
        return df[df[label_col].astype(str).str.lower().eq(style.lower())]
    if "intended_use_case" in df.columns:
        return df[df["intended_use_case"].astype(str).str.lower().eq(style.lower())]
    return df

def safe_default(key: str, options: list, fallback=ANY):
    v = st.session_state.get(key, fallback)
    return v if v in options else fallback

def _val(x):  # "Any" -> None
    return None if x == ANY else x

# ─────────────────────────── Evaluation (unchanged)
def split_df(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42,
             label_col: str = "intended_use_case_norm"):
    if label_col not in df.columns:
        if "intended_use_case" in df.columns:
            label_col = "intended_use_case"
        else:
            tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
            return tr.reset_index(drop=True), te.reset_index(drop=True)

    y = df[label_col].astype(str).str.strip().replace({"nan": "", "None": ""})
    y = y.mask(y.eq(""), "Student")
    try:
        tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        tr, te = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
    return tr.reset_index(drop=True), te.reset_index(drop=True)

def evaluate_precision_recall_at_k_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                              k: int = 5, alpha: float = 0.6,
                                              label_col: str = "intended_use_case_norm") -> pd.DataFrame:
    allowed = {"Business","Gaming","Creator"}
    test_df = test_df[test_df[label_col].isin(allowed)].copy()
    if test_df.empty:
        return pd.DataFrame(columns=["scenario","precision@k","recall@k","f1@k","mse","rmse"])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vec.fit(train_df["spec_text"].fillna(""))
    X_test = vec.transform(test_df["spec_text"].fillna(""))

    labels = ["Business","Gaming","Creator"]
    out = []
    for lab in labels:
        prefs = dict(use_case=lab, min_ram=8, min_storage=512, min_vram=0,
                     min_cpu_cores=4, min_year=2018, min_refresh=60)
        sim = cosine_similarity(vec.transform([build_pref_query(prefs)]), X_test).ravel()
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min()) if sim.max() > sim.min() else sim
        rb = rule_based_scores(test_df, lab)
        scores = alpha * sim_norm + (1 - alpha) * rb
        ranked = test_df.assign(score=scores).sort_values("score", ascending=False)
        truth_col = label_col if label_col in ranked.columns else "intended_use_case"
        is_rel = ranked[truth_col].astype(str).str.lower().eq(lab.lower()).to_numpy()
        hits_at_k = int(is_rel[:k].sum())
        total_rel = int(is_rel.sum())
        if total_rel == 0: continue
        precision = hits_at_k / max(k, 1)
        recall = hits_at_k / total_rel
        f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
        mse = float(np.mean((ranked["score"].to_numpy() - is_rel.astype(float)) ** 2))
        rmse = float(np.sqrt(mse))
        out.append({
            "scenario": lab,
            "precision@k": round(precision,3),
            "recall@k": round(recall,3),
            "f1@k": round(f1,3),
            "mse": round(mse,4),
            "rmse": round(rmse,4)
        })
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def evaluate_fixed(df: pd.DataFrame, label_col: str, *, test_size=0.30, k=5, alpha=0.55):
    tr_df, te_df = split_df(df, test_size=test_size, label_col=label_col)
    res = evaluate_precision_recall_at_k_train_test(tr_df, te_df, k=k, alpha=alpha, label_col=label_col)
    return res, len(tr_df), len(te_df), test_size, k, alpha

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

# Label column -> normalize THEN collapse to 3 buckets
LABEL_COL = "intended_use_case_norm"
if "intended_use_case" in df.columns:
    df[LABEL_COL] = df["intended_use_case"].apply(normalize_use_case).apply(map_to_three)
else:
    df[LABEL_COL] = "Business"  # safe default

# Build TF-IDF space (for interactive recommendations)
vec, X = build_tfidf(df["spec_text"])

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

# ── Preference block (data-driven by style)
st.write("## Find laptops")

# 1) Choose style
style_choice = st.radio("Preferred style", STYLE_CHOICES, horizontal=True)
style_bucket = STYLE_TO_BUCKET.get(style_choice, "Business")

# 2) Subset for this style (options + budget come from here)
subset = style_subset(df, style_bucket, label_col=LABEL_COL)

# Budget options from style subset
pc = pd.to_numeric(subset.get("price_myr"), errors="coerce").dropna().astype(int)
if pc.empty:
    budget = st.slider("Budget (MYR)", min_value=0, max_value=10000, value=(1500, 8000), step=100)
else:
    price_options = sorted(pd.unique(pc))
    budget = st.select_slider("Budget (MYR)", options=price_options,
                              value=(price_options[0], price_options[-1]))

results_count = st.slider("How many results to show?", 3, 30, 10, 1)

with st.expander("Advanced filters (optional)", expanded=False):
    col1, col2 = st.columns(2)

    stor_opts = opts_from_subset(subset, "storage_primary_capacity_GB")
    ram_opts  = opts_from_subset(subset, "ram_base_GB")
    vram_opts = opts_from_subset(subset, "gpu_vram_GB")
    ref_opts  = opts_from_subset(subset, "display_refresh_Hz")
    year_opts = opts_from_subset(subset, "year")  # exact-year choices

    # reset stale selections when style changes
    if "prev_style" not in st.session_state:
        st.session_state.prev_style = style_bucket
    just_switched = (st.session_state.prev_style != style_bucket)
    st.session_state.prev_style = style_bucket
    if just_switched:
        for k in ("min_storage","min_ram","min_vram","min_refresh","year_exact"):
            st.session_state[k] = ANY

    with col1:
        min_storage = st.select_slider(
            "Min Storage (GB)", options=stor_opts,
            key="min_storage", value=safe_default("min_storage", stor_opts)
        )
        min_vram = st.select_slider(
            "Min GPU VRAM (GB)", options=vram_opts,
            key="min_vram", value=safe_default("min_vram", vram_opts)
        )
        year_exact = st.select_slider(
            "Year (exact)", options=year_opts,
            key="year_exact", value=safe_default("year_exact", year_opts)
        )

    with col2:
        min_ram = st.select_slider(
            "Min RAM (GB)", options=ram_opts,
            key="min_ram", value=safe_default("min_ram", ram_opts)
        )
        min_refresh = st.select_slider(
            "Min Refresh (Hz)", options=ref_opts,
            key="min_refresh", value=safe_default("min_refresh", ref_opts)
        )

# Build prefs used for filtering + ranking
prefs = dict(
    budget_min=budget[0], budget_max=budget[1],
    use_case=style_bucket,
    min_storage=_val(st.session_state["min_storage"]),
    min_ram=_val(st.session_state["min_ram"]),
    min_vram=_val(st.session_state["min_vram"]),
    min_refresh=_val(st.session_state["min_refresh"]),
    min_cpu_cores=4,
    year_exact=_val(st.session_state["year_exact"]),
    alpha=0.6,
)

recs = None
if st.button("Show laptops"):
    with st.spinner("Finding good matches..."):
        recs = recommend_by_prefs(df, vec, X, prefs, "Hybrid", results_count)

if recs is not None:
    if recs.empty:
        st.warning("No matching laptops found for your choices.")
    else:
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

# ── Performance (fixed settings)
with st.expander("Performance (Precision@K, Recall@K, F1@K, MSE/RMSE)", expanded=True):
    res, n_tr, n_te, TS, K_FIXED, A_FIXED = evaluate_fixed(
        df, LABEL_COL, test_size=0.30, k=5, alpha=0.55
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
