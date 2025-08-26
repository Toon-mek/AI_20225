# app.py — Streamlit UI for Laptop Recommender (BMCS2003)
# Works with abc.py (backend). Supports Google Drive CSV, file upload, or bundled sample.

import os
import pandas as pd
import numpy as np
import streamlit as st
import gdown

# === import your backend ===
from abc import (
    load_data,
    recommend,
    train_test_eval,
    eval_confusion_labstyle,
    USE_PROFILES,
)

st.set_page_config(page_title="Laptop Recommender (BMCS2003)", layout="wide")
st.title("💻 Laptop Recommender – BMCS2003")
st.caption("Content-based scoring with lab-style train/test evaluation")

# ---------- Google Drive helpers ----------
@st.cache_data(show_spinner=False)
def _extract_drive_id(s: str) -> str:
    """Accept file ID or any common Drive link and return the file ID."""
    s = s.strip()
    if "drive.google.com" in s:
        if "id=" in s:
            return s.split("id=")[1].split("&")[0]
        if "/d/" in s:
            return s.split("/d/")[1].split("/")[0]
    return s  # assume it's already an ID

@st.cache_data(show_spinner=True)
def download_from_drive_to(path_out: str, file_id_or_link: str) -> str:
    """
    Download CSV from Google Drive to path_out.
    Make sure your file is shared as 'Anyone with the link: Viewer'.
    """
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fid = _extract_drive_id(file_id_or_link)
    url = f"https://drive.google.com/file/d/1ys7aa3wdxPcbk7QIzW9tmmnvSlR9HeG3/view?usp=drive_link"
    gdown.download(url, path_out, quiet=True)
    if not os.path.exists(path_out) or os.path.getsize(path_out) == 0:
        raise RuntimeError("Download failed or empty file. Check Drive sharing & File ID.")
    return path_out

@st.cache_data(show_spinner=False)
def read_csv_cached(local_path: str) -> pd.DataFrame:
    return pd.read_csv(local_path)

# ---------- Sidebar: data source + parameters ----------
with st.sidebar:
    st.header("Data Source")
    upl = st.file_uploader("Upload CSV (optional)", type=["csv"])

    use_drive = st.checkbox("Load from Google Drive", value=False)
    drive_input = st.text_input(
        "Google Drive File ID or Share Link",
        help="Example ID: 1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_  "
             "• Or paste a share link (the app will extract the ID)."
    )

    use_default = st.checkbox("Use bundled sample (fallback)", value=not use_drive)
    default_path = "data/laptop_market_2025_10000_with_chromebook.csv"

    st.markdown(
        "Expected columns:\n"
        "`Category, Model, CPU, GPU, RAM_GB, Storage_GB, Storage_Type, "
        "Screen_Size, Weight_kg, OS, Battery_Wh, TouchScreen, Price_USD, Price_MYR`."
    )

    st.divider()
    st.header("User Inputs")
    budget = st.number_input("Budget (MYR)", min_value=0, value=4000, step=100)
    purpose = st.selectbox("Purpose", options=list(USE_PROFILES.keys()), index=0)
    min_ram = st.selectbox("Minimum RAM (GB)", options=[0, 4, 8, 16, 32, 64], index=3)
    min_storage = st.selectbox("Minimum Storage (GB)", options=[0, 64, 128, 256, 512, 1024, 2048], index=4)
    must_touch = st.selectbox("Touch Screen?", options=["Any", "Yes", "No"], index=0)
    must_os = st.selectbox(
        "OS Preference",
        options=["Any", "Windows 11 Home", "Windows 11 Pro", "macOS 15", "ChromeOS",
                 "Windows 11 Home (ARM)", "Windows 11 Home / Linux"],
        index=0
    )
    topk = st.slider("Top-K recommendations", min_value=5, max_value=30, value=10)

    st.divider()
    st.header("Evaluation (Lab Style)")
    run_eval = st.checkbox("Run 50/50 train–test evaluation", value=False)
    k_eval = st.slider("k for hit-rate / confusion", min_value=1, max_value=20, value=5)
    show_cm = st.checkbox("Show confusion matrix", value=False)

# ---------- Load dataset (priority: upload > drive > default) ----------
df = None
try:
    if upl is not None:
        df = pd.read_csv(upl)
    elif use_drive and drive_input.strip():
        local_tmp = "data/_drive_dataset.csv"
        csv_path = download_from_drive_to(local_tmp, drive_input)
        df = read_csv_cached(csv_path)
    elif use_default:
        if not os.path.exists(default_path):
            st.warning("Bundled sample not found. Upload a CSV or use Google Drive.")
            st.stop()
        df = pd.read_csv(default_path)
    else:
        st.warning("Please upload a CSV, provide a Google Drive file, or enable the bundled sample.")
        st.stop()

    # validate & light clean using backend
    df = load_data(df) if isinstance(df, str) else df.drop_duplicates().reset_index(drop=True)

    # If user provided arbitrary columns, try to be forgiving:
    # (load_data in abc.py validates strictly; here we allow preview then rely on recommend() to filter)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ---------- Optional quick filters before recommend ----------
subset = df.copy()
if must_touch in ["Yes", "No"]:
    subset = subset[subset["TouchScreen"].astype(str).str.lower() == must_touch.lower()]
if must_os != "Any":
    subset = subset[subset["OS"].astype(str) == must_os]

# Build min_spec dictionary
min_specs = {}
if min_ram > 0:
    min_specs["RAM_GB"] = min_ram
if min_storage > 0:
    min_specs["Storage_GB"] = min_storage

# ---------- Recommendations ----------
st.subheader("Recommendations")
try:
    recs = recommend(
        subset,
        budget_myr=float(budget),
        purpose=purpose,
        min_specs=min_specs,
        top_k=topk
    )
    if recs.empty:
        st.warning("No recommendations match the filters. Try increasing budget or relaxing constraints.")
    else:
        show_cols = [
            "Category","Model","CPU","GPU","RAM_GB","Storage_GB","Storage_Type",
            "Screen_Size","Weight_kg","OS","Battery_Wh","TouchScreen","Price_MYR"
        ]
        existing_cols = [c for c in show_cols if c in recs.columns]
        st.dataframe(recs[existing_cols].reset_index(drop=True), use_container_width=True)
except Exception as e:
    st.error(f"Recommend failed: {e}")

# ---------- Evaluation ----------
st.divider()
st.subheader("📊 Evaluation (Lab-style 50/50 Split)")
st.caption(
    "Split 50/50. For each test row, set purpose from its category and budget≈its price, "
    "recommend top-k from train, and compute hit-rate. Optionally show confusion matrix."
)

if run_eval:
    purpose_map = {
        "gaming": "gaming",
        "ultrabook": "study_programming",
        "convertible": "office_browsing",
        "workstation": "content_creation",
        "compact": "study_programming",
        "chromebook": "chromebook_use",
    }
    try:
        res = train_test_eval(df, purpose_map, test_size=0.5, seed=42, k=k_eval)
        st.json(res)

        if show_cm:
            res_cm = eval_confusion_labstyle(df, purpose_map, test_size=0.5, seed=42, k=k_eval)
            st.write(f"Accuracy: {res_cm['accuracy']}  •  Precision(macro): {res_cm['precision_macro']}  •  Recall(macro): {res_cm['recall_macro']}")
            st.dataframe(res_cm["confusion_matrix"], use_container_width=True)
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
else:
    st.info("Tick the evaluation checkbox in the sidebar to run the 50/50 split test.")

st.divider()
st.caption("Tip: On Streamlit Cloud, set the main file to `app.py`. Ensure `gdown` is in `requirements.txt`.")

