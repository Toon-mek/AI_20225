# app.py — Streamlit UI for Laptop Recommender (BMCS2003)
# Works with abc.py (backend). Supports Google Drive CSV, file upload, or bundled sample.

import os
import pandas as pd
import numpy as np
import streamlit as st
import gdown

# === import your backend ===
from recommender import (
    load_data,
    recommend,
    train_test_eval,
    eval_confusion_labstyle,
    USE_PROFILES,
)
st.set_page_config(page_title="Laptop Recommender (BMCS2003)", layout="wide")
st.title("💻 Laptop Recommender – BMCS2003")
st.caption("Content-based scoring with lab-style train/test evaluation")

# ---------- CONFIG: set your Google Drive file here ----------
DRIVE_FILE_ID = "https://drive.google.com/file/d/1ys7aa3wdxPcbk7QIzW9tmmnvSlR9HeG3/view?usp=drive_link"  
LOCAL_CSV_PATH = "data/_drive_dataset.csv"

# ---------- Data loader (Drive-only) ----------
@st.cache_data(show_spinner=True)
def download_data_from_drive(file_id: str, dest_path: str) -> pd.DataFrame:
    """
    Download the CSV from Google Drive (by FILE ID) to dest_path and return as DataFrame.
    Cache is keyed by file_id, so updating the ID invalidates cache automatically.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=True)
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        raise RuntimeError("Download failed or empty file. Check Drive sharing & File ID.")
    df = pd.read_csv(dest_path)
    # light clean: standardize TouchScreen to Yes/No
    if "TouchScreen" in df.columns:
        df["TouchScreen"] = (
            df["TouchScreen"].astype(str).str.strip().str.lower().map({"yes": "Yes", "no": "No"}).fillna(df["TouchScreen"])
        )
    return df

# ---------- Sidebar: data source + parameters ----------
# ---------- Sidebar: only user inputs (no upload, no default toggle) ----------
with st.sidebar:
    st.header("User Inputs")
    st.caption("Dataset is loaded from Google Drive automatically.")
    st.text_input("Google Drive File ID", value=DRIVE_FILE_ID, key="file_id", help="Change if you move the file in Drive.")
    refresh = st.button("🔄 Re-download from Drive")

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

# ---------- Load dataset from Drive ----------
file_id = st.session_state.get("file_id", DRIVE_FILE_ID)
if refresh:
    # clear cache for a forced re-download
    download_data_from_drive.clear()

try:
    df = download_data_from_drive(file_id, LOCAL_CSV_PATH)
except Exception as e:
    st.error(f"Failed to download dataset from Drive: {e}")
    st.stop()

# ---------- Optional pre-filters ----------
subset = df.copy()
if "TouchScreen" in subset.columns and must_touch in ["Yes", "No"]:
    subset = subset[subset["TouchScreen"].astype(str).str.lower() == must_touch.lower()]
if "OS" in subset.columns and must_os != "Any":
    subset = subset[subset["OS"].astype(str) == must_os]

min_specs = {}
if "RAM_GB" in subset.columns and min_ram > 0:
    min_specs["RAM_GB"] = min_ram
if "Storage_GB" in subset.columns and min_storage > 0:
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
        show_cols = [c for c in show_cols if c in recs.columns]
        st.dataframe(recs[show_cols].reset_index(drop=True), use_container_width=True)
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
st.caption("Dataset is fetched from Google Drive every run (cached). Make sure the file sharing is set to 'Anyone with the link: Viewer'.")
