# app.py
import streamlit as st
import pandas as pd

from recommender import (
    load_data,
    recommend,
    evaluate_precision_at_k,
    evaluate_with_split,
)

DATA_PATH = "laptop_dataset_expanded_myr_full_clean.csv"

@st.cache_data(show_spinner=False)
def _load():
    return load_data(DATA_PATH)

st.set_page_config(page_title="Laptop Recommender (BMCS2009)", layout="wide")
st.title("💻 Laptop Recommender — BMCS2009")

df = _load()
st.caption(f"{len(df)} models loaded")

# ---------------------------
# Sidebar: Preferences
# ---------------------------
with st.sidebar:
    st.subheader("Preferences")

    price_min = int(max(0, float(df["price_myr"].min())) if "price_myr" in df else 0)
    price_max = int(min(20000, float(df["price_myr"].max())) if "price_myr" in df else 20000)
    budget = st.slider(
        "Budget (MYR)",
        min_value=0,
        max_value=price_max if price_max > 0 else 20000,
        value=(price_min if price_min > 0 else 1500, min(price_max, 10000) if price_max else 6000),
        step=100,
    )

    if "intended_use_case" in df and df["intended_use_case"].dropna().size:
        options = sorted({str(x) for x in df["intended_use_case"].dropna().tolist() if str(x).strip()})
    else:
        options = ["Student", "Gaming", "Creator", "Business", "Programming"]
    use_case = st.selectbox("Use case", options)

    colA, colB = st.columns(2)
    with colA:
        min_ram = st.number_input("Min RAM (GB)", value=8, min_value=0, step=4)
        min_storage = st.number_input("Min Storage (GB)", value=512, min_value=0, step=128)
        min_vram = st.number_input("Min GPU VRAM (GB)", value=0, min_value=0, step=2)
        min_cores = st.number_input("Min CPU Cores", value=4, min_value=0, step=1)
        min_battery_wh = st.number_input("Min Battery (Wh)", value=0, min_value=0, step=5)
    with colB:
        min_screen, max_screen = st.slider("Screen size (inches)", 10.0, 18.0, (13.0, 16.0), 0.1)
        max_weight = st.number_input("Max Weight (kg)", value=3.0, min_value=0.0, step=0.1)
        min_refresh = st.number_input("Min Refresh (Hz)", value=60, min_value=0, step=30)
        min_year = st.number_input("Min Release Year", value=2019, min_value=2015, max_value=2025, step=1)

    st.divider()
    algo = st.selectbox("Algorithm", ["Hybrid", "Content-Based", "Rule-Based"])
    top_k = st.slider("Top K", 3, 20, 10)
    alpha = st.slider("Hybrid α (Content weight)", 0.0, 1.0, 0.6, 0.05)

prefs = dict(
    budget_min=budget[0],
    budget_max=budget[1],
    use_case=use_case,
    min_ram=min_ram,
    min_storage=min_storage,
    min_vram=min_vram,
    min_cpu_cores=min_cores,
    min_battery_wh=min_battery_wh,
    min_refresh=min_refresh,
    min_screen=min_screen,
    max_screen=max_screen,
    max_weight=max_weight,
    min_year=min_year,
    alpha=alpha,
)

algo_key = {"Hybrid": "hybrid", "Content-Based": "content", "Rule-Based": "rule"}[algo]

# ---------------------------
# Recommend
# ---------------------------
with st.spinner("Scoring laptops..."):
    recs = recommend(df, prefs, algo=algo_key, top_k=top_k)

st.subheader("Results")
st.dataframe(recs, use_container_width=True)

csv = recs.to_csv(index=False).encode("utf-8")
st.download_button("Download results (CSV)", data=csv, file_name="laptop_recommendations.csv")

# ---------------------------
# Quick Evaluation (same data)
# ---------------------------
with st.expander("Quick Evaluation (Precision@K by use-case)"):
    k_eval = st.slider("K", 3, 20, 10, key="k_eval")
    labels = (
        sorted({str(x) for x in df["intended_use_case"].dropna().tolist() if str(x).strip()})
        if "intended_use_case" in df and df["intended_use_case"].dropna().size
        else ["Gaming", "Student", "Creator", "Business"]
    )
    scenarios = [(lab, {**prefs, "use_case": lab}) for lab in labels]
    eval_df = evaluate_precision_at_k(df, scenarios, algo=algo_key, k=k_eval)
    st.dataframe(eval_df, use_container_width=True)

# ---------------------------
# Practical Block: Train/Test + Tuning
# ---------------------------
with st.expander("Train/Test split (as in practical)"):
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, help="Portion of data used for TEST")
    k_tt = st.slider("K for Precision@K", 3, 20, 10, key="k_tt")
    if st.button("Run train/test evaluation"):
        with st.spinner("Splitting data, tuning α on TRAIN, evaluating on TEST..."):
            summary, alpha_table, test_results = evaluate_with_split(df, test_size=test_size, k=k_tt)
        st.write(f"**Train**: {summary['n_train']} rows | **Test**: {summary['n_test']} rows")
        st.write(f"**Best α (hybrid)**: {summary['best_alpha']:.2f}")
        st.write(f"**Mean Precision@{k_tt} on TEST**: {summary['mean_precision@k_test']}")
        st.markdown("**Alpha tuning on TRAIN**")
        st.dataframe(alpha_table, use_container_width=True)
        st.markdown(f"**Per-scenario Precision@{k_tt} on TEST**")
        st.dataframe(test_results, use_container_width=True)
