# recommender.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------
# Columns & Loading
# ---------------------------

EXPECTED_COLS = [
    # identity
    "brand", "series", "model", "year",
    # cpu
    "cpu_brand", "cpu_family", "cpu_model", "cpu_cores", "cpu_TDP_W",
    # gpu
    "gpu_brand", "gpu_model", "gpu_vram_GB",
    # memory
    "ram_base_GB", "ram_max_GB", "ram_type", "ram_slots",
    # storage
    "storage_primary_type", "storage_primary_capacity_GB",
    "storage_primary_interface", "storage_pcie_gen",
    # display
    "display_size_in", "display_resolution", "display_refresh_Hz",
    "display_panel", "display_aspect_ratio", "display_touchscreen",
    # battery/weight
    "battery_capacity_Wh", "battery_life_claimed_hr", "charger_W", "weight_kg",
    # io/wireless
    "ports_usb_a_count", "ports_usb_c_count", "ports_thunderbolt_count",
    "ports_hdmi_count", "ports_ethernet", "ports_sdcard",
    "wireless_wifi", "wireless_bt", "webcam_resolution",
    # os/price/use
    "os", "price_myr", "intended_use_case",
]

NAME_COLS = ["brand", "series", "model"]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and create safe defaults for any missing columns."""
    df = pd.read_csv(csv_path)

    # ensure all expected cols exist
    for c in EXPECTED_COLS:
        if c not in df.columns:
            if c.startswith("ports_") or c in ["display_touchscreen", "ports_ethernet", "ports_sdcard"]:
                df[c] = 0
            elif c in [
                "gpu_vram_GB", "ram_base_GB", "ram_max_GB", "cpu_cores", "display_size_in",
                "display_refresh_Hz", "battery_capacity_Wh", "battery_life_claimed_hr",
                "charger_W", "weight_kg", "storage_primary_capacity_GB", "price_myr", "year"
            ]:
                df[c] = np.nan
            else:
                df[c] = ""

    # normalize strings
    for c in ["display_resolution", "cpu_brand", "gpu_brand", "ram_type", "storage_primary_type"]:
        df[c] = df[c].astype(str).str.upper()
    df["os"] = df["os"].astype(str).str.title()

    # model must exist
    df["model"] = df["model"].fillna("").astype(str)
    df = df[df["model"].str.len() > 0].copy()

    # coerce numerics
    numeric_cols = [
        "year", "cpu_cores", "cpu_TDP_W", "gpu_vram_GB", "ram_base_GB", "ram_max_GB",
        "display_size_in", "display_refresh_Hz", "battery_capacity_Wh", "battery_life_claimed_hr",
        "charger_W", "weight_kg", "ports_usb_a_count", "ports_usb_c_count",
        "ports_thunderbolt_count", "ports_hdmi_count", "storage_primary_capacity_GB", "price_myr"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------
# Feature Engineering
# ---------------------------

def feature_pipeline(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical = [
        "cpu_brand", "cpu_family", "gpu_brand", "ram_type",
        "storage_primary_type", "storage_primary_interface", "display_resolution",
        "display_panel", "display_aspect_ratio", "os", "intended_use_case",
    ]
    categorical = [c for c in categorical if c in df.columns]

    numeric = [
        "year", "cpu_cores", "gpu_vram_GB", "ram_base_GB", "ram_max_GB",
        "display_size_in", "display_refresh_Hz", "battery_capacity_Wh",
        "battery_life_claimed_hr", "weight_kg", "storage_primary_capacity_GB", "price_myr",
        "ports_usb_a_count", "ports_usb_c_count", "ports_thunderbolt_count", "ports_hdmi_count",
    ]
    numeric = [c for c in numeric if c in df.columns]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return pre, categorical, numeric


def build_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, ColumnTransformer, List[str], List[str]]:
    pre, cats, nums = feature_pipeline(df)
    X = pre.fit_transform(df[cats + nums])
    return X, pre, cats, nums


# ---------------------------
# Preference Vector & Scoring
# ---------------------------

def _preference_vector(
    pre: ColumnTransformer,
    cats: List[str],
    nums: List[str],
    df_sample: pd.DataFrame,
    prefs: Dict,
) -> np.ndarray:
    """Build a single-row 'ideal' preference frame to transform with the pipeline."""
    row = {}

    # categorical picks
    cat_defaults = {
        "cpu_brand": prefs.get("cpu_brand"),
        "gpu_brand": prefs.get("gpu_brand"),
        "ram_type": prefs.get("ram_type"),
        "storage_primary_type": prefs.get("storage_type"),
        "display_resolution": prefs.get("display_resolution"),
        "display_panel": prefs.get("display_panel"),
        "display_aspect_ratio": prefs.get("display_aspect_ratio"),
        "os": prefs.get("os"),
        "intended_use_case": prefs.get("use_case"),
        "storage_primary_interface": None,
    }
    for c in cats:
        v = cat_defaults.get(c)
        if v is None:
            v = (
                df_sample[c].dropna().astype(str).value_counts().index[0]
                if df_sample.get(c) is not None and df_sample[c].dropna().size
                else ""
            )
        row[c] = v

    # numeric targets (center of desired ranges)
    num_defaults = {
        "year": prefs.get("min_year"),
        "cpu_cores": prefs.get("min_cpu_cores"),
        "gpu_vram_GB": prefs.get("min_vram"),
        "ram_base_GB": prefs.get("min_ram"),
        "ram_max_GB": max(prefs.get("min_ram", 0), 16) if prefs.get("min_ram") else None,
        "display_size_in": np.mean([prefs.get("min_screen", None), prefs.get("max_screen", None)]),
        "display_refresh_Hz": prefs.get("min_refresh"),
        "battery_capacity_Wh": prefs.get("min_battery_wh"),
        "battery_life_claimed_hr": None,
        "weight_kg": prefs.get("max_weight"),
        "storage_primary_capacity_GB": prefs.get("min_storage"),
        "price_myr": np.mean([prefs.get("budget_min"), prefs.get("budget_max")]),
        "ports_usb_a_count": None,
        "ports_usb_c_count": None,
        "ports_thunderbolt_count": None,
        "ports_hdmi_count": None,
    }
    for c in nums:
        row[c] = num_defaults.get(c, None)

    pref_df = pd.DataFrame([row])
    return pre.transform(pref_df[cats + nums])


def score_content_based(
    df: pd.DataFrame, X: np.ndarray, pre: ColumnTransformer, cats: List[str], nums: List[str], prefs: Dict
) -> np.ndarray:
    v = _preference_vector(pre, cats, nums, df, prefs)
    sim = cosine_similarity(X, v)[:, 0]
    return sim  # higher is better


def score_rule_based(df: pd.DataFrame, prefs: Dict) -> np.ndarray:
    """Heuristic score in [0,1], combining budget fit + use-case signals."""
    s = np.zeros(len(df), dtype=float)

    # Budget fit (soft)
    pmin, pmax = prefs.get("budget_min", 0), prefs.get("budget_max", float("inf"))
    price = pd.to_numeric(df["price_myr"], errors="coerce")
    in_budget = (price >= pmin) & (price <= pmax)
    budget_score = np.where(
        in_budget,
        1.0,
        np.clip(1.0 - (np.minimum(np.abs(price - np.clip(price, pmin, pmax)), 0.5 * pmax) / (0.5 * pmax)), 0, 1),
    )
    s += 0.35 * np.nan_to_num(budget_score, nan=0.0)

    def nz(col, default=0.0):
        return pd.to_numeric(df.get(col, default), errors="coerce").fillna(default).astype(float)

    ram = nz("ram_base_GB")
    vram = nz("gpu_vram_GB")
    cores = nz("cpu_cores")
    refresh = nz("display_refresh_Hz")
    weight = nz("weight_kg")
    battery = nz("battery_capacity_Wh")
    year = nz("year")
    storage = nz("storage_primary_capacity_GB")

    use = str(prefs.get("use_case") or "").strip().lower()

    # general desirability
    s += 0.10 * np.clip((year - 2018) / (2025 - 2018), 0, 1)
    s += 0.10 * np.clip(storage / 1024.0, 0, 1)  # 1TB ideal

    if use in ("gaming", "gamer"):
        s += 0.15 * np.clip((cores - 6) / 6.0, 0, 1)
        s += 0.18 * np.clip((vram - 6) / 6.0, 0, 1)
        s += 0.12 * np.clip((refresh - 120) / 60.0, 0, 1)
    elif use in ("creator", "content creation", "video editing", "designer"):
        s += 0.18 * np.clip((ram - 16) / 16.0, 0, 1)
        s += 0.12 * np.clip((vram - 4) / 8.0, 0, 1)
        res = df["display_resolution"].astype(str)
        hi = res.str.contains("2560|2880|3000|3200|3840|4K", case=False, na=False)
        s += 0.10 * hi.astype(float)
    elif use in ("student", "office", "productivity"):
        s += 0.12 * np.clip((ram - 8) / 8.0, 0, 1)
        s += 0.18 * np.clip((60 - weight) / 60.0, 0, 1)  # lighter better
        s += 0.10 * np.clip((battery - 50) / 40.0, 0, 1)
    elif use in ("business", "programming", "data"):
        s += 0.16 * np.clip((cores - 8) / 8.0, 0, 1)
        s += 0.14 * np.clip((ram - 16) / 16.0, 0, 1)

    # preference-aligned nudges
    if prefs.get("max_weight") is not None:
        s += 0.05 * (weight <= prefs["max_weight"]).astype(float)
    if prefs.get("min_battery_wh") is not None:
        s += 0.05 * (battery >= prefs["min_battery_wh"]).astype(float)
    if prefs.get("min_screen") is not None and prefs.get("max_screen") is not None and "display_size_in" in df:
        size = nz("display_size_in")
        s += 0.04 * ((size >= prefs["min_screen"]) & (size <= prefs["max_screen"])).astype(float)

    return np.clip(s, 0, 1)


def score_hybrid(
    df: pd.DataFrame,
    X: np.ndarray,
    pre: ColumnTransformer,
    cats: List[str],
    nums: List[str],
    prefs: Dict,
    alpha: float = 0.6,
) -> np.ndarray:
    cb = score_content_based(df, X, pre, cats, nums, prefs)
    rb = score_rule_based(df, prefs)
    # normalize CB to [0,1]
    if cb.max() > cb.min():
        cb = (cb - cb.min()) / (cb.max() - cb.min())
    return alpha * cb + (1 - alpha) * rb


# ---------------------------
# Public API: Recommend + Eval
# ---------------------------

def recommend(
    df: pd.DataFrame,
    prefs: Dict,
    algo: str = "hybrid",
    top_k: int = 10,
) -> pd.DataFrame:
    """Return a ranked list of laptops according to prefs."""
    view = df.copy()
    pmin, pmax = prefs.get("budget_min", 0), prefs.get("budget_max", float("inf"))
    if "price_myr" in view.columns:
        view = view[
            view["price_myr"].between(pmin, pmax, inclusive="both") | view["price_myr"].isna()
        ].copy()

    X, pre, cats, nums = build_matrix(view)

    if algo == "content":
        scores = score_content_based(view, X, pre, cats, nums, prefs)
    elif algo == "rule":
        scores = score_rule_based(view, prefs)
    else:
        scores = score_hybrid(view, X, pre, cats, nums, prefs, alpha=prefs.get("alpha", 0.6))

    view = view.copy()
    view["score"] = scores
    view = view.sort_values("score", ascending=False)

    show_cols = [
        c
        for c in [
            "brand", "series", "model", "year", "price_myr",
            "cpu_brand", "cpu_family", "cpu_model", "cpu_cores",
            "gpu_brand", "gpu_model", "gpu_vram_GB",
            "ram_base_GB", "ram_type",
            "storage_primary_type", "storage_primary_capacity_GB",
            "display_size_in", "display_resolution", "display_refresh_Hz", "display_panel",
            "battery_capacity_Wh", "weight_kg", "os", "intended_use_case", "score",
        ]
        if c in view.columns
    ]

    return view[show_cols].head(top_k).reset_index(drop=True)


def evaluate_precision_at_k(
    df: pd.DataFrame,
    scenarios: List[Tuple[str, Dict]],
    algo: str = "hybrid",
    k: int = 10,
) -> pd.DataFrame:
    """
    For each scenario (target use_case + prefs), ask the recommender for top-k,
    then compute Precision@k w.r.t. intended_use_case label.
    """
    results = []
    for label, prefs in scenarios:
        recs = recommend(df, prefs, algo=algo, top_k=k)
        if "intended_use_case" not in recs.columns:
            prec = np.nan
            matched = 0
        else:
            matched = (
                recs["intended_use_case"].astype(str).str.lower() == str(label).lower()
            ).sum()
            prec = matched / float(k)
        results.append({"scenario": label, "matched@k": int(matched), "precision@k": round(prec, 3)})
    return pd.DataFrame(results)


# ---------------------------
# Practical Add-ons: Split + Tuning
# ---------------------------

def split_df(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split by intended_use_case when available."""
    if "intended_use_case" in df.columns:
        stratify = df["intended_use_case"].fillna("unknown")
    else:
        stratify = None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tune_alpha(
    train_df: pd.DataFrame,
    labels: List[str] | None = None,
    k: int = 10,
    alphas: List[float] | None = None,
) -> Tuple[float, pd.DataFrame]:
    """Grid-search the hybrid alpha on TRAIN via mean Precision@k over scenarios."""
    if alphas is None:
        alphas = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]
    if labels is None:
        labels = sorted(
            [
                x
                for x in train_df.get("intended_use_case", pd.Series(dtype=str)).dropna().unique().tolist()
                if str(x).strip()
            ]
        )

    pref_base = dict(
        budget_min=float(train_df["price_myr"].quantile(0.05)) if "price_myr" in train_df else 0,
        budget_max=float(train_df["price_myr"].quantile(0.95)) if "price_myr" in train_df else 20000,
        min_ram=8, min_storage=512, min_vram=0, min_cpu_cores=4,
        min_battery_wh=0, min_refresh=60,
        min_screen=13.0, max_screen=16.0,
        max_weight=3.0, min_year=2018,
    )

    rows, best_alpha, best_score = [], 0.6, -1.0
    for a in alphas:
        precisions = []
        for lab in labels:
            prefs = {**pref_base, "use_case": lab, "alpha": a}
            recs = recommend(train_df, prefs, algo="hybrid", top_k=k)
            if "intended_use_case" in recs.columns and len(recs) > 0:
                match = (
                    recs["intended_use_case"].astype(str).str.lower() == str(lab).lower()
                ).sum()
                precisions.append(match / k)
        mean_p = float(np.mean(precisions)) if precisions else float("nan")
        rows.append({"alpha": a, "precision@k": round(mean_p, 3) if not math.isnan(mean_p) else np.nan})
        if not math.isnan(mean_p) and mean_p > best_score:
            best_score, best_alpha = mean_p, a

    return best_alpha, pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def evaluate_with_split(
    df: pd.DataFrame, test_size: float = 0.2, k: int = 10
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Practical-style workflow:
    1) Stratified TRAIN/TEST split
    2) Tune alpha on TRAIN
    3) Evaluate Precision@k on TEST using best alpha
    """
    train_df, test_df = split_df(df, test_size=test_size)
    labels = sorted(
        [
            x
            for x in train_df.get("intended_use_case", pd.Series(dtype=str)).dropna().unique().tolist()
            if str(x).strip()
        ]
    ) or ["Gaming", "Student", "Creator", "Business"]

    best_alpha, alpha_table = tune_alpha(train_df, labels=labels, k=k)

    scenarios = [
        (
            lab,
            {
                "budget_min": float(test_df["price_myr"].quantile(0.05)) if "price_myr" in test_df else 0,
                "budget_max": float(test_df["price_myr"].quantile(0.95)) if "price_myr" in test_df else 20000,
                "min_ram": 8, "min_storage": 512, "min_vram": 0, "min_cpu_cores": 4,
                "min_battery_wh": 0, "min_refresh": 60,
                "min_screen": 13.0, "max_screen": 16.0, "max_weight": 3.0, "min_year": 2018,
                "use_case": lab, "alpha": best_alpha,
            },
        )
        for lab in labels
    ]

    test_results = evaluate_precision_at_k(test_df, scenarios, algo="hybrid", k=k)
    summary = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "best_alpha": best_alpha,
        "mean_precision@k_test": round(float(test_results["precision@k"].mean()), 3)
        if not test_results.empty
        else None,
    }
    return summary, alpha_table, test_results
