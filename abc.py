"""
abc.py — AI backend for Laptop Recommender (BMCS2003)

✅ What this file provides
- load_data(): read & validate CSV
- recommend(): budget/spec filters + purpose-weighted scoring (content-style)
- train_test_eval(): 50/50 split hit_rate@k (lab style)
- eval_confusion_labstyle(): 50/50 split confusion matrix + accuracy/precision/recall
- CLI runner for quick checks:  python abc.py

Expected CSV columns:
  Category, Model, CPU, GPU, RAM_GB, Storage_GB, Storage_Type, Screen_Size,
  Weight_kg, OS, Battery_Wh, TouchScreen, Price_USD, Price_MYR
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# -----------------------------
# Config / Profiles
# -----------------------------
USE_PROFILES: Dict[str, Dict[str, float]] = {
    "gaming": {"cpu": 0.25, "gpu": 0.55, "ram": 0.15, "storage": 0.05},
    "study_programming": {"cpu": 0.45, "gpu": 0.10, "ram": 0.30, "storage": 0.15},
    "office_browsing": {"cpu": 0.35, "gpu": 0.05, "ram": 0.35, "storage": 0.25},
    "video_editing": {"cpu": 0.35, "gpu": 0.30, "ram": 0.20, "storage": 0.15},
    "content_creation": {"cpu": 0.40, "gpu": 0.25, "ram": 0.20, "storage": 0.15},
    "data_science": {"cpu": 0.45, "gpu": 0.25, "ram": 0.20, "storage": 0.10},
    "chromebook_use": {"cpu": 0.30, "gpu": 0.05, "ram": 0.40, "storage": 0.25},
}
REQUIRED_COLS = [
    "Category","Model","CPU","GPU","RAM_GB","Storage_GB","Storage_Type",
    "Screen_Size","Weight_kg","OS","Battery_Wh","TouchScreen","Price_USD","Price_MYR"
]
NUM_COLS = ["RAM_GB","Storage_GB","Battery_Wh","Screen_Size","Weight_kg","Price_MYR"]


# -----------------------------
# Data loading & validation
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Read CSV, validate required columns, drop duplicates, and remove invalid rows.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df = df.drop_duplicates().reset_index(drop=True)

    # Basic sanity checks
    df = df[df["Price_MYR"].astype(float) > 0]
    df = df[df["RAM_GB"].astype(float) > 0]
    df = df[df["Storage_GB"].astype(float) > 0]

    # Clean TouchScreen values to 'Yes'/'No'
    df["TouchScreen"] = df["TouchScreen"].astype(str).str.strip().str.lower().map(
        {"yes":"Yes","no":"No"}
    ).fillna(df["TouchScreen"])

    return df.reset_index(drop=True)


def normalize_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numeric columns to z-scores (mean=0, std=1).
    (Useful if you later add cosine similarity.)
    """
    for c in NUM_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing numeric column: {c}")
    scaler = StandardScaler()
    Xn = scaler.fit_transform(df[NUM_COLS])
    Xn = pd.DataFrame(Xn, columns=[f"z_{c}" for c in NUM_COLS], index=df.index)
    return Xn, scaler


# -----------------------------
# Purpose scoring (fast, explainable)
# -----------------------------
def _purpose_score_row(row: pd.Series, weights: Dict[str, float]) -> float:
    """
    Compute an interpretable score using string heuristics for CPU/GPU +
    normalized RAM/Storage caps and small QoL bonuses.
    """
    # CPU heuristic
    cpu_txt = str(row.get("CPU", "")).lower()
    if any(x in cpu_txt for x in ["ultra 9"," i9","ryzen 9"]): cpu_bonus = 1.0
    elif any(x in cpu_txt for x in ["ultra 7"," i7","ryzen 7"]): cpu_bonus = 0.8
    elif any(x in cpu_txt for x in ["ultra 5"," i5","ryzen 5"]): cpu_bonus = 0.6
    elif any(x in cpu_txt for x in [" i3","ryzen 3"]): cpu_bonus = 0.4
    elif any(x in cpu_txt for x in ["m1","m2","m3","m4","snapdragon"]): cpu_bonus = 0.7
    else: cpu_bonus = 0.5

    # GPU heuristic
    gpu_txt = str(row.get("GPU", "")).upper()
    if any(x in gpu_txt for x in ["RTX 5090","RTX 5080","RTX 50"]): gpu_bonus = 1.0
    elif any(x in gpu_txt for x in ["RTX 4070","RTX 4060","RTX 40","RTX 3050"]): gpu_bonus = 0.85
    elif ("GTX" in gpu_txt) or (" RX " in gpu_txt) or gpu_txt.startswith("RX "): gpu_bonus = 0.7
    elif any(x in gpu_txt for x in ["IRIS","UHD","ADRENO","SNAPDRAGON","APPLE"]): gpu_bonus = 0.4
    else: gpu_bonus = 0.5

    # Normalized spec caps
    ram_norm = min(float(row.get("RAM_GB", 0)) / 64.0, 1.0)
    storage_norm = min(float(row.get("Storage_GB", 0)) / 2048.0, 1.0)

    # Normalize weights
    w = weights.copy()
    total = sum(w.values()) or 1.0
    for k in w: w[k] /= total

    score = (
        w["cpu"] * cpu_bonus +
        w["gpu"] * gpu_bonus +
        w["ram"] * ram_norm +
        w["storage"] * storage_norm
    )

    # QoL bonuses
    if str(row.get("Storage_Type","")).upper() == "SSD": score += 0.02
    if str(row.get("TouchScreen","")).lower() == "yes": score += 0.01

    return float(score)


# -----------------------------
# Recommender
# -----------------------------
def recommend(
    df: pd.DataFrame,
    budget_myr: Optional[float] = None,
    purpose: Optional[str] = None,
    min_specs: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Filter by budget & min specs, then rank by purpose score (+ value/RM tie-breaker).
    Returns top_k rows as a DataFrame.
    """
    cand = df.copy()

    # Budget (10% wiggle)
    if budget_myr is not None:
        cand = cand[cand["Price_MYR"] <= float(budget_myr) * 1.10]

    # Hard minimum specs (ignore keys that don't exist)
    if min_specs:
        for key, val in min_specs.items():
            if key in cand.columns:
                cand = cand[cand[key] >= val]

    if cand.empty:
        return cand

    weights = USE_PROFILES.get(purpose, {"cpu":0.35,"gpu":0.30,"ram":0.20,"storage":0.15})

    cand = cand.assign(purpose_score=cand.apply(lambda r: _purpose_score_row(r, weights), axis=1))
    cand = cand.assign(value_score=cand["purpose_score"] / np.log1p(cand["Price_MYR"]))
    cand = cand.sort_values(
        ["purpose_score","value_score","RAM_GB","Storage_GB"],
        ascending=[False, False, False, False]
    )
    return cand.head(top_k).reset_index(drop=True)


# -----------------------------
# Lab-style evaluation (hit rate)
# -----------------------------
def train_test_eval(
    df: pd.DataFrame,
    purpose_map: Dict[str, str],
    test_size: float = 0.5,
    seed: int = 42,
    k: int = 5,
) -> Dict[str, Any]:
    """
    50/50 split. For each test row:
      purpose := purpose_map[row.Category]
      budget  := row.Price_MYR
      recommend top-k from TRAIN
    Hit if any recommended item shares the same Category as the test row.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    hits = 0
    total = 0

    for _, row in test_df.iterrows():
        cat = str(row["Category"]).lower()
        purpose = purpose_map.get(cat, "office_browsing")
        budget = float(row["Price_MYR"])
        recs = recommend(train_df, budget_myr=budget, purpose=purpose, min_specs=None, top_k=k)
        if not recs.empty and any(str(c).lower() == cat for c in recs["Category"].tolist()):
            hits += 1
        total += 1

    return {"hit_rate@{}".format(k): (hits/total if total else 0.0), "tested": total, "hits": hits}


# -----------------------------
# Lab-style evaluation (confusion matrix)
# -----------------------------
def _predict_category_from_recs(train_df: pd.DataFrame, purpose: str, budget_myr: float, k: int) -> str:
    recs = recommend(train_df, budget_myr=budget_myr, purpose=purpose, min_specs=None, top_k=k)
    if recs.empty:
        return "none"
    return str(recs["Category"].value_counts().idxmax()).lower()


def eval_confusion_labstyle(
    df: pd.DataFrame,
    purpose_map: Dict[str, str],
    test_size: float = 0.5,
    seed: int = 42,
    k: int = 5,
) -> Dict[str, Any]:
    """
    50/50 split. Predict ONE category per test row using mode(Category) of top-k TRAIN recs.
    Returns confusion matrix + accuracy + macro precision/recall.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        actual_cat = str(row["Category"]).lower()
        purpose = purpose_map.get(actual_cat, "office_browsing")
        budget = float(row["Price_MYR"])
        pred_cat = _predict_category_from_recs(train_df, purpose, budget, k)
        y_true.append(actual_cat)
        y_pred.append(pred_cat)

    labels = sorted(list(set([c for c in y_true if c != "none"] + [c for c in y_pred if c != "none"])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])

    y_pred_clean = [p if p in labels else "none" for p in y_pred]
    acc = (np.array(y_true) == np.array(y_pred_clean)).mean()
    prec = precision_score(y_true, y_pred_clean, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred_clean, labels=labels, average="macro", zero_division=0)

    return {
        "tested": len(y_true),
        "accuracy": float(round(acc, 4)),
        "precision_macro": float(round(prec, 4)),
        "recall_macro": float(round(rec, 4)),
        "labels": labels + (["none"] if "none" in y_pred_clean else []),
        "confusion_matrix": cm_df,
    }


# -----------------------------
# CLI runner (quick check)
# -----------------------------
if __name__ == "__main__":
    csv_path = "data/laptop_market_2025_10000_with_chromebook.csv"  # adjust if needed
    df = load_data(csv_path)

    # Map dataset categories to purposes for evaluation
    purpose_map = {
        "gaming":"gaming",
        "ultrabook":"study_programming",
        "convertible":"office_browsing",
        "workstation":"content_creation",
        "compact":"study_programming",
        "chromebook":"chromebook_use",
    }

    print("=== LAB-STYLE HIT RATE EVAL (50/50 split) ===")
    res = train_test_eval(df, purpose_map, test_size=0.5, seed=42, k=5)
    print(res)

    print("\n=== LAB-STYLE CONFUSION MATRIX EVAL (50/50 split) ===")
    res2 = eval_confusion_labstyle(df, purpose_map, test_size=0.5, seed=42, k=5)
    print(f"Tested: {res2['tested']}")
    print(f"Accuracy: {res2['accuracy']}")
    print(f"Precision (macro): {res2['precision_macro']}")
    print(f"Recall (macro): {res2['recall_macro']}")
    print("\nConfusion Matrix:")
    print(res2["confusion_matrix"])
