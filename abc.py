from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score
# cosine_similarity can be used later if you want a pure content-sim variant
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Purpose weight profiles
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

# Numeric columns we normalize when building vectors (handy if you add cosine later)
NUM_COLS = ["RAM_GB", "Storage_GB", "Battery_Wh", "Screen_Size", "Weight_kg", "Price_MYR"]

# -----------------------------
# Data loading & cleaning
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Read CSV, drop duplicates, and remove obviously invalid rows.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)

    # Basic sanity checks
    if "Price_MYR" not in df.columns or "RAM_GB" not in df.columns or "Storage_GB" not in df.columns:
        missing = {"Price_MYR", "RAM_GB", "Storage_GB"} - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df[df["Price_MYR"] > 0]
    df = df[df["RAM_GB"] > 0]
    df = df[df["Storage_GB"] > 0]
    df = df.reset_index(drop=True)
    return df


def normalize_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numeric columns to z-scores (mean=0, std=1).
    Useful if you want to compute cosine similarity later.
    """
    for c in NUM_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing numeric column: {c}")
    scaler = StandardScaler()
    Xn = scaler.fit_transform(df[NUM_COLS])
    Xn = pd.DataFrame(Xn, columns=[f"z_{c}" for c in NUM_COLS], index=df.index)
    return Xn, scaler


# -----------------------------
# Purpose scoring (simple & explainable)
# -----------------------------
def _purpose_score_row(row: pd.Series, weights: Dict[str, float]) -> float:
    """
    Compute an explainable score using:
      - CPU/GPU string heuristics (bonuses)
      - Normalized RAM/Storage caps
      - Small bonuses for SSD and Touchscreen
    """
    # CPU heuristic
    cpu_bonus = 0.0
    cpu = str(row.get("CPU", "")).lower()
    if any(x in cpu for x in ["ultra 9", " i9", "ryzen 9"]):
        cpu_bonus = 1.0
    elif any(x in cpu for x in ["ultra 7", " i7", "ryzen 7"]):
        cpu_bonus = 0.8
    elif any(x in cpu for x in ["ultra 5", " i5", "ryzen 5"]):
        cpu_bonus = 0.6
    elif any(x in cpu for x in [" i3", "ryzen 3"]):
        cpu_bonus = 0.4
    elif any(x in cpu for x in ["m1", "m2", "m3", "m4", "snapdragon"]):
        cpu_bonus = 0.7

    # GPU heuristic
    gpu_bonus = 0.0
    g = str(row.get("GPU", "")).upper()
    if any(x in g for x in ["RTX 5090", "RTX 5080", "RTX 50"]):
        gpu_bonus = 1.0
    elif any(x in g for x in ["RTX 4070", "RTX 4060", "RTX 40", "RTX 3050"]):
        gpu_bonus = 0.85
    elif ("GTX" in g) or (" RX " in g) or g.startswith("RX "):
        gpu_bonus = 0.7
    elif any(x in g for x in ["IRIS", "UHD", "ADRENO", "SNAPDRAGON"]):
        gpu_bonus = 0.4

    # Normalized RAM/Storage caps
    ram_norm = min(float(row.get("RAM_GB", 0)) / 64.0, 1.0)
    storage_norm = min(float(row.get("Storage_GB", 0)) / 2048.0, 1.0)

    # Normalize weights
    w = weights.copy()
    total = sum(w.values())
    for k in w:
        w[k] = w[k] / total if total > 0 else 0.0

    score = (
        w["cpu"] * cpu_bonus
        + w["gpu"] * gpu_bonus
        + w["ram"] * ram_norm
        + w["storage"] * storage_norm
    )

    # QoL bonuses
    if str(row.get("Storage_Type", "")).upper() == "SSD":
        score += 0.02
    if str(row.get("TouchScreen", "")).lower() == "yes":
        score += 0.01

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
    Filter candidates by budget and min specs, then rank by purpose score
    (with a value-for-money tie-breaker).
    """
    cand = df.copy()

    # Budget (10% wiggle room)
    if budget_myr is not None:
        cand = cand[cand["Price_MYR"] <= float(budget_myr) * 1.10]

    # Hard minimum specs
    if min_specs:
        for key, val in min_specs.items():
            if key in cand.columns:
                cand = cand[cand[key] >= val]

    if cand.empty:
        return cand

    # Choose weights
    weights = USE_PROFILES.get(
        purpose, {"cpu": 0.35, "gpu": 0.30, "ram": 0.20, "storage": 0.15}
    )

    # Purpose score + value-per-RM to break ties
    cand = cand.assign(purpose_score=cand.apply(lambda r: _purpose_score_row(r, weights), axis=1))
    cand = cand.assign(value_score=cand["purpose_score"] / np.log1p(cand["Price_MYR"]))
    cand = cand.sort_values(
        ["purpose_score", "value_score", "RAM_GB", "Storage_GB"],
        ascending=[False, False, False, False],
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
      - purpose := purpose_map[row.Category]
      - budget := row.Price_MYR
      - recommend top-k from TRAIN set
    Hit if any recommended row shares the same Category as the test row.
    Returns hit_rate@k, tested count, and hits.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    hits, total = 0, 0
    for _, row in test_df.iterrows():
        cat = str(row["Category"]).lower()
        purpose = purpose_map.get(cat, "office_browsing")
        budget = float(row["Price_MYR"])
        recs = recommend(train_df, budget_myr=budget, purpose=purpose, min_specs=None, top_k=k)
        if not recs.empty and any(str(c).lower() == cat for c in recs["Category"].tolist()):
            hits += 1
        total += 1

    hit_rate = hits / total if total > 0 else 0.0
    return {"hit_rate@{}".format(k): hit_rate, "tested": total, "hits": hits}


# -----------------------------
# Lab-style evaluation (confusion matrix)
# -----------------------------
def _predict_category_from_recs(train_df: pd.DataFrame, purpose: str, budget_myr: float, k: int) -> str:
    """
    Predict ONE category: most frequent category among top-k recs.
    """
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
    50/50 split. For each test row:
      - actual := its Category
      - purpose := purpose_map[Category]
      - budget := its Price_MYR
      - predicted := mode(Category) among top-k TRAIN recommendations
    Returns accuracy, macro precision/recall, and a confusion matrix DataFrame.
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

    # Build labels without "none" (but keep "none" in y_pred to reflect misses)
    labels = sorted(list(set([c for c in y_true if c != "none"] + [c for c in y_pred if c != "none"])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])

    # Macro precision/recall on real categories; treat "none" as wrong
    y_pred_clean = [p if p in labels else "none" for p in y_pred]
    acc = (np.array(y_true) == np.array(y_pred_clean)).mean()
    try:
        prec = precision_score(y_true, y_pred_clean, labels=labels, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred_clean, labels=labels, average="macro", zero_division=0)
    except Exception:
        prec, rec = 0.0, 0.0

    return {
        "tested": len(y_true),
        "accuracy": float(round(acc, 4)),
        "precision_macro": float(round(prec, 4)),
        "recall_macro": float(round(rec, 4)),
        "labels": labels + (["none"] if "none" in y_pred_clean else []),
        "confusion_matrix": cm_df,
    }


# -----------------------------
# CLI runner (optional)
# -----------------------------
if __name__ == "__main__":
    # Example quick run from terminal: python abc.py
    csv_path = "data/laptop_market_2025_10000_with_chromebook.csv"  # adjust if needed
    df = load_data(csv_path)

    # Map dataset categories to purposes for evaluation
    purpose_map = {
        "gaming": "gaming",
        "ultrabook": "study_programming",
        "convertible": "office_browsing",
        "workstation": "content_creation",
        "compact": "study_programming",
        "chromebook": "chromebook_use",
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
