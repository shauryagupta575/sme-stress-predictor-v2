"""Feature engineering with an explicit fit/transform split.

Why this module exists: the original pipeline engineered features inside a
notebook using statistics computed over the whole dataset (per-column medians
for sentinel fill, a 99th-percentile clip, a max-based normaliser) and never
persisted them. Anything scoring a new borrower later would recompute those
statistics from whatever data it happened to have, producing different feature
values than the model was trained on — silent train/serve skew.

Here, `fit_params` learns those statistics once and returns a plain dict that is
saved to JSON. `transform` takes that dict and is fully deterministic. The
training script and the dashboard call the same `transform`, so a borrower gets
the same numbers in both places.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import config as C

# ── sentinel handling ───────────────────────────────────────────────

def _demote_sentinels(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace the -99999 / -99998 'not reported' codes with NaN."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].replace(C.SENTINEL_VALUES, np.nan)
    return out


def fit_params(df: pd.DataFrame) -> dict[str, Any]:
    """Learn every dataset-level statistic the transform depends on.

    Call this on the training split only, then persist the result.
    """
    present = [c for c in C.RAW_NUMERIC_FEATURES if c in df.columns]
    clean = _demote_sentinels(df, present)

    medians = {c: float(clean[c].median()) for c in present}

    # Only carry indicators for columns where missingness is common enough to
    # carry signal; below the threshold it is noise and costs a column.
    missing_rates = {c: float(clean[c].isna().mean()) for c in present}
    candidates = sorted(
        c for c, r in missing_rates.items() if r > C.MISSING_INDICATOR_THRESHOLD
    )

    # Several column families share one sentinel pattern — every enquiry field is
    # unreported for the same 6,321 borrowers — so their indicators would be
    # identical columns. Keep one representative per distinct pattern.
    indicator_cols: list[str] = []
    seen_patterns: dict[bytes, str] = {}
    for col in candidates:
        pattern = clean[col].isna().to_numpy().tobytes()
        if pattern not in seen_patterns:
            seen_patterns[pattern] = col
            indicator_cols.append(col)

    # Normalisers for the ratio features, learned rather than recomputed.
    income = clean["NETMONTHLYINCOME"].replace(0, np.nan)
    tenure = clean["Time_With_Curr_Empr"]

    params: dict[str, Any] = {
        "medians": medians,
        "missing_rates": missing_rates,
        "indicator_cols": indicator_cols,
        "income_max": float(income.max()),
        "tenure_max": float(tenure.max()),
        "education_order": C.EDUCATION_ORDER,
    }

    # Clip bounds for the engineered ratios, learned on the fitted frame so the
    # 99th percentile is not recomputed per-batch at serve time.
    engineered = _engineer(clean, params, clip=None)
    clip_cols = [
        "enquiry_acceleration", "enquiry_rejection_proxy",
        "unsecured_loan_ratio", "recent_loan_hunger", "closure_rate",
        "utilisation_pressure", "credit_hunger_ratio", "exposure_to_income",
    ]
    params["clips"] = {
        c: [float(engineered[c].quantile(0.005)), float(engineered[c].quantile(0.995))]
        for c in clip_cols
        if c in engineered.columns
    }
    return params


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Ratio with a +1 denominator guard, matching the project's convention."""
    return num.fillna(0) / (den.fillna(0) + 1)


def _engineer(
    clean: pd.DataFrame, params: dict[str, Any], clip: dict | None
) -> pd.DataFrame:
    """Build the engineered feature block from a sentinel-demoted frame."""
    f = pd.DataFrame(index=clean.index)

    # ── Credit-seeking behaviour ────────────────────────────────────
    # Recency-weighted enquiry count. This was the project's strongest
    # predictor and its headline causal finding, so it is kept verbatim:
    # last 3 months weighted 4x, last 6 months 2x, last 12 months 1x.
    f["enquiry_acceleration"] = (
        clean["enq_L3m"].fillna(0) * 4
        + clean["enq_L6m"].fillna(0) * 2
        + clean["enq_L12m"].fillna(0) * 1
    )

    # Enquiries per active trade line: many enquiries yielding few accounts
    # suggests the borrower is being declined.
    f["enquiry_rejection_proxy"] = _safe_div(clean["tot_enq"], clean["Tot_Active_TL"])

    # Share of lifetime enquiries that fall in the last 6 months — a burst of
    # credit-seeking relative to the borrower's own baseline.
    f["enquiry_recency_share"] = _safe_div(clean["enq_L6m"], clean["tot_enq"])

    # Unsecured credit hunger: credit-card and personal-loan enquiries are the
    # ones a cash-squeezed borrower reaches for first.
    f["credit_hunger_ratio"] = _safe_div(
        clean["CC_enq_L6m"].fillna(0) + clean["PL_enq_L6m"].fillna(0),
        clean["enq_L6m"],
    )

    # Months since the last enquiry, inverted so that higher = more recent.
    f["enquiry_freshness"] = 1.0 / (clean["time_since_recent_enq"].fillna(999) + 1)

    # ── Balance-sheet utilisation ───────────────────────────────────
    # Both utilisation columns are majority-sentinel, so the fill value is
    # carried alongside a missingness indicator rather than trusted outright.
    f["utilisation_pressure"] = (
        clean["CC_utilization"].fillna(params["medians"]["CC_utilization"])
        + clean["PL_utilization"].fillna(params["medians"]["PL_utilization"])
    ) / 2.0

    f["exposure_to_income"] = _safe_div(
        clean["max_unsec_exposure_inPct"], clean["NETMONTHLYINCOME"] / 1000.0
    )

    f["balance_pressure"] = clean["pct_currentBal_all_TL"].fillna(
        params["medians"]["pct_currentBal_all_TL"]
    )

    # ── Portfolio structure ─────────────────────────────────────────
    f["unsecured_loan_ratio"] = _safe_div(clean["Unsecured_TL"], clean["Tot_Active_TL"])

    # Recent account opening, weighted toward the last 6 months.
    f["recent_loan_hunger"] = (
        clean["Total_TL_opened_L6M"].fillna(0) * 2
        + clean["Total_TL_opened_L12M"].fillna(0) * 1
    )

    # Healthy borrowers close accounts; a low closure rate alongside heavy
    # opening is the pattern worth flagging.
    f["closure_rate"] = _safe_div(
        clean["Tot_TL_closed_L12M"], clean["Total_TL_opened_L12M"]
    )
    f["open_close_imbalance"] = f["recent_loan_hunger"] - clean[
        "Tot_TL_closed_L12M"
    ].fillna(0)

    # Portfolio maturity: a thin, young file behaves differently from a long
    # established one at the same utilisation.
    f["portfolio_age_span"] = (
        clean["Age_Oldest_TL"].fillna(params["medians"]["Age_Oldest_TL"])
        - clean["Age_Newest_TL"].fillna(params["medians"]["Age_Newest_TL"])
    )
    f["newest_tl_age"] = clean["Age_Newest_TL"].fillna(
        params["medians"]["Age_Newest_TL"]
    )

    # Product-mix concentration in unsecured lines.
    unsecured_products = clean["CC_TL"].fillna(0) + clean["PL_TL"].fillna(0)
    f["unsecured_product_share"] = _safe_div(unsecured_products, clean["Total_TL"])

    # ── Borrower profile ────────────────────────────────────────────
    f["income_stability"] = (
        clean["NETMONTHLYINCOME"].fillna(params["medians"]["NETMONTHLYINCOME"])
        / (params["income_max"] + 1)
        + clean["Time_With_Curr_Empr"].fillna(params["medians"]["Time_With_Curr_Empr"])
        / (params["tenure_max"] + 1)
    )
    f["log_income"] = np.log1p(
        clean["NETMONTHLYINCOME"].fillna(params["medians"]["NETMONTHLYINCOME"]).clip(0)
    )
    f["AGE"] = clean["AGE"].fillna(params["medians"]["AGE"])
    f["Time_With_Curr_Empr"] = clean["Time_With_Curr_Empr"].fillna(
        params["medians"]["Time_With_Curr_Empr"]
    )

    if clip:
        for col, (lo, hi) in clip.items():
            if col in f.columns:
                f[col] = f[col].clip(lo, hi)

    return f.replace([np.inf, -np.inf], 0).fillna(0)


def transform(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Turn raw merged rows into the model's feature matrix.

    Deterministic given `params`. Used by both training and the dashboard.
    """
    present = [c for c in C.RAW_NUMERIC_FEATURES if c in df.columns]
    clean = _demote_sentinels(df, present)

    # Engineered block.
    out = _engineer(clean, params, clip=params.get("clips"))

    # Retained raw columns, median-filled from the persisted params.
    passthrough = [
        "tot_enq", "enq_L3m", "enq_L6m", "enq_L12m",
        "CC_utilization", "PL_utilization", "max_unsec_exposure_inPct",
        "Tot_Active_TL", "Tot_Closed_TL", "Total_TL",
        "Secured_TL", "Unsecured_TL",
        "pct_tl_open_L6M", "pct_tl_closed_L6M",
        "NETMONTHLYINCOME",
        "CC_Flag", "PL_Flag", "HL_Flag", "GL_Flag",
    ]
    for col in passthrough:
        if col in clean.columns and col not in out.columns:
            out[col] = clean[col].fillna(params["medians"].get(col, 0.0))

    # Missingness indicators — "not reported" as a first-class signal.
    for col in params["indicator_cols"]:
        if col in clean.columns:
            out[f"{col}_missing"] = clean[col].isna().astype(int)

    # Categoricals: education is ordinal, the rest are low-cardinality dummies.
    if "EDUCATION" in df.columns:
        out["education_level"] = (
            df["EDUCATION"].map(params["education_order"]).fillna(0).astype(int)
        )
    if "MARITALSTATUS" in df.columns:
        out["is_married"] = (df["MARITALSTATUS"] == "Married").astype(int)
    if "GENDER" in df.columns:
        out["is_male"] = (df["GENDER"] == "M").astype(int)

    for col in ("last_prod_enq2", "first_prod_enq2"):
        if col in df.columns:
            for level in ["CC", "PL", "ConsumerLoan", "AL", "HL", "others"]:
                out[f"{col}_{level}"] = (df[col] == level).astype(int)

    return out.astype(float)


# ── persistence ─────────────────────────────────────────────────────

def save_params(params: dict[str, Any], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)


def load_params(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def assert_no_leakage(feature_cols: list[str]) -> None:
    """Fail loudly if any label-defining column reached the feature matrix.

    Called from training and importable by tests. A leakage bug that surfaces as
    a suspiciously good AUC is much harder to catch than one that raises.
    """
    banned = set(C.LABEL_DEFINING_COLS)
    hits = sorted(c for c in feature_cols if c in banned)
    # Indicator columns derived from a banned source count as leakage too.
    hits += sorted(
        c for c in feature_cols
        if c.endswith("_missing") and c[: -len("_missing")] in banned
    )
    if hits:
        raise AssertionError(
            f"Label-defining columns leaked into the feature matrix: {hits}"
        )