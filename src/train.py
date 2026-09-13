"""Train the MSME stress model and report metrics that survive scrutiny.

Two changes from the original notebook approach:

1. Metrics come from stratified 5-fold cross-validation, reported as mean +/- sd.
   A single 80/20 split gives one number with no sense of its stability; at a
   10.6% event rate that number moves meaningfully with the seed.

2. Average precision is the headline metric, not ROC-AUC. With ~11% positives,
   AUC is dominated by the majority class and reads optimistically. Average
   precision and precision@top-k reflect what a credit team actually does:
   work a ranked queue from the top under fixed review capacity.

Run from the project root:  python -m src.train
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C
from . import features, labels

RANDOM_STATE = 42
N_FOLDS = 5

BASE_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: float) -> float:
    """Share of genuinely stressed accounts in the top k fraction by score."""
    n = max(1, int(len(scores) * k))
    top = np.argsort(scores)[::-1][:n]
    return float(np.asarray(y_true)[top].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: float) -> float:
    """Share of all stressed accounts captured within the top k fraction."""
    y_true = np.asarray(y_true)
    n = max(1, int(len(scores) * k))
    top = np.argsort(scores)[::-1][:n]
    total = y_true.sum()
    return float(y_true[top].sum() / total) if total else 0.0


def decile_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Stress rate and lift by score decile — the standard credit-risk view."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": scores})
    df["decile"] = pd.qcut(df["score"].rank(method="first"), 10, labels=False)
    df["decile"] = 10 - df["decile"]  # decile 1 = highest risk
    base = df["y"].mean()
    out = (
        df.groupby("decile")
        .agg(accounts=("y", "size"), stressed=("y", "sum"), stress_rate=("y", "mean"))
        .reset_index()
        .sort_values("decile")
    )
    out["lift"] = out["stress_rate"] / base
    return out


def gold_loan_cohort(raw: pd.DataFrame, y: pd.Series) -> list[dict]:
    """Stress rate by gold-loan count.

    Worth persisting separately from feature importance: gold loans are a
    collateral product, not a delinquency field, so nothing about this
    relationship can be explained by label leakage. Repeated pledging and
    redeeming of gold is a recognised working-capital bridge for small Indian
    traders, which makes the gradient interpretable rather than incidental.

    Bands are disjoint, not cumulative, so each bar is the stress rate of that
    band alone rather than of every band above it.

    IMPORTANT — read alongside gold_loan_within_exposure(). The marginal
    gradient here is confounded: gold-loan count is strongly correlated with
    total trade lines, and once exposure is held fixed the association reverses.
    This function is retained because the contrast between the two is the useful
    result, not because the marginal gradient means anything on its own.
    """
    base = float(y.mean())
    gold = raw["Gold_TL"].fillna(0)
    bands = [(0, 0, "None"), (1, 4, "1–4"), (5, 9, "5–9"),
             (10, 19, "10–19"), (20, 39, "20–39"), (40, None, "40+")]
    rows = []
    for lo, hi, label in bands:
        mask = (gold >= lo) if hi is None else (gold >= lo) & (gold <= hi)
        n = int(mask.sum())
        if n < 50:
            continue
        rate = float(y[mask].mean())
        rows.append(
            {
                "band": label,
                "accounts": n,
                "share_of_book": n / len(raw),
                "stress_rate": rate,
                "lift": rate / base if base else 0.0,
            }
        )
    return rows


def gold_loan_within_exposure(raw: pd.DataFrame, y: pd.Series) -> list[dict]:
    """Stress rate by gold-loan band, holding trade-line count fixed.

    A worked example of why the exposure control matters. The marginal
    gold-loan gradient rises monotonically and looks like a strong finding.
    Conditioned on exposure it flattens and then reverses — borrowers with many
    gold loans are no riskier than their same-exposure peers, and in the largest
    stratum they are slightly safer. Gold loans are secured credit, so having
    them may simply indicate pledgeable collateral.

    The marginal gradient was Simpson's paradox: gold-loan count and trade-line
    count move together, and the label rises with trade-line count.
    """
    band = labels.exposure_band(raw)
    gold = pd.cut(
        raw["Gold_TL"].fillna(0),
        bins=[-1, 0, 4, 9, 10_000],
        labels=["none", "1-4", "5-9", "10+"],
    )
    rows = []
    for b in C.EXPOSURE_BAND_LABELS:
        for g in ["none", "1-4", "5-9", "10+"]:
            mask = ((band == b) & (gold == g)).to_numpy()
            n = int(mask.sum())
            if n < 40:
                continue
            rows.append(
                {
                    "exposure_band": b,
                    "gold_band": g,
                    "accounts": n,
                    "stress_rate": float(y.to_numpy()[mask].mean()),
                }
            )
    return rows


def exposure_baseline(
    raw: pd.DataFrame, y: pd.Series, idx_train, idx_test
) -> dict:
    """Train a model on account counts and credit-file ages alone.

    This is the benchmark that matters. The label counts deterioration events,
    so a borrower holding more accounts has more chances to register one; a
    model can score well by doing little more than counting trade lines. The
    exposure-only AUC is the floor, and the full model's claim to value is the
    margin above it — not the margin above 0.5.
    """
    Xe_train, medians = labels.build_exposure_matrix(raw.loc[idx_train])
    Xe_test, _ = labels.build_exposure_matrix(raw.loc[idx_test], medians)

    pos_weight = (y.loc[idx_train] == 0).sum() / max(1, (y.loc[idx_train] == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
        eval_metric="aucpr", random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        scale_pos_weight=pos_weight,
    )
    model.fit(Xe_train, y.loc[idx_train], verbose=False)
    probs = model.predict_proba(Xe_test)[:, 1]
    yt = y.loc[idx_test].to_numpy()
    return {
        "features": list(Xe_train.columns),
        "auc": float(roc_auc_score(yt, probs)),
        "ap": float(average_precision_score(yt, probs)),
        "precision_top10": precision_at_k(yt, probs, 0.10),
    }


def no_overlap_variant(
    X: pd.DataFrame, y: pd.Series, idx_train, idx_test
) -> dict:
    """Retrain using only features whose window does not overlap the label's.

    This answers the sharpest objection to the project. The label covers the last
    12 months and so do many features, so a burst of enquiries could be the
    borrower reacting to their own delinquency rather than a signal preceding it.
    The snapshot carries no as-of dates, so the ordering is unidentifiable — but
    dropping every same-window feature gives a model that provably cannot be
    reading post-outcome behaviour. The gap between the two is the size of the
    doubt, stated as a number instead of a caveat.
    """
    keep = [c for c in X.columns if c not in set(C.WINDOW_OVERLAP_FEATURES)]
    dropped = [c for c in X.columns if c in set(C.WINDOW_OVERLAP_FEATURES)]
    Xn = X[keep]

    pos_weight = (y.loc[idx_train] == 0).sum() / max(1, (y.loc[idx_train] == 1).sum())
    model = xgb.XGBClassifier(**BASE_PARAMS, scale_pos_weight=pos_weight)
    model.fit(Xn.loc[idx_train], y.loc[idx_train], verbose=False)
    probs = model.predict_proba(Xn.loc[idx_test])[:, 1]
    yt = y.loc[idx_test].to_numpy()
    return {
        "n_features": len(keep),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "auc": float(roc_auc_score(yt, probs)),
        "ap": float(average_precision_score(yt, probs)),
        "precision_top10": precision_at_k(yt, probs, 0.10),
    }


def stratified_metrics(
    raw: pd.DataFrame, y: pd.Series, probs: np.ndarray, idx_test
) -> list[dict]:
    """Performance within comparable-exposure strata.

    An overall AUC above every within-stratum AUC is the signature of a
    confound: the model is partly ranking high-exposure borrowers above
    low-exposure ones rather than discriminating among peers.
    """
    band = labels.exposure_band(raw.loc[idx_test])
    yt = y.loc[idx_test].to_numpy()
    rows = []
    for name in C.EXPOSURE_BAND_LABELS:
        mask = (band == name).to_numpy()
        n, pos = int(mask.sum()), int(yt[mask].sum())
        if n < 100 or pos < 20 or pos == n:
            continue
        rows.append(
            {
                "band": name,
                "accounts": n,
                "stressed": pos,
                "stress_rate": float(yt[mask].mean()),
                "auc": float(roc_auc_score(yt[mask], probs[mask])),
                "precision_top10": precision_at_k(yt[mask], probs[mask], 0.10),
            }
        )
    return rows


def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Stratified 5-fold CV. Fixed n_estimators — no early stopping, so no fold
    gets to peek at its own validation set for the stopping decision."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    oof = np.zeros(len(y))

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        pos_weight = (y.iloc[tr] == 0).sum() / max(1, (y.iloc[tr] == 1).sum())
        model = xgb.XGBClassifier(**BASE_PARAMS, scale_pos_weight=pos_weight)
        model.fit(X.iloc[tr], y.iloc[tr], verbose=False)

        probs = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = probs
        yv = y.iloc[va].to_numpy()
        rows.append(
            {
                "fold": fold,
                "auc": roc_auc_score(yv, probs),
                "ap": average_precision_score(yv, probs),
                "precision_top5": precision_at_k(yv, probs, 0.05),
                "precision_top10": precision_at_k(yv, probs, 0.10),
                "recall_top10": recall_at_k(yv, probs, 0.10),
            }
        )
        print(
            f"  fold {fold}: AUC={rows[-1]['auc']:.4f}  AP={rows[-1]['ap']:.4f}  "
            f"P@10%={rows[-1]['precision_top10']:.1%}"
        )

    folds = pd.DataFrame(rows)
    summary = {
        f"{m}_{stat}": float(getattr(folds[m], stat)())
        for m in ["auc", "ap", "precision_top5", "precision_top10", "recall_top10"]
        for stat in ["mean", "std"]
    }
    return {"folds": rows, "summary": summary, "oof": oof}


def main() -> None:
    print("Loading merged source data...")
    raw = pd.read_csv(C.MERGED_DATA)

    y = labels.build_stress_label(raw)
    print()
    print(labels.label_report(raw, y))
    print()

    # Hold out a test set before fitting anything, including feature params.
    idx_train, idx_test = train_test_split(
        raw.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Feature statistics are learned on the training rows only, so the test set
    # contributes nothing to the medians, clips or normalisers.
    print("Fitting feature params on the training split only...")
    params = features.fit_params(raw.loc[idx_train])
    features.save_params(params, C.FEATURE_PARAMS)

    X = features.transform(raw, params)
    features.assert_no_leakage(list(X.columns))
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Leakage check passed ({len(C.LABEL_DEFINING_COLS)} columns excluded by rule)")
    print()

    X_train, y_train = X.loc[idx_train], y.loc[idx_train]
    X_test, y_test = X.loc[idx_test], y.loc[idx_test]

    print(f"Cross-validating on the training split ({N_FOLDS}-fold stratified)...")
    cv = cross_validate(X_train, y_train)
    s = cv["summary"]
    print()
    print("  CV summary (mean +/- sd across folds):")
    print(f"    ROC-AUC          : {s['auc_mean']:.4f} +/- {s['auc_std']:.4f}")
    print(f"    Average precision: {s['ap_mean']:.4f} +/- {s['ap_std']:.4f}")
    print(f"    Precision @ top 5%  : {s['precision_top5_mean']:.1%} +/- {s['precision_top5_std']:.1%}")
    print(f"    Precision @ top 10% : {s['precision_top10_mean']:.1%} +/- {s['precision_top10_std']:.1%}")
    print(f"    Recall @ top 10%    : {s['recall_top10_mean']:.1%} +/- {s['recall_top10_std']:.1%}")
    print()

    # Final model: fit on the full training split, evaluated once on the
    # untouched test set.
    print("Fitting final model on the full training split...")
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    model = xgb.XGBClassifier(**BASE_PARAMS, scale_pos_weight=pos_weight)
    model.fit(X_train, y_train, verbose=False)

    test_probs = model.predict_proba(X_test)[:, 1]
    yt = y_test.to_numpy()
    base_rate = float(yt.mean())
    test_metrics = {
        "auc": float(roc_auc_score(yt, test_probs)),
        "ap": float(average_precision_score(yt, test_probs)),
        "base_rate": base_rate,
        "precision_top5": precision_at_k(yt, test_probs, 0.05),
        "precision_top10": precision_at_k(yt, test_probs, 0.10),
        "precision_top20": precision_at_k(yt, test_probs, 0.20),
        "recall_top10": recall_at_k(yt, test_probs, 0.10),
        "recall_top20": recall_at_k(yt, test_probs, 0.20),
    }

    print()
    print("=== HELD-OUT TEST SET ===")
    print(f"  n = {len(yt):,}   base stress rate = {base_rate:.2%}")
    print(f"  ROC-AUC           : {test_metrics['auc']:.4f}")
    print(f"  Average precision : {test_metrics['ap']:.4f}  (baseline {base_rate:.4f})")
    print(
        f"  Precision @ top 5% : {test_metrics['precision_top5']:.1%}  "
        f"(lift {test_metrics['precision_top5']/base_rate:.2f}x)"
    )
    print(
        f"  Precision @ top 10%: {test_metrics['precision_top10']:.1%}  "
        f"(lift {test_metrics['precision_top10']/base_rate:.2f}x)"
    )
    print(f"  Recall @ top 10%   : {test_metrics['recall_top10']:.1%}")
    print()

    deciles = decile_table(yt, test_probs)
    print("Decile table (1 = highest risk):")
    print(deciles.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    # ── the benchmark that actually matters ─────────────────────────
    print("Training exposure-only baseline (account counts and file ages only)...")
    exposure = exposure_baseline(raw, y, idx_train, idx_test)
    margin = test_metrics["auc"] - exposure["auc"]
    print(f"  exposure-only AUC : {exposure['auc']:.4f}   ({len(exposure['features'])} features)")
    print(f"  full model AUC    : {test_metrics['auc']:.4f}   ({X.shape[1]} features)")
    print(f"  margin over exposure: {margin:+.4f}")
    print()

    # ── direction of the arrow ──────────────────────────────────────
    print("Retraining without same-window features (direction-of-arrow check)...")
    no_overlap = no_overlap_variant(X, y, idx_train, idx_test)
    print(
        f"  full model                       : AUC {test_metrics['auc']:.4f}  "
        f"({X.shape[1]} features)"
    )
    print(
        f"  no same-window features          : AUC {no_overlap['auc']:.4f}  "
        f"({no_overlap['n_features']} features, {no_overlap['n_dropped']} dropped)"
    )
    print(f"  exposure-only floor              : AUC {exposure['auc']:.4f}")
    print(
        f"  cost of removing the doubt       : {no_overlap['auc'] - test_metrics['auc']:+.4f}   "
        f"still above exposure floor by {no_overlap['auc'] - exposure['auc']:+.4f}"
    )
    print()

    strata = stratified_metrics(raw, y, test_probs, idx_test)
    print("Within comparable-exposure strata (trade lines held):")
    print(pd.DataFrame(strata).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    within = float(np.mean([s["auc"] for s in strata])) if strata else float("nan")
    print(f"  mean within-stratum AUC: {within:.4f}   vs overall {test_metrics['auc']:.4f}")
    print()

    importance = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    print("Top 15 features by gain:")
    for name, val in importance.head(15).items():
        print(f"  {val:.4f}  {name}")

    # ── persist artifacts ───────────────────────────────────────────
    joblib.dump(
        {"model": model, "feature_names": list(X_train.columns)}, C.MODEL_PATH
    )

    metrics = {
        "label_definition": (
            "Deterioration within a fixed 12-month window: any delinquency, or "
            "any NPA classification, in the last 12 months"
        ),
        "exposure_baseline": exposure,
        "exposure_margin": margin,
        "no_overlap_variant": no_overlap,
        "strata": strata,
        "mean_within_stratum_auc": within,
        "n_rows": len(raw),
        "n_features": int(X.shape[1]),
        "stress_rate": float(y.mean()),
        "excluded_for_leakage": C.LABEL_DEFINING_COLS,
        "cv": {"n_folds": N_FOLDS, "folds": cv["folds"], "summary": s},
        "test": test_metrics,
        "deciles": deciles.to_dict(orient="records"),
        "feature_importance": importance.round(6).to_dict(),
        "gold_loan_cohort": gold_loan_cohort(raw, y),
        "gold_loan_within_exposure": gold_loan_within_exposure(raw, y),
    }
    with open(C.METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Scored portfolio for the dashboard: out-of-sample rows only, with the
    # borrower ID carried through so a firm can actually be drilled into.
    scored = pd.DataFrame(
        {
            C.ID_COL: raw.loc[idx_test, C.ID_COL].to_numpy(),
            "stress_probability": test_probs,
            "actual_stress": yt,
        }
    )
    scored["risk_tier"] = pd.cut(
        scored["stress_probability"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Medium", "High", "Critical"],
    )
    scored = scored.sort_values("stress_probability", ascending=False)
    scored.to_csv(C.SCORED_PORTFOLIO, index=False)

    print()
    print("Actual stress rate by risk tier:")
    tiers = scored.groupby("risk_tier", observed=True)["actual_stress"].agg(
        ["count", "mean"]
    )
    print(tiers.to_string(float_format=lambda v: f"{v:.3f}"))

    print()
    print(f"Saved model         -> {C.MODEL_PATH}")
    print(f"Saved feature params-> {C.FEATURE_PARAMS}")
    print(f"Saved metrics       -> {C.METRICS_PATH}")
    print(f"Saved scored portfolio -> {C.SCORED_PORTFOLIO}")


if __name__ == "__main__":
    main()