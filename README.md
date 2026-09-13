# MSME Credit Stress Early-Warning

Ranks Indian MSME-proprietor borrowers by whether their credit deteriorated
**during the last 12 months**, using credit-seeking behaviour, balance-sheet
utilisation and portfolio structure — deliberately excluding the delinquency
record that defines the outcome.

This is a **detection and triage** model, not a forecast. The label looks
backwards over a fixed window, and the data is a single cross-sectional snapshot
with no as-of dates, so no forward-dated or lead-time claim is made. See
[Direction of the arrow](#direction-of-the-arrow) — it is the sharpest objection
to this project and it is answered with numbers, not prose.

**Held-out test performance** (10,268 borrowers, 16.76% base stress rate):

| Metric | 5-fold CV | Held-out test |
|---|---|---|
| ROC-AUC | 0.7655 ± 0.0045 | 0.7744 |
| Average precision | 0.3944 ± 0.0150 | 0.4058 |
| Precision @ top 5% | 52.7% ± 2.9% | 55.8% |
| Precision @ top 10% | 46.0% ± 1.5% | 47.5% |
| Recall @ top 10% | 27.4% ± 0.9% | 28.3% |

Top-decile lift is **2.83x** the base rate, and the risk tiers escalate
cleanly: 2.9% → 13.0% → 28.1% → 50.8% actual stress rate.

**The number that matters more than the AUC.** The label counts deterioration
events, so borrowers holding more accounts have mechanically more chances to
register one — a model can score well by little more than counting trade lines.
An exposure-only baseline given nothing but account counts and credit-file ages
reaches **0.7031**. The full model reaches 0.7744, a margin of
**+0.0713**, and stays between 0.736 and 0.768 *within* every
exposure stratum. Both figures are reported on the model card.

Average precision is the headline metric rather than ROC-AUC, because at a
16.8% event rate AUC is dominated by the majority class.
Precision@top-k reflects the actual workflow: a credit team works a ranked queue
under fixed review capacity.

---

## Quickstart

```bash
pip install -r requirements.txt
python -m src.train      # trains, evaluates, writes artifacts to models/
streamlit run app.py     # dashboard
```

## The label

Stress is **deterioration inside a fixed 12-month window**: within the last 12
months, any trade line went delinquent, or any was classified sub-standard,
doubtful or loss. Prevalence is 16.76% (8,603 of 51,336 borrowers).

Under RBI's Income Recognition and Asset Classification norms an account is
tracked as SMA-0 (1–30 days overdue), SMA-1 (31–60), SMA-2 (61–90) and NPA (90+).
The SMA framework is a rolling current-state view — what supervisors and lenders
monitor month to month — so a fixed recent window is the right shape for the label.

This label is the third attempt, and the first two are documented in
`reports/technical_report.md` because the diagnostics that killed them are the
substance of the project:

1. **Approval tier** (`Approved_Flag != "P2"`) — merged the best and worst tiers
   into one positive class. AUC 0.825, and close to meaningless: the tiers barely
   separate genuine stress (P2 8.3% vs P1 13.0%, P4 17.9%).
2. **Lifetime delinquency** ("ever 60+ days past due") — not exposure-comparable.
   Prevalence ran 4.8% at 1–2 trade lines against 26.4% at 11+, a 5.45x gradient,
   and an account-counting baseline alone reached AUC 0.782 against the full
   model's 0.8135. The model was mostly counting accounts.

The fixed window cut the exposure gradient to **2.39x** and the exposure-only
baseline to **0.7031**, raising the model's margin over it from +0.025 to
**+0.0713**. Headline AUC fell by 0.039; the model got more defensible.

## Direction of the arrow

The sharpest objection to this project, stated plainly because it does not go away.

The label covers the last 12 months. So do 14 of the 57 features — `enq_L3m`,
`Total_TL_opened_L6M`, `closure_rate` and others measure activity inside the same
window the outcome is measured in. A borrower who went delinquent in month 3 may
have made a burst of enquiries in months 4–12 *because* they were already in
distress, making the feature a consequence of the outcome rather than a signal
preceding it. The data is one cross-sectional snapshot with no as-of dates, so the
ordering cannot be recovered: **the direction is formally unidentifiable.**

What can be bounded is how much rides on it. Retraining with every same-window
feature removed:

| Model | Features | ROC-AUC |
|---|---|---|
| Exposure only (floor) | 6 | 0.7031 |
| **No same-window features** | **43** | **0.7340** |
| Full model | 57 | 0.7744 |

Of the +0.0713 the full model gains over the exposure floor, **57% depends on
same-window features** and could be post-outcome behaviour; **43% (+0.0309)** comes
from lifetime stocks, point-in-time utilisation and profile, which cannot be.

The defensible claim: a model using only information that cannot be a reaction to
the outcome still ranks deterioration at AUC 0.734 against a 0.703 floor. The full
0.774 is **concurrent detection**, not anticipation. Settling it needs panel data —
features as of month 0, outcomes over months 1–12 — which is a data-acquisition
problem, not a modelling one.

## Leakage controls

The label is built from delinquency and asset-classification fields, so all 29
columns that define or mechanically imply it are excluded from the feature set by
rule — enumerated in `src/config.py` and asserted in training via
`features.assert_no_leakage`, so a leak raises rather than showing up as a
suspiciously good AUC. The excluded families are:

- asset classification counts (`num_std`/`num_sub`/`num_dbt`/`num_lss`, all windows)
- delinquency and DPD history (counts, levels, recency)
- payment behaviour (`Tot_Missed_Pmnt`, `time_since_recent_payment`)
- bureau/underwriting outputs (`Credit_Score`, `Approved_Flag`)

What remains is credit-seeking behaviour, utilisation, portfolio structure and
borrower profile — 57 features. The substantive claim of the project, stated at the
strength the data supports: **payment distress is detectable without any payment
history.** Not "predictable" — that word would smuggle back the forward-dated claim
the direction-of-the-arrow section rules out.

## Train/serve parity

Feature statistics (medians, clip bounds, normalisers) are fitted on the training
split only and persisted to `models/feature_params.json`. Training and the
dashboard both call `src.features.transform` with those saved parameters, so a
borrower scores identically in both. Verified: maximum score difference between
the two paths is 2.2e-08 (float32 rounding).

## Missing data

The source encodes "not reported" as `-99999`/`-99998`. Rates are severe in
places — `CC_utilization` is 92.8% unreported and `PL_utilization` 86.6% — so
median-filling alone would turn those columns into near-constants. Each is filled
*and* paired with a binary missingness indicator, on the reasoning that an absent
credit-card utilisation record usually means a thin file rather than an average
one. That reasoning is only partly borne out: on its own the flag carries almost
no marginal signal (0.99x lift), so its model importance is interaction-driven.

## Layout

```
src/config.py      column groups and the leakage boundary
src/labels.py      label construction and its diagnostic report
src/features.py    fit/transform feature pipeline, persisted params
src/train.py       CV + held-out evaluation, writes all artifacts
src/theme.py       design tokens and chart chrome (CVD-validated)
app.py             Streamlit dashboard (5 pages)
notebooks/         exploratory work: EDA, survival, causal, LSTM, conformal, MLflow
reports/           technical report
```

The dashboard has five pages: an overview leading with the exposure benchmark,
portfolio triage with an interactive review-capacity slider, per-borrower
drill-down with live SHAP and what-if rescoring, batch CSV scoring, and a model
card carrying the metrics, leakage controls and limitations.

## Limitations

- **No lead-time claim.** The source is a cross-sectional snapshot with no default
  timestamps, so the model predicts *concurrent* stress classification, not a
  forward-dated event. An earlier framing of this project claimed a 90-day
  horizon; nothing in the data supports that and the claim has been withdrawn.
- **Retail proxy for firm risk.** Trained on retail credit-bureau records standing
  in for MSME proprietor risk. Not validated against firm-level financials, GST
  filing sequences, or TReDS payment data.
- **Utilisation is weak.** See missing data above; treat those inputs as low
  confidence.
- **Commodity volatility is cross-sectional.** The Agmarknet extract covers a
  limited window, so volatility could not be computed as a time series. It is not
  used as a model input.
- **Exposure is reduced, not eliminated.** The fixed window cut the prevalence
  gradient across exposure bands to 2.39x, but borrowers with more trade lines
  still show higher stress rates. Some of that is likely real risk, so it is
  reported and controlled for in evaluation rather than adjusted away.
- **Event count.** Top-decile metrics rest on 487 stressed accounts in the test
  set. Fold-level variation is reported rather than a single split.
- **A gold-loan finding was withdrawn.** Stress rate appeared to rise 5.0x with
  gold-loan count. Holding trade-line count fixed, the gradient reverses —
  Simpson's paradox. Documented in the technical report and in the dashboard.
- **Causal and survival analyses withdrawn.** Both were computed against the
  discarded labels; see the withdrawn-claims table in the technical report.
- **Exploratory notebooks are not production paths.** The survival analysis
  (notebook 06) simulates durations from variables that are then used as Cox
  predictors, which makes its C-index circular; it is retained as exploratory work
  only and no survival output feeds the model or dashboard.
