"""
Automation Bias Survey Analysis (Control vs Intervention w/ AI Confidence UI)

What this script does:
- Cleans + validates the dataset
- Builds a unified participant identifier:
    participant_id = student_id if present else email
- Creates automation-bias outcome variables:
    * follow_ai
    * user_correct
    * commission_error (followed AI when AI was wrong)  <-- primary
    * omission_error   (didn't follow AI when AI was right)
- Produces:
    * Summary tables
    * Repeated-measures stats via GEE (clustered by participant_id)
    * Bootstrap CIs + permutation tests at participant level
    * Multiple plots saved to ./outputs/

Run:
    pip install pandas numpy matplotlib scipy statsmodels
    python analyze_automation_bias.py --csv "Final responses.csv"

Optional:
    python analyze_automation_bias.py --csv "Final responses.csv" --outdir outputs --alpha 0.05
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency

# statsmodels is optional; we fall back gracefully if missing
try:
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# -----------------------------
# Helpers
# -----------------------------

@dataclass
class ColumnMap:
    student_id: str = "student_id"
    email: str = "email"  # CHANGE if your column is named differently
    group: str = "group"
    question_id: str = "question_id"
    ai_suggested_option: str = "ai_suggested_option"
    ai_was_correct: str = "ai_was_correct"
    user_final_choice: str = "user_final_choice"
    correct_answer: str = "correct_answer"
    time_taken_ms: str = "time_taken_ms"
    ai_confidence_shown: str = "ai_confidence_shown"
    submitted_at: Optional[str] = None  # if you later add a timestamp column


def _normalize_group(x: str) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip().lower()
    if "control" in s:
        return "CONTROL"
    if "inter" in s or "treat" in s:
        return "INTERVENTION"
    return str(x).strip().upper()


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerces common bool encodings: True/False, 1/0, 'true'/'false', 'yes'/'no'."""
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    true_set = {"true", "1", "yes", "y", "t"}
    false_set = {"false", "0", "no", "n", "f"}
    out = s.map(lambda v: True if v in true_set else (False if v in false_set else np.nan))
    return out.astype("boolean")


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054
    if alpha != 0.05:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)

    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def safe_to_str(series: pd.Series) -> pd.Series:
    """Stringify, but this WILL convert NaN->'nan'. Use only for non-ID fields."""
    return series.astype(str).str.strip()


def _clean_id_series(series: pd.Series) -> pd.Series:
    """Clean ID-like columns without turning NaN into 'nan' strings."""
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA, "NaN": pd.NA})
    return s


# -----------------------------
# Core analysis
# -----------------------------

def load_and_prepare(csv_path: str, cols: ColumnMap) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all").copy()

    # Required non-ID columns
    required_core = [
        cols.group, cols.question_id,
        cols.ai_suggested_option, cols.ai_was_correct,
        cols.user_final_choice, cols.correct_answer
    ]
    missing_core = [c for c in required_core if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing required columns: {missing_core}\nFound columns: {list(df.columns)}")

    # At least one identifier column should exist (student_id OR email)
    has_student = cols.student_id in df.columns
    has_email = cols.email in df.columns

    if not has_student and not has_email:
        raise ValueError(
            f"Need at least one identifier column: '{cols.student_id}' or '{cols.email}'.\n"
            f"Found columns: {list(df.columns)}"
        )

    # Normalize group
    df[cols.group] = df[cols.group].apply(_normalize_group)

    # Coerce ai_was_correct to boolean
    df[cols.ai_was_correct] = _coerce_bool(df[cols.ai_was_correct])

    # Canonicalize non-ID strings
    df[cols.ai_suggested_option] = safe_to_str(df[cols.ai_suggested_option])
    df[cols.user_final_choice] = safe_to_str(df[cols.user_final_choice])
    df[cols.correct_answer] = safe_to_str(df[cols.correct_answer])
    df[cols.question_id] = safe_to_str(df[cols.question_id])

    # Clean ID columns WITHOUT creating "nan" strings
    if has_student:
        df[cols.student_id] = _clean_id_series(df[cols.student_id])
    else:
        df[cols.student_id] = pd.NA

    if has_email:
        df[cols.email] = _clean_id_series(df[cols.email])
    else:
        df[cols.email] = pd.NA

    # Unified participant identifier
    df["participant_id"] = df[cols.student_id]
    df.loc[df["participant_id"].isna(), "participant_id"] = df[cols.email]

    # Drop rows we can't attribute to any participant
    df = df[df["participant_id"].notna()].copy()

    # Derived outcomes
    df["follow_ai"] = df[cols.user_final_choice] == df[cols.ai_suggested_option]
    df["user_correct"] = df[cols.user_final_choice] == df[cols.correct_answer]
    df["commission_error"] = df["follow_ai"] & (~df[cols.ai_was_correct].astype(bool))
    df["omission_error"] = (~df["follow_ai"]) & (df[cols.ai_was_correct].astype(bool))

    # Time normalization (optional)
    if cols.time_taken_ms in df.columns:
        df[cols.time_taken_ms] = pd.to_numeric(df[cols.time_taken_ms], errors="coerce")
        df["time_s"] = df[cols.time_taken_ms] / 1000.0
    else:
        df["time_s"] = np.nan

    # Confidence normalization (optional)
    if cols.ai_confidence_shown in df.columns:
        df[cols.ai_confidence_shown] = pd.to_numeric(df[cols.ai_confidence_shown], errors="coerce")
    else:
        df[cols.ai_confidence_shown] = np.nan

    return df


def summarize(df: pd.DataFrame, cols: ColumnMap, alpha: float = 0.05) -> pd.DataFrame:
    g = df.groupby(cols.group)

    def _agg(x: pd.DataFrame) -> pd.Series:
        n_trials = len(x)
        n_students = x["participant_id"].nunique()
        ai_correct = x[cols.ai_was_correct].mean(skipna=True)
        follow = x["follow_ai"].mean()
        acc = x["user_correct"].mean()
        comm = x["commission_error"].mean()
        omis = x["omission_error"].mean()
        med_time = x["time_s"].median(skipna=True)

        wrong = x[x[cols.ai_was_correct] == False]
        k = int(wrong["follow_ai"].sum())
        n = int(len(wrong))
        ci_lo, ci_hi = wilson_ci(k, n, alpha=alpha)

        return pd.Series({
            "n_trials": n_trials,
            "n_students": n_students,
            "ai_correct_rate": ai_correct,
            "follow_rate": follow,
            "accuracy": acc,
            "commission_rate_overall": comm,
            "omission_rate_overall": omis,
            "median_time_s": med_time,
            "commission_when_ai_wrong": (k / n) if n else np.nan,
            "commission_ai_wrong_n": n,
            "commission_ai_wrong_ci_low": ci_lo,
            "commission_ai_wrong_ci_high": ci_hi,
        })

    # Fix pandas warning with include_groups=False
    out = g.apply(_agg, include_groups=False).sort_index()
    return out


def conditional_summary(df: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    rows = []
    for grp, d in df.groupby(cols.group):
        wrong = d[d[cols.ai_was_correct] == False]
        corr = d[d[cols.ai_was_correct] == True]
        rows.append({
            "group": grp,
            "n_ai_wrong": len(wrong),
            "follow_when_ai_wrong": wrong["follow_ai"].mean() if len(wrong) else np.nan,
            "acc_when_ai_wrong": wrong["user_correct"].mean() if len(wrong) else np.nan,
            "n_ai_correct": len(corr),
            "follow_when_ai_correct": corr["follow_ai"].mean() if len(corr) else np.nan,
            "acc_when_ai_correct": corr["user_correct"].mean() if len(corr) else np.nan,
        })
    return pd.DataFrame(rows).set_index("group").sort_index()


def gee_logit(df: pd.DataFrame, cols: ColumnMap, outcome: str, predictors: list[str]) -> Optional[object]:
    if not HAS_STATSMODELS:
        return None

    d = df.copy()
    d = d[d[cols.group].isin(["CONTROL", "INTERVENTION"])].copy()
    d = d.reset_index(drop=True)

    d["group_bin"] = (d[cols.group] == "INTERVENTION").astype(int)

    y = pd.to_numeric(d[outcome], errors="coerce")

    X = pd.DataFrame({"Intercept": np.ones(len(d), dtype=float)}, index=d.index)

    for p in predictors:
        if p == "group_bin":
            X[p] = d["group_bin"].astype(float)

        elif p == "question_fe":
            q_series = d[cols.question_id].astype(str)
            q = pd.get_dummies(q_series, prefix="q", drop_first=True, dtype=float)
            X = pd.concat([X, q], axis=1)

        else:
            X[p] = pd.to_numeric(d[p], errors="coerce")

    groups = d["participant_id"].astype("string")

    valid = y.notna() & groups.notna()
    valid &= ~X.isna().any(axis=1)

    y = y.loc[valid].astype(int)
    X = X.loc[valid]
    groups = groups.loc[valid]

    X = X.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"GEE design matrix has non-numeric values after coercion. Bad columns: {bad_cols}")

    X = X.astype(float)

    model = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable())
    return model.fit()


def quick_independence_tests(df: pd.DataFrame, cols: ColumnMap, outcome: str) -> Dict[str, float]:
    """Quick tests ignoring repeated measures (sanity only)."""
    d = df[df[cols.group].isin(["CONTROL", "INTERVENTION"])].copy()
    tab = pd.crosstab(d[cols.group], d[outcome].astype(int))
    if tab.shape != (2, 2):
        return {"fisher_p": np.nan, "chi2_p": np.nan}

    _, fisher_p = fisher_exact(tab.values)
    chi2_p = chi2_contingency(tab.values, correction=False)[1]
    return {"fisher_p": fisher_p, "chi2_p": chi2_p}


def per_student_rates(df: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    """Aggregates to per-participant rates."""
    d = df[df[cols.group].isin(["CONTROL", "INTERVENTION"])].copy()

    wrong = d[d[cols.ai_was_correct] == False].copy()
    comm = wrong.groupby(["participant_id", cols.group])["follow_ai"].agg(["mean", "count"]).reset_index()
    comm = comm.rename(columns={"mean": "commission_rate_ai_wrong", "count": "n_ai_wrong"})

    acc = d.groupby(["participant_id", cols.group])["user_correct"].mean().reset_index(name="accuracy")
    follow = d.groupby(["participant_id", cols.group])["follow_ai"].mean().reset_index(name="follow_rate")
    time = d.groupby(["participant_id", cols.group])["time_s"].median().reset_index(name="median_time_s")

    out = acc.merge(follow, on=["participant_id", cols.group], how="left") \
             .merge(comm, on=["participant_id", cols.group], how="left") \
             .merge(time, on=["participant_id", cols.group], how="left")
    return out


def compare_per_student(per: pd.DataFrame, metric: str) -> Dict[str, float]:
    """Mann–Whitney comparison between groups on per-participant metric."""
    a = per[per["group"] == "CONTROL"][metric].dropna().values
    b = per[per["group"] == "INTERVENTION"][metric].dropna().values
    if len(a) < 3 or len(b) < 3:
        return {"mw_p": np.nan, "control_n": len(a), "intervention_n": len(b)}
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    return {"mw_p": p, "control_n": len(a), "intervention_n": len(b)}


# -----------------------------
# Added: bootstrap + permutation + balance
# -----------------------------

def _student_cluster_bootstrap(
    per: pd.DataFrame,
    metric: str,
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    c = per[per["group"] == "CONTROL"][["participant_id", metric]].dropna()
    t = per[per["group"] == "INTERVENTION"][["participant_id", metric]].dropna()

    c_vals = c[metric].to_numpy()
    t_vals = t[metric].to_numpy()

    if len(c_vals) < 5 or len(t_vals) < 5:
        return {"mean_control": np.nan, "mean_treat": np.nan, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        c_samp = rng.choice(c_vals, size=len(c_vals), replace=True)
        t_samp = rng.choice(t_vals, size=len(t_vals), replace=True)
        diffs[i] = np.mean(t_samp) - np.mean(c_samp)

    diffs.sort()
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))

    return {
        "mean_control": float(np.mean(c_vals)),
        "mean_treat": float(np.mean(t_vals)),
        "diff": float(np.mean(t_vals) - np.mean(c_vals)),
        "ci_low": lo,
        "ci_high": hi,
        "n_control": int(len(c_vals)),
        "n_treat": int(len(t_vals)),
        "n_boot": int(n_boot),
    }


def print_bootstrap_effects(per: pd.DataFrame, outdir: str) -> pd.DataFrame:
    rows = []
    for metric in ["commission_rate_ai_wrong", "accuracy", "follow_rate", "median_time_s"]:
        res = _student_cluster_bootstrap(per, metric=metric, n_boot=5000, seed=42)
        rows.append({"metric": metric, **res})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "bootstrap_effects.csv"), index=False)

    print("\n=== Cluster bootstrap (by participant) risk-difference CIs ===")
    for _, r in out.iterrows():
        m = r["metric"]
        diff = r["diff"]
        lo = r["ci_low"]
        hi = r["ci_high"]
        mc = r["mean_control"]
        mt = r["mean_treat"]
        if m != "median_time_s":
            print(f"- {m}: CONTROL={mc:.3f}  INTERVENTION={mt:.3f}  RD={diff:+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]")
        else:
            print(f"- {m}: CONTROL={mc:.2f}s INTERVENTION={mt:.2f}s  Diff={diff:+.2f}s  95%CI=[{lo:+.2f},{hi:+.2f}]")
    return out


def permutation_test_student_level(
    per: pd.DataFrame,
    metric: str,
    n_perm: int = 20000,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    d = per[["participant_id", "group", metric]].dropna().copy()
    d = d.drop_duplicates(subset=["participant_id"])

    y = d[metric].to_numpy()
    g = (d["group"] == "INTERVENTION").to_numpy()

    if g.sum() == 0 or (~g).sum() == 0:
        return {"p_perm": np.nan, "obs_diff": np.nan, "n_perm": n_perm}

    obs = float(y[g].mean() - y[~g].mean())

    n_t = int(g.sum())
    diffs = np.empty(n_perm, dtype=float)
    idx = np.arange(len(y))

    for i in range(n_perm):
        rng.shuffle(idx)
        g_perm = np.zeros(len(y), dtype=bool)
        g_perm[idx[:n_t]] = True
        diffs[i] = y[g_perm].mean() - y[~g_perm].mean()

    p = float((np.abs(diffs) >= abs(obs)).mean())
    return {"p_perm": p, "obs_diff": obs, "n_perm": n_perm, "n_students": int(len(y))}


def student_balance_table(df: pd.DataFrame, cols: ColumnMap, outdir: str) -> pd.DataFrame:
    d = df[df[cols.group].isin(["CONTROL", "INTERVENTION"])].copy()

    per_student = d.groupby(["participant_id", cols.group]).agg(
        n_trials=("follow_ai", "size"),
        median_time_s=("time_s", "median"),
    ).reset_index()

    tab = per_student.groupby(cols.group).agg(
        n_students=("participant_id", "nunique"),
        trials_per_student_mean=("n_trials", "mean"),
        trials_per_student_min=("n_trials", "min"),
        trials_per_student_max=("n_trials", "max"),
        median_time_s_median=("median_time_s", "median"),
    )

    tab.to_csv(os.path.join(outdir, "student_balance_table.csv"))
    print("\n=== Participant-level balance / quality (descriptive) ===")
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
    return tab


# -----------------------------
# Visualizations
# -----------------------------

def plot_group_bars_with_ci(df: pd.DataFrame, cols: ColumnMap, outdir: str, alpha: float = 0.05) -> None:
    ensure_outdir(outdir)

    groups = ["CONTROL", "INTERVENTION"]
    metrics = []

    for g in groups:
        d = df[(df[cols.group] == g) & (df[cols.ai_was_correct] == False)]
        k = int(d["follow_ai"].sum())
        n = int(len(d))
        lo, hi = wilson_ci(k, n, alpha=alpha)
        metrics.append(("Commission (AI wrong)", g, (k / n) if n else np.nan, lo, hi))

    for g in groups:
        d = df[(df[cols.group] == g) & (df[cols.ai_was_correct] == True)]
        k = int((~d["follow_ai"]).sum())
        n = int(len(d))
        lo, hi = wilson_ci(k, n, alpha=alpha)
        metrics.append(("Omission (AI correct)", g, (k / n) if n else np.nan, lo, hi))

    for g in groups:
        d = df[df[cols.group] == g]
        k = int(d["user_correct"].sum())
        n = int(len(d))
        lo, hi = wilson_ci(k, n, alpha=alpha)
        metrics.append(("Accuracy (overall)", g, (k / n) if n else np.nan, lo, hi))

    for g in groups:
        d = df[df[cols.group] == g]
        k = int(d["follow_ai"].sum())
        n = int(len(d))
        lo, hi = wilson_ci(k, n, alpha=alpha)
        metrics.append(("Follow AI (overall)", g, (k / n) if n else np.nan, lo, hi))

    mdf = pd.DataFrame(metrics, columns=["metric", "group", "rate", "ci_low", "ci_high"])

    for metric_name in mdf["metric"].unique():
        sub = mdf[mdf["metric"] == metric_name].copy()
        sub = sub.set_index("group").loc[groups].reset_index()

        x = np.arange(len(groups))
        y = sub["rate"].values
        yerr = np.vstack([y - sub["ci_low"].values, sub["ci_high"].values - y])

        plt.figure()
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=6)
        plt.xticks(x, groups)
        plt.ylim(0, 1)
        plt.title(metric_name)
        plt.ylabel("Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bar_{metric_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"), dpi=160)
        plt.close()


def plot_question_level(df: pd.DataFrame, cols: ColumnMap, outdir: str) -> None:
    ensure_outdir(outdir)
    d = df[df[cols.group].isin(["CONTROL", "INTERVENTION"])].copy()

    q_follow = d.groupby([cols.question_id, cols.group])["follow_ai"].mean().unstack(cols.group)
    q_acc = d.groupby([cols.question_id, cols.group])["user_correct"].mean().unstack(cols.group)

    qids = sorted(d[cols.question_id].unique())

    def _plot(matrix: pd.DataFrame, title: str, fname: str):
        matrix = matrix.reindex(qids)
        x = np.arange(len(matrix.index))
        width = 0.40

        plt.figure(figsize=(max(7, len(qids) * 1.1), 4))
        plt.bar(x - width / 2, matrix.get("CONTROL", pd.Series(index=qids, dtype=float)).values, width, label="CONTROL")
        plt.bar(x + width / 2, matrix.get("INTERVENTION", pd.Series(index=qids, dtype=float)).values, width, label="INTERVENTION")
        plt.xticks(x, matrix.index, rotation=0)
        plt.ylim(0, 1)
        plt.title(title)
        plt.ylabel("Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=160)
        plt.close()

    _plot(q_follow, "Follow AI by Question", "question_follow_ai.png")
    _plot(q_acc, "Accuracy by Question", "question_accuracy.png")


def plot_time_distributions(df: pd.DataFrame, cols: ColumnMap, outdir: str) -> None:
    ensure_outdir(outdir)
    d = df[df[cols.group].isin(["CONTROL", "INTERVENTION"])].copy()
    if d["time_s"].isna().all():
        return

    control = d[d[cols.group] == "CONTROL"]["time_s"].dropna().values
    treat = d[d[cols.group] == "INTERVENTION"]["time_s"].dropna().values

    plt.figure()
    # Fix matplotlib warning: use tick_labels
    plt.boxplot([control, treat], tick_labels=["CONTROL", "INTERVENTION"])
    plt.ylabel("Time (seconds)")
    plt.title("Response Time Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_boxplot.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.hist(control, bins=20, alpha=0.6, label="CONTROL")
    plt.hist(treat, bins=20, alpha=0.6, label="INTERVENTION")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Count")
    plt.title("Response Time Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_hist.png"), dpi=160)
    plt.close()


def plot_treatment_confidence_effect(df: pd.DataFrame, cols: ColumnMap, outdir: str) -> None:
    ensure_outdir(outdir)
    d = df[df[cols.group] == "INTERVENTION"].copy()
    if cols.ai_confidence_shown not in d.columns or d[cols.ai_confidence_shown].isna().all():
        return

    x = d[cols.ai_confidence_shown].values
    y = d["follow_ai"].astype(int).values
    yj = y + (np.random.rand(len(y)) - 0.5) * 0.1

    plt.figure()
    plt.scatter(x, yj)
    plt.yticks([0, 1], ["Did not follow AI", "Followed AI"])
    plt.xlabel("AI confidence shown")
    plt.title("Intervention: Confidence vs Following AI")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "treatment_confidence_scatter.png"), dpi=160)
    plt.close()


def plot_running_effect(df: pd.DataFrame, cols: ColumnMap, outdir: str, alpha: float = 0.05) -> None:
    ensure_outdir(outdir)
    d = df[df[cols.group].isin(["CONTROL", "INTERVENTION"])].copy()
    d = d[d[cols.ai_was_correct] == False].copy()

    diffs = []
    ns = []
    for i in range(20, len(d) + 1):
        pref = d.iloc[:i]

        def rate(grp: str) -> Tuple[float, int]:
            dd = pref[pref[cols.group] == grp]
            n = len(dd)
            if n == 0:
                return (np.nan, 0)
            return (dd["follow_ai"].mean(), n)

        r_t, n_t = rate("INTERVENTION")
        r_c, n_c = rate("CONTROL")
        if np.isnan(r_t) or np.isnan(r_c):
            continue

        diffs.append(r_t - r_c)
        ns.append(n_t + n_c)

    if len(diffs) < 5:
        return

    plt.figure()
    plt.plot(ns, diffs)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Total AI-wrong trials accumulated (both groups)")
    plt.ylabel("Running diff: (Intervention - Control) follow_ai rate")
    plt.title("Running Estimate of Commission-Error Difference (AI wrong only)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "running_commission_diff.png"), dpi=160)
    plt.close()


# -----------------------------
# Design checks (confounds)
# -----------------------------

def detect_confounds(df: pd.DataFrame, cols: ColumnMap) -> Dict[str, str]:
    notes = {}

    if cols.ai_confidence_shown in df.columns and not df[cols.ai_confidence_shown].isna().all():
        by_q = df.groupby(cols.question_id)[cols.ai_confidence_shown].nunique(dropna=True)
        if (by_q <= 1).all():
            notes["confidence_fixed_per_question"] = (
                "ai_confidence_shown is constant within each question_id. "
                "If question_id differs in difficulty, confidence effects are confounded with item effects."
            )

    by_q_corr = df.groupby(cols.question_id)[cols.ai_was_correct].nunique(dropna=True)
    if (by_q_corr <= 1).all():
        notes["ai_correctness_fixed_per_question"] = (
            "ai_was_correct is constant within each question_id. "
            "This means correctness is item-specific; be careful attributing effects to UI alone."
        )

    grp_counts = df[cols.group].value_counts(dropna=False).to_dict()
    notes["group_trial_counts"] = f"Trial counts by group: {grp_counts}"

    return notes


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the CSV export")
    parser.add_argument("--outdir", default="outputs", help="Directory to save plots/tables")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha level for CIs")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)

    cols = ColumnMap()  # edit here if your column names change
    df = load_and_prepare(args.csv, cols)

    # Missing-ID diagnostics
    print("\n=== Missing ID diagnostics ===")
    if cols.student_id in df.columns:
        print("Missing student_id rows (after cleaning):", int(df[cols.student_id].isna().sum()))
    if cols.email in df.columns:
        print("Missing email rows (after cleaning):", int(df[cols.email].isna().sum()))
    print("Missing participant_id rows:", int(df["participant_id"].isna().sum()))

    # Duplicate checks (by participant_id)
    print("\n=== Duplicate (participant_id, question_id) pairs ===")
    dups = (
        df.groupby(["participant_id", cols.question_id])
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    if len(dups) == 0:
        print("No duplicate participant-question pairs found.")
    else:
        print(dups.sort_values(["participant_id", cols.question_id]).to_string(index=False))

    print("\n=== Participants with >5 total trials ===")
    trial_counts = (
        df.groupby("participant_id")
        .size()
        .reset_index(name="n_trials")
        .query("n_trials > 5")
        .sort_values("n_trials", ascending=False)
    )
    if len(trial_counts) == 0:
        print("All participants have ≤5 trials.")
    else:
        print(trial_counts.to_string(index=False))

    student_balance_table(df, cols, outdir)

    n_trials = len(df)
    n_students = df["participant_id"].nunique()
    print(f"\nLoaded {n_trials} trials from {n_students} unique participants.\n")

    conf = detect_confounds(df, cols)
    print("=== Design / Confound Checks ===")
    for k, v in conf.items():
        print(f"- {k}: {v}")
    print()

    print("=== Group Summary ===")
    summ = summarize(df, cols, alpha=args.alpha)
    print(summ.to_string(float_format=lambda x: f"{x:.3f}"))
    summ.to_csv(os.path.join(outdir, "summary_by_group.csv"))

    print("\n=== Conditional (AI wrong vs correct) ===")
    cond = conditional_summary(df, cols)
    print(cond.to_string(float_format=lambda x: f"{x:.3f}"))
    cond.to_csv(os.path.join(outdir, "conditional_summary.csv"))

    per = per_student_rates(df, cols)
    per.to_csv(os.path.join(outdir, "per_student_metrics.csv"), index=False)

    print_bootstrap_effects(per.rename(columns={cols.group: "group"}), outdir)

    print("\n=== Randomization inference (participant-level permutation) ===")
    per2 = per.rename(columns={cols.group: "group"})
    for metric in ["commission_rate_ai_wrong", "accuracy", "median_time_s"]:
        res = permutation_test_student_level(per2, metric=metric, n_perm=20000, seed=123)
        print(f"- {metric}: diff={res['obs_diff']:+.3f}  p_perm={res['p_perm']:.4f}  (n={res['n_students']})")

    print("\n=== Per-participant comparisons (Mann–Whitney; robust/simple) ===")
    for metric in ["commission_rate_ai_wrong", "accuracy", "median_time_s"]:
        res = compare_per_student(per.rename(columns={cols.group: "group"}), metric)
        print(f"- {metric}: {res}")

    print("\n=== Quick (non-clustered) tests (sanity only) ===")
    wrong = df[df[cols.ai_was_correct] == False].copy()
    print("Commission error proxy (follow_ai on AI-wrong trials):", quick_independence_tests(wrong, cols, "follow_ai"))
    print("Overall accuracy:", quick_independence_tests(df, cols, "user_correct"))

    if HAS_STATSMODELS:
        print("\n=== Clustered Logistic GEE (primary stats) ===")

        wrong = df[df[cols.ai_was_correct] == False].copy()
        res1 = gee_logit(wrong, cols, outcome="follow_ai", predictors=["group_bin"])
        if res1 is not None:
            b = res1.params.get("group_bin", np.nan)
            se = res1.bse.get("group_bin", np.nan)
            OR = float(np.exp(b))
            lo = float(np.exp(b - 1.96 * se))
            hi = float(np.exp(b + 1.96 * se))
            p = float(res1.pvalues.get("group_bin", np.nan))
            print(f"Follow AI when AI wrong ~ group: OR={OR:.3f} 95%CI=[{lo:.3f},{hi:.3f}] p={p:.4f}")

        res2 = gee_logit(df, cols, outcome="user_correct", predictors=["group_bin"])
        if res2 is not None:
            b = res2.params.get("group_bin", np.nan)
            se = res2.bse.get("group_bin", np.nan)
            OR = float(np.exp(b))
            lo = float(np.exp(b - 1.96 * se))
            hi = float(np.exp(b + 1.96 * se))
            p = float(res2.pvalues.get("group_bin", np.nan))
            print(f"Accuracy ~ group: OR={OR:.3f} 95%CI=[{lo:.3f},{hi:.3f}] p={p:.4f}")

        res2_fe = gee_logit(df, cols, outcome="user_correct", predictors=["group_bin", "question_fe"])
        if res2_fe is not None:
            b = res2_fe.params.get("group_bin", np.nan)
            se = res2_fe.bse.get("group_bin", np.nan)
            OR = float(np.exp(b))
            lo = float(np.exp(b - 1.96 * se))
            hi = float(np.exp(b + 1.96 * se))
            p = float(res2_fe.pvalues.get("group_bin", np.nan))
            print(f"(With question FE) Accuracy ~ group: OR={OR:.3f} 95%CI=[{lo:.3f},{hi:.3f}] p={p:.4f}")

        res3 = gee_logit(wrong, cols, outcome="follow_ai", predictors=["group_bin", "question_fe"])
        if res3 is not None:
            b = res3.params.get("group_bin", np.nan)
            se = res3.bse.get("group_bin", np.nan)
            OR = float(np.exp(b))
            lo = float(np.exp(b - 1.96 * se))
            hi = float(np.exp(b + 1.96 * se))
            p = float(res3.pvalues.get("group_bin", np.nan))
            print(f"(With question FE) Follow AI when AI wrong ~ group: OR={OR:.3f} 95%CI=[{lo:.3f},{hi:.3f}] p={p:.4f}")

    else:
        print("\n(statsmodels not installed) Skipping GEE models.")
        print("Install with: pip install statsmodels")

    print("\nGenerating plots in:", outdir)
    plot_group_bars_with_ci(df, cols, outdir, alpha=args.alpha)
    plot_question_level(df, cols, outdir)
    plot_time_distributions(df, cols, outdir)
    plot_treatment_confidence_effect(df, cols, outdir)
    plot_running_effect(df, cols, outdir, alpha=args.alpha)

    print("\nDone. Outputs:")
    print(f"- {os.path.join(outdir, 'summary_by_group.csv')}")
    print(f"- {os.path.join(outdir, 'conditional_summary.csv')}")
    print(f"- {os.path.join(outdir, 'per_student_metrics.csv')}")
    print(f"- {os.path.join(outdir, 'bootstrap_effects.csv')}")
    print(f"- {os.path.join(outdir, 'student_balance_table.csv')}")
    print(f"- Plots: {outdir}/*.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e)
        sys.exit(1)
