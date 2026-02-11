"""
Compute minimal behavioral stats for OpenfMRI ds000102 flanker events TSV files.

Run:
  python analysis/compute_stats.py
"""

from __future__ import annotations

import csv
import glob
import math
import os
from collections import defaultdict
from statistics import mean, median


DATA_DIR = "data"
EVENT_GLOB = os.path.join(DATA_DIR, "sub-*_task-flanker_run-*_events.tsv")


def _safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def _trial_condition(row: dict) -> str | None:
    # Use `cond` codes (most consistent in these files):
    #   cond001/cond002 = congruent; cond003/cond004 = incongruent.
    cond = (row.get("cond") or "").strip().lower()
    if cond in {"cond001", "cond002"}:
        return "congruent"
    if cond in {"cond003", "cond004"}:
        return "incongruent"
    stim = (row.get("Stimulus") or "").strip().lower()
    if stim in {"congruent", "incongruent"}:
        return stim
    return None


def _is_correct(row: dict) -> bool | None:
    c = (row.get("correctness") or "").strip().lower()
    if c == "correct":
        return True
    if c == "incorrect":
        return False
    return None


def _rt_ms(row: dict) -> float | None:
    rt_s = _safe_float((row.get("response_time") or "").strip())
    if rt_s is None:
        return None
    return 1000.0 * rt_s


def paired_t(diff_values: list[float]) -> tuple[float, int]:
    """Return (t, df) for a one-sample t-test of diffs vs 0."""
    n = len(diff_values)
    df = n - 1
    md = mean(diff_values)
    var = sum((d - md) ** 2 for d in diff_values) / df
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = md / se
    return t, df


def main():
    files = sorted(glob.glob(EVENT_GLOB))
    if not files:
        raise SystemExit(f"No events files found at {EVENT_GLOB}")

    trials_by_subj: dict[str, list[dict]] = defaultdict(list)
    for fp in files:
        subj = os.path.basename(fp).split("_")[0]  # sub-01
        with open(fp, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                trials_by_subj[subj].append(row)

    included: list[str] = []
    excluded: list[tuple[str, str]] = []

    rt_mean_cong: dict[str, float] = {}
    rt_mean_incong: dict[str, float] = {}
    acc_cong: dict[str, float] = {}
    acc_incong: dict[str, float] = {}
    rt_median_cong: dict[str, float] = {}
    rt_median_incong: dict[str, float] = {}
    rt_logmean_cong: dict[str, float] = {}
    rt_logmean_incong: dict[str, float] = {}

    rt_trim_lo = 200.0
    rt_trim_hi = 1500.0

    for subj, rows in sorted(trials_by_subj.items()):
        n_trials_total = 0
        n_miss = 0
        n_correct = 0
        valid_trials = []

        for row in rows:
            cond = _trial_condition(row)
            if cond is None:
                continue
            n_trials_total += 1

            corr = _is_correct(row)
            if corr is None:
                n_miss += 1
                continue
            if corr:
                n_correct += 1

            valid_trials.append((cond, corr, _rt_ms(row)))

        if n_trials_total == 0:
            excluded.append((subj, "no_trials"))
            continue

        miss_rate = n_miss / n_trials_total
        acc_overall = n_correct / n_trials_total

        if acc_overall < 0.80:
            excluded.append((subj, f"low_accuracy({acc_overall:.3f})"))
            continue
        if miss_rate > 0.20:
            excluded.append((subj, f"high_miss({miss_rate:.3f})"))
            continue

        cong_correct_rts = []
        incong_correct_rts = []
        cong_acc = []
        incong_acc = []

        for cond, corr, rt in valid_trials:
            if cond == "congruent":
                cong_acc.append(1.0 if corr else 0.0)
            else:
                incong_acc.append(1.0 if corr else 0.0)

            if (not corr) or rt is None:
                continue
            if not (rt_trim_lo <= rt <= rt_trim_hi):
                continue
            (cong_correct_rts if cond == "congruent" else incong_correct_rts).append(rt)

        if not cong_correct_rts or not incong_correct_rts:
            excluded.append((subj, "no_correct_rts_after_trimming"))
            continue

        included.append(subj)
        rt_mean_cong[subj] = mean(cong_correct_rts)
        rt_mean_incong[subj] = mean(incong_correct_rts)
        rt_median_cong[subj] = median(cong_correct_rts)
        rt_median_incong[subj] = median(incong_correct_rts)
        acc_cong[subj] = mean(cong_acc) if cong_acc else float("nan")
        acc_incong[subj] = mean(incong_acc) if incong_acc else float("nan")
        rt_logmean_cong[subj] = mean([math.log(x) for x in cong_correct_rts])
        rt_logmean_incong[subj] = mean([math.log(x) for x in incong_correct_rts])

    diffs = [rt_mean_incong[s] - rt_mean_cong[s] for s in included]
    t, df = paired_t(diffs)
    dz = mean(diffs) / (math.sqrt(sum((d - mean(diffs)) ** 2 for d in diffs) / df))
    median_effects = [rt_median_incong[s] - rt_median_cong[s] for s in included]
    log_effects = [rt_logmean_incong[s] - rt_logmean_cong[s] for s in included]

    print("=== ds000102 flanker behavioral reanalysis (minimal) ===")
    print(f"N included: {len(included)}")
    print(f"N excluded: {len(excluded)}")
    if excluded:
        print("Excluded (subject, reason):")
        for subj, reason in excluded:
            print(f"  {subj}\t{reason}")
    print()
    print(f"Mean RT congruent (ms): {mean([rt_mean_cong[s] for s in included]):.2f}")
    print(f"Mean RT incongruent (ms): {mean([rt_mean_incong[s] for s in included]):.2f}")
    print(f"Congruency effect (ms): {mean(diffs):.2f}")
    print(f"t({df}): {t:.3f}")
    print(f\"Cohen's dz: {dz:.3f}\")
    print()
    print(f\"Median congruency effect (ms): {mean(median_effects):.2f}\")
    print(f\"Log-RT congruency effect: {mean(log_effects):.4f}\")
    print()
    print(f\"Mean accuracy congruent: {mean([acc_cong[s] for s in included]):.3f}\")
    print(f\"Mean accuracy incongruent: {mean([acc_incong[s] for s in included]):.3f}\")


if __name__ == \"__main__\":
    main()

