"""Aggregates results/runs.jsonl into tables, statistical tests and figures.

  python analyze.py --results results_v2
Every run is re-scored with the current evaluate.py, so metric changes never need a re-run.
Repeats are averaged inside each task before task-level statistics (bootstrap, McNemar).
"""
import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluate import POST_HOC, evaluate  # noqa: E402
from tasks import TASK_VERSION, build_tasks  # noqa: E402

ORDER = ["FP16", "BF16", "Q8_0", "Q4_K_M", "ORACLE"]
ERR_ORDER = ["format", "tool_selection", "extra_lookup", "argument", "observation", "skipped_step",
             "over_execution", "premature_stop", "max_steps", "state", "final_answer"]
B = 10000
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ helpers
def boot_ci(x, b=B):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def task_means(df, col="success"):
    return df.groupby(["model", "task_id"], observed=True)[col].mean().unstack("model")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--think", default=None, choices=["on", "off"])
    ap.add_argument("--any-version", action="store_true", help="do not drop runs from another task version")
    a = ap.parse_args()
    res = Path(a.results)
    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_txt = []

    def say(*s):
        line = " ".join(str(x) for x in s)
        print(line)
        out_txt.append(line)

    rows = [json.loads(l) for l in (res / "runs.jsonl").open(encoding="utf-8")]
    n_all = len(rows)
    if not a.any_version:
        rows = [r for r in rows if r.get("task_version", "v1") == TASK_VERSION]
    if len(rows) < n_all:
        say(f"[note] dropped {n_all - len(rows)} runs from another task version")
    tmap = {lang: {t["id"]: t for t in build_tasks(lang)} for lang in {r["lang"] for r in rows}}
    for r in rows:
        r["metrics"] = evaluate(tmap[r["lang"]][r["task_id"]], r["record"])
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("record", "metrics")} | r["metrics"] for r in rows])
    if a.lang:
        df = df[df.lang == a.lang]
    if a.think:
        df = df[df.think == (a.think == "on")]
    df["success"] = df.success.astype(int)
    models = [m for m in ORDER if m in set(df.model)] + sorted(set(df.model) - set(ORDER))
    stat_models = [m for m in models if m != "ORACLE"]
    df["model"] = pd.Categorical(df.model, models)
    meta_p = res / "model_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.precision", 3)

    # ---------------- 1. by gold length (1 / 3 / 6 calls) with task-level bootstrap CI
    tab = []
    for (m, n), d in df.groupby(["model", "n_gold"], observed=True):
        tm = d.groupby("task_id").success.mean()
        lo, hi = boot_ci(tm.values)
        rep = d.groupby("repeat").success.mean()
        tab.append({"model": m, "n_gold": n, "n_tasks": tm.size, "n_runs": len(d), "success": tm.mean(),
                    "ci_lo": lo, "ci_hi": hi, "success_std_over_repeats": rep.std(),
                    "step_accuracy": d.step_accuracy.mean(), "f1_func": d.f1_func.mean(),
                    "f1_param": d.f1_param.mean(), "full_match": d.full_match.mean(),
                    "valid_call_rate": d.valid_call_rate.mean(), "extra_lookups": d.extra_lookups.mean(),
                    "wall_s": d.wall_s.mean(), "gen_tps": d.gen_tps.mean()})
    tab = pd.DataFrame(tab)
    tab.to_csv(res / "summary_by_length.csv", index=False)
    say("\n=== 1. success by number of gold calls (task-level 95% bootstrap CI) ===")
    say(tab.to_string(index=False))

    # ---------------- 2. overall + decay slope on real gold-call count + efficiency
    ov = []
    for m in models:
        d = df[df.model == m]
        tm = d.groupby(["task_id", "n_gold"]).success.mean().reset_index()
        slope = np.polyfit(tm.n_gold, tm.success, 1)[0] if tm.n_gold.nunique() > 1 else np.nan
        ov.append({"model": m, "success": tm.success.mean(), "step_accuracy": d.step_accuracy.mean(),
                   "f1_func": d.f1_func.mean(), "f1_param": d.f1_param.mean(), "full_match": d.full_match.mean(),
                   "decay_slope_per_call": slope, "task_latency_s": d.wall_s.mean(), "gen_tps": d.gen_tps.mean(),
                   "size_mb": meta.get(m, {}).get("size_mb"), "peak_ram_mb": meta.get(m, {}).get("peak_rss_mb"),
                   "load_mode": meta.get(m, {}).get("load_mode")})
    ov = pd.DataFrame(ov)
    ov.to_csv(res / "summary_overall.csv", index=False)
    say("\n=== 2. overall (decay slope = change in success per additional gold call) ===")
    say(ov.to_string(index=False))

    # ---------------- 3. by number of observation-derived arguments
    der = df.groupby(["model", "n_derived"], observed=True).agg(
        n_tasks=("task_id", "nunique"), success=("success", "mean"), step_accuracy=("step_accuracy", "mean")
    ).reset_index()
    der.to_csv(res / "summary_by_derived.csv", index=False)
    say("\n=== 3. success by number of arguments that must be read from earlier tool results ===")
    say(der.pivot(index="n_derived", columns="model", values="success").to_string())
    say("n_tasks per n_derived: " + str(der.groupby("n_derived").n_tasks.max().to_dict()))

    # ---------------- 4. logistic regression: precision x length interaction
    say("\n=== 4. logit(success) ~ precision * (n_gold - 1), task-clustered SE ===")
    try:
        import statsmodels.formula.api as smf
        d = df[df.model.isin(stat_models)].copy()
        d["model"] = d.model.astype(str)
        ref = "FP16" if "FP16" in stat_models else stat_models[0]
        d["length"] = d.n_gold - 1
        groups = pd.factorize(d.task_id)[0]
        fit = smf.logit(f"success ~ C(model, Treatment('{ref}')) * length", d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        coef = pd.DataFrame({"coef": fit.params, "odds_ratio": np.exp(fit.params), "se": fit.bse,
                             "p": fit.pvalues})
        say(coef.to_string())
        inter = [i for i, n in enumerate(fit.params.index) if ":" in n]
        if inter:
            R = np.zeros((len(inter), len(fit.params)))
            for r_, i in enumerate(inter):
                R[r_, i] = 1
            w = fit.wald_test(R, scalar=True)
            say(f"joint Wald test, all precision x length terms = 0: chi2={float(w.statistic):.3f}, "
                f"df={len(inter)}, p={float(w.pvalue):.4f}")
        coef.to_csv(res / "stats_logit.csv")
    except Exception as e:  # noqa: BLE001 - e.g. perfect separation with tiny data
        say(f"logistic regression failed: {e!r}")

    # ---------------- 5. pairwise McNemar + paired bootstrap of task-mean differences
    say("\n=== 5. pairwise precision comparison (repeats averaged per task) ===")
    say("McNemar uses per-task majority success (>= 0.5 of repeats); diff CI uses task means.")
    pw = []
    for n in [None] + sorted(df.n_gold.unique()):
        d = df if n is None else df[df.n_gold == n]
        tmn = task_means(d)
        for m1, m2 in combinations([m for m in stat_models if m in tmn.columns], 2):
            x = tmn[[m1, m2]].dropna()
            if x.empty:
                continue
            diff = (x[m1] - x[m2]).values
            lo, hi = boot_ci(diff)
            b1, b2 = x[m1] >= 0.5, x[m2] >= 0.5
            b, c = int((b1 & ~b2).sum()), int((~b1 & b2).sum())
            pw.append({"n_gold": "all" if n is None else n, "pair": f"{m1} - {m2}", "n_tasks": len(x),
                       "diff": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                       "only_first": b, "only_second": c, "mcnemar_p": mcnemar_exact(b, c)})
    pw = pd.DataFrame(pw)
    pw.to_csv(res / "stats_pairwise.csv", index=False)
    if len(pw):
        say(pw.to_string(index=False))

    # ---------------- 6. compounding check: p^T vs observed
    say("\n=== 6. compounding: predicted p^T vs observed success ===")
    say("p1 = 1-call task success; p_step = pooled per-call accuracy on all tasks (matched / gold calls)")
    cp = []
    for m in stat_models:
        d = df[df.model == m]
        p1 = d[d.n_gold == 1].success.mean()
        p_step = (d.step_accuracy * d.n_gold).sum() / d.n_gold.sum()
        for T_ in sorted(d.n_gold.unique()):
            cp.append({"model": m, "T": T_, "observed": d[d.n_gold == T_].success.mean(),
                       "pred_p1^T": p1 ** T_, "pred_pstep^T": p_step ** T_, "p1": p1, "p_step": p_step})
    cp = pd.DataFrame(cp)
    cp.to_csv(res / "compounding.csv", index=False)
    say(cp.to_string(index=False))

    # ---------------- 7. first error: type distribution, relative position
    err = df[df.had_error]
    et = pd.crosstab([err.model, err.n_gold], err.first_error_type, normalize="index")
    et = et[[c for c in ERR_ORDER if c in et.columns]]
    et.to_csv(res / "error_types.csv")
    herr = df[df.had_hard_error]
    ht = pd.crosstab([herr.model, herr.n_gold], herr.first_hard_error_type, normalize="index")
    ht = ht[[c for c in ERR_ORDER if c in ht.columns]]
    ht.to_csv(res / "error_types_hard.csv")
    say("\n=== 7a. first error type (all deviations, incl. extra_lookup) ===")
    say(et.to_string())
    say("\n=== 7b. first HARD error type (extra_lookup ignored) ===")
    say(ht.to_string())

    pos = []
    for col_s, col_t, name in [("first_error_step", "first_error_type", "first_error"),
                               ("first_hard_error_step", "first_hard_error_type", "first_hard_error")]:
        d = df[df[col_t].notna() & ~df[col_t].isin(POST_HOC)].copy()
        d["rel"] = d[col_s] / d.n_gold
        g = d.groupby(["model", "n_gold"], observed=True).rel.agg(["size", "mean", "median"]).reset_index()
        g.insert(0, "which", name)
        pos.append(g)
    pos = pd.concat(pos)
    pos.to_csv(res / "first_error_position.csv", index=False)
    say("\n=== 7c. relative first-error position = step / gold calls (post-hoc types excluded) ===")
    say(pos.to_string(index=False))

    # ---------------- 8. error-free progress curve (longest tasks)
    long_n = int(df.n_gold.max())
    d = df[df.n_gold == long_n]
    surv = pd.DataFrame({m: [(d[d.model == m].progress >= k).mean() for k in range(long_n + 1)]
                         for m in models}, index=pd.Index(range(long_n + 1), name="k"))
    surv.to_csv(res / "progress_curve.csv")
    say(f"\n=== 8. share of {long_n}-call runs that reproduced >= k gold calls before the first hard error ===")
    say(surv.to_string())

    # ---------------- 9. recovery with denominators
    rec = df[df.had_hard_error].groupby(["model", "n_gold"], observed=True).agg(
        n_error_runs=("recovered", "size"), n_recovered=("recovered", "sum")).reset_index()
    rec["recovery_rate"] = rec.n_recovered / rec.n_error_runs
    rec.to_csv(res / "recovery.csv", index=False)
    say("\n=== 9. recovery after a hard error (n = runs with a hard error) ===")
    say(rec.to_string(index=False))

    (res / "analysis.txt").write_text("\n".join(out_txt), encoding="utf-8")

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        t = tab[tab.model == m]
        ax.errorbar(t.n_gold, t.success, yerr=[t.success - t.ci_lo, t.ci_hi - t.success],
                    marker="o", capsize=3, label=m)
    ax.set(xlabel="Gold tool calls per task", ylabel="Task success rate", ylim=(-0.02, 1.05),
           xticks=sorted(df.n_gold.unique()), title="Success vs. task length (95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "success_by_length.png", dpi=150)

    for tbl, fname, title in [(et, "error_types.png", "First error type"),
                              (ht, "error_types_hard.png", "First hard error type")]:
        if len(tbl):
            fig, ax = plt.subplots(figsize=(9, 4))
            tbl.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set(ylabel="Share of runs with an error", xlabel="(model, gold calls)", title=title)
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / fname, dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        ax.step(surv.index, surv[m], where="post", marker="o", label=m)
    ax.set(xlabel="k (gold calls reproduced)", ylabel=f"Share of {long_n}-call runs", ylim=(-0.02, 1.05),
           title="Error-free progress before the first hard error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "progress_curve.png", dpi=150)

    pts = [(m, meta[m]["size_mb"], df[(df.model == m) & (df.n_gold == long_n)].success.mean())
           for m in models if m in meta]
    if pts:
        fig, ax = plt.subplots(figsize=(5, 4))
        for m, s, y in pts:
            ax.scatter(s, y, s=80)
            ax.annotate(m, (s, y), textcoords="offset points", xytext=(5, 5))
        ax.set(xlabel="Model size (MB)", ylabel=f"Success ({long_n}-call tasks)", ylim=(0, 1.05),
               title="Efficiency vs long-horizon reliability")
        fig.tight_layout()
        fig.savefig(fig_dir / "tradeoff.png", dpi=150)
    print(f"\nsaved tables/figures to {res} (text log: analysis.txt)")


if __name__ == "__main__":
    main()
