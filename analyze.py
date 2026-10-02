"""Aggregates results/runs.jsonl into tables and figures.

  python analyze.py                 # all runs
  python analyze.py --lang ko --think off
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluate import evaluate  # noqa: E402
from tasks import build_tasks  # noqa: E402

ORDER = ["FP16", "BF16", "Q8_0", "Q4_K_M", "ORACLE"]
ERR_ORDER = ["format", "tool_selection", "argument", "observation", "skipped_step", "premature_stop", "max_steps", "over_execution",
             "final_answer", "state"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--think", default=None, choices=["on", "off"])
    a = ap.parse_args()
    res = Path(a.results)
    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)

    rows = [json.loads(l) for l in (res / "runs.jsonl").open(encoding="utf-8")]
    # re-score every run with the current evaluate.py (metric changes need no re-run)
    task_maps = {lang: {t["id"]: t for t in build_tasks(lang)} for lang in {r["lang"] for r in rows}}
    for r in rows:
        r["metrics"] = evaluate(task_maps[r["lang"]][r["task_id"]], r["record"])
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("record", "metrics")} | r["metrics"] for r in rows])
    if a.lang:
        df = df[df.lang == a.lang]
    if a.think:
        df = df[df.think == (a.think == "on")]
    models = [m for m in ORDER if m in df.model.unique()] + sorted(set(df.model) - set(ORDER))
    meta_p = res / "model_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    # ---- per model x horizon (mean over tasks, then mean/std over repeats)
    per_rep = df.groupby(["model", "horizon", "repeat"]).success.mean().reset_index()
    sr = per_rep.groupby(["model", "horizon"]).success.agg(success_mean="mean", success_std="std")
    err = df[df.had_error]
    tab = df.groupby(["model", "horizon"]).agg(
        n=("success", "size"), step_accuracy=("step_accuracy", "mean"),
        f1_func=("f1_func", "mean"), f1_param=("f1_param", "mean"), full_match=("full_match", "mean"),
        valid_call_rate=("valid_call_rate", "mean"), wall_s=("wall_s", "mean"),
        completion_tokens=("completion_tokens", "mean"), gen_tps=("gen_tps", "mean")).join(sr)
    tab = tab.join(err.groupby(["model", "horizon"]).first_error_step.mean().rename("first_error_step"))
    tab = tab.join(err.groupby(["model", "horizon"]).recovered.mean().rename("recovery_rate"))
    tab = tab.reset_index()
    tab["model"] = pd.Categorical(tab.model, models)
    tab = tab.sort_values(["model", "horizon"])
    tab.to_csv(res / "summary_by_horizon.csv", index=False)

    # ---- per model overall + efficiency + decay slope
    ov = []
    for m in models:
        d = df[df.model == m]
        t = tab[tab.model == m]
        slope = np.polyfit(t.horizon, t.success_mean, 1)[0] if t.horizon.nunique() > 1 else np.nan
        ov.append({"model": m, "success": d.success.mean(), "step_accuracy": d.step_accuracy.mean(),
                   "f1_func": d.f1_func.mean(), "f1_param": d.f1_param.mean(), "full_match": d.full_match.mean(),
                   "valid_call_rate": d.valid_call_rate.mean(), "decay_slope_per_step": slope,
                   "first_error_step": d[d.had_error].first_error_step.mean(),
                   "recovery_rate": d[d.had_error].recovered.mean(),
                   "task_latency_s": d.wall_s.mean(), "gen_tps": d.gen_tps.mean(),
                   "size_mb": meta.get(m, {}).get("size_mb"), "peak_ram_mb": meta.get(m, {}).get("peak_rss_mb")})
    ov = pd.DataFrame(ov)
    ov.to_csv(res / "summary_overall.csv", index=False)

    et = pd.crosstab([err.model, err.horizon], err.first_error_type, normalize="index")
    et = et[[c for c in ERR_ORDER if c in et.columns]]
    et.to_csv(res / "error_types.csv")

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 3)
    print("\n=== by horizon ===\n", tab.to_string(index=False))
    print("\n=== overall ===\n", ov.to_string(index=False))
    print("\n=== first error type (share of runs with an error) ===\n", et)

    # ---- figures
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        t = tab[tab.model == m]
        ax.errorbar(t.horizon, t.success_mean, yerr=t.success_std.fillna(0), marker="o", capsize=3, label=m)
    ax.set(xlabel="Task horizon (steps)", ylabel="Task success rate", ylim=(0, 1.05), xticks=[1, 3, 5],
           title="Success decay by precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "success_by_horizon.png", dpi=150)

    if len(et):
        fig, ax = plt.subplots(figsize=(8, 4))
        et.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set(ylabel="Share of first errors", xlabel="(model, horizon)", title="Where does failure start?")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "error_types.png", dpi=150)

    long_h = df.horizon.max()
    pts = [(m, meta[m]["size_mb"], df[(df.model == m) & (df.horizon == long_h)].success.mean())
           for m in models if m in meta]
    if pts:
        fig, ax = plt.subplots(figsize=(5, 4))
        for m, s, y in pts:
            ax.scatter(s, y, s=80)
            ax.annotate(m, (s, y), textcoords="offset points", xytext=(5, 5))
        ax.set(xlabel="Model size (MB)", ylabel=f"Success rate ({long_h}-step)", ylim=(0, 1.05),
               title="Efficiency vs long-horizon reliability")
        fig.tight_layout()
        fig.savefig(fig_dir / "tradeoff.png", dpi=150)
    print(f"\nsaved tables to {res} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
