"""Tables and figures for bfcl_runner.py results.   python bfcl_analyze.py [--results bfcl_results]"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["FP16", "BF16", "Q8_0", "Q4_K_M", "ORACLE"]
BIN_ORDER = ["short", "medium", "long"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bfcl_results")
    a = ap.parse_args()
    res = Path(a.results)
    (res / "figures").mkdir(exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "turns"}
                       for r in map(json.loads, (res / "runs.jsonl").open(encoding="utf-8"))])
    models = [m for m in ORDER if m in set(df.model)] + sorted(set(df.model) - set(ORDER))
    meta_p = res / "model_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    df["bin"] = pd.Categorical(df["bin"], BIN_ORDER)
    df["model"] = pd.Categorical(df.model, models)

    tab = df.groupby(["model", "bin"], observed=True).agg(
        n=("valid", "size"), accuracy=("valid", "mean"),
        turns_passed=("turns_passed_before_fail", "mean"), error_call_rate=("error_call_rate", "mean"),
        format_errors=("format_errors", "mean"), latency_s=("latency_s", "mean")).reset_index()
    tab.to_csv(res / "bfcl_by_bin.csv", index=False)

    ov = df.groupby("model", observed=True).agg(
        accuracy=("valid", "mean"), turns_passed=("turns_passed_before_fail", "mean"),
        first_failed_turn=("first_failed_turn", "mean"), error_call_rate=("error_call_rate", "mean"),
        latency_s=("latency_s", "mean"), gen_tps=("gen_tps", "mean")).reset_index()
    ov["size_mb"] = ov.model.map(lambda m: meta.get(m, {}).get("size_mb"))
    ov["peak_ram_mb"] = ov.model.map(lambda m: meta.get(m, {}).get("peak_rss_mb"))
    ov.to_csv(res / "bfcl_overall.csv", index=False)

    fr = pd.crosstab([df.model, df["bin"]], df.first_fail_reason.fillna("pass"), normalize="index")
    fr.to_csv(res / "bfcl_fail_reasons.csv")

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 3)
    print("\n=== BFCL multi-turn by horizon bin ===\n", tab.to_string(index=False))
    print("\n=== overall ===\n", ov.to_string(index=False))
    print("\n=== first failure reason ===\n", fr)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        t = tab[tab.model == m]
        ax.plot(t["bin"].astype(str), t.accuracy, marker="o", label=m)
    ax.set(xlabel="Gold calls per entry (short 2-4 / medium 5-6 / long 7+)", ylabel="BFCL multi-turn accuracy",
           ylim=(0, 1.05), title="BFCL multi-turn accuracy by horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(res / "figures" / "bfcl_accuracy_by_bin.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    fr.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set(ylabel="Share of entries", xlabel="(model, bin)", title="BFCL: where does the first failure come from?")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "figures" / "bfcl_fail_reasons.png", dpi=150)
    print(f"\nsaved to {res}")


if __name__ == "__main__":
    main()
