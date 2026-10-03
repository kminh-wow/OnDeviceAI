"""Human audit of the automatic grader.

1) python audit.py sample --results results_v2 --n 40
   -> audit_sheet.csv (blind: no automatic verdicts) + audit_key.csv (automatic verdicts)
   Fill human_answer_ok / human_success in audit_sheet.csv with 1 or 0 (open in Excel, save as CSV UTF-8).
2) python audit.py score --results results_v2
   -> agreement rate and Cohen's kappa between human and automatic grading
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate
from tasks import TASK_VERSION, build_tasks


def compact_calls(calls):
    return " | ".join(f"{c['name']}({json.dumps(c.get('args'), ensure_ascii=False)})" for c in calls)


def compact_state(st):
    return json.dumps({k: v for k, v in st.items() if v}, ensure_ascii=False)


def sample(a):
    res = Path(a.results)
    rows = [json.loads(l) for l in (res / "runs.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["model"] != "ORACLE" and r.get("task_version", "v1") == TASK_VERSION]
    tmap = {lang: {t["id"]: t for t in build_tasks(lang)} for lang in {r["lang"] for r in rows}}
    rng = np.random.default_rng(a.seed)
    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault((r["model"], r["horizon"]), []).append(i)
    per = a.n // len(cells)
    picked = []
    for key in sorted(cells):
        picked += list(rng.choice(cells[key], min(per, len(cells[key])), replace=False))
    rest = [i for i in range(len(rows)) if i not in set(picked)]
    picked += list(rng.choice(rest, a.n - len(picked), replace=False))
    rng.shuffle(picked)

    sheet, key = [], []
    for k, i in enumerate(picked, 1):
        r = rows[i]
        t = tmap[r["lang"]][r["task_id"]]
        m = evaluate(t, r["record"])
        sheet.append({"audit_id": k, "task_id": r["task_id"], "query": t["query"],
                      "expected_answer_contains": " AND ".join("/".join(g[:3]) for g in t["answer"]) or "-",
                      "expected_final_state": compact_state(t["expected_state"]) or "-",
                      "gold_calls": compact_calls([{"name": g["name"], "args": g["args"]} for g in t["gold"]]),
                      "agent_calls": compact_calls(r["record"]["calls"]),
                      "agent_final_state": compact_state(r["record"]["state"]) or "-",
                      "agent_final_answer": r["record"]["final_answer"],
                      "human_answer_ok": "", "human_success": "", "note": ""})
        key.append({"audit_id": k, "model": r["model"], "repeat": r["repeat"], "horizon": r["horizon"],
                    "auto_answer_ok": int(m["answer_ok"]), "auto_success": int(m["success"])})
    pd.DataFrame(sheet).to_csv(res / "audit_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(res / "audit_key.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {len(sheet)} rows -> {res / 'audit_sheet.csv'} (blind) and audit_key.csv")


def kappa(x, y):
    x, y = np.asarray(x, int), np.asarray(y, int)
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score(a):
    res = Path(a.results)
    s = pd.read_csv(res / "audit_sheet.csv", encoding="utf-8-sig")
    k = pd.read_csv(res / "audit_key.csv", encoding="utf-8-sig")
    d = s.merge(k, on="audit_id")
    for h, auto in [("human_answer_ok", "auto_answer_ok"), ("human_success", "auto_success")]:
        x = d[[h, auto]].dropna()
        x = x[x[h].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
        po, kp = kappa(x[h].astype(float), x[auto])
        print(f"{h:16s} n={len(x):3d}  agreement={po:.3f}  Cohen's kappa={kp:.3f}")
        dis = d.loc[x.index][x[h].astype(float) != x[auto]]
        if len(dis):
            print("   disagreements (audit_id):", dis.audit_id.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else score(args)
