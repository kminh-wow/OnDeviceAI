#!/usr/bin/env python3
"""Independent re-analysis of the workflow-en-v4 logs (interim report).

Recomputes every number and figure in the interim report directly from the
raw JSONL logs in ``logs/main`` (and ``logs/pilot``), plus the harness summary
CSVs for the pre-specified paired tests. Nothing here reuses harness metrics
except where the report says so (Table 4 and the interaction tests).

Usage:
    python analysis/reanalyze.py --repo . --out analysis/out

Outputs: numbers.json (every reported value), fig1..fig5 PDFs.
Requires: numpy, pandas, scipy, statsmodels, matplotlib.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

MODELS = {"self-FP16": "FP16", "self-Q8": "Q8_0", "self-Q4": "Q4_K_M"}
ORDER = ["FP16", "Q8_0", "Q4_K_M"]
CONDS = ["free", "normal_restart", "error_restart", "signaled_error_restart"]
RESTART = {"error_restart", "signaled_error_restart"}
SEED = 20261018
COLORS = {"FP16": "#2D7DD2", "Q8_0": "#EC6B3A", "Q4_K_M": "#1FAF7A"}


# --------------------------------------------------------------------------- load
def classify_candidates(spec):
    req = spec["request"]
    cons, cands = req["constraints"], spec["candidates"]

    def eligible(c):
        s = c["summary"]
        for k, v in cons.items():
            field = k[4:]
            if k.startswith("max_") and s[field] > v:
                return False
            if k.startswith("min_") and s[field] < v:
                return False
        return True

    def key(c):
        return tuple(-c["summary"][r["field"]] if r["direction"] == "max" else c["summary"][r["field"]]
                     for r in req["ranking"])

    elig = [c for c in cands if eligible(c)]
    best = min(elig, key=key)
    labels = {c["candidate_id"]: ("correct" if c is best else "other_eligible" if eligible(c) else "violating")
              for c in cands}
    return labels, min(cands, key=key)["candidate_id"], len(elig)


def scenario_of(base_id):
    return re.sub(r"_h\d+$", "", base_id.removeprefix("recovery_"))


def load_runs(repo):
    tasks = {t["id"]: t for t in json.load(open(os.path.join(repo, "tasks", "tasks.json"), encoding="utf-8"))}
    rows = []
    for path in sorted(glob.glob(os.path.join(repo, "logs", "main", "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        label = name.split("_")[0]
        cond = name[len(label) + 1:]
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            t, tr, sc = d["task"], d["trace"], d["score"]
            spec = tasks[t["id"]]["execution_spec"]
            lab, uncons_best, n_elig = classify_candidates(spec)
            opt_kind = {o["option_id"]: o["kind"] for c in spec["candidates"] for o in c["options"]}
            target = t["success_condition"]["completed_request_equals"]
            saved = tr["final_state"]["completed_requests"]
            rows.append(dict(
                model=MODELS[label], cond=cond, task_id=t["id"], scen=scenario_of(t["base_id"]),
                domain=t["domain"], horizon=tr["horizon"], error_kind=tr.get("error_kind") or "-",
                ctx=tr.get("context_delivery"), success=bool(sc["strict_success"]),
                indep_success=(len(saved) == 1 and list(saved.values())[0] == target),
                max_calls=tr["max_calls"], events=tr["events"], initial_messages=tr["initial_messages"],
                termination=tr["termination_reason"], infra=tr["infrastructure_error"],
                internal=tr["internal_error"], injected=tr.get("injected_actions") or [], spec=spec,
                lab=lab, uncons_best=uncons_best, n_elig=n_elig, opt_kind=opt_kind,
                req_kind=spec["request"]["required_option_kind"], final_saved=saved,
                final_answer=tr["final_answer"], peak_gpu=tr.get("peak_gpu_memory_mb"),
                init_hash=hashlib.sha256(json.dumps(tr["initial_messages"], sort_keys=True,
                                                    ensure_ascii=False).encode()).hexdigest(),
            ))
    runs = pd.DataFrame(rows)
    runs["n_resp"] = runs.events.str.len()
    return runs


def classify_saved(r, saved):
    cls = r.lab.get(saved["candidate_id"], "unknown")
    if cls == "correct" and r.opt_kind.get(saved["option_id"]) != r.req_kind:
        return "option_error"
    return cls


def final_class(r):
    if not r.final_saved:
        return "empty"
    return classify_saved(r, list(r.final_saved.values())[0])


def raw_message(e):
    rr = e.get("raw_response") or {}
    ch = (rr.get("choices") or [{}])[0] if isinstance(rr, dict) else {}
    return ch.get("message") or {}


def intended_tool(e):
    a = e.get("action") or {}
    if a.get("type") == "tool_call":
        return a["name"]
    if a.get("type") == "finish":
        return "__finish__"
    msg = raw_message(e)
    if msg.get("tool_calls"):
        return msg["tool_calls"][0].get("function", {}).get("name")
    content = msg.get("content") if isinstance(msg.get("content"), str) else ""
    m = re.search(r'"name"\s*:\s*"([A-Za-z_]+)"', content)
    return m.group(1) if m else None


def build_responses(runs):
    """One row per model response with the message that preceded it."""
    rows = []
    for ri, r in runs.iterrows():
        msgs = [dict(role=m["role"], name=m.get("name"), content=m.get("content") or "",
                     tool=bool(m.get("tool_calls")), injected=any(c.get("id", "").startswith("injected")
                                                                 for c in (m.get("tool_calls") or [])))
                for m in r.initial_messages]
        # mark tool results that answer an injected call
        for i, m in enumerate(msgs):
            m["after_injected"] = m["role"] == "tool" and i > 0 and msgs[i - 1]["injected"]
        had_fmt = False
        for e in r.events:
            prev = msgs[-1]
            if prev["role"] == "tool":
                pk = "tool:" + (prev["name"] or "")
            elif prev["role"] == "user":
                c = prev["content"]
                pk = ("user:audit" if c.startswith("Automated audit") else
                      "user:feedback" if c.startswith("Your last response") else "user:query")
            else:
                pk = prev["role"]
            msg = raw_message(e)
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            a = e.get("action") or {}
            fmt = e.get("error") == "model_output_error"
            timings = (e.get("raw_response") or {}).get("timings") if isinstance(e.get("raw_response"), dict) else None
            rows.append(dict(
                ri=ri, model=r.model, cond=r.cond, horizon=r.horizon, error_kind=r.error_kind, ctx=r.ctx,
                step=e["step"], prev=pk, prev_injected=prev.get("after_injected", False),
                ntool=sum(1 for m in msgs if m["role"] == "assistant" and m["tool"]),
                fmt=fmt, model_error=e.get("model_error"), detail=e.get("error_detail"),
                atype=a.get("type"), aname=a.get("name"), intended=intended_tool(e), content=content,
                executed=bool(e.get("tool_executed")), tool_error=e.get("tool_error"),
                latency=e.get("latency_ms"), prompt_tokens=(e.get("usage") or {}).get("prompt_tokens"),
                prompt_ms=(timings or {}).get("prompt_ms"), prompt_n=(timings or {}).get("prompt_n"),
                pred_ms=(timings or {}).get("predicted_ms"), pred_n=(timings or {}).get("predicted_n"),
                cache_n=(timings or {}).get("cache_n"), before_first_fmt=not had_fmt,
            ))
            if fmt or e.get("error") in ("invalid_action", "invalid_arguments"):
                msgs += [dict(role="assistant", content=content, tool=False, injected=False, name=None),
                         dict(role="user", content="Your last response", tool=False, injected=False, name=None)]
                had_fmt = had_fmt or fmt
            elif a.get("type") == "tool_call" and e.get("tool_executed"):
                msgs += [dict(role="assistant", content="", tool=True, injected=False, name=None),
                         dict(role="tool", name=a["name"], content="", tool=False, injected=False)]
            elif a.get("type") == "finish":
                msgs.append(dict(role="assistant", content=a.get("answer", ""), tool=False, injected=False, name=None))
    resp = pd.DataFrame(rows)
    resp["server_ms"] = resp.prompt_ms + resp.pred_ms
    return resp


# ------------------------------------------------------------------------ helpers
def pct(a, b, nd=1):
    return round(100.0 * a / b, nd) if b else None


def by_model(series_fn):
    return {m: series_fn(m) for m in ORDER}


def confirms(r, ok_only=True):
    out = []
    for e in r.events:
        a = e.get("action") or {}
        if e.get("tool_executed") and a.get("name", "").startswith("confirm"):
            res = e.get("tool_result") or {}
            if res.get("ok") or not ok_only:
                out.append((e, res))
    return out


def provided_results(r):
    if r.ctx == "user_message":
        u = [m for m in r.initial_messages if m["role"] == "user"][0]["content"]
        return [(a["tool"], a["result"]) for a in json.loads(u[u.index('[{"tool"'):])]
    return [(m["name"], json.loads(m["content"])) for m in r.initial_messages if m["role"] == "tool"]


def h1_drafts(r):
    ds = [res for tool, res in provided_results(r) if tool.startswith("prepare")]
    return [d["draft_id"] for d in ds], tuple(classify_saved(r, d["proposed_result"]) for d in ds)


def cluster_boot_mean(diff, groups, B, rng):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    sums = np.array([diff[groups == g].sum() for g in uniq])
    cnts = np.array([(groups == g).sum() for g in uniq])
    stats = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats.append(sums[pick].sum() / cnts[pick].sum())
    return np.percentile(stats, [2.5, 97.5])


def mcnemar(b, c):
    return binomtest(int(b), int(b + c)).pvalue if b + c else 1.0


# ---------------------------------------------------------------------- analyses
def analyze(repo, out, logit_boot=2000):
    rng = np.random.default_rng(SEED)
    N = {}
    runs = load_runs(repo)
    resp = build_responses(runs)
    runs["final_class"] = runs.apply(final_class, axis=1)
    runs["any_fmt"] = resp.groupby("ri").fmt.any().reindex(runs.index).fillna(False)

    # ---------------------------------------------------------------- integrity
    restart = runs[runs.cond != "free"]
    hash_groups = restart.groupby(["cond", "task_id", "error_kind"]).init_hash.nunique()
    N["integrity"] = dict(
        runs=len(runs), responses=len(resp), responses_by_model=resp.groupby("model").size().to_dict(),
        scoring_agree=int((runs.success == runs.indep_success).sum()),
        budget_ok=bool((runs.max_calls == runs.horizon + 3).all() and (runs.n_resp <= runs.max_calls).all()),
        infra=int(runs.infra.notna().sum()), internal=int(runs.internal.notna().sum()),
        truncated=resp[resp.model_error == "output_truncated"].groupby("model").size().to_dict(),
        max_prompt_tokens=int(resp.prompt_tokens.max()), cache_n_max=float(resp.cache_n.max()),
        identical_restart_inputs=f"{int((hash_groups == 1).sum())}/{len(hash_groups)}",
        violator_rank1=bool(runs.apply(lambda r: r.lab[r.uncons_best] == "violating", axis=1).all()),
        two_eligible=bool((runs.n_elig == 2).all()),
    )
    templates = {json.load(open(p))["server_properties"]["chat_template"]
                 for p in glob.glob(os.path.join(repo, "logs", "main", "*.preflight.json"))}
    N["integrity"]["template_sha256"] = [hashlib.sha256(t.encode()).hexdigest()[:8] for t in templates]
    # pilot
    pil_same = pil_tot = 0
    pil_ts, main_ts, pil_exh, pil_fmt = [], [], [], []
    for p in sorted(glob.glob(os.path.join(repo, "logs", "pilot", "*.jsonl"))):
        cond = os.path.basename(p)[len("pilot-self-FP16_"):-6]
        main = {}
        for line in open(os.path.join(repo, "logs", "main", f"self-FP16_{cond}.jsonl"), encoding="utf-8"):
            d = json.loads(line)
            main[(d["task"]["id"], d["trace"].get("error_kind"))] = d
            main_ts.append(d["metadata"]["created_at_unix"])
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            pil_ts.append(d["metadata"]["created_at_unix"])
            m = main[(d["task"]["id"], d["trace"].get("error_kind"))]
            # compare what the model produced; tool-call ids are random per request
            seq = lambda x: [((raw_message(e).get("content") or "").strip(),
                              [(c["function"]["name"], c["function"]["arguments"])
                               for c in (raw_message(e).get("tool_calls") or [])]) for e in x["trace"]["events"]]
            pil_tot += 1
            pil_same += seq(d) == seq(m)
            if cond == "free":
                pil_exh.append(d["trace"]["termination_reason"] == "budget_exhausted")
                pil_fmt.append(sum(e.get("error") == "model_output_error" for e in d["trace"]["events"]))
    kst = dt.timezone(dt.timedelta(hours=9))
    fmt_ts = lambda t: dt.datetime.fromtimestamp(t, kst).strftime("%m-%d %H:%M")
    N["pilot"] = dict(runs=pil_tot, identical=pil_same, first=fmt_ts(min(pil_ts)), last=fmt_ts(max(pil_ts)),
                      main_first=fmt_ts(min(main_ts)), free_budget_exhausted=f"{sum(pil_exh)}/{len(pil_exh)}",
                      free_mean_fmt=round(float(np.mean(pil_fmt)), 2))

    # ---------------------------------------------------------------- success
    succ = runs.groupby(["model", "cond", "horizon", "error_kind"]).success.agg(["sum", "count"])
    N["success_cells"] = {f"{m}|{c}|h{h}|{k}": [int(s), int(n)] for (m, c, h, k), (s, n) in succ.iterrows()}
    N["success_by_cond"] = {f"{m}|{c}": [int(s), int(n)] for (m, c), (s, n)
                            in runs.groupby(["model", "cond"]).success.agg(["sum", "count"]).iterrows()}

    # ---------------------------------------------------------------- h1 position
    h1 = runs[runs.horizon == 1].copy()
    dd = h1.apply(h1_drafts, axis=1)
    h1["draft_ids"], h1["draft_cls"] = dd.str[0], dd.str[1]
    h1["correct_pos"] = h1.draft_cls.apply(lambda c: c.index("correct") + 1)
    pos_of = lambda r, did: r.draft_ids.index(did) + 1 if did in r.draft_ids else None
    h1["conf_pos"] = h1.apply(lambda r: [pos_of(r, e["action"]["arguments"].get("draft_id")) for e, _ in confirms(r)], axis=1)
    f1 = h1[h1.cond == "free"]
    N["h1"] = dict(
        rotations={"/".join(k): int(v) for k, v in h1.draft_cls.value_counts().items()},
        free_first_pos={m: {int(p): int(n) for p, n in Counter(x[0] for x in f1[f1.model == m].conf_pos if x).items()}
                        | {"n": int((f1[f1.model == m].conf_pos.str.len() > 0).sum())} for m in ORDER},
        free_success_by_correct_pos={m: f1[f1.model == m].groupby("correct_pos").success.sum().astype(int).to_dict() for m in ORDER},
        first_draft_policy=f"{int((f1[f1.model == 'FP16'].correct_pos == 1).sum())}/60",
        free_first_class={m: Counter(classify_saved(r, confirms(r)[0][1]["saved_result"])
                                     for _, r in f1[f1.model == m].iterrows() if confirms(r)) for m in ORDER},
    )

    def inj_pos(r):
        for m in r.initial_messages:
            for c in (m.get("tool_calls") or []):
                if c["id"].startswith("injected"):
                    return pos_of(r, json.loads(c["function"]["arguments"])["draft_id"])
    er = h1[h1.cond.isin(RESTART)].copy()
    er["inj"] = er.apply(inj_pos, axis=1)
    e1 = er[er.cond == "error_restart"]
    nx = e1[(e1.inj < 3) & (e1.conf_pos.str.len() > 0)]
    N["h1"]["next_slot"] = {m: f"{int((g.conf_pos.str[0] == g.inj + 1).sum())}/{len(g)}" for m, g in nx.groupby("model")}
    N["h1"]["recovery_by_inj"] = {f"{m}|{k}": {int(p): f"{int(g.success.sum())}/{len(g)}" for p, g in gg.groupby("inj")}
                                  for (m, k), gg in e1.groupby(["model", "error_kind"])}
    v1 = e1[(e1.error_kind == "violating_candidate") & (e1.inj == 1)]
    N["h1"]["viol_inj1_sequences"] = {m: {str(tuple(s)) + ("✓" if ok else "✗"): int(n) for (s, ok), n in
                                          Counter(zip(map(tuple, g.conf_pos), g.success)).most_common(5)}
                                      for m, g in v1.groupby("model")}
    discard_after_correct = 0
    for _, r in e1[e1.model.isin(["FP16", "Q8_0"])].iterrows():
        cls = [classify_saved(r, res["saved_result"]) for _, res in confirms(r)]
        if "correct" in cls and r.final_class == "empty":
            discard_after_correct += 1
    N["h1"]["fp16_q8_correct_then_discard"] = discard_after_correct

    # ---------------------------------------------------------------- h3/h5 selection
    sel = runs[(runs.cond == "free") & runs.horizon.isin([3, 5])].copy()

    def first_valid(r):
        allc = confirms(r, ok_only=False)
        if not allc or not allc[0][1].get("ok"):
            return None
        return allc[0][1]["saved_result"]
    sel["fv"] = sel.apply(first_valid, axis=1)
    sel["fv_cls"] = sel.apply(lambda r: r.lab.get(r.fv["candidate_id"]) if r.fv else None, axis=1)
    sel["fv_opt_ok"] = sel.apply(lambda r: (r.opt_kind.get(r.fv["option_id"]) == r.req_kind) if r.fv else None, axis=1)
    order_cls = lambda r: [r.lab[c["candidate_id"]] for c in r.spec["candidates"]]
    sel["vpos"] = sel.apply(lambda r: order_cls(r).index("violating") + 1, axis=1)
    perms = sel[sel.model == "FP16"].groupby("horizon").apply(lambda g: g.apply(lambda r: tuple(order_cls(r)), axis=1).value_counts().tolist())

    def earlier(r):
        rem = [c["candidate_id"] for c in r.spec["candidates"] if r.lab[c["candidate_id"]] != "violating"]
        return r.fv["candidate_id"] == rem[0]
    nv = sel[sel.fv_cls.isin(["correct", "other_eligible"])]
    first_tool = lambda r: next((x for x in (intended_tool(e) for e in r.events) if x), None)
    s3 = sel[sel.horizon == 3]
    N["selection"] = dict(
        permutations={int(h): v for h, v in perms.items()},
        n_valid={f"{m}|h{h}": int(g.fv.notna().sum()) for (m, h), g in sel.groupby(["model", "horizon"])},
        class_share={f"{m}|h{h}": {k: round(v, 3) for k, v in g.fv_cls.value_counts(normalize=True).items()}
                     for (m, h), g in sel.groupby(["model", "horizon"])},
        class_counts={f"{m}|h{h}": g.fv_cls.value_counts().to_dict() for (m, h), g in sel.groupby(["model", "horizon"])},
        option_wrong={f"{m}|h{h}": f"{int((g.fv_opt_ok == False).sum())}/{int(g.fv.notna().sum())}"
                      for (m, h), g in sel.groupby(["model", "horizon"])},
        violating_by_pos={f"{m}|h{h}": g[g.fv_cls == "violating"].vpos.value_counts().sort_index().to_dict()
                          for (m, h), g in sel.groupby(["model", "horizon"])},
        earlier_remaining={f"{m}|h{h}": f"{int(g.apply(earlier, axis=1).sum())}/{len(g)}"
                           for (m, h), g in nv.groupby(["model", "horizon"])},
        free_h3_requery={m: f"{int(g.apply(lambda r: (first_tool(r) or '').startswith('find'), axis=1).sum())}/{len(g)}"
                         for m, g in s3.groupby("model")},
    )
    # normal h3 vs free h5 duplication
    fcc = lambda r: (confirms(r, False)[0][1].get("saved_result", {}) or {}).get("candidate_id") if confirms(r, False) else "-"
    dup = {}
    for m in ORDER:
        a = runs[(runs.model == m) & (runs.cond == "normal_restart") & (runs.horizon == 3)].set_index("scen")
        b = runs[(runs.model == m) & (runs.cond == "free") & (runs.horizon == 5)].set_index("scen").reindex(a.index)
        dup[m] = dict(first_confirm=int((a.apply(fcc, axis=1) == b.apply(fcc, axis=1)).sum()),
                      success=int((a.success == b.success).sum()))
    N["selection"]["normal_h3_vs_free_h5"] = dup
    # selection accuracy difference (free h3, all 60 runs; no valid confirm = not correct)
    acc = s3.assign(c=(s3.fv_cls == "correct").astype(int)).pivot_table(index="scen", columns="model", values="c")
    d = (acc["Q4_K_M"] - acc["FP16"]).values
    N["selection"]["acc_q4_minus_fp16_h3"] = [round(100 * d.mean(), 1)] + list(np.round(100 * cluster_boot_mean(d, acc.index, 3000, rng), 1))

    # conditional logit
    def long_rows(g):
        rows = []
        for _, r in g.iterrows():
            if not r.fv:
                continue
            for pos, c in enumerate(r.spec["candidates"]):
                cid = c["candidate_id"]
                rows.append(dict(grp=r.scen, y=int(cid == r.fv["candidate_id"]), p1=int(pos == 0), p2=int(pos == 1),
                                 v=int(r.lab[cid] == "violating"), g=int(r.lab[cid] == "correct")))
        return pd.DataFrame(rows)

    def fit(df):
        res = ConditionalLogit(df.y, df[["p1", "p2", "v", "g"]], groups=df.grp).fit(disp=0, method="bfgs", maxiter=500)
        return res.params.values
    logit = {}
    for (m, h), g in sel.groupby(["model", "horizon"]):
        df = long_rows(g)
        est = np.exp(fit(df))
        by = {k: v for k, v in df.groupby("grp")}
        keys = list(by)
        bs, bad = [], 0
        for _ in range(logit_boot):
            parts = []
            for j, k in enumerate(rng.choice(keys, len(keys), replace=True)):
                x = by[k].copy()
                x["grp"] = f"{k}#{j}"
                parts.append(x)
            try:
                p = fit(pd.concat(parts))
                if not np.all(np.isfinite(p)) or np.any(np.abs(p) > 15):
                    bad += 1
                    continue
                bs.append(np.exp(p))
            except Exception:
                bad += 1
        bs = np.array(bs)
        if len(bs):
            lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
        else:
            lo = hi = np.full(4, np.nan)
        logit[f"{m}|h{h}"] = dict(n=int(df.grp.nunique()), excluded=bad,
                                  OR=[[round(float(e), 2), round(float(l), 2), round(float(u), 2)] for e, l, u in zip(est, lo, hi)])
    N["logit"] = logit

    # ---------------------------------------------------------------- format errors
    fmt = resp[resp.fmt]
    plain = resp[resp.model_error == "unparsed_tool_call"].copy()

    def parse(c):
        try:
            return json.loads(c.strip())
        except Exception:
            return None
    plain["p"] = plain.content.apply(parse)
    plain["valid"] = plain.p.apply(lambda d: isinstance(d, dict) and isinstance(d.get("name"), str) and isinstance(d.get("arguments"), dict))
    workflow = resp[resp.before_first_fmt & resp.intended.notna()
                    & ~resp.intended.isin(["inspect_state", "__finish__", "discard_saved_result"])
                    & (resp.model_error != "multiple_tool_calls")]
    N["format"] = dict(
        rate={m: pct(fmt[fmt.model == m].shape[0], (resp.model == m).sum()) for m in ORDER},
        kinds={m: fmt[fmt.model == m].model_error.value_counts().to_dict() for m in ORDER},
        details={m: fmt[fmt.model == m].detail.value_counts().to_dict() for m in ORDER},
        plain_inspect_share=round(float(plain.p.apply(lambda d: d == {"name": "inspect_state", "arguments": {}}).mean()), 3),
        plain_valid={m: round(float(plain[plain.model == m].valid.mean()), 4) for m in ORDER},
        workflow_first_try={m: f"{int(workflow[workflow.model == m].fmt.sum())}/{int((workflow.model == m).sum())}" for m in ORDER},
        think_close={m: int(fmt[(fmt.model == m) & fmt.content.str.contains("</think>", regex=False)].shape[0]) for m in ORDER},
    )
    ins = resp[(resp.intended == "inspect_state") & resp.before_first_fmt]
    conf = ins.prev.str.startswith("tool:confirm")

    def cell(mask):
        g = ins[mask].groupby("model").fmt.agg(["sum", "count"])
        return {m: [int(g.loc[m, "sum"]), int(g.loc[m, "count"])] if m in g.index else [0, 0] for m in ORDER}
    N["table3"] = dict(
        after_audit=cell(ins.prev == "user:audit"),
        after_model_confirm=cell(conf & ~ins.prev_injected),
        after_injected_confirm=cell(conf & ins.prev_injected),
        confirm_ntool_le3=cell(conf & (ins.ntool <= 3)),
        confirm_ntool_ge4=cell(conf & (ins.ntool >= 4)),
        h1_free_json=cell(conf & (ins.horizon == 1) & (ins.cond == "free")),
        h1_normal_history=cell(conf & (ins.horizon == 1) & (ins.cond == "normal_restart")),
    )
    t3 = N["table3"]
    fp = lambda k: t3[k]["FP16"]
    N["format"]["fisher_turns_fp16"] = fisher_exact([[fp("confirm_ntool_le3")[0], fp("confirm_ntool_le3")[1] - fp("confirm_ntool_le3")[0]],
                                                     [fp("confirm_ntool_ge4")[0], fp("confirm_ntool_ge4")[1] - fp("confirm_ntool_ge4")[0]]])[1]
    N["format"]["fisher_h1_fp16"] = fisher_exact([[fp("h1_free_json")[0], fp("h1_free_json")[1] - fp("h1_free_json")[0]],
                                                  [fp("h1_normal_history")[0], fp("h1_normal_history")[1] - fp("h1_normal_history")[0]]])[1]
    fig3a = ins[conf].groupby(["cond", "horizon", "model"]).fmt.agg(["sum", "count"])
    # identical-input first responses of error_restart
    e1r = resp[(resp.cond == "error_restart") & (resp.step == 1)].copy()
    e1r["task"] = runs.loc[e1r.ri, "task_id"].values
    piv = e1r.pivot_table(index=["task", "error_kind"], columns="model", values=["intended", "fmt"], aggfunc="first")
    same = piv[(piv["intended"] == "inspect_state").all(axis=1)]["fmt"]
    b = int(((same.FP16 == True) & (same.Q4_K_M == False)).sum())
    c = int(((same.FP16 == False) & (same.Q4_K_M == True)).sum())
    N["format"]["identical_input"] = dict(n=len(same), fails={m: int(same[m].sum()) for m in ORDER},
                                          fp16_q8_same=int((same.FP16 & same.Q8_0).sum()), mcnemar_fp16_q4=mcnemar(b, c))
    rs = resp.sort_values(["ri", "step"]).copy()
    rs["prev_fmt"] = rs.groupby("ri").fmt.shift(1)
    rs["prev_content"] = rs.groupby("ri").content.shift(1)
    after = rs[rs.prev_fmt == True]
    fp_rep = after[(after.model == "FP16") & after.fmt]
    exh = runs[runs.termination == "budget_exhausted"]
    last = rs.groupby("ri").tail(1).set_index("ri")
    lat = resp.groupby(["model", "fmt"]).latency.sum().unstack()
    keyrun = runs.set_index(["cond", "task_id", "error_kind", "model"]).any_fmt.unstack("model")
    b = int((keyrun.FP16 & ~keyrun.Q8_0).sum())
    c = int((~keyrun.FP16 & keyrun.Q8_0).sum())
    N["format"].update(
        persistence={m: pct(after[after.model == m].fmt.sum(), (after.model == m).sum()) for m in ORDER},
        fp16_identical_repeat=pct((fp_rep.content == fp_rep.prev_content).sum(), len(fp_rep)),
        budget_exhausted=len(exh), exhausted_ending_fmt=int(last.loc[exh.index].fmt.sum()),
        reject_latency_share={m: pct(lat.loc[m, True], lat.loc[m, True] + lat.loc[m, False], 0) for m in ORDER},
        runs_with_fmt={m: int(keyrun[m].sum()) for m in ORDER}, mcnemar_runs_fp16_q8=mcnemar(b, c),
        error_restart_first_fmt={m: int(e1r[e1r.model == m].fmt.sum()) for m in ORDER},
        err_h3_first_fmt={m: int(e1r[(e1r.model == m) & (e1r.horizon == 3)].fmt.sum()) for m in ORDER},
    )

    # ---------------------------------------------------------------- post-inspect behaviour (new)
    pi = resp[(resp.prev == "tool:inspect_state") & resp.before_first_fmt].copy()
    pi["run_success"] = runs.loc[pi.ri, "success"].values
    kind = lambda r: ("finish" if r.atype == "finish" else "format" if r.fmt else
                      "repair" if str(r.intended).startswith(("confirm", "discard", "prepare")) else "other")
    pi["next"] = pi.apply(kind, axis=1)
    failed = pi[~pi.run_success]
    rate_fail = {m: failed[failed.model == m].next.value_counts(normalize=True).to_dict() for m in ORDER}
    n_fail_fmt = runs[runs.any_fmt & ~runs.success].groupby("model").size().to_dict()
    N["post_inspect"] = dict(
        all={m: {k: round(v, 3) for k, v in pi[pi.model == m].next.value_counts(normalize=True).items()} | {"n": int((pi.model == m).sum())} for m in ORDER},
        failed_runs={m: {k: round(v, 3) for k, v in rate_fail[m].items()} | {"n": int((failed.model == m).sum())} for m in ORDER},
        failed_runs_with_fmt=n_fail_fmt,
        est_missed_repairs={m: round(n_fail_fmt[m] * rate_fail[m].get("repair", 0.0), 1) for m in ORDER},
    )

    # ---------------------------------------------------------------- completion, repair, audit
    fin = resp[resp.atype == "finish"]
    neg = re.compile(r"\b(not|doesn't|does not|fail\w*|incorrect|wrong|unable|cannot|can't|error|violat\w*|issue|problem|mismatch)\b", re.I)

    def inspected_then_finished(r):
        seen = False
        for e in r.events:
            a = e.get("action") or {}
            if a.get("name") == "inspect_state" and e.get("tool_executed"):
                seen = True
            if a.get("type") == "finish":
                return seen
        return False
    itf = runs[runs.apply(inspected_then_finished, axis=1)]
    state_change = lambda r: sum(1 for e in r.events if e.get("tool_executed")
                                 and (e.get("action") or {}).get("name", "").startswith(("confirm", "discard")))
    e3 = runs[(runs.cond == "error_restart") & (runs.horizon == 3)]
    acted = runs[runs.cond.isin(RESTART)].copy()
    acted["acted"] = acted.apply(lambda r: state_change(r) > 0, axis=1)
    holm_rows = []
    for (m, k, h), g in acted.groupby(["model", "error_kind", "horizon"]):
        p = g.pivot_table(index="scen", columns="cond", values="success", aggfunc="first")
        bb = int(((p.error_restart == 1) & (p.signaled_error_restart == 0)).sum())
        cc = int(((p.error_restart == 0) & (p.signaled_error_restart == 1)).sum())
        holm_rows.append(dict(model=m, kind=k, h=int(h), b=bb, c=cc, p=mcnemar(bb, cc)))
    hr = pd.DataFrame(holm_rows)
    hr["holm"] = multipletests(hr.p, method="holm")[1]
    sig_fin = runs[(runs.cond == "signaled_error_restart") & (runs.termination == "finished")]
    claims = sig_fin.final_answer.str.contains(r"satisf|meets the request|correct result|complete", case=False, regex=True)
    N["completion"] = dict(
        finish_responses=len(fin), finish_by_model=fin.groupby("model").size().to_dict(),
        negative_keyword=int(fin.content.str.contains(neg).sum()),
        echo_only_one_record={m: pct(fin[fin.model == m].content.str.contains("only one record", case=False).sum(), (fin.model == m).sum(), 0) for m in ORDER},
        inspect_then_finish={m: f"{int(itf[itf.model == m].success.sum())}/{int((itf.model == m).sum())}" for m in ORDER},
        err_h3_state_changes={m: int(e3[e3.model == m].apply(state_change, axis=1).sum()) for m in ORDER},
        err_h3_no_fmt_runs={m: int((~e3[e3.model == m].any_fmt).sum()) for m in ORDER},
        err_h3_no_fmt_finished_without_change={m: int(((~g.any_fmt) & (g.termination == "finished") & (g.apply(state_change, axis=1) == 0)).sum()) for m, g in e3.groupby("model")},
        acted_rate={f"{m}|h{h}|{k}|{c}": pct(g.acted.sum(), len(g)) for (m, h, k, c), g in acted.groupby(["model", "horizon", "error_kind", "cond"])},
        acted_success={f"{m}|h{h}|{c}": f"{int(g[g.acted].success.sum())}/{int(g.acted.sum())}" for (m, h, c), g in acted.groupby(["model", "horizon", "cond"])},
        signal_mcnemar=hr.round(5).to_dict("records"),
        signaled_finish_claims={m: f"{int(claims[sig_fin.model == m].sum())}/{int((sig_fin.model == m).sum())}" for m in ORDER},
        q4_sig_h3=dict(acted=pct(acted[(acted.model == "Q4_K_M") & (acted.horizon == 3) & (acted.cond == "signaled_error_restart")].acted.sum(), 120),
                       acted_final=acted[(acted.model == "Q4_K_M") & (acted.horizon == 3) & (acted.cond == "signaled_error_restart") & acted.acted].final_class.value_counts().to_dict()),
    )

    # ---------------------------------------------------------------- CLEAN first action and penalties
    nr = runs[runs.cond == "normal_restart"]
    tasks = {t["id"]: t for t in json.load(open(os.path.join(repo, "tasks", "tasks.json"), encoding="utf-8"))}

    def gold_match(r):
        t = tasks[r.task_id]
        gold = t["gold_actions"][t["checkpoint_step"]]
        a = r.events[0].get("action") or {}
        return a.get("type") == "tool_call" and a["name"] == gold["tool"] and a.get("arguments") == gold["arguments"]
    N["clean_first_action"] = dict(
        gold_match={f"{m}|h{h}": pct(g.apply(gold_match, axis=1).sum(), len(g)) for (m, h), g in nr.groupby(["model", "horizon"])},
        executable={f"{m}|h{h}": pct(g.apply(lambda r: bool(r.events[0].get("tool_executed")) or (r.events[0].get("action") or {}).get("type") == "finish", axis=1).sum(), len(g)) for (m, h), g in nr.groupby(["model", "horizon"])},
    )
    pen = pd.read_csv(os.path.join(repo, "logs", "main", "summary_penalty.csv"))
    inter = pd.read_csv(os.path.join(repo, "logs", "main", "summary_interaction.csv"))
    N["penalty_harness"] = pen[["horizon", "error_condition", "model_label", "reference_recovery_penalty", "model_recovery_penalty",
                                "penalty_difference", "difference_ci95", "sign_flip_p", "sign_flip_holm_p"]].to_dict("records")
    N["interaction_harness"] = inter[["condition", "short_horizon", "long_horizon", "model_label", "interaction",
                                      "interaction_ci95", "sign_flip_p", "sign_flip_holm_p"]].to_dict("records")
    # own penalty point estimates for the cross-check
    own = {}
    for h in (1, 3):
        for k in ("violating_candidate", "wrong_option"):
            for c in RESTART:
                L = {m: 100 * (runs[(runs.model == m) & (runs.cond == "normal_restart") & (runs.horizon == h)].success.mean()
                               - runs[(runs.model == m) & (runs.cond == c) & (runs.horizon == h) & (runs.error_kind == k)].success.mean())
                     for m in ORDER}
                own[f"h{h}|{c}|{k}"] = {m: round(v, 1) for m, v in L.items()}
    N["penalty_own"] = own

    # ---------------------------------------------------------------- behaviour, net effects
    pc = resp[resp.prev.str.startswith("tool:confirm") & ~resp.prev_injected]
    N["post_confirm"] = {m: dict(n=int((pc.model == m).sum()),
                                 inspect=pct((pc[pc.model == m].intended == "inspect_state").sum(), (pc.model == m).sum(), 0),
                                 finish=pct((pc[pc.model == m].atype == "finish").sum(), (pc.model == m).sum(), 0),
                                 format=pct(pc[pc.model == m].fmt.sum(), (pc.model == m).sum(), 0)) for m in ORDER}
    discarded = lambda r: any(e.get("tool_executed") and (e.get("action") or {}).get("name") == "discard_saved_result" for e in r.events)
    q4f1 = runs[(runs.model == "Q4_K_M") & (runs.cond == "free") & (runs.horizon == 1)].copy()
    q4f1["disc"] = q4f1.apply(discarded, axis=1)
    q4f1["fc"] = q4f1.apply(lambda r: classify_saved(r, confirms(r)[0][1]["saved_result"]) if confirms(r) else None, axis=1)
    multi = resp[resp.model_error == "multiple_tool_calls"]
    mruns = runs.loc[multi.ri.unique()]
    ncalls = [next(e.get("attempted_tool_call_count") for e in r.events if e.get("model_error") == "multiple_tool_calls")
              for _, r in mruns.iterrows()]
    N["behaviour"] = dict(
        q4_free_h1_discard=f"{int(q4f1.disc.sum())}/60",
        q4_discard_by_first=q4f1[q4f1.fc.notna()].groupby("fc").disc.agg(lambda s: f"{int(s.sum())}/{len(s)}").to_dict(),
        free_h1_discard_runs=runs[(runs.cond == "free") & (runs.horizon == 1)].groupby("model").apply(lambda g: int(g.apply(discarded, axis=1).sum())).to_dict(),
        parallel=dict(n=len(multi), where=multi.groupby(["model", "cond", "horizon", "step"]).size().rename(lambda x: str(x)).to_dict(),
                      call_counts=dict(Counter(ncalls)), final_empty=int((mruns.final_class == "empty").sum())),
        empty_final=runs.groupby("model").final_class.apply(lambda s: int((s == "empty").sum())).to_dict(),
        tool_errors=resp[resp.tool_error.notna()].groupby(["model", "tool_error"]).size().rename(lambda x: str(x)).to_dict(),
        domain_success={m: {d: pct(g.success.sum(), len(g)) for d, g in runs[runs.model == m].groupby("domain")} for m in ORDER},
    )
    P = runs.pivot_table(index=["cond", "task_id", "error_kind", "scen"], columns="model", values="success", aggfunc="first").astype(int).reset_index()
    net = {}
    for m in ("Q8_0", "Q4_K_M"):
        for c in ["all"] + CONDS:
            sub = P if c == "all" else P[P.cond == c]
            d = (sub[m] - sub["FP16"]).values
            net[f"{m}|{c}"] = [round(100 * d.mean(), 1)] + list(np.round(100 * cluster_boot_mean(d, sub.scen, 3000, rng), 1)) + [len(sub)]
    N["net_effect"] = net

    def aseq(r):
        out = []
        for e in r.events:
            a = e.get("action") or {}
            out.append((a.get("type"), a.get("name"), json.dumps(a.get("arguments"), sort_keys=True)) if a else ("ERR", e.get("model_error")))
        return tuple(out)
    runs["aseq"] = runs.apply(aseq, axis=1)
    S = runs.set_index(["cond", "task_id", "error_kind", "model"]).aseq.unstack("model")
    agree = {}
    for m in ("Q8_0", "Q4_K_M"):
        a, b2 = P["FP16"], P[m]
        agree[m] = dict(agree=pct((a == b2).sum(), len(a)), phi=round(float(np.corrcoef(a, b2)[0, 1]), 2),
                        same_action_sequence=pct((S.FP16 == S[m]).sum(), len(S)))
    N["agreement"] = agree

    # ---------------------------------------------------------------- efficiency
    man = json.load(open(os.path.join(repo, "models", "manifest.json")))
    sizes = {v["quantization"]: v["size_mb"] for v in man["variants"]}
    g = resp.groupby("model")
    fixed = resp[(resp.cond != "free") & (resp.step == 1)]
    eff = {}
    for m, q in zip(ORDER, ["F16", "Q8_0", "Q4_K_M"]):
        eff[m] = dict(
            file_mib=round(sizes[q]), gpu_mb=float(runs[runs.model == m].peak_gpu.median()),
            gen_ms_per_tok=round(float((resp[resp.model == m].pred_ms / resp[resp.model == m].pred_n).median()), 2),
            prompt_ms=round(float(g.prompt_ms.mean()[m])), server_ms=round(float(g.server_ms.mean()[m])),
            prompt_share=round(float(g.prompt_ms.sum()[m] / g.server_ms.sum()[m]), 3),
            fixed_prompt_ms=round(float(fixed[fixed.model == m].prompt_ms.mean()), 1),
            fixed_prompt_tokens=round(float(fixed[fixed.model == m].prompt_n.mean())),
            fixed_n=int((fixed.model == m).sum()), responses=int((resp.model == m).sum()),
        )
    N["efficiency"] = eff
    N["kv_cache_mib"] = 28 * 2 * 8 * 128 * 2 * 8192 / 2 ** 20

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(N), f, ensure_ascii=False, indent=1)
    make_figures(out, runs, h1, fig3a, after, resp, N)
    return N


def _clean(o):
    """JSON-safe copy: string keys, Python scalars."""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _clean(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ---------------------------------------------------------------------- figures
def make_figures(out, runs, h1, fig3a, after, resp, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK-Regular" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "pdf.fonttype": 3, "legend.frameon": False})
    w = 0.26

    # Figure 1 ------------------------------------------------------------
    cells = [("free", 1, None, "free\nh1"), ("free", 3, None, "free\nh3"), ("free", 5, None, "free\nh5"),
             ("normal_restart", 1, None, "normal\nh1"), ("normal_restart", 3, None, "normal\nh3"),
             ("error_restart", 1, "violating_candidate", "error\nh1·위반"), ("error_restart", 1, "wrong_option", "error\nh1·옵션"),
             ("error_restart", 3, None, "error\nh3"), ("signaled_error_restart", 1, "violating_candidate", "signal\nh1·위반"),
             ("signaled_error_restart", 1, "wrong_option", "signal\nh1·옵션"), ("signaled_error_restart", 3, None, "signal\nh3")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (c, h, k, _) in enumerate(cells):
        for j, m in enumerate(ORDER):
            sub = runs[(runs.model == m) & (runs.cond == c) & (runs.horizon == h)]
            if k:
                sub = sub[sub.error_kind == k]
            ax.bar(i + (j - 1) * w, 100 * sub.success.mean(), w, color=COLORS[m], label=m if i == 0 else None)
        if h == 1:
            ax.hlines(100 / 3, i - 0.42, i + 0.42, colors="k", linestyles=":", linewidth=0.9)
        else:
            ax.hlines(100 / 6, i - 0.42, i + 0.42, colors="k", linestyles=":", linewidth=0.9)
            ax.hlines(100 / 3, i - 0.42, i + 0.42, colors="#777777", linestyles="--", linewidth=0.8)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([x[3] for x in cells], fontsize=7)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 60)
    ax.legend(ncol=3, loc="upper left", fontsize=7)
    ax.text(0.99, 0.97, "점선: 단회 균등 조합 선택   파선: 옵션이 맞을 때 후보 균등 선택", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_success.pdf"))
    plt.close(fig)

    # Figure 2 ------------------------------------------------------------
    f1 = h1[h1.cond == "free"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for j, m in enumerate(ORDER):
        g = f1[f1.model == m]
        firsts = [x[0] for x in g.conf_pos if x]
        axes[0].bar(np.arange(3) + (j - 1) * w, [100 * firsts.count(p) / len(firsts) for p in (1, 2, 3)], w, color=COLORS[m], label=m)
        axes[1].bar(np.arange(3) + (j - 1) * w, [100 * g[g.correct_pos == p].success.mean() for p in (1, 2, 3)], w, color=COLORS[m])
    for a, t, xl, yl in ((axes[0], "(a) 선택 위치", "처음 확정한 draft의 위치", "비율 (%)"),
                         (axes[1], "(b) 정답 위치별 성공률", "정답 draft의 위치", "성공률 (%)")):
        a.set_xticks(range(3))
        a.set_xticklabels(["첫째", "둘째", "셋째"])
        a.set_title(t, loc="left", fontsize=8)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.set_ylim(0, 105)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_position.pdf"))
    plt.close(fig)

    # Figure 3 ------------------------------------------------------------
    order = [("free", 1), ("free", 3), ("free", 5), ("normal_restart", 1), ("normal_restart", 3),
             ("error_restart", 1), ("error_restart", 3), ("signaled_error_restart", 1), ("signaled_error_restart", 3)]
    names = {"free": "free", "normal_restart": "normal", "error_restart": "error", "signaled_error_restart": "signal"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [3.2, 1]})
    for j, m in enumerate(ORDER):
        vals = []
        for c, h in order:
            try:
                s, n = fig3a.loc[(c, h, m)]
            except KeyError:
                s, n = 0, 0
            vals.append(100 * s / n if n >= 5 else np.nan)
        axes[0].bar(np.arange(len(order)) + (j - 1) * w, vals, w, color=COLORS[m], label=m)
        tot = resp[resp.model == m].fmt.mean() * 100
        pers = after[after.model == m].fmt.mean() * 100
        axes[1].bar(np.arange(2) + (j - 1) * w, [tot, pers], w, color=COLORS[m])
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([f"{names[c]}\nh{h}" for c, h in order], fontsize=7)
    axes[0].set_title("(a) 확정 결과 직후 inspect_state 시도", loc="left", fontsize=8)
    axes[0].set_ylabel("형식 실패율 (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(["전체 응답", "직전이\n형식 오류"], fontsize=7)
    axes[1].set_title("(b) 지속성", loc="left", fontsize=8)
    axes[1].set_ylabel("형식 오류 비율 (%)")
    axes[1].set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_format.pdf"))
    plt.close(fig)

    # Figure 4 ------------------------------------------------------------
    rows = ["all", "free", "normal_restart", "error_restart", "signaled_error_restart"]
    labels = {"all": "전체", "free": "free", "normal_restart": "normal_restart", "error_restart": "error_restart",
              "signaled_error_restart": "signaled"}
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    for i, r in enumerate(rows[::-1]):
        for j, m in enumerate(("Q8_0", "Q4_K_M")):
            e, lo, hi, _ = N["net_effect"][f"{m}|{r}"]
            y = i + (0.13 if j == 0 else -0.13)
            ax.plot([lo, hi], [y, y], color=COLORS[m], linewidth=2)
            ax.plot(e, y, "o", color=COLORS[m], markersize=4, label=m if i == 0 else None)
    ax.axvline(0, color="k", linestyle=":", linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[r] for r in rows[::-1]])
    ax.grid(axis="x", color="#e6e6e6")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("FP16 대비 성공률 차이 (%p, 95% 구간)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_net.pdf"))
    plt.close(fig)

    # Figure 5 ------------------------------------------------------------
    eff = N["efficiency"]
    metrics = [("file_mib", "모델 파일"), ("gpu_mb", "GPU 메모리"), ("gen_ms_per_tok", "토큰당 생성"),
               ("server_ms", "응답당 서버 시간"), ("prompt_ms", "응답당 프롬프트 처리"), ("fixed_prompt_ms", "고정 입력\n프롬프트 처리")]
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for j, m in enumerate(ORDER):
        ax.bar(np.arange(len(metrics)) + (j - 1) * w, [100 * eff[m][k] / eff["FP16"][k] for k, _ in metrics], w, color=COLORS[m], label=m)
    ax.axhline(100, color="k", linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([x[1] for x in metrics], fontsize=7)
    ax.set_ylabel("FP16 대비 (%)")
    ax.set_ylim(0, 120)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_efficiency.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--logit-boot", type=int, default=2000, help="bootstrap resamples for Table 2 (0 = skip)")
    args = ap.parse_args()
    analyze(args.repo, args.out, args.logit_boot)
    print(f"wrote {args.out}/numbers.json and figures")
