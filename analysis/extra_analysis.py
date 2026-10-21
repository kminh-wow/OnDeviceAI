"""State-matched follow-up analyses (공유 문서 핵심 발견 11-13, 4번·10번 보강).

- div: first divergence point between FP16 and each quantized model (identical inputs up to that response)
- pos: h1 trials where position strategies cannot reach the correct draft
- con: conditional logit with precision interactions for constraint following (scenario bootstrap)
- fmt: response-level logistic model of format failures
- tok: per-token prompt/generation cost and time per success

Usage (from the repository root):
    python analysis/extra_analysis.py --repo . --out analysis/out
Reuses the loaders in analysis/reanalyze.py. Writes analysis/out/extra_<parts>.json.
"""
import argparse, importlib.util, json, os, sys, warnings
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")
SEED = 20261018
M = ["FP16", "Q8_0", "Q4_K_M"]


def load_ra(repo):
    spec = importlib.util.spec_from_file_location("ra", os.path.join(repo, "analysis", "reanalyze.py"))
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    return ra


def mcnemar(b, c):
    return binomtest(int(b), int(b + c)).pvalue if b + c else 1.0


# ------------------------------------------------------------------ step annotation
def annotate_steps(ra, r):
    """Per response: raw signature, action category, state class before the step."""
    # initial saved state
    saved = None
    for m in r.initial_messages:
        if m["role"] == "tool" and m.get("name", "").startswith("confirm"):
            res = json.loads(m["content"])
            if res.get("ok"):
                saved = res["saved_result"]
    drafts = {}
    for m in r.initial_messages:
        if m["role"] == "tool" and m.get("name", "").startswith("prepare"):
            res = json.loads(m["content"])
            drafts[res["draft_id"]] = res["proposed_result"]
    if r.ctx == "user_message":
        for tool, res in ra.provided_results(r):
            if tool.startswith("prepare"):
                drafts[res["draft_id"]] = res["proposed_result"]
    out = []
    for e in r.events:
        state = "empty" if saved is None else ra.classify_saved(r, saved)
        msg = ra.raw_message(e)
        sig = ((msg.get("content") or "").strip(),
               tuple((c["function"]["name"], c["function"]["arguments"]) for c in (msg.get("tool_calls") or [])))
        a = e.get("action") or {}
        res = e.get("tool_result") or {}
        if e.get("error") == "model_output_error":
            cat = "format"
        elif a.get("type") == "finish":
            cat = "finish_ok" if state == "correct" else "finish_wrong"
        elif e.get("tool_executed"):
            n = a.get("name", "")
            if not res.get("ok", False):
                cat = "tool_error"
            elif n.startswith("confirm"):
                cat = "confirm_" + ra.classify_saved(r, res["saved_result"])
                saved = res["saved_result"]
            elif n == "discard_saved_result":
                cat = "discard_correct" if state == "correct" else "discard_wrong"
                saved = None
            elif n == "inspect_state":
                cat = "inspect"
            elif n.startswith("prepare"):
                drafts[res["draft_id"]] = res["proposed_result"]
                cat = "prepare_" + ra.classify_saved(r, res["proposed_result"])
            elif "candidate_id" in res:
                cat = "detail_" + r.lab.get(res["candidate_id"], "?")
            else:
                cat = "lookup"
        else:
            cat = "other"
        out.append(dict(sig=sig, cat=cat, state=state))
    return out


ERR = {"format", "finish_wrong", "confirm_violating", "confirm_option_error", "confirm_other_eligible",
       "discard_correct", "tool_error"}
GOOD = {"confirm_correct", "finish_ok", "discard_wrong"}


def divergence(ra, runs, N):
    steps = {i: annotate_steps(ra, r) for i, r in runs.iterrows()}
    key = runs.set_index(["cond", "task_id", "error_kind", "model"])
    idx = {k: i for k, i in zip(key.index, runs.index)}
    res = {}
    for q in ("Q8_0", "Q4_K_M"):
        rows = []
        for (c, t, k, m), i in idx.items():
            if m != "FP16":
                continue
            j = idx[(c, t, k, q)]
            a, b = steps[i], steps[j]
            n = next((s for s in range(min(len(a), len(b))) if a[s]["sig"] != b[s]["sig"]), None)
            if n is None:
                if len(a) == len(b):
                    rows.append(dict(cond=c, diverged=False))
                    continue
                n = min(len(a), len(b))
            ca = a[n]["cat"] if n < len(a) else "budget_end"
            cb = b[n]["cat"] if n < len(b) else "budget_end"
            rows.append(dict(cond=c, diverged=True, step=n + 1, cat_fp=ca, cat_q=cb, state=a[min(n, len(a) - 1)]["state"],
                             cosmetic=(ca == cb), succ_fp=bool(runs.loc[i, "success"]), succ_q=bool(runs.loc[j, "success"])))
        d = pd.DataFrame(rows)
        dv = d[d.diverged.astype(bool)].copy()
        for col in ("cosmetic", "succ_fp", "succ_q"):
            dv[col] = dv[col].astype(bool)
        real = dv[~dv.cosmetic]

        def asym(df, exclude_format=False):
            errset = ERR - ({"format"} if exclude_format else set())
            ea = df.cat_fp.isin(errset)
            eb = df.cat_q.isin(errset)
            if exclude_format:  # drop divergences where either side is a format error
                keep = ~(df.cat_fp.eq("format") | df.cat_q.eq("format"))
                ea, eb, df = ea[keep], eb[keep], df[keep]
            return dict(n=int(len(df)), only_fp16_error=int((ea & ~eb).sum()), only_q_error=int((~ea & eb).sum()),
                        both_error=int((ea & eb).sum()), neither=int((~ea & ~eb).sum()),
                        p=mcnemar(int((ea & ~eb).sum()), int((~ea & eb).sum())))
        ga, gb = real.cat_fp.isin(GOOD), real.cat_q.isin(GOOD)
        res[q] = dict(
            pairs=len(d), diverged=int(dv.shape[0]), cosmetic=int(dv.cosmetic.sum()), real=int(len(real)),
            step_mean=round(float(dv.step.mean()), 2), step_median=float(dv.step.median()),
            diverged_by_cond=dv.groupby("cond").size().to_dict(),
            real_by_cond=real.groupby("cond").size().to_dict(),
            cross_top=[(f"{x}|{y}", int(n)) for (x, y), n in Counter(zip(real.cat_fp, real.cat_q)).most_common(14)],
            error_asym=asym(real), error_asym_no_format=asym(real, True),
            good_asym=dict(only_fp16_good=int((ga & ~gb).sum()), only_q_good=int((~ga & gb).sum()),
                           p=mcnemar(int((ga & ~gb).sum()), int((~ga & gb).sum()))),
            outcome=dict(fp16_only=int((real.succ_fp & ~real.succ_q).sum()), q_only=int((~real.succ_fp & real.succ_q).sum()),
                         both=int((real.succ_fp & real.succ_q).sum()), neither=int((~real.succ_fp & ~real.succ_q).sum())),
            cats_fp=real.cat_fp.value_counts().to_dict(), cats_q=real.cat_q.value_counts().to_dict(),
        )
        # per condition asymmetry
        res[q]["error_asym_by_cond"] = {c: asym(g) for c, g in real.groupby("cond")}
        res[q]["error_asym_no_format_by_cond"] = {c: asym(g, True) for c, g in real.groupby("cond")}
    N["divergence"] = res


# ------------------------------------------------------------------ position-incongruent
def position(ra, runs, N):
    h1 = runs[runs.horizon == 1].copy()
    dd = h1.apply(ra.h1_drafts, axis=1)
    h1["ids"], h1["cls"] = dd.str[0], dd.str[1]
    h1["cpos"] = h1.cls.apply(lambda c: c.index("correct") + 1)

    def conf_list(r):
        out = []
        for e, res in ra.confirms(r):
            did = e["action"]["arguments"].get("draft_id")
            out.append((r.ids.index(did) + 1) if did in r.ids else None)
        return out
    h1["conf"] = h1.apply(conf_list, axis=1)
    out = {}
    for c in ("free", "normal_restart"):
        g = h1[(h1.cond == c) & (h1.cpos != 1)]
        g = g.assign(first_ok=g.apply(lambda r: bool(r.conf) and r.conf[0] == r.cpos, axis=1))
        piv = g.pivot_table(index="task_id", columns="model", values="first_ok", aggfunc="first")
        out[c] = dict(n=int(len(g) / 3),
                      first_confirm_correct={m: int(g[g.model == m].first_ok.sum()) for m in M},
                      success={m: int(g[g.model == m].success.sum()) for m in M},
                      first_confirm_first_slot={m: int(g[g.model == m].conf.apply(lambda x: bool(x) and x[0] == 1).sum()) for m in M},
                      mcnemar_q4_vs_fp16=mcnemar(int((piv.Q4_K_M & ~piv.FP16).sum()), int((~piv.Q4_K_M & piv.FP16).sum())),
                      mcnemar_q8_vs_fp16=mcnemar(int((piv.Q8_0 & ~piv.FP16).sum()), int((~piv.Q8_0 & piv.FP16).sum())))

    def inj_pos(r):
        for m in r.initial_messages:
            for cc in (m.get("tool_calls") or []):
                if cc["id"].startswith("injected"):
                    did = json.loads(cc["function"]["arguments"])["draft_id"]
                    return r.ids.index(did) + 1
    for c in ("error_restart", "signaled_error_restart"):
        g = h1[h1.cond == c].copy()
        g["inj"] = g.apply(inj_pos, axis=1)
        g["next_correct"] = g.apply(lambda r: r.inj < 3 and r.cls[r.inj] == "correct", axis=1)
        inc = g[~g.next_correct].copy()
        inc["re"] = inc.conf.apply(lambda x: x[0] if x else None)
        inc["first_ok"] = inc.re == inc.cpos
        inc["key"] = inc.task_id + "|" + inc.error_kind
        piv = inc.pivot_table(index="key", columns="model", values="first_ok", aggfunc="first")
        out[c] = dict(n=int(len(inc) / 3),
                      reconfirmed={m: int(inc[inc.model == m].re.notna().sum()) for m in M},
                      first_reconfirm_correct={m: int(inc[inc.model == m].first_ok.sum()) for m in M},
                      success={m: int(inc[inc.model == m].success.sum()) for m in M},
                      mcnemar_q4_vs_fp16=mcnemar(int((piv.Q4_K_M & ~piv.FP16).sum()), int((~piv.Q4_K_M & piv.FP16).sum())),
                      mcnemar_q8_vs_fp16=mcnemar(int((piv.Q8_0 & ~piv.FP16).sum()), int((~piv.Q8_0 & piv.FP16).sum())))
        # congruent trials for contrast
        con = g[g.next_correct]
        out[c]["congruent"] = dict(n=int(len(con) / 3), success={m: int(con[con.model == m].success.sum()) for m in M})
    N["position_incongruent"] = out


# ------------------------------------------------------------------ constraint following
def constraint(ra, runs, N, B):
    rng = np.random.default_rng(SEED)
    sel = runs[(runs.cond == "free") & runs.horizon.isin([3, 5])].copy()

    def first_valid(r):
        allc = ra.confirms(r, ok_only=False)
        if not allc or not allc[0][1].get("ok"):
            return None
        return allc[0][1]["saved_result"]["candidate_id"]
    sel["fv"] = sel.apply(first_valid, axis=1)
    out = {}
    for h in (3, 5):
        g = sel[sel.horizon == h]
        rows = []
        for _, r in g.iterrows():
            if not isinstance(r.fv, str):
                continue
            for pos, cnd in enumerate(r.spec["candidates"]):
                cid = cnd["candidate_id"]
                row = dict(scen=r.scen, grp=f"{r.scen}|{r.model}", y=int(cid == r.fv), p1=int(pos == 0), p2=int(pos == 1),
                           v=int(r.lab[cid] == "violating"), g=int(r.lab[cid] == "correct"))
                for m, tag in (("Q8_0", "q8"), ("Q4_K_M", "q4")):
                    for f in ("p1", "p2", "v", "g"):
                        row[f"{f}_{tag}"] = row[f] * int(r.model == m)
                rows.append(row)
        df = pd.DataFrame(rows)
        cols = ["p1", "p2", "v", "g"] + [f"{f}_{t}" for t in ("q8", "q4") for f in ("p1", "p2", "v", "g")]

        def fit(d):
            return ConditionalLogit(d.y, d[cols], groups=d.grp).fit(disp=0, method="bfgs", maxiter=1000).params.values
        est = fit(df)
        by = {k: v for k, v in df.groupby("scen")}
        keys = list(by)
        bs, bad = [], 0
        for _ in range(B):
            parts = []
            for j, k in enumerate(rng.choice(keys, len(keys), replace=True)):
                x = by[k].copy()
                x["grp"] = x["grp"] + f"#{j}"
                parts.append(x)
            try:
                p = fit(pd.concat(parts))
                if not np.all(np.isfinite(p)) or np.any(np.abs(p) > 15):
                    bad += 1
                    continue
                bs.append(p)
            except Exception:
                bad += 1
        bs = np.array(bs)
        ci = np.percentile(bs, [2.5, 97.5], axis=0) if len(bs) else np.full((2, len(cols)), np.nan)
        out[f"h{h}"] = {c: [round(float(np.exp(e)), 2), round(float(np.exp(l)), 2), round(float(np.exp(u)), 2)]
                        for c, e, l, u in zip(cols, est, ci[0], ci[1])}
        out[f"h{h}"]["excluded"] = bad
        # descriptive paired differences in violation and correct-choice rates (no valid confirm counts as 0)
        ok = lambda r: isinstance(r.fv, str)
        g2 = g.assign(viol=g.apply(lambda r: int(ok(r) and r.lab[r.fv] == "violating"), axis=1),
                      corr_=g.apply(lambda r: int(ok(r) and r.lab[r.fv] == "correct"), axis=1),
                      valid=g.apply(lambda r: int(ok(r)), axis=1))
        for metric in ("viol", "corr_"):
            piv = g2.pivot_table(index="scen", columns="model", values=metric)
            for m in ("Q8_0", "Q4_K_M"):
                d = (piv[m] - piv["FP16"]).values
                b2 = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
                out[f"h{h}"][f"{metric}_{m}_minus_fp16"] = [round(100 * d.mean(), 1)] + list(np.round(100 * np.percentile(b2, [2.5, 97.5]), 1))
        out[f"h{h}"]["rates"] = {m: dict(viol=int(g2[g2.model == m].viol.sum()), corr=int(g2[g2.model == m]["corr_"].sum()),
                                         valid=int(g2[g2.model == m].valid.sum())) for m in M}
    N["constraint"] = out


# ------------------------------------------------------------------ format error model
def format_model(ra, runs, resp, N):
    d = resp[resp.before_first_fmt].copy()

    def kind(r):
        if r.prev.startswith("tool:confirm"):
            return "confirm_injected" if r.prev_injected else "confirm_model"
        if r.prev.startswith("tool:"):
            return "tool_other"
        return {"user:audit": "audit", "user:query": "query"}.get(r.prev, "other")
    d["prev_kind"] = d.apply(kind, axis=1)
    d["fmt_i"] = d.fmt.astype(int)
    d["scen"] = runs.loc[d.ri, "scen"].values
    d["ntool_c"] = d.ntool - 4
    desc = d.groupby(["prev_kind", "model"]).fmt_i.agg(["sum", "count"])
    out = {"descriptive": {f"{k}|{m}": [int(s), int(n)] for (k, m), (s, n) in desc.iterrows()}}
    # model on responses that follow a tool result (audit/query have ~no failures)
    t = d[d.prev_kind.isin(["confirm_model", "confirm_injected", "tool_other"])].copy()
    formula = "fmt_i ~ C(model, Treatment('FP16')) * ntool_c + C(prev_kind, Treatment('confirm_model'))"
    fit = smf.glm(formula, data=t, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(t.scen)[0]})
    tab = pd.DataFrame({"OR": np.exp(fit.params), "lo": np.exp(fit.conf_int()[0]), "hi": np.exp(fit.conf_int()[1]), "p": fit.pvalues})
    out["n"] = int(len(t))
    out["coef"] = {k: [round(float(v.OR), 3), round(float(v.lo), 3), round(float(v.hi), 3), float(f"{v.p:.3g}")] for k, v in tab.iterrows()}
    # predicted failure probability after a model-executed confirm by tool-call count
    grid = pd.DataFrame([dict(model=m, ntool_c=n - 4, prev_kind="confirm_model") for m in M for n in (2, 4, 6, 8, 10)])
    grid["p"] = fit.predict(grid)
    out["predicted_confirm_model"] = {f"{r.model}|{r.ntool_c + 4}": round(float(r.p), 3) for r in grid.itertuples()}
    out["ntool_range_by_prev"] = {k: [int(g.ntool.min()), int(g.ntool.max())] for k, g in t.groupby("prev_kind")}
    N["format_model"] = out


# ------------------------------------------------------------------ token cost
def token_cost(ra, runs, resp, N):
    out = {}
    for m in M:
        g = resp[(resp.model == m) & resp.prompt_ms.notna()]
        pr = sm.OLS(g.prompt_ms, sm.add_constant(g.prompt_n)).fit()
        ge = sm.OLS(g.pred_ms, sm.add_constant(g.pred_n)).fit()
        per_run = resp[resp.model == m].groupby("ri")[["prompt_ms", "pred_ms", "latency"]].sum()
        rr = runs[runs.model == m]
        out[m] = dict(
            prompt_ms_per_1k_tok=round(float(pr.params["prompt_n"] * 1000), 1),
            prompt_ms_per_1k_ci=[round(float(x * 1000), 1) for x in pr.conf_int().loc["prompt_n"]],
            prompt_intercept_ms=round(float(pr.params["const"]), 1),
            gen_ms_per_tok=round(float(ge.params["pred_n"]), 3),
            gen_ms_per_tok_ci=[round(float(x), 3) for x in ge.conf_int().loc["pred_n"]],
            gen_intercept_ms=round(float(ge.params["const"]), 1),
            r2_prompt=round(float(pr.rsquared), 3), r2_gen=round(float(ge.rsquared), 3),
            mean_prompt_tokens=round(float(g.prompt_n.mean())), mean_gen_tokens=round(float(g.pred_n.mean()), 1),
            responses_per_run=round(float(rr.n_resp.mean()), 2),
            server_s_per_run=round(float((per_run.prompt_ms + per_run.pred_ms).mean() / 1000), 2),
            wall_s_per_run=round(float(per_run.latency.mean() / 1000), 2),
            server_s_per_success=round(float((per_run.prompt_ms + per_run.pred_ms).sum() / 1000 / rr.success.sum()), 1),
            successes=int(rr.success.sum()),
        )
    N["token_cost"] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for the constraint model")
    ap.add_argument("--only", default="", help="comma list of div,pos,con,fmt,tok (default: all)")
    args = ap.parse_args()
    ra = load_ra(args.repo)
    runs = ra.load_runs(args.repo)
    resp = ra.build_responses(runs)
    runs["final_class"] = runs.apply(ra.final_class, axis=1)
    N = {}
    todo = args.only.split(",") if args.only else ["div", "pos", "con", "fmt", "tok"]
    if "div" in todo: divergence(ra, runs, N)
    if "pos" in todo: position(ra, runs, N)
    if "fmt" in todo: format_model(ra, runs, resp, N)
    if "tok" in todo: token_cost(ra, runs, resp, N)
    if "con" in todo: constraint(ra, runs, N, args.boot)
    os.makedirs(args.out, exist_ok=True)
    name = "extra_" + ("_".join(todo)) + ".json"
    json.dump(ra._clean(N), open(os.path.join(args.out, name), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(ra._clean(N), ensure_ascii=False, indent=1))
