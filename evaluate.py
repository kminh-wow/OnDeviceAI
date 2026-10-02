"""Scores one agent run against the task's gold trajectory.

first_error_type:
  format          tool call could not be parsed (bad JSON / leaked <tool_call> text)
  tool_selection  called a tool that is not (or no longer) needed, or a non-existent tool
  argument        right tool, wrong argument taken from the query (date, city, people ...)
  observation     right tool, wrong argument that should come from an earlier tool result
                  (place_id, free-slot time, weather-dependent choice ...) -> state tracking
  skipped_step    same as observation, but the tool that provides that value was never called
  premature_stop  answered before finishing the required calls
  max_steps       ran out of turns
  over_execution  all required calls were already done, but the agent kept acting (termination failure)
  final_answer    all calls correct, but the final answer misses the required fact

Paper-style metrics (for comparison with NESTFUL / Granite-FC / BFCL):
  f1_func         F1 over called function names vs gold names (multiset)            [Granite, NESTFUL]
  f1_param        F1 over (tool, argument, value) triples on gold's key arguments   [Granite, NESTFUL]
  partial_match   = step_accuracy, share of gold calls reproduced exactly           [NESTFUL partial seq. acc.]
  full_match      all gold calls reproduced, no wrong extra calls                   [NESTFUL full seq. acc.]
  success         full_match + correct final state + required answer (= win rate with state check)
"""
from collections import Counter
from tools_env import DEFAULTS, TOOL_SPECS


def normalize_args(name, args):
    if name not in TOOL_SPECS or not isinstance(args, dict):
        return None
    spec = TOOL_SPECS[name]
    return {k: f(args[k]) for k, f in {**spec["required"], **spec["optional"]}.items()
            if args.get(k) not in (None, "")}


def _mismatch(gold, pargs):
    d = DEFAULTS.get(gold["name"], {})
    return [k for k, v in gold["args"].items() if pargs.get(k, d.get(k)) != v]


def _cart(items):
    agg = {}
    for c in items:
        agg[c["product_id"]] = agg.get(c["product_id"], 0) + c["quantity"]
    return [{"product_id": k, "quantity": v} for k, v in agg.items()]


def state_ok(expected, actual):
    for key in ("reservations", "events", "cart"):
        exp, act = expected.get(key, []), actual.get(key, [])
        if key == "cart":
            exp, act = _cart(exp), _cart(act)
        if len(exp) != len(act):
            return False
        used = set()
        for e in exp:
            j = next((j for j, a in enumerate(act) if j not in used
                      and all(a.get(k) == v for k, v in e.items())), None)
            if j is None:
                return False
            used.add(j)
    return True


def answer_ok(groups, answer):
    if not groups:
        return True
    a = (answer or "").lower().replace(",", "")
    return all(any(v.lower().replace(",", "") in a for v in g) for g in groups)


def evaluate(task, rec):
    gold = task["gold"]
    remaining, matched = list(range(len(gold))), []
    first_step, first_type, redundant = None, None, 0

    for idx, c in enumerate(rec["calls"], 1):
        etype = None
        if c["status"] == "invalid":
            if c.get("format_error") or c.get("args") is None:
                etype = "format"
            elif c["name"] not in TOOL_SPECS:
                etype = "tool_selection"
            else:
                etype = "argument"
        else:
            pargs = normalize_args(c["name"], c["args"]) or {}
            j = next((j for j in remaining if gold[j]["name"] == c["name"]
                      and not _mismatch(gold[j], pargs)), None)
            if j is not None:
                remaining.remove(j)
                matched.append(j)
            elif any(gold[j]["name"] == c["name"] and not _mismatch(gold[j], pargs) for j in matched):
                redundant += 1  # harmless repeat of an already-correct call
            else:
                cands = [gold[j] for j in remaining if gold[j]["name"] == c["name"]]
                if not cands:
                    etype = "tool_selection"
                else:
                    jb = min((j for j in remaining if gold[j]["name"] == c["name"]),
                             key=lambda j: len(_mismatch(gold[j], pargs)))
                    mm = _mismatch(gold[jb], pargs)
                    skipped = [j for j in remaining if j < jb
                               and gold[j]["name"].startswith(("get_", "search_"))]
                    if set(mm) <= set(gold[jb]["derived"]):
                        etype = "skipped_step" if skipped else "observation"
                    else:
                        etype = "argument"
        if etype and not remaining:  # all required calls already done, but kept acting
            etype = "over_execution"
        if etype and first_step is None:
            first_step, first_type = idx, etype

    s_ok = state_ok(task["expected_state"], rec["state"])
    a_ok = answer_ok(task["answer"], rec.get("final_answer"))
    n_calls = len(rec["calls"])
    if first_step is None:
        if remaining or not rec.get("terminated"):
            first_step = n_calls + 1
            first_type = "max_steps" if not rec.get("terminated") else "premature_stop"
        elif not s_ok:
            first_step, first_type = n_calls + 1, "state"
        elif not a_ok:
            first_step, first_type = n_calls + 1, "final_answer"

    success = not remaining and s_ok and a_ok and bool(rec.get("terminated"))

    # ---- paper-style metrics
    valid_calls = [c for c in rec["calls"] if c["status"] != "invalid" and c["name"] in TOOL_SPECS]
    pn, gn = Counter(c["name"] for c in valid_calls), Counter(g["name"] for g in gold)
    keys = {}
    for g in gold:
        keys.setdefault(g["name"], set()).update(g["args"])
    pp = Counter()
    for c in valid_calls:
        na = normalize_args(c["name"], c["args"]) or {}
        d = DEFAULTS.get(c["name"], {})
        for k in keys.get(c["name"], ()):
            pp[(c["name"], k, str(na.get(k, d.get(k))))] += 1
    gp = Counter((g["name"], k, str(v)) for g in gold for k, v in g["args"].items())

    def f1(p, g):
        ov = sum((p & g).values())
        pr, rc = (ov / sum(p.values()) if p else 0.0), (ov / sum(g.values()) if g else 1.0)
        return 2 * pr * rc / (pr + rc) if pr + rc else (1.0 if not g and not p else 0.0)

    full_match = not remaining and (len(valid_calls) - redundant) == len(gold) \
        and len(valid_calls) == len(rec["calls"])
    valid = [c for c in rec["calls"] if c["status"] != "invalid"]
    return {
        "success": success,
        "step_accuracy": len(matched) / len(gold),
        "valid_call_rate": (len(valid) / n_calls) if n_calls else None,
        "n_calls": n_calls,
        "n_gold": len(gold),
        "redundant_calls": redundant,
        "state_ok": s_ok,
        "answer_ok": a_ok,
        "first_error_step": first_step,
        "first_error_type": first_type,
        "had_error": first_step is not None,
        "recovered": success and first_step is not None,
        "f1_func": f1(pn, gn),
        "f1_param": f1(pp, gp),
        "partial_match": len(matched) / len(gold),
        "full_match": full_match,
    }
