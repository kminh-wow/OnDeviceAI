"""Scores one agent run against the task's gold trajectory.

first_error_type (first deviation from the gold trajectory, in call order):
  format          tool call could not be parsed (bad JSON / leaked tool-call text)
  tool_selection  called a non-existent tool, or a WRITE tool (book_place, create_event, add_to_cart)
                  that the task does not need
  extra_lookup    called a READ tool (get_*, search_*) that the task does not need (harmless for state)
  argument        right tool, wrong argument taken from the query (date, city, people ...)
  observation     right tool, wrong argument that must come from an earlier tool result
                  (place_id, free-slot time, weather-dependent choice ...) -> state tracking
  skipped_step    same as observation, but the read tool that provides that value was never called
  over_execution  all required calls were already done, but the agent kept making write / invalid calls
  premature_stop  answered before finishing the required calls          (recorded at step n_calls+1)
  max_steps       ran out of turns                                      (recorded at step n_calls+1)
  state           all calls matched but the final state differs         (recorded at step n_calls+1)
  final_answer    all calls correct, but the final answer misses the fact (recorded at step n_calls+1)

first_hard_error_* : same, but ignoring extra_lookup (the first error that can actually hurt).
progress           : gold calls reproduced before the first hard error (for "error-free up to k" curves).
recovered          : success although a hard error happened (extra_lookup / over_execution do not count).

Paper-style metrics (NESTFUL / Granite-FC / BFCL):
  f1_func, f1_param, partial_match (= step_accuracy), full_match, success (win rate with state check)
"""
import re
from collections import Counter

from tools_env import DEFAULTS, TOOL_SPECS

READ_PREFIX = ("get_", "search_")
POST_HOC = {"premature_stop", "max_steps", "state", "final_answer"}  # recorded at n_calls + 1


def normalize_args(name, args):
    if name not in TOOL_SPECS or not isinstance(args, dict):
        return None
    spec = TOOL_SPECS[name]
    return {k: f(args[k]) for k, f in {**spec["required"], **spec["optional"]}.items()
            if args.get(k) not in (None, "")}


def _mismatch(gold, pargs):
    d, alt = DEFAULTS.get(gold["name"], {}), gold.get("alt", {})
    bad = []
    for k, v in gold["args"].items():
        pv = pargs.get(k, d.get(k))
        if k in alt:
            if pv not in alt[k]:
                bad.append(k)
        elif pv != v:
            bad.append(k)
    return bad


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


def _contains(text, v):
    v = v.lower().replace(",", "")
    pat = re.escape(v)
    plain_num = v.isdigit()  # e.g. minutes / KRW / temperature: must not be part of a time or date
    if v[:1].isdigit():
        pat = (r"(?<![\d.:\-])" if plain_num else r"(?<![\d.])") + pat
    if v[-1:].isdigit():
        pat = pat + (r"(?![\d:\-])" if plain_num else r"(?![\d])")
    return re.search(pat, text) is not None


def answer_ok(groups, answer):
    """Every group must be matched by one of its variants. Numbers must not be part of a longer number."""
    if not groups:
        return True
    a = (answer or "").lower().replace(",", "")
    return all(any(_contains(a, v) for v in g) for g in groups)


def evaluate(task, rec):
    gold = task["gold"]
    remaining, matched = list(range(len(gold))), []
    first_step = first_type = hard_step = hard_type = None
    redundant = extra_lookups = 0
    progress = None

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
                cands = [j for j in remaining if gold[j]["name"] == c["name"]]
                if not cands:
                    etype = "extra_lookup" if c["name"].startswith(READ_PREFIX) else "tool_selection"
                else:
                    jb = min(cands, key=lambda j: len(_mismatch(gold[j], pargs)))
                    mm = _mismatch(gold[jb], pargs)
                    skipped = [j for j in remaining if j < jb and gold[j]["name"].startswith(READ_PREFIX)]
                    if set(mm) <= set(gold[jb]["derived"]):
                        etype = "skipped_step" if skipped else "observation"
                    else:
                        etype = "argument"
        if etype and not remaining and etype != "extra_lookup":
            etype = "over_execution"
        if etype == "extra_lookup":
            extra_lookups += 1
        if etype and first_step is None:
            first_step, first_type = idx, etype
        if etype and etype != "extra_lookup" and hard_step is None:
            hard_step, hard_type, progress = idx, etype, len(matched)

    s_ok = state_ok(task["expected_state"], rec["state"])
    a_ok = answer_ok(task["answer"], rec.get("final_answer"))
    n_calls = len(rec["calls"])
    post = None
    if remaining or not rec.get("terminated"):
        post = "max_steps" if not rec.get("terminated") else "premature_stop"
    elif not s_ok:
        post = "state"
    elif not a_ok:
        post = "final_answer"
    if post:
        if first_step is None:
            first_step, first_type = n_calls + 1, post
        if hard_step is None:
            hard_step, hard_type = n_calls + 1, post
    if progress is None:
        progress = len(matched)

    success = not remaining and s_ok and a_ok and bool(rec.get("terminated"))

    # ---- paper-style metrics
    valid_calls = [c for c in rec["calls"] if c["status"] != "invalid" and c["name"] in TOOL_SPECS]
    pn, gn = Counter(c["name"] for c in valid_calls), Counter(g["name"] for g in gold)
    keys, alts = {}, {}
    for g in gold:
        keys.setdefault(g["name"], set()).update(g["args"])
        for k, vals in g.get("alt", {}).items():
            alts[(g["name"], k)] = (vals, g["args"][k])
    pp = Counter()
    for c in valid_calls:
        na = normalize_args(c["name"], c["args"]) or {}
        d = DEFAULTS.get(c["name"], {})
        for k in keys.get(c["name"], ()):
            v = na.get(k, d.get(k))
            if (c["name"], k) in alts and v in alts[(c["name"], k)][0]:
                v = alts[(c["name"], k)][1]
            pp[(c["name"], k, str(v))] += 1
    gp = Counter((g["name"], k, str(v)) for g in gold for k, v in g["args"].items())

    def f1(p, g):
        ov = sum((p & g).values())
        pr, rc = (ov / sum(p.values()) if p else 0.0), (ov / sum(g.values()) if g else 1.0)
        return 2 * pr * rc / (pr + rc) if pr + rc else (1.0 if not g and not p else 0.0)

    full_match = not remaining and (len(valid_calls) - redundant) == len(gold) \
        and len(valid_calls) == len(rec["calls"])
    valid = [c for c in rec["calls"] if c["status"] != "invalid"]
    had_hard = hard_type is not None
    return {
        "success": success,
        "step_accuracy": len(matched) / len(gold),
        "valid_call_rate": (len(valid) / n_calls) if n_calls else None,
        "n_calls": n_calls,
        "n_gold": len(gold),
        "n_derived": task.get("n_derived"),
        "redundant_calls": redundant,
        "extra_lookups": extra_lookups,
        "state_ok": s_ok,
        "answer_ok": a_ok,
        "first_error_step": first_step,
        "first_error_type": first_type,
        "first_hard_error_step": hard_step,
        "first_hard_error_type": hard_type,
        "progress": progress,
        "had_error": first_step is not None,
        "had_hard_error": had_hard,
        "recovered": success and had_hard and hard_type != "over_execution",
        "f1_func": f1(pn, gn),
        "f1_param": f1(pp, gp),
        "partial_match": len(matched) / len(gold),
        "full_match": full_match,
    }
