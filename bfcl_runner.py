"""External validation on BFCL multi-turn (multi_turn_base, same data as BFCL V3 multi-turn).

Uses BFCL's own stateful API classes (bfcl/func_source_code, Apache-2.0) and re-implements
BFCL's multi-turn checker (state check + execution-response check after every turn).
Protocol follows BFCL's FC handler: tools only (no extra system prompt), a turn ends when the
model answers without tool calls or after 20 steps; there is no repair prompt.

Horizon cannot be fixed in advance on BFCL, so entries are binned by the number of gold calls:
  short = 2-4, medium = 5-6, long = 7+   (stratified sample, same entries for every precision)

  python bfcl_runner.py --oracle                                   # sanity check, should be 100%
  python bfcl_runner.py --models ..\\qwen3-1.7b-q4_k_m.gguf --per-bin 2 --mmap   # smoke
  python bfcl_runner.py --models ..\\qwen3-1.7b-f16.gguf ..\\qwen3-1.7b-q8_0.gguf ..\\qwen3-1.7b-q4_k_m.gguf --mmap
"""
import argparse
import ast
import copy
import importlib
import inspect
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import requests

from run_experiment import RamMonitor, precision_label, start_server

HERE = Path(__file__).parent / "bfcl"
CLASS_MODULE = {"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
                "TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
                "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control"}
STATELESS = {"MathAPI"}
TYPE_MAP = {"integer": "integer", "number": "number", "float": "number", "string": "string",
            "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array", "dict": "object",
            "object": "object", "tuple": "array", "any": "string"}
MAX_STEPS = 20
BINS = [("short", 2, 4), ("medium", 5, 6), ("long", 7, 99)]
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
LEAK_RE = re.compile(r'<tool_call>|</tool_call>|\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"')


# ------------------------------------------------------------------ data
def load_entries():
    q = [json.loads(l) for l in (HERE / "data" / "multi_turn_base.json").open(encoding="utf-8")]
    a = {x["id"]: x["ground_truth"] for x in map(json.loads, (HERE / "data" / "multi_turn_base_answer.json")
                                                 .open(encoding="utf-8"))}
    out = []
    for e in q:
        e["ground_truth"] = a[e["id"]]
        n = sum(len(t) for t in e["ground_truth"])
        e["n_gold_calls"] = n
        e["bin"] = next(b for b, lo, hi in BINS if lo <= n <= hi)
        out.append(e)
    return out


def sample(entries, per_bin, seed=0):
    rng, picked = random.Random(seed), []
    for b, _, _ in BINS:
        ids = sorted((e for e in entries if e["bin"] == b), key=lambda e: int(e["id"].rsplit("_", 1)[1]))
        picked += rng.sample(ids, min(per_bin, len(ids)))
    return picked


def _conv(schema):
    s = dict(schema)
    if "type" in s:
        s["type"] = TYPE_MAP.get(s["type"], "string")
    if "properties" in s:
        s["properties"] = {k: _conv(v) for k, v in s["properties"].items()}
    if isinstance(s.get("items"), dict):
        s["items"] = _conv(s["items"])
    return s


def tools_for(entry):
    tools = []
    for cls in entry["involved_classes"]:
        for line in (HERE / "data" / "func_doc" / f"{CLASS_MODULE[cls]}.json").open(encoding="utf-8"):
            fd = json.loads(line)
            params = _conv(fd["parameters"])
            params["type"] = "object"
            tools.append({"type": "function", "function": {
                "name": fd["name"], "description": fd["description"], "parameters": params}})
    return tools


# ------------------------------------------------------------------ execution (mirrors BFCL)
def make_instances(entry):
    inst, methods = {}, {}
    for cls in entry["involved_classes"]:
        mod = importlib.import_module(f"bfcl.func_source_code.{CLASS_MODULE[cls]}")
        obj = getattr(mod, cls)()
        if cls not in STATELESS:
            obj._load_scenario(copy.deepcopy(entry["initial_config"].get(cls, {})), long_context=False)
        inst[cls] = obj
        for name, m in inspect.getmembers(obj, predicate=inspect.ismethod):
            if not name.startswith("_"):
                methods[name] = m
    return inst, methods


def to_str(res):
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        try:
            return json.dumps(res)
        except (TypeError, ValueError):
            return str(res)
    return str(res)


def call_method(methods, name, args):
    if name in ("kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"):
        return f"Error during execution: Function call {name} is not allowed."
    if name not in methods:
        return f"Error during execution: name '{name}' is not defined"
    try:
        return to_str(methods[name](**args))
    except Exception as e:  # noqa: BLE001 - same behaviour as BFCL's executor
        return f"Error during execution: {e}"


def parse_call(call_str, methods):
    """'sort('a.txt')' -> ('sort', {'file_name': 'a.txt'}) using the method signature."""
    node = ast.parse(call_str, mode="eval").body
    name = node.func.id
    args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    if node.args and name in methods:
        params = [p for p in inspect.signature(methods[name]).parameters]
        for p, v in zip(params, node.args):
            args[p] = ast.literal_eval(v)
    return name, args


def state_diff(model_inst, gt_inst):
    diffs = {}
    for cls, g in gt_inst.items():
        m = model_inst[cls]
        for k, v in vars(g).items():
            if not k.startswith("_") and getattr(m, k, None) != v:
                diffs[f"{cls}.{k}"] = True
    return diffs


def multiset_missing(needed, have):
    pool, missing = list(have), []
    for x in needed:
        try:
            pool.remove(x)
        except ValueError:
            missing.append(x)
    return missing


# ------------------------------------------------------------------ agent
def chat(url, messages, tools, a, seed):
    payload = {"messages": messages, "tools": tools, "tool_choice": "auto", "temperature": a.temperature,
               "top_p": 0.8, "top_k": 20, "max_tokens": 1024, "seed": seed,
               "chat_template_kwargs": {"enable_thinking": a.think}}
    t0 = time.perf_counter()
    r = requests.post(url + "/v1/chat/completions", json=payload, timeout=1800)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def run_entry(entry, a, url=None, seed=0):
    tools = tools_for(entry)
    m_inst, m_methods = make_instances(entry)
    g_inst, g_methods = make_instances(entry)
    messages, all_model_results, turns, stats = [], [], [], {"latency": 0.0, "pred_n": 0, "pred_ms": 0.0,
                                                             "completion_tokens": 0, "llm_calls": 0}
    for t_idx, (user_msgs, gt_calls) in enumerate(zip(entry["question"], entry["ground_truth"])):
        messages.extend(user_msgs)
        calls, fmt_err, steps, forced, server_err = [], 0, 0, False, None
        if a.oracle:
            for cs in gt_calls:
                name, args = parse_call(cs, m_methods)
                res = call_method(m_methods, name, args)
                calls.append({"name": name, "args": args, "result": res})
        else:
            while True:
                try:
                    data, dt = chat(url, messages, tools, a, seed)
                except Exception as e:  # noqa: BLE001
                    server_err = repr(e)
                    break
                stats["latency"] += dt
                stats["llm_calls"] += 1
                tim, usage = data.get("timings") or {}, data.get("usage") or {}
                stats["pred_n"] += tim.get("predicted_n") or 0
                stats["pred_ms"] += tim.get("predicted_ms") or 0
                stats["completion_tokens"] += usage.get("completion_tokens") or 0
                msg = data["choices"][0]["message"]
                raw = msg.get("content") or ""
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    content = THINK_RE.sub("", raw).strip()
                    if LEAK_RE.search(content):
                        fmt_err += 1  # BFCL: decode failure -> turn ends
                    messages.append({"role": "assistant", "content": content})
                    break
                messages.append({"role": "assistant", "content": raw, "tool_calls": tcs})
                bad_json = False
                for k, tc in enumerate(tcs):
                    fn = tc.get("function") or {}
                    name, rawargs = fn.get("name"), fn.get("arguments")
                    try:
                        args = rawargs if isinstance(rawargs, dict) else (json.loads(rawargs) if rawargs else {})
                        res = call_method(m_methods, name, args)
                    except (TypeError, json.JSONDecodeError):
                        args, res, bad_json = None, "Error: arguments are not valid JSON", True
                    calls.append({"name": name, "args": args, "result": res})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"c{t_idx}_{steps}_{k}",
                                     "content": res})
                if bad_json:
                    fmt_err += 1
                    break
                steps += 1
                if steps > MAX_STEPS:
                    forced = True
                    break
        all_model_results.extend(c["result"] for c in calls)
        gt_results = [call_method(g_methods, *parse_call(cs, g_methods)) for cs in gt_calls]

        reason = None
        if gt_calls:  # BFCL skips turns whose ground truth is empty
            if not calls:
                reason = "empty_turn"
            elif state_diff(m_inst, g_inst):
                reason = "state_mismatch"
            elif multiset_missing(gt_results, all_model_results):
                reason = "response_mismatch"
        turns.append({"turn": t_idx, "n_gold": len(gt_calls), "n_calls": len(calls), "steps": steps,
                      "forced_quit": forced, "format_errors": fmt_err, "server_error": server_err,
                      "pass": reason is None, "fail_reason": reason, "calls": calls,
                      "state_diff": list(state_diff(m_inst, g_inst)) if reason == "state_mismatch" else []})
        if server_err:
            break

    first_fail = next((t for t in turns if not t["pass"]), None)
    n_calls = sum(t["n_calls"] for t in turns)
    err_calls = sum(1 for t in turns for c in t["calls"] if str(c["result"]).startswith(("Error", "{\"error\"")))
    return {"valid": first_fail is None and len(turns) == len(entry["question"]),
            "first_failed_turn": first_fail["turn"] if first_fail else None,
            "first_fail_reason": first_fail["fail_reason"] if first_fail else None,
            "turns_passed_before_fail": (first_fail["turn"] if first_fail else len(turns)) / len(entry["question"]),
            "n_calls": n_calls, "error_call_rate": err_calls / n_calls if n_calls else None,
            "format_errors": sum(t["format_errors"] for t in turns),
            "forced_quits": sum(t["forced_quit"] for t in turns),
            "latency_s": stats["latency"], "llm_calls": stats["llm_calls"],
            "completion_tokens": stats["completion_tokens"],
            "gen_tps": stats["pred_n"] / stats["pred_ms"] * 1000 if stats["pred_ms"] else None,
            "turns": turns}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--server", default=r"..\llama-bin\llama-server.exe")
    ap.add_argument("--per-bin", type=int, default=15, help="entries per horizon bin (max: 52/86/62)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--ctx", type=int, default=16384, help="BFCL tool docs are long; 16K recommended")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ngl", type=int, default=None, help="GPU layers, e.g. 99 with the Vulkan build")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--load-mode", default=None, choices=["auto", "none", "mmap", "mlock"])
    ap.add_argument("--out", default="bfcl_results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_path, meta_path = out / "runs.jsonl", out / "model_meta.json"
    entries = sample(load_entries(), a.per_bin, a.sample_seed)
    done = set()
    if runs_path.exists():
        done = {(r["model"], r["think"], r["id"], r["repeat"]) for r in map(json.loads, runs_path.open(encoding="utf-8"))}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    models = ["ORACLE"] if a.oracle else a.models
    if not models:
        ap.error("give --models ... or --oracle")
    for model in models:
        label = "ORACLE" if a.oracle else precision_label(model)
        todo = [(e, r) for r in range(a.repeats) for e in entries if (label, a.think, e["id"], r) not in done]
        if not todo:
            print(f"[{label}] nothing to do")
            continue
        proc = log = mon = url = None
        load_s = 0.0
        if not a.oracle:
            proc, log, load_s, url = start_server(a, model, label, out)
            mon = RamMonitor(proc.pid)
            mon.start()
            print(f"[{label}] server ready in {load_s:.1f}s")
        try:
            with runs_path.open("a", encoding="utf-8") as f:
                for n, (e, rep) in enumerate(todo, 1):
                    t0 = time.perf_counter()
                    res = run_entry(e, a, url, a.seed + rep)
                    row = {"model": label, "model_path": model, "think": a.think, "id": e["id"], "repeat": rep,
                           "bin": e["bin"], "n_gold_calls": e["n_gold_calls"], "n_turns": len(e["question"]),
                           "classes": e["involved_classes"], "wall_s": time.perf_counter() - t0, **res}
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    print(f"[{label}] {n}/{len(todo)} {e['id']} ({e['bin']}, {e['n_gold_calls']} calls) "
                          f"valid={res['valid']} fail_turn={res['first_failed_turn']} "
                          f"reason={res['first_fail_reason']} {row['wall_s']:.0f}s")
        finally:
            if proc:
                mon.stop()
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
                meta[label] = {"path": model, "size_mb": os.path.getsize(model) / 2**20, "load_time_s": load_s,
                               "peak_rss_mb": mon.peak / 2**20, "ctx": a.ctx}
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"[{label}] peak RAM {meta[label]['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
