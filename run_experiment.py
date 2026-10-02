"""Runs all tasks for each GGUF model and logs trajectories + efficiency.

Example (from the agent_quant folder):
  python run_experiment.py --models ..\\qwen3-1.7b-f16.gguf ..\\qwen3-1.7b-q8_0.gguf ..\\qwen3-1.7b-q4_k_m.gguf
  python run_experiment.py --oracle            # pipeline check, no model needed
Results are appended to results/runs.jsonl; already finished (model, task, repeat) are skipped.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests

from agent import AgentConfig, run_oracle, run_task
from evaluate import evaluate
from tasks import build_tasks


def precision_label(path):
    n = os.path.basename(path).lower()
    for k, v in [("q4_k_m", "Q4_K_M"), ("q8_0", "Q8_0"), ("bf16", "BF16"), ("f16", "FP16"), ("f32", "FP32")]:
        if k in n:
            return v
    return os.path.splitext(os.path.basename(path))[0]


class RamMonitor(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.proc, self.peak, self._halt = psutil.Process(pid), 0, threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except psutil.Error:
                break
            time.sleep(0.2)

    def stop(self):
        self._halt.set()
        try:  # Windows keeps an exact peak working set
            self.peak = max(self.peak, getattr(self.proc.memory_info(), "peak_wset", 0))
        except psutil.Error:
            pass


def start_server(a, model, label, out):
    cmd = [a.server, "-m", model, "--jinja", "-c", str(a.ctx), "--port", str(a.port), "-np", "1"]
    if not a.mmap:
        cmd.append("--no-mmap")
    if a.threads:
        cmd += ["-t", str(a.threads)]
    log = open(out / f"server_{label}.log", "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{a.port}"
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited. See {out / f'server_{label}.log'}")
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.time() - t0 > 600:
            proc.kill()
            raise RuntimeError("llama-server did not become ready in 600 s")
        time.sleep(1)
    return proc, log, time.time() - t0, url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--server", default=r"..\llama-bin\llama-server.exe")
    ap.add_argument("--oracle", action="store_true", help="replay gold trajectories (no server)")
    ap.add_argument("--lang", default="ko", choices=["ko", "en"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking mode")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 3, 5])
    ap.add_argument("--limit", type=int, default=0, help="max tasks per horizon (0 = all)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mmap", action="store_true", help="allow mmap (default: --no-mmap for honest RAM)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_path, meta_path = out / "runs.jsonl", out / "model_meta.json"

    tasks = [t for t in build_tasks(a.lang) if t["horizon"] in a.horizons]
    if a.limit:
        mixed = sorted(tasks, key=lambda t: (t["id"].rsplit("-", 1)[1], t["domain"]))  # spread over domains
        tasks = [t for h in a.horizons for t in [x for x in mixed if x["horizon"] == h][:a.limit]]

    done = set()
    if runs_path.exists():
        for line in runs_path.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["model"], r["lang"], r["think"], r["task_id"], r["repeat"]))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    models = ["ORACLE"] if a.oracle else a.models
    if not models:
        ap.error("give --models ... or --oracle")

    for model in models:
        label = "ORACLE" if a.oracle else precision_label(model)
        todo = [(t, r) for r in range(a.repeats) for t in tasks
                if (label, a.lang, a.think, t["id"], r) not in done]
        if not todo:
            print(f"[{label}] nothing to do")
            continue
        proc = log = mon = None
        url, load_s = None, 0.0
        if not a.oracle:
            proc, log, load_s, url = start_server(a, model, label, out)
            mon = RamMonitor(proc.pid)
            mon.start()
            print(f"[{label}] server ready in {load_s:.1f}s")
        try:
            with runs_path.open("a", encoding="utf-8") as f:
                for n, (t, rep) in enumerate(todo, 1):
                    t0 = time.perf_counter()
                    if a.oracle:
                        rec = run_oracle(t, a.lang)
                    else:
                        cfg = AgentConfig(base_url=url, temperature=a.temperature, max_turns=a.max_turns,
                                          think=a.think, seed=a.seed + rep)
                        rec = run_task(t, cfg, a.lang)
                    wall = time.perf_counter() - t0
                    m = evaluate(t, rec)
                    pn = sum(x.get("predicted_n") or 0 for x in rec["turns"])
                    pms = sum(x.get("predicted_ms") or 0 for x in rec["turns"])
                    row = {"model": label, "model_path": model, "lang": a.lang, "think": a.think,
                           "task_id": t["id"], "domain": t["domain"], "horizon": t["horizon"], "repeat": rep,
                           "seed": a.seed + rep, "temperature": a.temperature, "wall_s": wall,
                           "n_turns": len(rec["turns"]),
                           "completion_tokens": sum(x.get("completion_tokens") or 0 for x in rec["turns"]),
                           "prompt_tokens_processed": sum(x.get("prompt_n") or 0 for x in rec["turns"]),
                           "gen_tps": (pn / pms * 1000) if pms else None,
                           "metrics": m, "record": rec}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{label}] {n}/{len(todo)} {t['id']} r{rep} "
                          f"success={m['success']} step_acc={m['step_accuracy']:.2f} "
                          f"err={m['first_error_type']} {wall:.1f}s")
        finally:
            if proc:
                mon.stop()
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
                meta[label] = {"path": model, "size_mb": os.path.getsize(model) / 2**20,
                               "load_time_s": load_s, "peak_rss_mb": mon.peak / 2**20,
                               "mmap": a.mmap, "ctx": a.ctx}
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"[{label}] peak RAM {meta[label]['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
