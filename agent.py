"""ReAct-style tool-calling loop on top of llama-server's OpenAI-compatible API.

The loop does not force any path. It only controls state (message history + world),
the maximum number of turns, and termination (= a turn without tool calls),
which is the same role LangGraph plays in the proposal.
"""
import json
import re
import time
from dataclasses import dataclass

import requests

from tools_env import TOOL_SCHEMAS, World

SYSTEM = {
    "ko": ("너는 사용자의 요청을 처리하는 비서야. 제공된 도구로 정보를 확인하고 작업을 수행해. "
           "요청에 필요한 정보는 모두 주어져 있으니 사용자에게 되묻지 마. 도구를 쓰겠다고 말만 하지 말고 반드시 실제로 "
           "도구를 호출해. 도구 결과에 없는 정보는 추측하지 마. 요청된 작업을 모두 도구로 실행하기 전에는 최종 답변을 하지 마. "
           "모든 작업이 끝나면 최종 답변을 한국어로 간단히 말해줘. "
           "날짜는 YYYY-MM-DD, 시간은 HH:MM 형식을 사용해."),
    "en": ("You are an assistant that completes the user's request. Use the provided tools to look up "
           "information and take actions. The request contains everything you need, so do not ask the user "
           "questions. Do not just say you will use a tool; actually call it. Do not guess information that is not "
           "in tool results. Do not give a final answer until every requested action has been executed with tools. "
           "Then give a short final answer. Use YYYY-MM-DD for dates and HH:MM for times."),
}
REPAIR = {
    "ko": "도구 호출 형식을 해석할 수 없었어. 제공된 function calling 형식으로 다시 호출해줘.",
    "en": "Your tool call could not be parsed. Please call tools using the provided function-calling format.",
}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
LEAK_RE = re.compile(r'<tool_call>|</tool_call>|\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"')


@dataclass
class AgentConfig:
    base_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 1024
    max_turns: int = 12
    think: bool = False
    seed: int = 0
    timeout: int = 900


def _chat(cfg, messages):
    payload = {"messages": messages, "tools": TOOL_SCHEMAS, "tool_choice": "auto",
               "temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k,
               "max_tokens": cfg.max_tokens, "seed": cfg.seed,
               "chat_template_kwargs": {"enable_thinking": cfg.think}}
    t0 = time.perf_counter()
    r = requests.post(cfg.base_url + "/v1/chat/completions", json=payload, timeout=cfg.timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def _record(task, world, calls, turns, final, terminated, max_turns):
    return {"calls": calls, "turns": turns, "final_answer": final, "terminated": terminated,
            "hit_max": (not terminated) and len(turns) >= max_turns, "state": world.state()}


def run_task(task, cfg, lang="ko"):
    world = World(**task["world_params"])
    messages = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": task["query"]}]
    calls, turns, final, terminated = [], [], None, False

    for turn in range(1, cfg.max_turns + 1):
        try:
            data, dt = _chat(cfg, messages)
        except Exception as e:  # server error / timeout -> stop this task
            turns.append({"turn": turn, "error": repr(e)})
            break
        msg = data["choices"][0]["message"]
        usage, tim = data.get("usage") or {}, data.get("timings") or {}
        raw_content = msg.get("content") or ""
        content = THINK_RE.sub("", raw_content).strip()
        turns.append({"turn": turn, "latency_s": dt,
                      "prompt_tokens": usage.get("prompt_tokens"),
                      "completion_tokens": usage.get("completion_tokens"),
                      "prompt_n": tim.get("prompt_n"), "prompt_ms": tim.get("prompt_ms"),
                      "predicted_n": tim.get("predicted_n"), "predicted_ms": tim.get("predicted_ms"),
                      "content": content, "reasoning": msg.get("reasoning_content")})

        tcs = msg.get("tool_calls") or []
        if tcs:
            messages.append({"role": "assistant", "content": raw_content, "tool_calls": tcs})
            for k, tc in enumerate(tcs):
                fn = tc.get("function") or {}
                name, raw = fn.get("name"), fn.get("arguments")
                args, fmt_err = None, False
                if isinstance(raw, dict):
                    args = raw
                else:
                    try:
                        args = json.loads(raw) if raw else {}
                    except (TypeError, json.JSONDecodeError):
                        fmt_err = True
                if fmt_err:
                    result, status = {"error": "arguments are not valid JSON"}, "invalid"
                else:
                    result, status = world.call(name, args)
                calls.append({"turn": turn, "name": name, "args_raw": raw if isinstance(raw, str) else json.dumps(raw),
                              "args": args, "status": status, "format_error": fmt_err, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{turn}_{k}",
                                 "content": json.dumps(result, ensure_ascii=False)})
            continue

        if LEAK_RE.search(content):  # tool call written as plain text -> format error, ask once more
            calls.append({"turn": turn, "name": None, "args_raw": content[:500], "args": None,
                          "status": "invalid", "format_error": True, "result": None})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": REPAIR[lang]})
            continue

        final, terminated = content, True
        break

    return _record(task, world, calls, turns, final, terminated, cfg.max_turns)


def run_oracle(task, lang="ko"):
    """Replays the gold trajectory. Used only to sanity-check the pipeline (should score 100%)."""
    world = World(**task["world_params"])
    calls = []
    for g in task["gold"]:
        args = {**g["args"], **g["extra"]}
        result, status = world.call(g["name"], args)
        calls.append({"turn": 1, "name": g["name"], "args_raw": json.dumps(args, ensure_ascii=False),
                      "args": args, "status": status, "format_error": False, "result": result})
    final = " / ".join(grp[0] for grp in task["answer"]) or "done"
    turns = [{"turn": 1, "latency_s": 0.0}]
    return _record(task, world, calls, turns, final, True, 99)
