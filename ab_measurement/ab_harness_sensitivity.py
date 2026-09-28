"""Collection harness for the CFTA sensitivity matrix: instruction complexity x model
thinking mode (paper Section 5.5).

This is an *additive* experiment on top of ab_harness.py; it neither modifies nor
overwrites the Section-5.2 frozen artifacts ab_raw_v*.jsonl / ab_meta.json (the
reproducibility contract is append-only). Goal: stretch the single configuration cell of
the headline result along two moderator axes to answer the authors'/reviewers' questions--

  1) as instruction complexity rises (L1 single fast tool -> L2 single heavy tool ->
     L3 multi-tool chain), does the CFTA TTUA advantage grow? Does TTFR stay flat?
  2) when the backend's thinking mode toggles (non-reasoning vs reasoning model), how do
     the two arms' latencies change? Does enabling reasoning reveal a CFTA first-token /
     authoritative-answer advantage?

Design:
  - the arms (Sync / CFTA-fast / CFTA-deferred) fully reuse ab_harness's real
    implementation;
  - intent routing uses the production rule classifier (this experiment does not touch
    the router; that is a separate axis);
  - the thinking switch has two implementation modes (see --thinking-mode): "model"
    swaps agent.model_key (off = non-reasoning model, on = reasoning model, used for
    DeepSeek flash <-> pro), and "extra_body" flips extra_body.thinking.type on the same
    model via monkeypatched model_registry.chat / chat_stream (registry's
    call_kwargs.update(kwargs) forwards it to litellm.acompletion);
  - the safety guard (read-only whitelist + execution-layer interception + max_steps=8 +
    trial timeout) is inherited unchanged;
  - default backends: deepseek-v4-flash (off) / deepseek-v4-pro (on), both
    function-calling-capable.

Usage (smoke):
  python ab_harness_sensitivity.py --thinking both --rep 1 --limit 2
Full matrix:
  python ab_harness_sensitivity.py --thinking both --rep 2
Output: ab_raw_sensitivity.jsonl + ab_meta_sensitivity.json (frozen Section-5.2 files
untouched).

NOTE: this driver imports ab_harness.py, which imports the full WeClaw agent codebase
(not shipped in this snapshot); it is provided as a read-only methodology reference and
cannot run from this snapshot alone.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ab_harness  # noqa: E402   reuse build_agent / run_sync / run_cfta / route / guards

# ---- thinking-switch injection channel (module-level; set by run_matrix before each trial) ----
# type=None -> not injected (unchanged behavior, identical to the Section-5.2 default);
# "enabled"/"disabled" -> injected
_THINK: dict = {"type": None}

# thinking-axis implementation modes:
#   "model"      -> swap agent.model_key (off = non-reasoning model, on = reasoning model),
#                   used for DeepSeek flash <-> pro
#   "extra_body" -> same model, flip only extra_body.thinking.type (for GLM-style
#                   mixed-thinking models)
_CONFIG: dict = {"mode": "model", "model_map": {"off": "", "on": ""}}

# records whether a call actually carried the thinking parameter (log evidence that the
# switch really took effect, not fabricated)
_THINK_EVIDENCE: dict = {"injected": 0}


def install_thinking_toggle(registry) -> None:
    """monkeypatch: inject _THINK['type'] as extra_body.thinking into model calls.

    Only wraps one layer; the original method logic (concurrency gate / retry /
    accounting) is untouched. When _THINK['type'] is empty it passes through, so the
    behavior is 100% identical to the unpatched registry.
    """
    orig_chat = registry.chat
    orig_stream = registry.chat_stream

    def _merge(kw: dict) -> dict:
        if not _THINK.get("type"):
            return kw
        eb = dict(kw.get("extra_body") or {})
        eb["thinking"] = {"type": _THINK["type"]}
        kw["extra_body"] = eb
        _THINK_EVIDENCE["injected"] += 1
        return kw

    @functools.wraps(orig_chat)
    async def chat_patched(*a, **kw):
        return await orig_chat(*a, **_merge(kw))

    @functools.wraps(orig_stream)
    async def stream_patched(*a, **kw):
        # chat_stream is an async generator: no yield inside finally; forward chunk by chunk
        async for chunk in orig_stream(*a, **_merge(kw)):
            yield chunk

    registry.chat = chat_patched
    registry.chat_stream = stream_patched


def is_bad(rec: dict, ceiling: float = 59.5) -> bool:
    if rec.get("error"):
        return True
    return (rec.get("ttfr_s") or 0) >= ceiling or (rec.get("ttua_s") or 0) >= ceiling


async def one_trial(agent, p: dict, arm: str, rep: int, thinking: str,
                    timeout: float) -> dict:
    prompt = p["text"]
    conf = agent_intent_conf(prompt)
    if _CONFIG["mode"] == "model":
        # realize the thinking switch by swapping models; no extra_body injection
        agent.model_key = _CONFIG["model_map"][thinking]
        _THINK["type"] = None
    else:
        # same model, flip only thinking.type: both states are sent explicitly, the sole
        # difference is that value (cause isolation)
        _THINK["type"] = {"off": "disabled", "on": "enabled"}.get(thinking, thinking)
    _THINK_EVIDENCE["injected"] = 0
    t_wall = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    holder: dict = {}
    try:
        coro = (ab_harness.run_sync(agent, prompt, holder) if arm == "sync"
                else ab_harness.run_cfta(agent, prompt, conf, holder))
        m = await asyncio.wait_for(coro, timeout)
        err = None
    except asyncio.TimeoutError:
        m = {"ttfr_s": holder.get("first"), "ttua_s": None,
             "fast_end_s": holder.get("fast_end"),
             "n_chunks": holder.get("chunks", 0), "out_chars": holder.get("chars", 0),
             "tool_fired": len(ab_harness._REC["tools"]) > 0,
             "tools": list(ab_harness._REC["tools"])}
        err = f"timeout_{timeout}s"
    except Exception as e:  # keep failed rows, never drop or fabricate
        m = {"ttfr_s": None, "ttua_s": None, "fast_end_s": None,
             "n_chunks": 0, "out_chars": 0, "tool_fired": False, "tools": []}
        err = f"{type(e).__name__}: {e}"
    return {
        "trial_id": f"{p['prompt_id']}_{arm}_{thinking}_r{rep}",
        "prompt_id": p["prompt_id"],
        "complexity": p.get("complexity", ""),
        "thinking": thinking,          # "off" | "on"
        "arm": arm,
        "rep": rep,
        "model_key": agent.model_key,
        "intent_confidence": round(float(conf), 3),
        "path": ab_harness.route(conf),
        "tool_fired": m["tool_fired"],
        "tools": m["tools"],
        "ttfr_s": m["ttfr_s"],
        "ttua_s": m["ttua_s"],
        "fast_end_s": m["fast_end_s"],
        "n_chunks": m["n_chunks"],
        "out_chars": m["out_chars"],
        "think_injected": _THINK_EVIDENCE["injected"],  # evidence: injection count for this trial
        "wall_ts": t_wall,
        "error": err,
    }


def agent_intent_conf(prompt: str) -> float:
    try:
        return float(ab_harness.detect_intent_with_confidence(prompt).confidence)
    except Exception:
        return 0.0


async def run_matrix(args):
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        sys.exit(f"error: prompt set not found {prompts_path}")
    prompts = [json.loads(l) for l in
               prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        # smoke mode: take the first prompt per complexity tier, stop once limit is reached
        picked, seen = [], set()
        for p in prompts:
            c = p.get("complexity")
            if c in seen:
                continue
            seen.add(c)
            picked.append(p)
            if len(picked) >= args.limit:
                break
        prompts = picked
    if not prompts:
        sys.exit("error: prompt set is empty")

    agent, model_key, removed_tools = await ab_harness.build_agent(args.model)
    _CONFIG["mode"] = args.thinking_mode
    _CONFIG["model_map"] = {"off": args.model_off, "on": args.model_on}
    if args.thinking_mode == "extra_body":
        install_thinking_toggle(agent.model_registry)
    thinking_axis = {"both": ["off", "on"], "on": ["on"], "off": ["off"]}[args.thinking]
    mm = (f"{args.thinking_mode}:{args.model_off}/{args.model_on}"
          if args.thinking_mode == "model" else f"{args.thinking_mode}@{model_key}")
    print(f"[OK] agent ready thinking_axis={mm} prompts={len(prompts)} "
          f"thinking={thinking_axis} rep={args.rep} timeout={args.timeout}s")

    # warm-up (discarded): heat once per state
    dummy = {"prompt_id": "warmup", "text": "Hello, what is the weather like today?", "complexity": "L1"}
    for th in thinking_axis:
        for _ in range(args.warmup):
            await one_trial(agent, dummy, "sync", 0, th, args.timeout)
            await one_trial(agent, dummy, "cfta", 0, th, args.timeout)
    print("[OK] warm-up done (not counted as data)")

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    n_written = 0
    consec_bad = 0
    n_cooldowns = 0
    start = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for th in thinking_axis:
            order = prompts[:]
            rng.shuffle(order)
            for pi, p in enumerate(order, 1):
                for rep in range(1, args.rep + 1):
                    arms = ["sync", "cfta"] if rep % 2 == 1 else ["cfta", "sync"]
                    for arm in arms:
                        rec = await one_trial(agent, p, arm, rep, th, args.timeout)
                        rec["order_slot"] = "S_first" if arms[0] == "sync" else "C_first"
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        n_written += 1
                        print(f"  [{th}] {p['prompt_id']} {arm} r{rep}: "
                              f"ttfr={rec['ttfr_s']} ttua={rec['ttua_s']} "
                              f"fired={rec['tool_fired']} inj={rec['think_injected']}"
                              f"{' ERR=' + str(rec['error']) if rec['error'] else ''}")
                        await asyncio.sleep(args.sleep)
                        if is_bad(rec):
                            consec_bad += 1
                        else:
                            consec_bad = 0
                        if consec_bad >= args.fail_threshold:
                            n_cooldowns += 1
                            print(f"  [cool] {consec_bad} consecutive throttle/timeouts -> "
                                  f"cooldown {args.cooldown}s")
                            await asyncio.sleep(args.cooldown)
                            consec_bad = 0
                print(f"  -- {p['prompt_id']} ({p.get('complexity')}) done, "
                      f"{n_written} rows so far")

    elapsed = time.time() - start
    meta = {
        "experiment": "cfta_sensitivity_complexity_x_thinking",
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": ab_harness.git_commit(),
        "model_key": model_key,
        "thinking_mode": args.thinking_mode,
        "thinking_model_map": ({"off": args.model_off, "on": args.model_on}
                               if args.thinking_mode == "model" else None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "n_prompts": len(prompts),
        "complexity_tiers": sorted({p.get("complexity", "") for p in prompts}),
        "thinking_axis": thinking_axis,
        "rep": args.rep,
        "warmup": args.warmup,
        "seed": args.seed,
        "sleep_sec": args.sleep,
        "trials_written": n_written,
        "elapsed_sec": round(elapsed, 1),
        "throttle_cooldowns": n_cooldowns,
        "fail_threshold": args.fail_threshold,
        "cooldown_sec": args.cooldown,
        "prompts_file": prompts_path.name,
        "prompts_sha1": hashlib.sha1(prompts_path.read_bytes()).hexdigest()[:12],
        "safety": {
            "readonly_whitelist": True,
            "removed_tools_n": len(removed_tools),
            "max_steps": 8,
            "trial_timeout_sec": args.timeout,
            "note": "inherits ab_harness's read-only whitelist and execution-layer "
                    "interception; this experiment does not overwrite the Section-5.2 "
                    "frozen files",
        },
        "thinking_injection": "extra_body.thinking.type via monkeypatched registry.chat/chat_stream",
    }
    Path(args.meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] collection done: {n_written} rows -> {out_path}")
    print(f"[OK] metadata -> {args.meta} (git={meta['git_commit'][:12]})")


def main():
    ap = argparse.ArgumentParser(description="CFTA sensitivity matrix (complexity x thinking)")
    ap.add_argument("--prompts", default=str(HERE / "prompts_sensitivity_complexity.jsonl"))
    ap.add_argument("--model", default="deepseek-v4-flash", help="base/warm-up model key")
    ap.add_argument("--thinking-mode", choices=["model", "extra_body"], default="model",
                    help="model=swap models (flash <-> pro); extra_body=flip thinking.type "
                         "on the same model")
    ap.add_argument("--model-off", default="deepseek-v4-flash",
                    help="non-reasoning model (mode=model)")
    ap.add_argument("--model-on", default="deepseek-v4-pro",
                    help="reasoning model (mode=model)")
    ap.add_argument("--thinking", choices=["both", "on", "off"], default="both")
    ap.add_argument("--rep", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: 1 prompt per complexity, up to limit")
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--fail-threshold", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=180.0)
    ap.add_argument("--out", default=str(HERE / "ab_raw_sensitivity.jsonl"))
    ap.add_argument("--meta", default=str(HERE / "ab_meta_sensitivity.json"))
    args = ap.parse_args()
    asyncio.run(run_matrix(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted: written rows are kept (resume or full rerun).")
        sys.exit(130)
