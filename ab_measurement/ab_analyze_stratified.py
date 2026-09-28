"""按 tool_fired 分层的真实配对分析（复用 ab_analyze_pooled 的 R1/R2 严格过滤）。

不改过滤规则，只在"干净配对集"上按 prompt 是否触发工具切两层，
对 TTUA / TTFR 分别跑 paired t / Wilcoxon / d_z / bootstrap CI。
输出可贴 LaTeX 的数值 + 存档 JSON。
"""
import importlib.util
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Read-only methodology reference: imports the private WeClaw monorepo (src/) and
# its temp/ raw logs; NOT runnable from this snapshot alone. Set WECLAW_REPO to run.
HERE = Path(__file__).resolve().parent                         # ab_measurement/ (ships here)
REPO = Path(os.environ.get("WECLAW_REPO", str(HERE.parent)))   # root of the WeClaw checkout
TEMP = REPO / "temp"

_spec = importlib.util.spec_from_file_location("abh", HERE / "ab_harness.py")
abh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abh)
WHITELIST = abh.READONLY_WHITELIST

CEIL = 59.5
FILES = ["_ab_full_v2_raw.jsonl", "_ab_full_v3_raw.jsonl", "_ab_full_v4_raw.jsonl"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def trial_bad(r):
    return (r.get("ttfr_s") or 0) >= CEIL or (r.get("ttua_s") or 0) >= CEIL or r.get("error")


def trial_unsafe_attempt(r):
    return any((t not in WHITELIST and t != "unknown") for t in r.get("tools", []) if t)


all_rows = []
for fn in FILES:
    for r in load(TEMP / fn):
        all_rows.append(r)

unsafe_prompts = {r["prompt_id"] for r in all_rows if trial_unsafe_attempt(r)}
clean = [r for r in all_rows
         if r["prompt_id"] not in unsafe_prompts and not trial_bad(r)]

agg = defaultdict(lambda: defaultdict(list))
fired = defaultdict(bool)
for r in clean:
    agg[r["prompt_id"]][r["arm"]].append(r)
    if r.get("tool_fired"):
        fired[r["prompt_id"]] = True

pairs = [p for p in agg if agg[p].get("sync") and agg[p].get("cfta")]


def metric_mean(recs, metric):
    vals = [x[metric] for x in recs if x.get(metric) is not None]
    return st.mean(vals) if vals else None


def block(group_pairs, metric):
    s, c = [], []
    for p in group_pairs:
        sv = metric_mean(agg[p]["sync"], metric)
        cv = metric_mean(agg[p]["cfta"], metric)
        if sv is not None and cv is not None:
            s.append(sv)
            c.append(cv)
    s, c = np.array(s), np.array(c)
    d = s - c
    n = len(d)
    if n < 2:
        return {"n": n, "note": "insufficient"}
    t, pt = stats.ttest_rel(s, c)
    try:
        w, pw = stats.wilcoxon(s, c)
    except ValueError:
        w, pw = float("nan"), float("nan")
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    rng = np.random.default_rng(20260927)
    boots = [float(np.mean(rng.choice(d, n, replace=True))) for _ in range(5000)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "n": n,
        "sync_mean": round(float(s.mean()), 2), "sync_sd": round(float(s.std(ddof=1)), 2),
        "cfta_mean": round(float(c.mean()), 2), "cfta_sd": round(float(c.std(ddof=1)), 2),
        "delta_mean": round(float(d.mean()), 2),
        "agg_reduction_pct": round((1 - c.mean() / s.mean()) * 100, 1),
        "median_reduction_pct": round(float(np.median([(a - b) / a * 100 for a, b in zip(s, c) if a > 0])), 1),
        "t": round(float(t), 3), "p_t": round(float(pt), 4),
        "W": round(float(w), 1), "p_w": round(float(pw), 4),
        "d_z": round(float(dz), 3), "boot_ci95": [round(lo, 2), round(hi, 2)],
    }


fired_p = [p for p in pairs if fired[p]]
chat_p = [p for p in pairs if not fired[p]]

res = {
    "all_TTUA": block(pairs, "ttua_s"),
    "all_TTFR": block(pairs, "ttfr_s"),
    "fired_TTUA": block(fired_p, "ttua_s"),
    "fired_TTFR": block(fired_p, "ttfr_s"),
    "chat_TTUA": block(chat_p, "ttua_s"),
    "chat_TTFR": block(chat_p, "ttfr_s"),
    "counts": {"n_pairs": len(pairs), "n_fired": len(fired_p), "n_chat": len(chat_p)},
}
print(json.dumps(res, ensure_ascii=False, indent=2))
(TEMP / "_ab_strat_by_fired.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n-> temp/_ab_strat_by_fired.json")
