"""C1-A 最终严格过滤配对分析（合并 v2+v3+v4 真实健康 trial）。

诚实剔除规则（预先声明，非事后挑选）：
 R1 剔除 trial：ttfr/ttua 碰到 >=59.5s 客户端超时天花板，或带 error（含 p023 慢 RAG 超时）。
 R2 剔除整条 prompt：其任一 trial 的 tools 数组出现过非只读白名单工具名
    （即模型幻觉出 shell/file/quant/browser/... 即便被执行层护栏 DENIED，
     该样本的 A/B 语义已不纯，整条不纳入，保数据可辩护性）。
剩余 prompt 若 sync/cfta 双臂均有有效 rep，则构成一个配对（rep 取均值）。
统计：paired t / Wilcoxon / Cohen's d_z / bootstrap 95%CI。
"""
import importlib.util
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# These analysis scripts are read-only methodology references: they import the
# private WeClaw monorepo (src/) and its temp/ raw logs, so they are NOT runnable
# from this snapshot alone. Set WECLAW_REPO to a local checkout to run them.
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
    # "unknown" 是事件名提取失败的默认占位，非具体危险工具，不计入
    return any((t not in WHITELIST and t != "unknown") for t in r.get("tools", []) if t)


all_rows = []
for fn in FILES:
    for r in load(TEMP / fn):
        r["_src"] = fn
        all_rows.append(r)

# R2: prompts to exclude (any trial ever attempted a non-whitelist tool)
unsafe_prompts = {r["prompt_id"] for r in all_rows if trial_unsafe_attempt(r)}
# R1: drop bad trials; drop unsafe prompts entirely
clean = [r for r in all_rows
         if r["prompt_id"] not in unsafe_prompts and not trial_bad(r)]

print(f"总 trial: {len(all_rows)}  (来自 {len(FILES)} 个 run)")
print(f"R2 整条剔除的 prompt（含幻觉非白名单工具尝试）: {sorted(unsafe_prompts)}")
print(f"R1+R2 后有效 trial: {len(clean)}")

agg = defaultdict(lambda: defaultdict(list))
for r in clean:
    agg[r["prompt_id"]][r["arm"]].append(r)

pairs = [p for p in agg if agg[p].get("sync") and agg[p].get("cfta")]
print(f"有效配对 prompt（双臂都有干净数据）: {len(pairs)}")


def paired_block(metric):
    s, c = [], []
    used = []
    for p in sorted(pairs):
        sv = [x[metric] for x in agg[p]["sync"] if x[metric] is not None]
        cv = [x[metric] for x in agg[p]["cfta"] if x[metric] is not None]
        if sv and cv:
            s.append(st.mean(sv))
            c.append(st.mean(cv))
            used.append(p)
    s, c = np.array(s), np.array(c)
    d = s - c
    n = len(d)
    t, pt = stats.ttest_rel(s, c)
    try:
        w, pw = stats.wilcoxon(s, c)
    except ValueError:
        w, pw = float("nan"), float("nan")
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    rng = np.random.default_rng(20260927)
    boots = [float(np.mean(rng.choice(d, n, replace=True))) for _ in range(5000)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    red = [(a - b) / a * 100 for a, b in zip(s, c) if a > 0]
    return {
        "metric": metric, "n_pairs": n,
        "sync_mean": round(float(s.mean()), 2), "sync_sd": round(float(s.std(ddof=1)), 2),
        "cfta_mean": round(float(c.mean()), 2), "cfta_sd": round(float(c.std(ddof=1)), 2),
        "delta_mean": round(float(d.mean()), 2), "delta_median": round(float(np.median(d)), 2),
        "reduction_pct_mean": round(float(np.mean(red)), 1),
        "reduction_pct_median": round(float(np.median(red)), 1),
        "cohens_d_paired": round(float(dz), 3),
        "delta_boot_ci95": [round(lo, 3), round(hi, 3)],
        "paired_t": {"t": round(float(t), 3), "p": round(float(pt), 4)},
        "wilcoxon": {"W": round(float(w), 1), "p": round(float(pw), 4)},
        "per_prompt_sync_cfta": {p: [round(float(a), 2), round(float(b), 2)] for p, a, b in zip(used, s, c)},
    }


res = {"TTFR": paired_block("ttfr_s"), "TTUA": paired_block("ttua_s")}
for m in ("TTFR", "TTUA"):
    b = res[m]
    print(f"\n[{m}] n={b['n_pairs']}")
    print(f"  sync {b['sync_mean']}±{b['sync_sd']}  cfta {b['cfta_mean']}±{b['cfta_sd']}")
    print(f"  delta_mean {b['delta_mean']}  median {b['delta_median']}  red%_median {b['reduction_pct_median']}")
    print(f"  t={b['paired_t']}  W={b['wilcoxon']}  dz={b['cohens_d_paired']}  bootCI={b['delta_boot_ci95']}")

out = TEMP / "_ab_final_summary.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n→ {out}")
