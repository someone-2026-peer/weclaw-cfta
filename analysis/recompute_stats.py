"""Standalone reproduction of the CFTA end-to-end A/B paired statistics.

This script has NO dependency on the WeClaw agent codebase -- it needs only
NumPy and SciPy. It reads the raw wall-clock A/B logs shipped in this snapshot
(../ab_measurement/ab_raw_v{2}.jsonl, ab_raw_v3_partial_throttled.jsonl,
ab_raw_v4.jsonl) and re-derives every paired figure reported in section 5.2 of
the paper (n pairs, TTFR null, TTUA reduction, paired t, Wilcoxon, Cohen's d_z,
bootstrap 95% CI, and the tool-fired stratification).

Reproduction command (from the repo root):
    python analysis/recompute_stats.py

The read-only whitelist below is inlined verbatim from ab_harness.py so that the
R2 exclusion rule (drop any prompt whose model attempted a non-whitelist tool)
can be applied without importing the harness.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# --- inlined from ab_harness.py READONLY_WHITELIST (verbatim) ---
READONLY_WHITELIST = {
    "weather", "datetime_tool", "calculator", "statistics", "tool_info",
    "chat_history", "knowledge_rag", "literature_search", "search", "poetry",
    "baike_lookup", "enterprise_query", "medical_lookup", "chinese_dictionary",
    "pinyin_dict", "bilibili_search", "system_monitor",
    "fred_query",
    "tool_audit", "log_viewer", "codebase_search", "experience_recall",
    "oss_pdf_search", "local_paper_search", "research_lineage", "research_landscape",
    "research_lookup", "contrarian_finder", "idea_migration",
    "methodology_deconstructor", "citation_storyteller", "journal_intelligence",
    "paper_lifecycle", "qualitative_analysis", "batch_paper_analyzer",
    "education_tool", "english_conversation", "english_vocab", "study_solver",
    "lunar_calendar", "daily_quote", "nutrition_query", "recipe_library",
    "attachment_search",
}

CEIL = 59.5            # R1: client 60s timeout ceiling
SEED = 20260927        # fixed bootstrap seed
N_BOOT = 5000

HERE = Path(__file__).resolve().parent
AB_DIR = HERE.parent / "ab_measurement"
FILES = [
    AB_DIR / "ab_raw_v2.jsonl",
    AB_DIR / "ab_raw_v3_partial_throttled.jsonl",
    AB_DIR / "ab_raw_v4.jsonl",
]


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def trial_bad(r):
    return (r.get("ttfr_s") or 0) >= CEIL or (r.get("ttua_s") or 0) >= CEIL or r.get("error")


def trial_unsafe_attempt(r):
    # "unknown" is the default placeholder when event-name extraction fails, not a
    # concrete dangerous tool, so it is not counted against the whitelist.
    return any((t not in READONLY_WHITELIST and t != "unknown") for t in r.get("tools", []) if t)


def main():
    all_rows = []
    for fn in FILES:
        all_rows.extend(load(fn))

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

    def block(group, metric):
        s, c = [], []
        for p in group:
            sv = metric_mean(agg[p]["sync"], metric)
            cv = metric_mean(agg[p]["cfta"], metric)
            if sv is not None and cv is not None:
                s.append(sv)
                c.append(cv)
        s, c = np.array(s), np.array(c)
        d = s - c
        n = len(d)
        if n < 2:
            return {"n": n, "note": "insufficient pairs"}
        t, pt = stats.ttest_rel(s, c)
        try:
            w, pw = stats.wilcoxon(s, c)
        except ValueError:
            w, pw = float("nan"), float("nan")
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
        rng = np.random.default_rng(SEED)
        boots = [float(np.mean(rng.choice(d, n, replace=True))) for _ in range(N_BOOT)]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        return {
            "n": n,
            "sync_mean": round(float(s.mean()), 2), "sync_sd": round(float(s.std(ddof=1)), 2),
            "cfta_mean": round(float(c.mean()), 2), "cfta_sd": round(float(c.std(ddof=1)), 2),
            "delta_mean": round(float(d.mean()), 2),
            "agg_reduction_pct": round((1 - c.mean() / s.mean()) * 100, 1),
            "median_reduction_pct": round(float(np.median([(a - b) / a * 100 for a, b in zip(s, c) if a > 0])), 1),
            "paired_t": {"t": round(float(t), 3), "p": round(float(pt), 4)},
            "wilcoxon": {"W": round(float(w), 1), "p": round(float(pw), 4)},
            "cohens_d_paired": round(float(dz), 3),
            "boot_ci95": [round(lo, 2), round(hi, 2)],
        }

    fired_p = [p for p in pairs if fired[p]]
    chat_p = [p for p in pairs if not fired[p]]
    out = {
        "counts": {"n_pairs": len(pairs), "n_fired": len(fired_p), "n_chat": len(chat_p)},
        "TTFR": block(pairs, "ttfr_s"),
        "TTUA": block(pairs, "ttua_s"),
        "TTUA_tool_fired": block(fired_p, "ttua_s"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # self-check against the shipped aggregate summary
    ref_path = AB_DIR / "ab_summary_final_pooled.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_t = ref["TTUA"]["paired_t"]["p"]
        got_t = out["TTUA"]["paired_t"]["p"]
        ok = abs(out["TTUA"]["n"] - ref["TTUA"]["n_pairs"]) == 0 and abs(got_t - ref_t) < 0.02
        print(f"\n[self-check] recompute TTUA p={got_t} vs shipped ref p={ref_t} "
              f"(n={out['TTUA']['n']} vs {ref['TTUA']['n_pairs']}) -> "
              f"{'CONSISTENT' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
