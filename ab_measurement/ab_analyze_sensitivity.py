"""Re-derive the complexity x thinking sensitivity matrix (paper Section 5.5)
from ab_raw_sensitivity.jsonl.

Pure standard-library statistics (no dependency on the agent codebase). Aggregates
paired sync vs CFTA TTUA / TTFR per (complexity, thinking) cell; the "useful answer"
comparison only includes pairs where the CFTA arm actually fired a tool, and each
cell's CFTA fire rate is reported separately (the fire rate is itself an observed
quantity of the complexity x thinking design).

Usage:
  python ab_analyze_sensitivity.py [--raw ab_raw_sensitivity.jsonl]
Output: prints the matrix table + writes sensitivity_summary.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
# This sensitivity matrix ran on DeepSeek with throttle_cooldowns=0 (no 429s), so the
# 59.5 s ceiling that Section 5.2 uses to filter throttle artifacts does not apply here:
# only error/None trials are dropped and genuinely slow tool latencies (e.g. the 74.8 s
# web search) are kept. The ceiling sits just above the collection trial timeout (90 s),
# i.e. equivalent to dropping only None/timeout.
CEILING = 90.5


def load(raw: str):
    rows = []
    for l in Path(raw).read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l:
            continue
        rows.append(json.loads(l))
    return rows


def valid(rec, metric):
    if rec.get("error"):
        return False
    v = rec.get(metric)
    return v is not None and v < CEILING


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "ab_raw_sensitivity.jsonl"))
    args = ap.parse_args()
    rows = load(args.raw)

    # index: (prompt_id, thinking, rep, arm) -> rec
    idx = {}
    for r in rows:
        idx[(r["prompt_id"], r["thinking"], r["rep"], r["arm"])] = r

    tiers = ["L1", "L2", "L3"]
    think = ["off", "on"]

    # per-cell aggregation
    cells = {}
    for c in tiers:
        for th in think:
            pids = sorted({r["prompt_id"] for r in rows
                           if r.get("complexity") == c and r.get("thinking") == th})
            pair_s, pair_c, pair_sfr, pair_cfr = [], [], [], []
            n_attempt = n_cfta_fired = 0
            for pid in pids:
                for rep in sorted({r["rep"] for r in rows if r["prompt_id"] == pid}):
                    s = idx.get((pid, th, rep, "sync"))
                    cf = idx.get((pid, th, rep, "cfta"))
                    if not s or not cf:
                        continue
                    n_attempt += 1
                    if cf.get("tool_fired"):
                        n_cfta_fired += 1
                    # TTUA: sync valid + CFTA valid and actually fired a tool
                    # (otherwise it is not an authoritative-answer comparison)
                    if valid(s, "ttua_s") and valid(cf, "ttua_s") and cf.get("tool_fired"):
                        pair_s.append(s["ttua_s"])
                        pair_c.append(cf["ttua_s"])
                    if valid(s, "ttfr_s") and valid(cf, "ttfr_s"):
                        pair_sfr.append(s["ttfr_s"])
                        pair_cfr.append(cf["ttfr_s"])

            def block(a, b):
                if not a:
                    return {"n": 0}
                ma, mb = st.mean(a), st.mean(b)
                per = [ (x - y) / x * 100 for x, y in zip(a, b) if x > 0 ]
                return {
                    "n": len(a),
                    "sync_mean": round(ma, 2), "cfta_mean": round(mb, 2),
                    "delta_pct_of_means": round((ma - mb) / ma * 100, 1),
                    "median_reduction_pct": round(st.median([ (x-y)/x*100 for x,y in zip(a,b) if x>0 ]), 1) if a else None,
                    "mean_per_pair_reduction_pct": round(st.mean(per), 1) if per else None,
                    "cfta_faster_count": sum(1 for x, y in zip(a, b) if y < x),
                }

            cells[f"{c}|{th}"] = {
                "tier": c, "thinking": th,
                "n_attempt": n_attempt, "cfta_fire_rate": round(n_cfta_fired / n_attempt, 2) if n_attempt else None,
                "TTUA_fired_only": block(pair_s, pair_c),
                "TTFR_all_valid": block(pair_sfr, pair_cfr),
            }

    # print the matrix
    print("=== CFTA sensitivity matrix: complexity x thinking (DeepSeek flash=off / pro=on) ===")
    hdr = f"{'tier':4} {'thk':4} | {'CFTA fire':>8} | {'TTUA sync':>9} {'cfta':>6} {'d%mean':>6} {'median%':>7} {'wins':>5} | {'TTFR sync':>9} {'cfta':>6} {'d%mean':>6}"
    print(hdr); print("-" * len(hdr))
    for c in tiers:
        for th in think:
            d = cells[f"{c}|{th}"]
            t = d["TTUA_fired_only"]; f = d["TTFR_all_valid"]
            def g(b, k): return b.get(k, "-")
            print(f"{c:4} {th:4} | {str(d['cfta_fire_rate']):>8} | "
                  f"{g(t,'sync_mean'):>9} {g(t,'cfta_mean'):>6} {g(t,'delta_pct_of_means'):>6} "
                  f"{g(t,'median_reduction_pct'):>7} {str(g(t,'cfta_faster_count'))+'/'+str(g(t,'n')):>5} | "
                  f"{g(f,'sync_mean'):>9} {g(f,'cfta_mean'):>6} {g(f,'delta_pct_of_means'):>6}")

    (HERE / "sensitivity_summary.json").write_text(
        json.dumps(cells, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[OK] details -> sensitivity_summary.json")


if __name__ == "__main__":
    main()
