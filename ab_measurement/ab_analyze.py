"""C1-A 真实 A/B 配对统计分析（从 ab_raw.jsonl 计算，不做任何模拟）。

流程：
  1) 读 ab_raw.jsonl，剔除 error / ttfr_s is None 的 trial（计数并报告）。
  2) 按 (prompt_id, arm) 对 rep 求均值 → 每 prompt 的 TTFR/TTUA(sync,cfta)。
  3) 逐 prompt 配对：ΔTTFR = TTFR_sync − TTFR_cfta；ΔTTUA 同理。
  4) 真实检验：paired t-test、Wilcoxon signed-rank、Cohen's d(paired)、
     bootstrap 95% CI（mean Δ）。TTFR 与 TTUA 分别报告（回应评审 C2 指标拆分）。
  5) 分层：按 tool_fired（cfta 臂是否触发工具）分别配对。
产物：ab_summary.json + 终端打印可贴 LaTeX 的数值。

用法：python ab_analyze.py [--raw ab_raw.jsonl] [--out ab_summary.json] [--boot 5000] [--seed 20260927]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_grouped(raw_path: Path):
    recs = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(recs)
    valid = [r for r in recs if not r.get("error") and r.get("ttfr_s") is not None]
    dropped = total - len(valid)

    # (prompt_id, arm) -> list of (ttfr, ttua)
    acc: dict[tuple[str, str], list] = defaultdict(list)
    fired: dict[str, bool] = {}  # prompt 是否在任一臂触发过工具（用于分层）
    for r in valid:
        acc[(r["prompt_id"], r["arm"])].append((r["ttfr_s"], r["ttua_s"]))
        if r.get("tool_fired"):
            fired[r["prompt_id"]] = True

    per_prompt: dict[str, dict] = {}
    for pid in {p for p, _ in acc}:
        if (pid, "sync") not in acc or (pid, "cfta") not in acc:
            continue  # 配对不完整则跳过该 prompt（保持 paired 语义）
        s = acc[(pid, "sync")]
        c = acc[(pid, "cfta")]
        per_prompt[pid] = {
            "ttfr_sync": statistics.mean(x[0] for x in s),
            "ttfr_cfta": statistics.mean(x[0] for x in c),
            "ttua_sync": statistics.mean(x[1] for x in s),
            "ttua_cfta": statistics.mean(x[1] for x in c),
            "tool_fired": bool(fired.get(pid, False)),
            "n_rep_sync": len(s),
            "n_rep_cfta": len(c),
        }
    return recs, total, dropped, per_prompt


def bootstrap_ci(deltas, n_boot, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3),
            round(float(np.percentile(means, 97.5)), 3)]


def paired_block(sync_vals, cfta_vals, n_boot, seed, label):
    s = np.asarray(sync_vals, dtype=float)
    c = np.asarray(cfta_vals, dtype=float)
    deltas = s - c
    n = int(deltas.size)
    block = {"metric": label, "n_pairs": n}
    if n < 2:
        block["note"] = "配对样本不足，跳过检验"
        return block
    block.update({
        "sync_mean": round(float(s.mean()), 2),
        "sync_sd": round(float(s.std(ddof=1)), 2),
        "cfta_mean": round(float(c.mean()), 2),
        "cfta_sd": round(float(c.std(ddof=1)), 2),
        "delta_mean": round(float(deltas.mean()), 2),
        "delta_median": round(float(np.median(deltas)), 2),
    })
    # 相对下降：基于每对 (cfta/sync) 的比值
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (s - c) / s
    rel = rel[np.isfinite(rel)]
    if rel.size:
        block["reduction_pct_mean"] = round(float(rel.mean() * 100), 1)
        block["reduction_pct_median"] = round(float(np.median(rel) * 100), 1)
    # 配对 t 检验（真实数据，null 非构造性为假）
    if float(deltas.std(ddof=1)) > 0:
        t, p = sps.ttest_rel(s, c)
        block["paired_t"] = {"t": round(float(t), 3), "p": float(f"{p:.4g}")}
        w, pw = sps.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided") \
            if np.count_nonzero(deltas) else (float("nan"), float("nan"))
        block["wilcoxon"] = {"W": float(w), "p": float(f"{pw:.4g}")}
        d_cohen = float(deltas.mean() / deltas.std(ddof=1))  # 配对 Cohen's d (dz)
        block["cohens_d_paired"] = round(d_cohen, 2)
    block["delta_boot_ci95"] = bootstrap_ci(deltas, n_boot, seed)
    return block


def main():
    ap = argparse.ArgumentParser(description="C1-A A/B 配对统计分析")
    ap.add_argument("--raw", default=str(HERE / "ab_raw.jsonl"))
    ap.add_argument("--out", default=str(HERE / "ab_summary.json"))
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        sys.exit(f"错误：找不到 {raw_path}，请先运行 ab_harness.py 采集真实数据")

    _, total, dropped, per_prompt = load_grouped(raw_path)
    if not per_prompt:
        sys.exit("错误：无有效配对样本（检查 ab_raw.jsonl 是否为真实数据）")

    pids = sorted(per_prompt)
    def col(k):
        return [per_prompt[p][k] for p in pids]

    ttfr = paired_block(col("ttfr_sync"), col("ttfr_cfta"), args.boot, args.seed, "TTFR")
    ttua = paired_block(col("ttua_sync"), col("ttua_cfta"), args.boot, args.seed, "TTUA")

    # 分层：按 tool_fired
    fired_pids = [p for p in pids if per_prompt[p]["tool_fired"]]
    chat_pids = [p for p in pids if not per_prompt[p]["tool_fired"]]
    strata = {}
    for name, grp in [("tool_fired", fired_pids), ("no_tool", chat_pids)]:
        if len(grp) >= 2:
            strata[name] = {
                "n": len(grp),
                "ttfr": paired_block([per_prompt[p]["ttfr_sync"] for p in grp],
                                     [per_prompt[p]["ttfr_cfta"] for p in grp],
                                     args.boot, args.seed, f"TTFR::{name}"),
            }

    summary = {
        "meta": {"raw_total_trials": total, "dropped_error_or_null": dropped,
                 "n_pairs": len(pids), "boot": args.boot, "seed": args.seed},
        "TTFR": ttfr,
        "TTUA": ttua,
        "stratified": strata,
        "per_prompt": {p: {k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in per_prompt[p].items()} for p in pids},
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 终端报告
    print("=" * 64)
    print(f"C1-A 实测分析  有效配对 prompt n={len(pids)}（剔除 {dropped}/{total} 无效 trial）")
    print("=" * 64)
    for blk in (ttfr, ttua):
        print(f"\n[{blk['metric']}] n={blk['n_pairs']}")
        for k in ("sync_mean", "sync_sd", "cfta_mean", "cfta_sd", "delta_mean",
                  "delta_median", "reduction_pct_mean", "reduction_pct_median",
                  "cohens_d_paired", "delta_boot_ci95"):
            if k in blk:
                print(f"  {k:22s} = {blk[k]}")
        for k in ("paired_t", "wilcoxon"):
            if k in blk:
                print(f"  {k:22s} = {blk[k]}")

    print("\n可用 LaTeX 片段（TTFR）：")
    if "paired_t" in ttfr:
        print("  Mean TTFR sync $\\pm$ sd: "
              f"{ttfr['sync_mean']} $\\pm$ {ttfr['sync_sd']} s; "
              f"CFTA: {ttfr['cfta_mean']} $\\pm$ {ttfr['cfta_sd']} s; "
              f"reduction {ttfr.get('reduction_pct_mean')}\\% "
              f"(paired $t$={ttfr['paired_t']['t']}, $p$={ttfr['paired_t']['p']}; "
              f"Cohen's $d_z$={ttfr.get('cohens_d_paired')})")
    print(f"\n✓ 完整结果 → {args.out}")


if __name__ == "__main__":
    main()
