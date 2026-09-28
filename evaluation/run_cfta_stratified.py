"""CFTA 补充分析 — 分层延迟收益（按工具执行时长分层）

补充 run_cfta_experiments.py：按工具真实执行时长分层，
展示 CFTA 在不同延迟区间的收益差异（回答"何时 CFTA 帮助最大"）。
"""
from __future__ import annotations

import json
import random
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scipy import stats as sps  # noqa: E402

HOME = Path.home()
AUDIT_DB = HOME / ".weclaw" / "tool_audit.db"
HISTORY_DB = HOME / ".weclaw" / "history.db"
OUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 20260808
EXCLUDE_TOOLS = {"voice_output"}


def main():
    # 工具执行时长（success, 排除 voice_output）
    db = sqlite3.connect(AUDIT_DB)
    dvals = [ms / 1000.0 for (ms,) in db.execute(
        "SELECT duration_ms FROM tool_audit_log WHERE status='success'")]
    db.close()
    dvals = sorted(d for d in dvals if d >= 0)

    # 模型响应时长（与主实验同方法，用相同种子抽样）
    from datetime import datetime

    def parse_ts(s):
        return datetime.fromisoformat(s)

    db = sqlite3.connect(HISTORY_DB)
    rows = db.execute(
        "SELECT role, tool_calls_json, created_at FROM messages ORDER BY session_id, id"
    ).fetchall()
    db.close()
    model_times = []
    last_user_ts = None
    for role, tc_json, ts in rows:
        if role == "user":
            last_user_ts = parse_ts(ts)
        elif role == "assistant" and last_user_ts is not None:
            if not tc_json or tc_json in ("", "[]", "null"):
                dt = (parse_ts(ts) - last_user_ts).total_seconds()
                if 0.5 <= dt <= 120:
                    model_times.append(dt)
            last_user_ts = None

    rng = random.Random(RANDOM_SEED)
    tiers = [
        ("fast (≤0.5s)", lambda d: d <= 0.5),
        ("medium (0.5–3.5s)", lambda d: 0.5 < d <= 3.5),
        ("slow (>3.5s)", lambda d: d > 3.5),
    ]
    print(f"{'tier':22s} {'n':>6s} {'sync':>8s} {'cfta':>8s} {'reduction':>10s} {'pct':>7s}")
    tier_results = {}
    for name, cond in tiers:
        sub = [d for d in dvals if cond(d)]
        if not sub:
            continue
        n = min(1000, len(sub))
        ds = rng.sample(sub, n)
        ms = [rng.choice(model_times) for _ in range(n)]
        sync = [2 * m + d for m, d in zip(ms, ds)]
        cfta = list(ms)
        pct = [(s - c) / s for s, c in zip(sync, cfta)]
        t, p = sps.ttest_rel(sync, cfta)
        tier_results[name] = {
            "n": n,
            "tool_dur_mean": round(statistics.mean(ds), 2),
            "sync_mean": round(statistics.mean(sync), 2),
            "cfta_mean": round(statistics.mean(cfta), 2),
            "reduction_pct_mean": round(statistics.mean(pct) * 100, 1),
            "reduction_pct_median": round(statistics.median(pct) * 100, 1),
            "t": round(t, 2), "p": float(f"{p:.3e}"),
            "population_share": round(len(sub) / len(dvals) * 100, 1),
        }
        print(f"{name:22s} {n:6d} {statistics.mean(sync):8.2f} "
              f"{statistics.mean(cfta):8.2f} {statistics.mean(sync)-statistics.mean(cfta):10.2f} "
              f"{statistics.mean(pct)*100:6.1f}%")

    out_path = OUT_DIR / "experiment_stratified.json"
    out_path.write_text(json.dumps(tier_results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
