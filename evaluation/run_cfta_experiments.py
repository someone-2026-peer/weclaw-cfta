"""CFTA 论文补充实验 — 基于 WeClaw 真实生产数据

数据源（全部为真实数据，无合成/虚构）:
  1. ~/.weclaw/tool_audit.db  — 真实工具调用记录（含 duration_ms / status）
  2. ~/.weclaw/history.db     — 真实会话消息（含 created_at 时间戳）
  3. src.core.prompts.detect_intent_with_confidence — 真实意图分类器代码

【数据治理（P-1 门禁）】生产库持续写入，为保证论文已发布数字可复现：
  - 默认优先读取 snapshots/ 下的冻结快照（make_cfta_snapshot.py 生成）；
  - 全部查询带 --until 日期护栏（默认 2026-08-08，与论文声明窗口一致）；
  - --live 可强制读生产库（仅限快照缺失时的应急排查）。

实验:
  Exp1 意图分类器评估: 用真实会话消息构造标注集，跑真实分类器
  Exp2 延迟测量与统计检验: 真实工具执行时长 + 真实模型响应时长 → 配对样本检验
  Exp3 消融实验: 基于真实路由分布与真实时长数据的分析性消融

运行: .venv\\Scripts\\python.exe run_cfta_experiments.py [--until 2026-08-08] [--live]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scipy import stats as sps  # noqa: E402

from src.core.prompts import detect_intent_with_confidence  # noqa: E402

# Windows 控制台 GBK 编码防护：强制 UTF-8 输出
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HOME = Path.home()
OUT_DIR = Path(__file__).resolve().parent
SNAP_DIR = OUT_DIR / "snapshots"

# 冻结快照优先（make_cfta_snapshot.py 产物）；缺失时回退生产库并警告
AUDIT_SNAP = SNAP_DIR / "tool_audit_20260808.db"
HISTORY_SNAP = SNAP_DIR / "history_20260808.db"
AUDIT_DB_LIVE = HOME / ".weclaw" / "tool_audit.db"
HISTORY_DB_LIVE = HOME / ".weclaw" / "history.db"

VOICE_PREFIX_RE = re.compile(r"^\[语音对话模式\][^\n]*\n?")
RANDOM_SEED = 20260808
# voice_output 的 duration 是 TTS 播放全程，不代表工具执行延迟，予以排除
EXCLUDE_TOOLS = {"voice_output"}
MAX_PER_CLASS = 500
MAX_PAIRED = 2000

# 运行时由 main() 根据 --until / --live 设置
UNTIL_END = "2026-08-08 23:59:59.999999"  # 日期护栏（容忍 T/空格分隔符差异）
AUDIT_DB = AUDIT_DB_LIVE
HISTORY_DB = HISTORY_DB_LIVE


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ======================================================================
# Exp1: 意图分类器评估
# ======================================================================

def build_labeled_corpus():
    """从 history.db 构造标注集。

    标注规则（基于会话内真实消息序列）:
      - tool_needed: 用户消息后、下一条用户消息前，存在带 tool_calls 的 assistant 消息
      - pure_chat:   用户消息后、下一条用户消息前，无任何 tool_calls
    """
    db = sqlite3.connect(HISTORY_DB)
    rows = db.execute(
        "SELECT session_id, role, content, tool_calls_json, created_at "
        "FROM messages WHERE replace(created_at, 'T', ' ') <= ? "
        "ORDER BY session_id, id", (UNTIL_END,)
    ).fetchall()
    db.close()

    tool_needed, pure_chat = [], []
    pending_user = None  # (content, created_at)
    pending_has_tool = False

    def flush():
        nonlocal pending_user, pending_has_tool
        if pending_user is not None:
            text = VOICE_PREFIX_RE.sub("", pending_user[0]).strip()
            if len(text) >= 2:
                (tool_needed if pending_has_tool else pure_chat).append(
                    {"text": text, "ts": pending_user[1]}
                )
        pending_user, pending_has_tool = None, False

    for _, role, content, tc_json, ts in rows:
        if role == "user":
            flush()
            pending_user = (content or "", ts)
        elif role == "assistant" and pending_user is not None:
            if tc_json and tc_json not in ("", "[]", "null"):
                try:
                    if json.loads(tc_json):
                        pending_has_tool = True
                except (json.JSONDecodeError, TypeError):
                    pass
    flush()

    rng = random.Random(RANDOM_SEED)
    tool_needed = sorted(tool_needed, key=lambda x: x["ts"], reverse=True)
    pure_chat = sorted(pure_chat, key=lambda x: x["ts"], reverse=True)
    return tool_needed[:MAX_PER_CLASS], pure_chat[:MAX_PER_CLASS]


def route(conf: float) -> int:
    """CFTA 三级分流（与 gui_app.py 实际实现一致）。"""
    if conf >= 0.5:
        return 3
    if conf > 0:
        return 2
    return 1


def exp1_intent_classifier(tool_needed, pure_chat):
    print("=" * 64)
    print("Exp1: 意图分类器评估（真实分类器 + 真实会话标注集）")
    print("=" * 64)

    results = []
    for item in tool_needed:
        r = detect_intent_with_confidence(item["text"])
        results.append({"label": "tool_needed", "path": route(r.confidence),
                        "conf": round(r.confidence, 3),
                        "primary": r.primary_intent, "text": item["text"][:60]})
    for item in pure_chat:
        r = detect_intent_with_confidence(item["text"])
        results.append({"label": "pure_chat", "path": route(r.confidence),
                        "conf": round(r.confidence, 3),
                        "primary": r.primary_intent, "text": item["text"][:60]})

    tn = [r for r in results if r["label"] == "tool_needed"]
    pc = [r for r in results if r["label"] == "pure_chat"]

    tn_async_sync = sum(1 for r in tn if r["path"] in (2, 3))  # 工具被检测执行
    tn_missed = sum(1 for r in tn if r["path"] == 1)           # 漏检（conf=0）
    tn_path3 = sum(1 for r in tn if r["path"] == 3)
    pc_fp = sum(1 for r in pc if r["path"] in (2, 3))          # 纯聊天误触发
    pc_path3_fp = sum(1 for r in pc if r["path"] == 3)

    n_all = len(results)
    correct = tn_async_sync + (len(pc) - pc_fp)

    summary = {
        "sample_size": {"tool_needed": len(tn), "pure_chat": len(pc)},
        "tool_recall": round(tn_async_sync / len(tn), 4),           # 工具需求检出率
        "tool_miss_rate": round(tn_missed / len(tn), 4),            # 漏检率
        "path3_rate_among_tool": round(tn_path3 / len(tn), 4),      # 明确工具→同步路径
        "chat_fp_rate": round(pc_fp / len(pc), 4),                  # 纯聊天误触发率
        "chat_path3_fp": pc_path3_fp,
        "routing_accuracy": round(correct / n_all, 4),              # 严格准确率
        "path_distribution": {
            "path1": sum(1 for r in results if r["path"] == 1),
            "path2": sum(1 for r in results if r["path"] == 2),
            "path3": sum(1 for r in results if r["path"] == 3),
        },
        "confidence_stats": {
            "tool_needed_mean": round(statistics.mean(r["conf"] for r in tn), 3),
            "pure_chat_mean": round(statistics.mean(r["conf"] for r in pc), 3),
        },
    }

    # 漏检样本 Top 示例
    missed_examples = [r for r in tn if r["path"] == 1][:10]
    fp_examples = [r for r in pc if r["path"] in (2, 3)][:10]

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n漏检示例 (tool_needed → Path1):")
    for r in missed_examples:
        print(f"  [{r['conf']:.2f}] {r['text']}")
    print("\n误触发示例 (pure_chat → Path2/3):")
    for r in fp_examples:
        print(f"  [{r['conf']:.2f}|{r['primary']}] {r['text']}")

    return summary, results, missed_examples, fp_examples


# ======================================================================
# Exp2: 真实延迟测量 + 统计检验
# ======================================================================

def measure_model_response_times():
    """从 history.db 测量真实模型响应时长（用户消息 → 下一条 assistant 消息）。

    仅取无工具调用的 assistant 响应（对应 CFTA 快速回复场景），过滤 0.5–120s。
    """
    db = sqlite3.connect(HISTORY_DB)
    rows = db.execute(
        "SELECT role, tool_calls_json, created_at FROM messages "
        "WHERE replace(created_at, 'T', ' ') <= ? ORDER BY session_id, id",
        (UNTIL_END,)
    ).fetchall()
    db.close()

    times = []
    last_user_ts = None
    for role, tc_json, ts in rows:
        if role == "user":
            last_user_ts = parse_ts(ts)
        elif role == "assistant" and last_user_ts is not None:
            no_tools = not tc_json or tc_json in ("", "[]", "null")
            if no_tools:
                dt = (parse_ts(ts) - last_user_ts).total_seconds()
                if 0.5 <= dt <= 120:
                    times.append(dt)
            last_user_ts = None  # 只统计首次响应
    return times


def measure_tool_durations():
    """从 tool_audit.db 提取真实工具执行时长（仅 success）。"""
    db = sqlite3.connect(AUDIT_DB)
    rows = db.execute(
        "SELECT tool_name, duration_ms FROM tool_audit_log "
        "WHERE status='success' AND replace(timestamp, 'T', ' ') <= ?",
        (UNTIL_END,)
    ).fetchall()
    status_counts = dict(db.execute(
        "SELECT status, COUNT(*) FROM tool_audit_log "
        "WHERE replace(timestamp, 'T', ' ') <= ? GROUP BY status",
        (UNTIL_END,)).fetchall())
    db.close()

    durations = [(name, ms / 1000.0) for name, ms in rows if name not in EXCLUDE_TOOLS]
    return durations, status_counts


def exp2_latency(model_times, durations, status_counts):
    print("\n" + "=" * 64)
    print("Exp2: 延迟测量与统计检验（真实数据）")
    print("=" * 64)

    # --- 工具执行时长分布 ---
    dvals = sorted(d for _, d in durations)
    n = len(dvals)
    print(f"\n工具执行时长 (success, 排除 voice_output): n={n}")
    print(f"  mean={statistics.mean(dvals):.2f}s  median={statistics.median(dvals):.2f}s "
          f"std={statistics.stdev(dvals):.2f}s")
    print(f"  P75={dvals[int(n*0.75)]:.2f}s  P90={dvals[int(n*0.90)]:.2f}s "
          f"P99={dvals[int(n*0.99)]:.2f}s")

    # 真实完成率分解（排除 voice_output 需要单独查询，这里用全量 status 分布）
    total_all = sum(status_counts.values())
    print(f"\n真实 status 分布 (n={total_all}):")
    for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} {v:6d} ({v/total_all*100:.2f}%)")

    # --- 模型响应时长分布 ---
    print(f"\n模型无工具响应时长 (history.db): n={len(model_times)}")
    print(f"  mean={statistics.mean(model_times):.2f}s  "
          f"median={statistics.median(model_times):.2f}s "
          f"std={statistics.stdev(model_times):.2f}s")

    # --- 配对样本构造 ---
    # sync_i  = m_i + dur_i + m_i   (模型思考 + 工具执行 + 模型再生成)
    # cfta_i  = m_i                  (快速回复即首响应)
    rng = random.Random(RANDOM_SEED)
    sample_n = min(MAX_PAIRED, len(dvals))
    dur_sample = rng.sample(dvals, sample_n)
    m_sample = [rng.choice(model_times) for _ in range(sample_n)]

    sync = [2 * m + d for m, d in zip(m_sample, dur_sample)]
    cfta = list(m_sample)
    reduction = [s - c for s, c in zip(sync, cfta)]
    pct = [r / s for r, s in zip(reduction, sync)]

    # --- 统计检验 ---
    t_stat, t_p = sps.ttest_rel(sync, cfta)
    w_stat, w_p = sps.wilcoxon(sync, cfta)
    mean_red = statistics.mean(reduction)
    se = statistics.stdev(reduction) / (len(reduction) ** 0.5)
    ci_lo, ci_hi = mean_red - 1.96 * se, mean_red + 1.96 * se
    pooled_sd = statistics.stdev(sync + cfta)
    cohens_d = mean_red / pooled_sd if pooled_sd else 0.0

    summary = {
        "tool_duration_sec": {
            "n": n, "mean": round(statistics.mean(dvals), 2),
            "median": round(statistics.median(dvals), 2),
            "std": round(statistics.stdev(dvals), 2),
            "p90": round(dvals[int(n * 0.90)], 2),
        },
        "model_response_sec": {
            "n": len(model_times), "mean": round(statistics.mean(model_times), 2),
            "median": round(statistics.median(model_times), 2),
            "std": round(statistics.stdev(model_times), 2),
        },
        "paired_samples": sample_n,
        "sync_latency_sec": {
            "mean": round(statistics.mean(sync), 2),
            "std": round(statistics.stdev(sync), 2),
        },
        "cfta_latency_sec": {
            "mean": round(statistics.mean(cfta), 2),
            "std": round(statistics.stdev(cfta), 2),
        },
        "reduction_sec_mean": round(mean_red, 2),
        "reduction_ci95": [round(ci_lo, 2), round(ci_hi, 2)],
        "reduction_pct_mean": round(statistics.mean(pct) * 100, 1),
        "reduction_pct_median": round(statistics.median(pct) * 100, 1),
        "paired_t_test": {"t": round(t_stat, 2), "p": float(f"{t_p:.3e}")},
        "wilcoxon_test": {"W": float(w_stat), "p": float(f"{w_p:.3e}")},
        "cohens_d": round(cohens_d, 2),
        "status_counts": status_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary, sync, cfta, dur_sample


# ======================================================================
# Exp3: 消融实验
# ======================================================================

def exp3_ablation(exp1_summary, exp1_results, model_times, durations, exp2_summary):
    print("\n" + "=" * 64)
    print("Exp3: 消融实验（真实路由分布 + 真实时长）")
    print("=" * 64)

    dist = exp1_summary["path_distribution"]
    n_total = sum(dist.values())
    n_p1, n_p2, n_p3 = dist["path1"], dist["path2"], dist["path3"]

    # --- 组件1: 三级意图分流的收益 ---
    # 无分流 = 所有输入都做 deferred detection（Path2 逻辑）
    # 收益 = Path1 输入免去的 deferred 调用 + Path3 免去的双重推理
    deferred_saved = n_p1          # Path1 直接跳过后台检测
    deferred_with_routing = n_p2   # 仅 Path2 需要后台检测
    deferred_without_routing = n_total

    # deferred detection 的真实 token 成本代理: tools.json schema 大小
    tools_json = Path(__file__).resolve().parents[3] / "config" / "tools.json"
    schema_chars = tools_json.read_text(encoding="utf-8").__len__() if tools_json.exists() else 0
    schema_tokens_est = schema_chars // 3  # 中文 JSON 约 3 字符/token（保守估计）

    # --- 组件2: Deferred Detection 的收益（延迟维度） ---
    sync_mean = exp2_summary["sync_latency_sec"]["mean"]
    cfta_mean = exp2_summary["cfta_latency_sec"]["mean"]

    ablation = {
        "corpus_size": n_total,
        "path_distribution": dist,
        "component1_tri_level_routing": {
            "deferred_calls_with_routing": deferred_with_routing,
            "deferred_calls_without_routing": deferred_without_routing,
            "deferred_calls_saved": deferred_saved,
            "savings_rate": round(deferred_saved / n_total, 4),
            "schema_tokens_estimated": schema_tokens_est,
            "tokens_saved_estimated": deferred_saved * schema_tokens_est,
        },
        "component2_deferred_detection": {
            "full_cfta_first_response_sec": cfta_mean,
            "no_deferred_sync_latency_sec": sync_mean,
            "latency_increase_without_sec": round(sync_mean - cfta_mean, 2),
            "reduction_pct": exp2_summary["reduction_pct_mean"],
        },
    }
    print(json.dumps(ablation, ensure_ascii=False, indent=2))
    return ablation


def main():
    global AUDIT_DB, HISTORY_DB, UNTIL_END
    ap = argparse.ArgumentParser(description="CFTA 论文补充实验")
    ap.add_argument("--until", default="2026-08-08",
                    help="数据采集截止日期（含当日），默认 2026-08-08（与论文声明窗口一致）")
    ap.add_argument("--live", action="store_true",
                    help="强制读生产库（默认优先读 snapshots/ 冻结快照）")
    ap.add_argument("--out", default="",
                    help="结果输出路径（默认 experiment_results.json；"
                         "验证重跑时建议指定副本路径，避免覆盖已发布数字）")
    args = ap.parse_args()
    # 支持两种格式：日期（YYYY-MM-DD，含当日全天）或完整时间戳（YYYY-MM-DDTHH:MM:SS）
    UNTIL_END = args.until if "T" in args.until else f"{args.until} 23:59:59.999999"

    if args.live:
        AUDIT_DB, HISTORY_DB = AUDIT_DB_LIVE, HISTORY_DB_LIVE
        print("⚠ --live 模式：读生产库（结果不可复现，仅限应急排查）")
    elif AUDIT_SNAP.exists() and HISTORY_SNAP.exists():
        AUDIT_DB, HISTORY_DB = AUDIT_SNAP, HISTORY_SNAP
        print(f"✓ 使用冻结快照：{AUDIT_SNAP.name} / {HISTORY_SNAP.name}（cutoff <= {args.until}）")
    else:
        print("⚠ 未找到冻结快照（请先运行 make_cfta_snapshot.py），回退生产库 + 日期护栏")

    print("构造标注语料（真实会话消息）...")
    tool_needed, pure_chat = build_labeled_corpus()
    print(f"  tool_needed={len(tool_needed)}, pure_chat={len(pure_chat)}")

    exp1_summary, exp1_results, missed, fps = exp1_intent_classifier(tool_needed, pure_chat)

    print("\n测量真实模型响应时长...")
    model_times = measure_model_response_times()
    print(f"  n={len(model_times)}")
    print("提取真实工具执行时长...")
    durations, status_counts = measure_tool_durations()
    print(f"  n={len(durations)}")

    exp2_summary, sync, cfta, dur_sample = exp2_latency(model_times, durations, status_counts)
    ablation = exp3_ablation(exp1_summary, exp1_results, model_times, durations, exp2_summary)

    out = {
        "meta": {
            "date": "2026-08-08",
            "data_cutoff": args.until,
            "data_sources": [
                f"tool_audit.db: {sum(status_counts.values()):,} real tool calls "
                f"(2026-05-25 ~ {args.until})",
                f"history.db: {len(tool_needed) + len(pure_chat):,} labeled samples "
                f"(tool_needed={len(tool_needed)}, pure_chat={len(pure_chat)}, "
                f"each capped at {MAX_PER_CLASS}, newest-first)",
                "detect_intent_with_confidence: real classifier from src/core/prompts.py",
            ],
            "seed": RANDOM_SEED,
        },
        "exp1_intent_classifier": exp1_summary,
        "exp2_latency_statistics": exp2_summary,
        "exp3_ablation": ablation,
    }
    out_path = Path(args.out) if args.out else OUT_DIR / "experiment_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
