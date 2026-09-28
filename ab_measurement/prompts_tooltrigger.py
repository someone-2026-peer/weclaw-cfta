"""生成并冻结 C1-A 实测的工具触发型 prompt 集（真实语料，非合成）。

数据来源与 run_cfta_experiments.py 一致：history.db（优先 ../snapshots 冻结快照）
的 tool_needed 标注集 —— 即"用户消息后、下一条用户消息前存在带 tool_calls 的
assistant 消息"的真实会话片段。按 detect_intent_with_confidence 的 primary_intent
分层抽样，保证 23 类意图覆盖度；一经写出即固化为研究材料。

用法:
  python prompts_tooltrigger.py --n 40 --seed 20260927 [--until 2026-08-08] [--live]
产物:
  prompts_tooltrigger.jsonl  每行 {"prompt_id","text","text_hash","primary_intent","confidence"}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# REPO_ROOT: this script ships under ab_measurement/ inside a snapshot of the private
# WeClaw monorepo; when placed back at its original depth (ab_measurement/.../docs/)
# parents[4] resolves to the repo root, which is where the `src` package lives.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.core.prompts import detect_intent_with_confidence  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
SNAP_HIST = PAPER_DIR / "snapshots" / "history_20260808.db"
LIVE_HIST = Path.home() / ".weclaw" / "history.db"
VOICE_PREFIX_RE = re.compile(r"^\[语音对话模式\][^\n]*\n?")

OUT_DEFAULT = HERE / "prompts_tooltrigger.jsonl"


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def collect_tool_needed(db_path: Path, until_end: str) -> list[dict]:
    """从 history.db 抽取"确有工具调用"的真实用户消息（同 Exp1 标注规则）。"""
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT session_id, role, content, tool_calls_json, created_at "
        "FROM messages WHERE replace(created_at, 'T', ' ') <= ? "
        "ORDER BY session_id, id",
        (until_end,),
    ).fetchall()
    db.close()

    out: list[dict] = []
    pending_user: tuple[str, str] | None = None
    pending_has_tool = False

    def flush():
        nonlocal pending_user, pending_has_tool
        if pending_user is not None and pending_has_tool:
            text = VOICE_PREFIX_RE.sub("", pending_user[0]).strip()
            if 4 <= len(text) <= 120:  # 过短无意义、过长偏离典型请求
                out.append({"text": text, "ts": pending_user[1]})
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
    return out


def stratified_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    """按 primary_intent 分层：每类先取一条代表，再轮转补足到 n。"""
    rng = random.Random(seed)
    # 去重（相同文本只留一条），保留最新
    seen: dict[str, dict] = {}
    for item in sorted(pool, key=lambda x: x["ts"], reverse=True):
        seen.setdefault(item["text"], item)
    uniq = list(seen.values())

    by_intent: dict[str, list[dict]] = defaultdict(list)
    for item in uniq:
        try:
            r = detect_intent_with_confidence(item["text"])
            intent = r.primary_intent or "_none"
            conf = round(float(r.confidence), 3)
        except Exception:
            intent, conf = "_err", 0.0
        item = {**item, "primary_intent": intent, "confidence": conf}
        # 仅保留分类器判为工具触发（conf>0）的样本，符合"工具触发型"研究材料定义
        if conf > 0:
            by_intent[intent].append(item)

    for lst in by_intent.values():
        rng.shuffle(lst)

    selected: list[dict] = []
    intents = sorted(by_intent, key=lambda k: -len(by_intent[k]))
    # 轮转取样，保证覆盖面
    while len(selected) < n and any(by_intent[i] for i in intents):
        for i in intents:
            if by_intent[i] and len(selected) < n:
                selected.append(by_intent[i].pop())
    return selected[:n]


def main():
    ap = argparse.ArgumentParser(description="C1-A prompt 集生成（真实语料分层抽样）")
    ap.add_argument("--n", type=int, default=40, help="目标 prompt 条数（默认 40）")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--until", default="2026-08-08")
    ap.add_argument("--live", action="store_true", help="强制读生产 history.db")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    until_end = args.until if "T" in args.until else f"{args.until} 23:59:59.999999"
    db = LIVE_HIST if args.live else (SNAP_HIST if SNAP_HIST.exists() else LIVE_HIST)
    if not db.exists():
        sys.exit(f"错误：找不到 history.db（尝试：{db}）。请先运行 make_cfta_snapshot.py。")
    print(f"✓ 数据源: {db}（cutoff <= {args.until}）")

    pool = collect_tool_needed(db, until_end)
    print(f"  tool_needed 候选（去重前）: {len(pool)}")
    sel = stratified_sample(pool, args.n, args.seed)
    if len(sel) < args.n:
        print(f"⚠ 候选不足，仅取 {len(sel)} 条（可减小 --n 或放宽过滤）")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(sel, start=1):
            pid = f"p{idx:03d}"
            h = hashlib.sha1(item["text"].encode("utf-8")).hexdigest()[:12]
            rec = {
                "prompt_id": pid,
                "text": item["text"],
                "text_hash": h,
                "primary_intent": item["primary_intent"],
                "confidence": item["confidence"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    intents = sorted({s["primary_intent"] for s in sel})
    print(f"✓ 已冻结 {len(sel)} 条 prompt → {out_path}")
    print(f"  覆盖意图类别 {len(intents)}: {intents}")


if __name__ == "__main__":
    main()
