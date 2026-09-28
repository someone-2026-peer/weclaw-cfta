"""C1-A 真实端到端 A/B 采集 harness（Sync vs. CFTA）。

对同一批"工具触发型"prompt，分别运行 WeClaw 两条真实执行路径，
用 time.perf_counter() 记录两个独立墙钟指标：
  - TTFR：用户可见首字符延迟
  - TTUA：含工具数据的权威答案就绪延迟

真实 API（已核对源码）：
  Sync 基线  : Agent.chat_stream(prompt)                agent.py:3902
  CFTA 快速   : Agent.chat_stream_voice_fast(prompt)     agent.py:5185
  CFTA 异步工具: Agent.process_deferred_tools(...)        agent.py:5557
  意图分流阈值 : 与 gui_app.py:813 一致（conf>=0.5→路径3, >0→路径2, else 路径1）
Headless 引导复用 src/app.py:200-258 CLI 入口（无需 Qt/GUI）。

【铁律】无法初始化真实模型/工具注册表时直接报错退出，绝不产出占位/伪造行。
产物：ab_raw.jsonl（一行一 trial）+ ab_meta.json（环境凭证）。
用法：
  python ab_harness.py --prompts prompts_tooltrigger.jsonl --rep 5 --warmup 2 \
                       [--model KEY] [--sleep 3] [--out ab_raw.jsonl]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 环境变量（模型 API key）：优先仓库根 .env，其次 data/.env
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    if (REPO_ROOT / "data" / ".env").exists():
        load_dotenv(REPO_ROOT / "data" / ".env")
except Exception:
    pass

# 真实核心组件（活体代码，与生产同路径）
from src.core.agent import Agent  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.tools.registry import create_default_registry  # noqa: E402
from src.tools.base import ToolResult, ToolResultStatus  # noqa: E402
from src.core.config import resolve_default_model_key  # noqa: E402
from src.core.prompts import detect_intent_with_confidence  # noqa: E402

HERE = Path(__file__).resolve().parent

# ---- 安全护栏：只读工具白名单（deny-by-default）---------------------
# 采集在真实执行工具时绝不允许产生本机/外发副作用，因此从注册表彻底
# 摘除一切非只读工具（shell/file/python_runner/notify/email/wechat/browser/
# media_*/doc生成/控制类/quant_trading 跑飞源 等），仅保留下列查询/检索/本地计算类。
# 依据：实测暴露 shell/file 为常驻 CORE 工具，按意图过滤挡不住，必须注册表级摘除。
# 【pilot 泄漏修正】data_visualization/data_processor/stock_query/financial_report
# 经 _resolve_dependencies + 懒加载 DI 会把 shell/python_runner 重新注入并真实执行
# （rep=1 pilot 中 p010 真执行了 shell+python），已从白名单剔除。
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

# canary 模式下置为 True：拒绝一切工具执行（仅用于护栏拦截自检）
_DENY_ALL = False


def install_execution_guard(registry) -> None:
    """执行层硬护栏：包装 registry.call_function（三处执行入口共用唯一通道，
    agent.py:3725/4934/5751）。凡解析出的 tool_name 不在只读白名单，直接返回
    DENIED，绝不落入 tool.safe_execute —— 彻底封堵依赖注入/DI 绕过注册表裁剪。
    """
    original = registry.call_function

    async def guarded(func_name: str, arguments: dict):
        if _DENY_ALL:
            return ToolResult(
                status=ToolResultStatus.DENIED,
                error=f"[AB-SAFETY] deny-all mode; refused {func_name}",
            )
        resolved = registry.resolve_function_name(func_name)
        tool_name = resolved[0] if resolved else str(func_name).split("_")[0].split(".")[0]
        if tool_name not in READONLY_WHITELIST:
            print(f"  [AB-SAFETY] BLOCKED execution: {func_name} (tool={tool_name})")
            return ToolResult(
                status=ToolResultStatus.DENIED,
                error=f"[AB-SAFETY] tool '{tool_name}' not in readonly whitelist",
            )
        return await original(func_name, arguments)

    registry.call_function = guarded


def apply_readonly_whitelist(registry) -> list[str]:
    """从注册表（含 _tools / _lazy_tools / _tool_configs）移除一切非白名单工具。
    返回被移除的工具名。unregister 已清 _tools/_func_map/缓存；额外 pop 懒加载与配置。
    """
    removed: list[str] = []
    for name in list(registry.list_all_tool_names()):
        if name in READONLY_WHITELIST:
            continue
        try:
            registry.unregister(name)
        except Exception:
            pass
        getattr(registry, "_lazy_tools", {}).pop(name, None)
        getattr(registry, "_tool_configs", {}).pop(name, None)
        removed.append(name)
    try:
        registry.invalidate_schema_cache()
    except Exception:
        pass
    return sorted(removed)


# 每次 trial 的工具调用收集器（由 EventBus 回调写入）
_REC: dict = {"tools": []}


def _on_tool_call(_etype, data):
    """记录 Sync/CFTA 运行期间发生的工具调用名（用于 tool_fired 与分层）。"""
    name = None
    if isinstance(data, dict):
        name = data.get("tool_name") or data.get("name")
    else:
        name = getattr(data, "tool_name", None) or getattr(data, "name", None)
    _REC["tools"].append(name or "unknown")


async def build_agent(model_key: str = ""):
    """headless 构造 Agent（与 src/app.py:200-258 一致）；无可用模型则抛错。"""
    model_registry = ModelRegistry()
    models = model_registry.list_models()
    if not models:
        raise RuntimeError("未找到任何可用模型配置 —— 拒绝产出占位数据。")
    default_key = model_key or resolve_default_model_key(model_registry) or models[0].key
    tool_registry = create_default_registry()

    # 【安全护栏·第一层】注册表级摘除非白名单工具（防 schema 暴露与正常调度）
    removed = apply_readonly_whitelist(tool_registry)
    # 【安全护栏·第二层】执行层硬拦截（防 _resolve_dependencies/DI 绕过注册表）
    install_execution_guard(tool_registry)
    print(f"✓ 只读白名单：摘除 {len(removed)} 个非只读工具；保留 {len(tool_registry.list_all_tool_names())} 个；执行层护栏已安装")

    cron_tool = tool_registry.get_tool("cron")
    if cron_tool and hasattr(cron_tool, "set_agent_dependencies"):
        cron_tool.set_agent_dependencies(model_registry, tool_registry)

    intent_mode, intent_llm_model = "rule", ""
    max_tools, max_fail = 3, 3
    try:
        from src.core.config import AppConfig
        opt = AppConfig.load().get("agent.tool_optimization", {}) or {}
        intent_mode = opt.get("intent_mode", "rule")
        intent_llm_model = opt.get("intent_llm_model", resolve_default_model_key())
        max_tools = opt.get("max_tools_per_call", 3)
        max_fail = opt.get("max_consecutive_failures", 3)
    except Exception:
        pass

    agent = Agent(
        model_registry=model_registry,
        tool_registry=tool_registry,
        model_key=default_key,
        intent_mode=intent_mode,
        intent_llm_model=intent_llm_model,
        max_tools_per_call=max_tools,
        max_consecutive_failures=max_fail,
        max_steps=8,  # 【护栏】防 ReAct 跑飞（实测有金融 prompt 循环 18 次工具）
    )
    agent.event_bus.on("tool_call", _on_tool_call)
    return agent, default_key, removed


async def _fresh_session(agent):
    """开新会话以隔离 trial（避免历史长度影响 prompt tokens 与延迟）。"""
    try:
        agent.session_manager.create_session(title=f"ab-{int(time.time()*1000)}")
    except Exception:
        pass


async def run_sync(agent, prompt: str, holder: dict | None = None) -> dict:
    """Sync 臂：完整 ReAct 流式。实测表明 chat_stream 会提前 yield 首字，
    故 TTFR(首字) 与 TTUA(流耗尽=权威答案) 分开记录；holder 供超时回退取局部值。"""
    holder = holder if holder is not None else {}
    await _fresh_session(agent)
    _REC["tools"] = []
    t0 = time.perf_counter()
    first = None
    chunks = 0
    chars = 0
    async for c in agent.chat_stream(prompt):
        if first is None:
            first = time.perf_counter() - t0
            holder["first"] = round(first, 3)
        chunks += 1
        chars += len(c)
        holder["chunks"] = chunks
        holder["chars"] = chars
    end = time.perf_counter() - t0
    ttfr = first if first is not None else end
    return {
        "ttfr_s": round(ttfr, 3),
        "ttua_s": round(end, 3),
        "fast_end_s": None,
        "n_chunks": chunks,
        "out_chars": chars,
        "tool_fired": len(_REC["tools"]) > 0,
        "tools": list(_REC["tools"]),
    }


async def run_cfta(agent, prompt: str, conf: float, holder: dict | None = None) -> dict:
    """CFTA 臂：快速聊天（无工具）+ 可选后台异步工具检测。"""
    holder = holder if holder is not None else {}
    await _fresh_session(agent)
    _REC["tools"] = []
    t0 = time.perf_counter()
    first = None
    full = ""
    chunks = 0
    async for c in agent.chat_stream_voice_fast(prompt):
        if first is None:
            first = time.perf_counter() - t0
            holder["first"] = round(first, 3)
        chunks += 1
        full += c
        holder["chunks"] = chunks
        holder["chars"] = len(full)
    fast_end = time.perf_counter() - t0
    holder["fast_end"] = round(fast_end, 3)
    ttfr = first if first is not None else fast_end
    ttua = fast_end
    # 路径2（0<conf<0.5）与路径3（conf>=0.5）均可能触发工具；
    # 与 gui_app 一致：conf>0 才后台检测（路径1 纯聊天不检测）
    if conf > 0:
        session_id = agent.session_manager.current_session.id
        res = await agent.process_deferred_tools(prompt, full, session_id)
        after = time.perf_counter() - t0
        holder["deferred_end"] = round(after, 3)
        if res:
            ttua = after
    return {
        "ttfr_s": round(ttfr, 3),
        "ttua_s": round(ttua, 3),
        "fast_end_s": round(fast_end, 3),
        "n_chunks": chunks,
        "out_chars": len(full),
        "tool_fired": len(_REC["tools"]) > 0,
        "tools": list(_REC["tools"]),
    }


def route(conf: float) -> int:
    if conf >= 0.5:
        return 3
    if conf > 0:
        return 2
    return 1


async def one_trial(agent, p: dict, arm: str, rep: int, timeout: float = 90.0) -> dict:
    prompt = p["text"]
    conf = detect_intent_with_confidence(prompt).confidence
    t_wall = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    holder: dict = {}
    try:
        coro = run_sync(agent, prompt, holder) if arm == "sync" else run_cfta(agent, prompt, conf, holder)
        m = await asyncio.wait_for(coro, timeout)
        err = None
    except asyncio.TimeoutError:
        # 超时：保留已测到的局部值（首字/快速回复尾），TTUA 记 None 供分析剔除或按上限处理
        m = {"ttfr_s": holder.get("first"), "ttua_s": None,
             "fast_end_s": holder.get("fast_end"),
             "n_chunks": holder.get("chunks", 0), "out_chars": holder.get("chars", 0),
             "tool_fired": len(_REC["tools"]) > 0, "tools": list(_REC["tools"])}
        err = f"timeout_{timeout}s"
    except Exception as e:  # 记录失败但保留行，不丢弃
        m = {"ttfr_s": None, "ttua_s": None, "fast_end_s": None,
             "n_chunks": 0, "out_chars": 0, "tool_fired": False, "tools": []}
        err = f"{type(e).__name__}: {e}"
    rec = {
        "trial_id": f"{p['prompt_id']}_{arm}_r{rep}",
        "prompt_id": p["prompt_id"],
        "arm": arm,
        "rep": rep,
        "model_key": agent.model_key,
        "intent_confidence": round(float(conf), 3),
        "path": route(conf),
        "tool_fired": m["tool_fired"],
        "tools": m["tools"],
        "ttfr_s": m["ttfr_s"],
        "ttua_s": m["ttua_s"],
        "fast_end_s": m["fast_end_s"],
        "n_chunks": m["n_chunks"],
        "out_chars": m["out_chars"],
        "wall_ts": t_wall,
        "error": err,
    }
    return rec


def git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return "unknown"


async def main():
    ap = argparse.ArgumentParser(description="C1-A 真实 A/B 采集 harness")
    ap.add_argument("--prompts", default=str(HERE / "prompts_tooltrigger.jsonl"))
    ap.add_argument("--rep", type=int, default=5, help="每 prompt 每 arm 重复次数")
    ap.add_argument("--warmup", type=int, default=2, help="每 arm 预热丢弃次数")
    ap.add_argument("--model", default="", help="覆盖模型 key")
    ap.add_argument("--sleep", type=float, default=3.0, help="trial 间 sleep 秒")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--timeout", type=float, default=90.0, help="单 trial 墙钟超时秒（防跑飞）")
    ap.add_argument("--fail-threshold", type=int, default=3, help="连续多少次超时/触顶即判定限流，触发冷却")
    ap.add_argument("--cooldown", type=float, default=180.0, help="限流冷却秒数")
    ap.add_argument("--out", default=str(HERE / "ab_raw.jsonl"))
    args = ap.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        sys.exit(f"错误：prompt 集不存在 {prompts_path}，请先运行 prompts_tooltrigger.py")
    prompts = [json.loads(l) for l in prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not prompts:
        sys.exit("错误：prompt 集为空")

    agent, model_key, removed_tools = await build_agent(args.model)
    print(f"✓ Agent 就绪，模型={model_key}，prompt 数={len(prompts)}，rep={args.rep}，timeout={args.timeout}s")

    # 预热（丢弃）
    dummy = {"prompt_id": "warmup", "text": "你好，今天天气怎么样？", "confidence": 0.0}
    for _ in range(args.warmup):
        await one_trial(agent, dummy, "sync", 0, args.timeout)
        await one_trial(agent, dummy, "cfta", 0, args.timeout)
    print("✓ 预热完成（不计入数据）")

    rng = random.Random(args.seed)
    order = prompts[:]
    rng.shuffle(order)

    out_path = Path(args.out)
    n_written = 0
    start = time.time()
    consec_bad = 0
    n_cooldowns = 0

    def is_bad(rec):
        # 超时/异常，或 首字/权威答案 碰到客户端超时天花板（>=59.5s）= 限流伪影
        if rec.get("error"):
            return True
        return (rec.get("ttfr_s") or 0) >= 59.5 or (rec.get("ttua_s") or 0) >= 59.5

    with out_path.open("w", encoding="utf-8") as f:
        for pi, p in enumerate(order, 1):
            for rep in range(1, args.rep + 1):
                # counterbalance：奇 rep 先 sync 后 cfta，偶 rep 反之
                arms = ["sync", "cfta"] if rep % 2 == 1 else ["cfta", "sync"]
                for arm in arms:
                    rec = await one_trial(agent, p, arm, rep, args.timeout)
                    rec["order_slot"] = "S_first" if arms[0] == "sync" else "C_first"
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_written += 1
                    await asyncio.sleep(args.sleep)
                    # 【限流自动退避】连续 bad 次达阈 → 冷却让速率窗口恢复
                    if is_bad(rec):
                        consec_bad += 1
                    else:
                        consec_bad = 0
                    if consec_bad >= args.fail_threshold:
                        n_cooldowns += 1
                        print(f"  ⏳ 连续 {consec_bad} 次限流/超时 → 冷却 {args.cooldown}s（第 {n_cooldowns} 次）")
                        await asyncio.sleep(args.cooldown)
                        consec_bad = 0
            print(f"  [{pi}/{len(order)}] {p['prompt_id']} 完成（累计 {n_written} 行）")

    elapsed = time.time() - start
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model_key": model_key,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "n_prompts": len(prompts),
        "rep": args.rep,
        "warmup": args.warmup,
        "seed": args.seed,
        "sleep_sec": args.sleep,
        "trials_written": n_written,
        "elapsed_sec": round(elapsed, 1),
        "throttle_cooldowns": n_cooldowns,
        "fail_threshold": args.fail_threshold,
        "cooldown_sec": args.cooldown,
        "prompts_file": str(prompts_path.name),
        "prompts_sha1": hashlib.sha1(prompts_path.read_bytes()).hexdigest()[:12],
        "safety": {
            "readonly_whitelist": True,
            "removed_tools_n": len(removed_tools),
            "removed_tools": removed_tools,
            "max_steps": 8,
            "trial_timeout_sec": args.timeout,
        },
        "data_cutoff_note": "prompt 集源自 history.db 快照（<=2026-08-08）；本实测为实时 A/B 墙钟采样（仅只读工具）",
    }
    (HERE / "ab_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 采集完成：{n_written} 行 → {out_path}")
    print(f"✓ 环境凭证 → ab_meta.json（git={meta['git_commit'][:12]}）")
    print("下一步：抽查 ab_raw.jsonl 合理性后运行 ab_analyze.py。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n中断：已写出的行保留在 ab_raw.jsonl（可续跑或全量重跑）。")
        sys.exit(130)
