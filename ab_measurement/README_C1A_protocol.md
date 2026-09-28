# C1-A 真实 A/B 实测方案（Sync vs. CFTA 端到端延迟）

> 目的：把论文 §5.1 的"成本模型解析投影"（`L_sync=2m+d`, `L_cfta=m`）升级为
> **真实端到端 A/B 实测**，使 54.1% 首响延迟下降、配对检验与效应量建立在
> 真实墙钟采样之上，彻底经得起 TCJ 评审（审稿人 C1/C2 硬伤）。

## 0. 铁律（不可违反）

1. **绝不造数**。本目录只交付"真实采集脚手架 + 分析脚本 + SOP"；
   `ab_raw.jsonl` 必须来自真实运行 WeClaw 双模式的墙钟采样。
2. 在拿到真实 `ab_raw.jsonl` 之前，**论文一字不改**（当前已降级为"投影"的诚实口径保持不动）。
   实测完成、数字落定后，才按 §6 的"替换位点清单"回填，并撤销 hedging 措辞。
3. 复现凭证使用**新** git tag `paper-cfta-ab-v1`（不要复用 `paper-cfta-eval-v1`，
   后者已锁定投影脚本，改动它会破坏已发布可复现契约）。
4. 采集脚本 `ab_harness.py` 若检测到无法初始化真实模型/工具注册表，**直接报错退出**，
   不产出任何占位/伪造行。

---

## 1. 被测量定义（区分 C2  conflated 指标）

对每条 prompt、每个 arm，用 `time.perf_counter()` 记录两个**互相独立**的墙钟指标：

| 指标 | 定义 | Sync 臂取值点 | CFTA 臂取值点 |
|---|---|---|---|
| **TTFR**（time-to-first-response，用户可见首字符） | 从发出输入到 UI 出现第一个可读 token | `chat_stream()` 首个 yield chunk | `chat_stream_voice_fast()` 首个 yield chunk |
| **TTUA**（time-to-useful-answer，含工具数据的权威答案就绪） | 权威答案（含工具结果）完整可用 | `chat_stream()` 流耗尽 | `process_deferred_tools()` 返回非空摘要；若未触发工具则=快速回复流耗尽 |

关键点（评审 C2）：Sync 模式的 ReAct 在工具链完成前 **不 yield 任何文本**（见 `agent.py:3915-3918`），
因此其 TTFR≈TTUA；CFTA 把工具移出关键路径，**只在 TTFR 维度取胜**，其 TTUA 因多一次后台
检测调用 **可能 ≥ Sync**。实测必须同时报两个指标并如实呈现这一权衡，不得混为单一"延迟"。

## 2. 两臂真实接口（已核对源码，非臆造）

**统一 headless 引导**（复用 `src/app.py:200-258` CLI 入口，无需 Qt/GUI）：
```python
model_registry = ModelRegistry()
tool_registry  = create_default_registry()
agent = Agent(model_registry=..., tool_registry=..., model_key=default_key,
              intent_mode=..., ...)   # 与 CLI 一致
```

**Arm S（Sync 基线 = 传统 ReAct）**
- `async for chunk in agent.chat_stream(prompt)`（`agent.py:3902`）
- 首 chunk → TTFR_S；流耗尽 → TTUA_S；是否触发工具由审计/返回步骤判定。

**Arm C（CFTA）**
- `conf = detect_intent_with_confidence(prompt).confidence`（`prompts.py`，分流阈值与 `gui_app.py:813` 一致）
- `async for chunk in agent.chat_stream_voice_fast(prompt)`（`agent.py:5185`）→ 首 chunk = TTFR_C，流尾 = fast_end
- 若 `conf > 0`：`res = await agent.process_deferred_tools(prompt, fast_reply, session_id)`（`agent.py:5557`）
  - `res is not None` → TTUA_C = 返回时刻，tool_fired=True
  - `res is None`（静默退出/无工具）→ TTUA_C = fast_end，tool_fired=False

**会话隔离**：每条 trial 前 `agent.session_manager.create_session(...)`（`session.py:177`）开新会话，
保证各 trial 上下文长度一致、互不污染（历史长度直接影响 prompt tokens 与延迟）。

## 3. 实验设计（保证可发表）

- **Prompt 集**：固定 `PROMPT_N`（默认 40）条**真实工具触发型**输入。
  优先由 `prompts_tooltrigger.py` 从 `history.db` 的 `tool_needed` 标注语料（同 `run_cfta_experiments.py` 的
  `build_labeled_corpus`）按 23 类意图分层抽样并冻结为 `prompts_tooltrigger.jsonl`（含固定 seed）；
  该文件一经生成即作为研究材料固化，保证任何人重跑用同一批输入。
- **重复**：每 prompt × 每 arm × `REP`（默认 5）次，取每 prompt 每 arm 的均值做配对。
- **配平顺序（counterbalancing）**：奇数次 rep 先跑 S 后跑 C，偶数次先 C 后 S —— 抵消预热/网络/热漂移。
- **预热**：正式采集前每 arm 各丢弃 `WARMUP`（默认 2）次（用一条无害 prompt）。
- **随机化**：prompt 顺序用固定 seed 洗牌。
- **节律**：trial 间 `sleep`（默认 3s）避免限流耦合；整轮记录起止 wall-clock。
- **环境冻结**：同一模型 key、同一硬件、同一网络；`ab_meta.json` 记录 git commit、模型 key、
  机器规格、采集日期时段、`py` 版本、种子。
- **失败处理**：异常 trial 记 `error` 字段并**保留**（不静默删除）；分析时按预设规则（error/timeout 剔除并计数）处理。

统计功效提示：40 prompt × 配对设计，若 TTFR 下降真实存在且效应中等（d≈0.5），
配对 t 检验在 α=0.05 下功效充足；`REP=5` 用于压制单次网络抖动。若 `REP` 资源受限，优先加 prompt 数。

## 4. 原始数据 schema（`ab_raw.jsonl`，一行一 trial）

```json
{"trial_id":"p007_S_r3","prompt_id":"p007","arm":"sync","rep":3,"order_slot":"S_first",
 "model_key":"...","intent_confidence":0.62,"path":3,"tool_fired":true,"tools":["web_search"],
 "ttfr_s":18.4,"ttua_s":21.1,"fast_end_s":null,"out_chars":512,"n_chunks":40,
 "wall_ts":"2026-09-27T14:03:11+08:00","error":null}
```
`prompt_text_hash`（SHA1 前 12 位）另存于 `prompts_tooltrigger.jsonl`，原始 prompt 文本不入 jsonl（隐私）。

## 5. 分析（`ab_analyze.py`）→ `ab_summary.json`

- 先按 `(prompt_id, arm)` 对 `REP` 次求均值（削网络抖动），得到每 prompt 的 TTFR_S/TTFR_C/TTUA_S/TTUA_C。
- **真实配对检验**（null 不再"by construction"为假）：
  - ΔTTFR = TTFR_S − TTFR_C，逐 prompt 配对 → **paired t-test** + **Wilcoxon signed-rank** + **Cohen's d(paired)** + **bootstrap 95% CI**。
  - ΔTTUA 同法 → 预期接近 0 或为负（CFTA 不显著更差即达标），**如实报告**。
  - 相对下降 = mean(TTFR_C)/mean(TTFR_S)，报 mean/median 及 CI。
- **分层**：按 `tool_fired` 及工具时长档（fast/medium/slow，若 jsonl 带工具名可 join 审计时长）分别配对。
- 产出可直接贴入 LaTeX 的数值块（`ab_summary.json` + 终端打印表格行）。

## 6. 论文替换位点清单（拿到真实数字后执行）

| # | 位置 | 现状（投影口径） | 实测后改为 |
|---|---|---|---|
| R1 | §5.1 Measurement methodology | "analytical projection … not a direct end-to-end A/B measurement"；MC 采样 `2m+d`/`m` | 改为真实 A/B 描述：两臂、TTFR/TTUA 定义、n prompt×REP、counterbalancing、git tag `paper-cfta-ab-v1` |
| R2 | §5.1 Conservative measurement | "measured to full-response completion … conservative lower bounds" | 用实测 TTFR（首 token）替换，删投影措辞 |
| R3 | Table 2 (tab:latency) | "Projected time-to-first-response … not a significance test"；95% CI 来自 MC | 实测 mean±sd、真实配对 t/Wilcoxon 的 p、Cohen's d、bootstrap CI；caption 去 "Projected" |
| R4 | §5.1 正文单调性段 | "follows analytically from L_sync−L_cfta=m+d … not independent empirical confirmation" | 若分层实测复现单调性则改为实证，否则保留谨慎措辞 |
| R5 | Table 3 (tab:stratified) | "Projected … cost model of §5.1" | 实测分档数值；caption 去 "Projected" |
| R6 | 摘要 & §8 结论 | "analytical projection" 相关降级措辞 | 视实测结果决定是否恢复"实证"口径（数字必须来自 jsonl） |
| R7 | §6.2 Discussion | 局限里"未做端到端 A/B"相关句 | 更新为已做，并新增 TTUA 权衡讨论 |
| R8 | 新增小节 | 无 | 建议加 "Threats to validity"：网络方差、单模型、prompt 集代表性 |

> **数值一致性规范**：回填时正文、表格、摘要、cover letter、box statement 五处数字必须逐一对齐 `ab_summary.json`，
> 禁止任一处残留旧投影值。生成后跑一次全文 grep 旧值（21.32/9.30/54.1%/50.4%/d=0.52/74.1%/50.5%/55.8%）确保无孤儿。

## 6.1 执行状态（2026-09-28 已完成回填）

**真实结果与本协议 §1 的预设相反**：协议原假设 CFTA 在 TTFR 取胜、TTUA 持平；实测为 **TTFR 零结果**（sync 2.45 vs cfta 2.23 s，paired t p=0.51，bootCI 含 0），**TTUA 才是真收益**（sync 11.72 vs cfta 7.58 s，均值降 35.3%，paired t p=0.012，d_z=0.43，bootCI[1.33,7.31]；Wilcoxon p=0.090、中位降 11%，收益集中于 sync 长尾，正文已并列诚实披露）。数据源：v2+v3+v4 合并，预声明 R1（剔 >=59.5s/超时/error）+ R2（剔幻觉非白名单工具整条）过滤后 **38 配对**。分层（`ab_analyze_stratified.py` -> `ab_summary_stratified_by_fired.json`）：tool_fired 36 对降 36% p=0.012；无工具 2 对仅描述。

R1-R8 已全部落地到 `paper_tcj.tex`（摘要/贡献3/§5.1/§5.2 两表/Fig2/§5.4 对比表+消融表/§6.1/§6.2/§7/伦理/数据可得性），并同步 `cover_letter_tcj_20260927.md`(+`.docx`) 与 `cover_letter_box_statement`；全文 grep 旧投影值已归零，XeLaTeX 两遍编译通过（17 页）。

## 7. 运行步骤

复现统计只需本快照：在仓库根执行 `python analysis/recompute_stats.py`（纯 NumPy/SciPy，不导入 WeClaw 代码，直接读 `ab_measurement/ab_raw_v{2,3,4}.jsonl` 复算 §5.2 全部配对数字）。

以下步骤 1–3 属于**采集/审计**通道，需要私有 WeClaw 代码库（`src/`，应请求提供）与可用模型 key，评审无需运行：

```powershell
cd ab_measurement
# 1) 生成并冻结 prompt 集（真实语料分层抽样，需 history.db 快照）
python prompts_tooltrigger.py --n 42 --seed 20260927
# 2) 真实采集（需可用模型 key + WeClaw 代码库；产出 ab_raw_v*.jsonl + ab_meta.json）
python ab_harness.py --prompts prompts_tooltrigger.jsonl --rep 3 --warmup 2
# 3) 配对分析（R1/R2 严格过滤，产出 LaTeX 数值块；等价于快照根的 recompute_stats.py）
python ab_analyze_pooled.py
```

采集完成后：先人工抽查 `ab_raw.jsonl` 合理性（Sync 的 TTFR 明显高于 CFTA、无异常 0 值、
error 比例可接受），再执行 §6 回填与重编译，最后打 tag `paper-cfta-ab-v1`。

## 8. 与"投影口径"的隔离

`run_cfta_experiments.py`（Exp2 投影）保留不动，作为历史可复现工件；
本 `ab_measurement/` 是**独立的真实实测通道**。二者产出的数字若不同，**以真实实测为准**并同步论文，
投影版仅在小节脚注说明"早期分析性投影已被端到端实测取代"。
