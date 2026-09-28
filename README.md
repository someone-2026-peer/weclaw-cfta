# CFTA — Public Evaluation Snapshot

Frozen, review-relevant artifacts for the manuscript:

> **CFTA (Chat-First, Tools-Async): Application-Layer Asynchronous Orchestration for Tool-Augmented LLM Agents**
> Submission to *The Computer Journal* (OUP).

This repository is a **self-contained proof snapshot**: it bundles the manuscript source, the raw
end-to-end wall-clock A/B logs, the aggregate statistics, the experiment protocol, and two
**standalone analysis scripts that re-derive the paper's latency figures directly from the raw
logs** — `analysis/recompute_stats.py` (every paired figure in §5.2, NumPy/SciPy only) and
`ab_measurement/ab_analyze_sensitivity.py` (every cell of the §5.5 sensitivity matrix, Python
standard library only) — **without importing the WeClaw agent codebase.** This lets a reviewer
independently verify both the headline latency result and its sensitivity decomposition.

Reproducibility contract: the manuscript and all artifacts are archived under the release **tag
`paper-cfta-ab-v1`**; the §5.5 sensitivity-probe artifacts (raw 72-trial logs, metadata, prompt
set, aggregate summary, analysis and collection scripts) were added in a later revision and are
archived under **tag `paper-cfta-ab-v2`**.

---

## Directory structure

```
.
├── README.md                      # this file
├── paper/                         # manuscript source + figures
│   ├── paper_tcj.tex              #   the paper (XeLaTeX / OUP oup-authoring-template)
│   ├── compile_tcj.bat            #   two-pass XeLaTeX build
│   ├── fig1_architecture.png      #   §3 architecture diagram
│   └── fig2_timeline.png          #   §5.2 timeline comparison figure
├── ab_measurement/                # the real end-to-end A/B channel (§5.1–§5.2)
│   ├── ab_raw_v2.jsonl            #   raw wall-clock trials — v2 run
│   ├── ab_raw_v3_partial_throttled.jsonl   #  v3 run (partial, throttle-affected)
│   ├── ab_raw_v4.jsonl            #   v4 run
│   ├── ab_summary_final_pooled.json          # pooled R1/R2-filtered paired stats
│   ├── ab_summary_pooled_v2v3.json           # intermediate aggregate
│   ├── ab_summary_stratified_by_fired.json   # tool_fired stratification (Table 3)
│   ├── ab_summary_v2.json                     # v2-only aggregate
│   ├── ab_meta.json                            # run config: seed, N, platform, guard state
│   ├── prompts_tooltrigger.jsonl / _safe.jsonl # the frozen 42-prompt A/B set
│   ├── README_C1A_protocol.md                  # the pre-declared protocol (R1–R8)
│   ├── ab_harness.py                           # live-collection harness (needs codebase)
│   ├── ab_analyze.py / _pooled.py / _stratified.py  # analysis (needs codebase + temp raw)
│   ├── safety_canary_guard.py                   # read-only guard self-check
│   ├── prompts_tooltrigger.py                   # corpus-sampling prompt builder (needs DB)
│   ├── ab_raw_sensitivity.jsonl                 # §5.5 sensitivity probe: 72 raw trials
│   ├── ab_meta_sensitivity.json                 #   probe run config (thinking mode, models)
│   ├── prompts_sensitivity_complexity.jsonl     #   the 9-prompt complexity ladder (L1/L2/L3)
│   ├── sensitivity_summary.json                 #   per-cell aggregate matrix
│   ├── ab_analyze_sensitivity.py                # ★ standalone: re-derive §5.5 table (stdlib only)
│   └── ab_harness_sensitivity.py                # probe collection driver (needs codebase)
├── evaluation/                    # telemetry methodology (§5.3, Tables status/routing/classifier)
│   ├── experiment_results.json    #   aggregate telemetry statistics
│   ├── experiment_stratified.json #   stratified telemetry statistics
│   ├── run_cfta_experiments.py    #   telemetry analysis (imports codebase)
│   └── run_cfta_stratified.py     #   stratified telemetry analysis
└── analysis/
    └── recompute_stats.py         # ★ standalone: re-derive §5.2 from the raw logs
```

---

## Reproduce the headline result (no codebase required)

From the repository root, with Python ≥ 3.10 and NumPy/SciPy installed:

```bash
python analysis/recompute_stats.py
```

The script reads `ab_measurement/ab_raw_v{2,3,4}.jsonl`, applies the **pre-declared** R1/R2
exclusion rules (drop trials pinned at the 59.5 s client timeout or carrying an error; drop any
prompt whose output attempted a non-read-only tool), pairs per-prompt arm means, and reports the
paired statistics. It then self-checks against `ab_summary_final_pooled.json` and prints
`CONSISTENT` when the independently recomputed figures match the published ones.

Expected output (matches paper Table 2 / Table 3):

| Metric | Sync (s) | CFTA (s) | Δ | p (paired t) | d_z |
|---|---|---|---|---|---|
| TTFR (first token) | 2.45 | 2.23 | 0.22 | 0.51 (**null**) | 0.11 |
| TTUA (useful answer) | 11.72 | 7.58 | −35.3 % | **0.012** | 0.43 |
| TTUA — tool fired (n = 36) | 12.13 | 7.76 | −36 % | **0.012** | 0.44 |

Over **n = 38** clean prompt pairs. A nonparametric Wilcoxon signed-rank check on ΔTTUA gives
p = 0.090 (the mean reduction is driven by a long right tail in the synchronous arm; median
per-pair reduction ≈ 11 %). This is disclosed in §5.2 rather than hidden.

### Reproduce the sensitivity matrix (Section 5.5, tag `paper-cfta-ab-v2`)

From the repository root, with Python ≥ 3.10 (standard library only, no external packages):

```bash
cd ab_measurement && python ab_analyze_sensitivity.py
```

The script reads `ab_raw_sensitivity.jsonl` (9 complexity-ladder prompts × 2 thinking modes ×
2 arms × 2 repetitions = 72 trials, 0 errors, 0 throttle cooldowns), pairs sync vs CFTA per
(prompt, thinking, repetition), compares TTUA only on CFTA-fired pairs and TTFR on all valid
pairs, and prints the six-cell matrix that underlies the paper's §5.5 sensitivity table. It also writes
`sensitivity_summary.json`. The collection driver `ab_harness_sensitivity.py` imports
`ab_harness.py` (which needs the private codebase) and is shipped as a read-only methodology
reference, like the other collection-channel scripts below.

Compile the manuscript separately (needs a TeX distribution with the OUP `oup-authoring-template.cls`,
xeCJK, and a CJK font for the one Chinese query example):

```bash
cd paper && ./compile_tcj.bat        # or: xelatex paper_tcj.tex  (twice)
```

---

## What is public vs. available on request

**Public (this snapshot):**
- The manuscript source and figures.
- The **raw A/B wall-clock logs** and the aggregate statistics.
- The fixed 42-prompt A/B set and the pre-declared protocol.
- The **standalone `recompute_stats.py`**, which fully reproduces every §5.2 paired figure from the
  raw logs with no dependency on the agent codebase.
- The **§5.5 sensitivity-probe artifacts** (raw 72-trial logs, metadata, 9-prompt complexity
  ladder, aggregate summary) and the **standalone `ab_analyze_sensitivity.py`**, which re-derives
  every cell of the sensitivity table with the Python standard library only.
- The telemetry aggregate JSONs (§5.3).

**Available on request (to the editor / reviewers):**
- The full **WeClaw codebase** (`src/`), pinned at the private development-repository tag
  `paper-cfta-eval-v1`. The A/B harness (`ab_harness.py`) and the telemetry analysis scripts
  (`run_cfta_experiments.py`) import this codebase and are shipped here only as read-only
  **methodology references** — they are not runnable from this snapshot alone.
- The **read-only telemetry snapshots** (~140 MB single-user usage databases: `tool_audit.db`,
  `history.db`). These contain sensitive personal usage data and are **not** publicly released. The
  derived aggregate statistics needed to reproduce Tables 4/5/6 are reported in §5; the underlying
  databases can be provided to the editor or reviewers on reasonable request.

> Note on the `ab_measurement/analysis` scripts: some reference `WECLAW_REPO` / `parents[4]` to
> locate the private codebase. Set `WECLAW_REPO` to a local checkout if you have one; otherwise
> `analysis/recompute_stats.py` is the intended, dependency-free reproduction path.

---

## Notes on the raw logs (privacy)

The A/B `*.jsonl` records only tool **names**, timing (`ttfr_s`, `ttua_s`), arm labels, trial status
and the model key — **not** the raw prompt/response text. The prompt set is drawn from a single-user
deployment and was filtered through a read-only tool whitelist enforced at both the registry and the
unique execution entry point, so no side-effecting tool (shell, file write, browser) could run during
measurement. `safety_canary_guard.py` includes a deliberately sensitive dummy path
(`C:/Windows/System32/config/sam`) used only to **prove** the guard denies such calls; nothing is
actually accessed.
