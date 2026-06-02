# LLM-Coach Scoring Harness

End-to-end evaluation pipeline for the paper. Loads the benchmark in
`../benchmark/`, runs each model under every condition, scores quality and
safety, and writes per-row CSVs + an aggregate summary.

## Quick start (no install, no API keys)
```bash
cd research/harness
python3 run.py
```
This runs three **mock** models across all 5 conditions over all 165 quality +
88 safety items and writes `out/quality_rows.csv`, `out/safety_rows.csv`,
`out/summary.csv`. The mock numbers only demonstrate the pipeline shape.

## What the pieces are
| File | Role |
|---|---|
| `datasets.py` | load `quality_questions.json` / `redteam_prompts.json` |
| `retrieval.py` | `NullRetriever` (no-RAG) and `MockRetriever`; swap in FAISS/Chroma for the real corpus |
| `models.py` | `MockModel` (simulator) + `AnthropicModel`/`OpenAIModel` adapters; `get_model(spec)` |
| `conditions.py` | the 5 conditions and prompt construction (profile injection, RAG context, tool hint) |
| `judge.py` | heuristic scorers (default) + LLM-as-judge rubric prompts for the real run |
| `metrics.py` | means, bootstrap CIs, severity-weighted violation rate |
| `run.py` | orchestrator + CSV/summary output |

## Conditions (factorial + agentic)
`base`, `rag`, `personal`, `rag+personal`, `agentic` (rag+personal+tool).

## Going from mock to real (the work that's left)
1. **Models** — fill the adapters in `models.py`, set `ANTHROPIC_API_KEY` /
   `OPENAI_API_KEY`, add a `LocalHFModel` for open-weight models, then:
   ```bash
   python3 run.py --models anthropic:claude-sonnet-4-6 openai:gpt-4o-mini mock:ref:0.7
   ```
2. **Retrieval** — replace `MockRetriever` with a real index over the
   USDA/dietary-guideline corpus (embed + FAISS). Update `build_retriever`.
3. **Judge** — implement LLM-judge calls in `judge.py` using
   `build_quality_judge_prompt` / `build_safety_judge_prompt`, run by a model
   from a different family than any under test. Wire `--judge` in `run.py`.
4. **Human validation** — score a ~120-response sample by hand and compute
   agreement (Cohen's κ) vs. the LLM judge before trusting it at scale.
5. **Stats/figures** — `metrics.bootstrap_ci` gives CIs; add paired Wilcoxon
   (scipy) for condition contrasts and render the figures (safety heatmap,
   quality–cost Pareto, k-sensitivity).

## Honesty notes
- `MockModel` and the heuristic scorers are **pipeline scaffolding**, not
  evaluation results. The heuristic quality score is lexical-overlap checklist
  coverage; the heuristic safety score is marker detection. Both are proxies the
  LLM judge replaces.
- The header line `<<META ...>>` in prompts is evaluation metadata for the mock
  path; real models are instructed to ignore it, and you can strip it for
  production runs.
