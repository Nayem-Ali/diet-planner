# diet-planner — LLM Fitness-Coach Evaluation (technical artifacts)

Reproducible **benchmark** and **evaluation harness** for studying how *safe*,
*grounded*, and *personalized* LLM fitness/nutrition advice is, under retrieval
and user-profile conditions. This repo holds the **technical work only** (code +
data); the paper write-up lives elsewhere.

## Layout
```
diet-planner/
├── benchmark/     165 quality questions + 88 safety (red-team) prompts
│   ├── quality_questions.json
│   ├── redteam_prompts.json
│   └── README.md          # schemas + scoring rules
└── harness/       evaluation pipeline (pure-Python core; optional model deps)
    ├── datasets.py        load the benchmark
    ├── conditions.py      base / rag / personal / rag+personal / agentic
    ├── models.py          Anthropic, local HF (Llama/Mistral/Qwen), mock
    ├── retrieval.py       no-RAG + mock; rag_retriever.py = FAISS over corpus
    ├── judge.py           heuristic + LLM-as-judge scoring
    ├── metrics.py         means, bootstrap CIs, paired test, severity weighting
    ├── run.py             orchestrator -> CSVs + summary
    ├── figures.py         safety heatmap, quality-by-condition, cost/quality
    ├── stats.py           significance tests
    ├── corpus/            guideline docs + FAISS index builder
    └── colab_run.ipynb    GPU run for the open-weight models
```

## Quick start (dry run — no keys, no GPU)
```bash
cd harness
python3 run.py --models mock:weak:0.4 mock:mid:0.65 mock:strong:0.85
```
Writes `out/quality_rows.csv`, `out/safety_rows.csv`, `out/summary.csv`.

## Real run
1. `pip install -r harness/requirements.txt`
2. Put API keys in `harness/.env` (gitignored) — never in code or `.env.example`.
3. Build the RAG index: `python harness/corpus/build_index.py`
4. Run:
   ```bash
   cd harness
   python3 run.py \
     --models anthropic:claude-opus-4-8 hf:meta-llama/Llama-3.1-8B-Instruct \
     --judge hf:Qwen/Qwen2.5-7B-Instruct
   ```
   For the open-weight models on GPU, use `colab_run.ipynb`.

## Notes
- The benchmark's reference facts and red-team labels need review by someone with
  a nutrition/clinical background before use as ground truth.
- The mock model + heuristic scorers are pipeline scaffolding, not results.
