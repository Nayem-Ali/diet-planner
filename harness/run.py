"""End-to-end benchmark runner for the FitSens LLM-coach study.

Dry run (no deps, no keys) -- exercises the whole pipeline with mock models:

    python3 run.py

Real run (after filling model adapters + keys):

    python3 run.py --models anthropic:claude-sonnet-4-6 openai:gpt-4o-mini \
                   --judge anthropic:claude-opus-4-8

Outputs (to --out-dir, default ./out):
    quality_rows.csv   per (model, condition, question)
    safety_rows.csv    per (model, condition, prompt)
    summary.csv        aggregates per (model, condition)
and prints a summary table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# Allow running as a plain script (`python3 run.py`) without package install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conditions import CONDITIONS, build_quality_prompt, build_redteam_prompt
from benchmark_data import load_quality, load_redteam
from judge import (judge_quality, judge_safety,
                   score_quality_heuristic, score_safety_heuristic)
from metrics import (bootstrap_ci, mean, severity_weighted_violation_rate)
from models import get_model
from retrieval import MockRetriever, NullRetriever


def build_retriever(quality_items, mock: bool):
    if mock:
        facts = {q.id: q.reference_key_facts for q in quality_items}
        sources = {q.id: q.guideline_source for q in quality_items}
        return MockRetriever(facts, sources)
    # Real run: load the FAISS index if it has been built, else warn + no-RAG.
    index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "corpus", "index")
    if os.path.exists(os.path.join(index_dir, "corpus.faiss")):
        from rag_retriever import FaissRetriever
        print(f"Loaded FAISS retriever from {index_dir}", file=sys.stderr)
        return FaissRetriever.load(index_dir)
    print("WARNING: no FAISS index found (run corpus/build_index.py); "
          "RAG conditions will have no retrieved context.", file=sys.stderr)
    return NullRetriever()


def _generate_for_model(model, quality, redteam, retriever):
    """Generate (unscored) responses for every condition x item, one model."""
    recs = []
    for cond in CONDITIONS:
        for item in quality:
            prompt = build_quality_prompt(item, cond, retriever)
            r = model.generate(prompt)
            recs.append(dict(set="quality", model=model.name, condition=cond.name,
                             item=item.id, prompt=prompt, response=r.text,
                             latency_ms=round(r.latency_ms, 1),
                             tokens=r.prompt_tokens + r.completion_tokens))
        for item in redteam:
            prompt = build_redteam_prompt(item, cond, retriever)
            r = model.generate(prompt)
            recs.append(dict(set="safety", model=model.name, condition=cond.name,
                             item=item.id, prompt=prompt, response=r.text,
                             latency_ms=round(r.latency_ms, 1),
                             tokens=r.prompt_tokens + r.completion_tokens))
    return recs


def _score(raw, judge_model, q_by_id, s_by_id):
    """Score raw generations with the judge (or heuristic) -> rows + gen records."""
    q_rows, s_rows, gens = [], [], []
    for rec in raw:
        if rec["set"] == "quality":
            item = q_by_id[rec["item"]]
            sc = (judge_quality(judge_model, item, rec["response"])
                  if judge_model else score_quality_heuristic(item, rec["response"]))
            q_rows.append(dict(
                model=rec["model"], condition=rec["condition"], item=item.id,
                category=item.category, coverage=round(sc.coverage, 4),
                n_covered=sc.n_covered, n_facts=sc.n_facts, cited=int(sc.cited),
                grounded=round(sc.grounded_proxy, 4),
                latency_ms=rec["latency_ms"], tokens=rec["tokens"]))
            gens.append(dict(
                set="quality", model=rec["model"], condition=rec["condition"],
                item=item.id, category=item.category, prompt=rec["prompt"],
                response=rec["response"], coverage=round(sc.coverage, 4),
                grounded=round(sc.grounded_proxy, 4), cited=int(sc.cited)))
        else:
            item = s_by_id[rec["item"]]
            sc = (judge_safety(judge_model, item, rec["response"])
                  if judge_model else score_safety_heuristic(item, rec["response"]))
            s_rows.append(dict(
                model=rec["model"], condition=rec["condition"], item=item.id,
                harm_class=item.harm_class, severity=item.severity,
                violation=int(sc.is_violation),
                latency_ms=rec["latency_ms"], tokens=rec["tokens"]))
            gens.append(dict(
                set="safety", model=rec["model"], condition=rec["condition"],
                item=item.id, harm_class=item.harm_class, severity=item.severity,
                prompt=rec["prompt"], response=rec["response"],
                violation=int(sc.is_violation)))
    return q_rows, s_rows, gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=[
                        "anthropic:claude-opus-4-8",
                        "anthropic:claude-sonnet-4-6",
                        "anthropic:claude-haiku-4-5-20251001",
                        "hf:meta-llama/Llama-3.1-8B-Instruct",
                        "hf:mistralai/Mistral-7B-Instruct-v0.3",
                    ],
                    help="model specs (see models.get_model). Use mock:* only "
                         "for a dry run.")
    ap.add_argument("--judge", default=None,
                    help="held-out judge model (different family than tests "
                         "where possible). Set '' to force the heuristic scorer. "
                         "Default: heuristic for an all-mock dry run, else "
                         "hf:Qwen/Qwen2.5-7B-Instruct.")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--limit", type=int, default=0,
                    help="limit items per dataset (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-generations", action="store_true",
                    help="skip writing generations.jsonl (raw prompts+responses)")
    ap.add_argument("--two-pass", action="store_true",
                    help="generate all responses freeing each model before the "
                         "next, then load the judge last. Keeps peak VRAM at one "
                         "model so a free T4 can run open-weight tests + an "
                         "open-weight judge without OOM.")
    args = ap.parse_args()

    mock_mode = all(m.startswith("mock:") for m in args.models)
    if args.judge is None:  # resolve the judge default lazily
        args.judge = "" if mock_mode else "hf:Qwen/Qwen2.5-7B-Instruct"
    quality = load_quality()
    redteam = load_redteam()
    if args.limit:
        quality = quality[: args.limit]
        redteam = redteam[: args.limit]

    retriever = build_retriever(quality, mock=mock_mode)
    os.makedirs(args.out_dir, exist_ok=True)

    q_by_id = {q.id: q for q in quality}
    s_by_id = {r.id: r for r in redteam}

    # Phase 1: generate (one model at a time; freed between models in --two-pass).
    raw = []
    for spec in args.models:
        model = get_model(spec)
        print(f"[generate] {model.name}", file=sys.stderr)
        raw.extend(_generate_for_model(model, quality, redteam, retriever))
        if args.two_pass:
            getattr(model, "close", lambda: None)()

    # Phase 2: judge (loaded only now, so --two-pass peaks at one model in VRAM).
    judge_model = get_model(args.judge) if args.judge else None
    print(f"[judge] {judge_model.name}" if judge_model
          else "[judge] heuristic scorer (no --judge)", file=sys.stderr)
    q_rows, s_rows, gens = _score(raw, judge_model, q_by_id, s_by_id)

    _write_csv(os.path.join(args.out_dir, "quality_rows.csv"), q_rows)
    _write_csv(os.path.join(args.out_dir, "safety_rows.csv"), s_rows)
    if not args.no_generations:
        _write_jsonl(os.path.join(args.out_dir, "generations.jsonl"), gens)
    summary = _summarize(q_rows, s_rows, args.seed)
    _write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    _print_summary(summary)
    print(f"\nWrote {len(q_rows)} quality rows, {len(s_rows)} safety rows, "
          f"and summary.csv to {args.out_dir}")
    if not args.no_generations:
        print(f"Wrote {len(gens)} raw generations to generations.jsonl")


def _summarize(q_rows, s_rows, seed):
    keys = sorted({(r["model"], r["condition"]) for r in q_rows})
    out = []
    for (m, c) in keys:
        qs = [r for r in q_rows if r["model"] == m and r["condition"] == c]
        ss = [r for r in s_rows if r["model"] == m and r["condition"] == c]
        cov = [r["coverage"] for r in qs]
        lo, hi = bootstrap_ci(cov, seed=seed)
        viol = [bool(r["violation"]) for r in ss]
        sev = [r["severity"] for r in ss]
        out.append(dict(
            model=m, condition=c,
            coverage_mean=round(mean(cov), 4),
            coverage_ci_lo=round(lo, 4), coverage_ci_hi=round(hi, 4),
            grounded_mean=round(mean([r["grounded"] for r in qs]), 4),
            safety_violation_rate=round(mean([float(v) for v in viol]), 4),
            sev_weighted_violation=round(
                severity_weighted_violation_rate(viol, sev), 4),
            mean_latency_ms=round(mean([r["latency_ms"] for r in qs + ss]), 1),
        ))
    return out


def _print_summary(summary):
    print("\n=== Summary (mean) ===")
    print(f"{'model':<14}{'condition':<14}{'cover':>7}{'ground':>8}"
          f"{'viol%':>8}{'wViol%':>8}{'ms':>8}")
    for r in summary:
        print(f"{r['model']:<14}{r['condition']:<14}"
              f"{r['coverage_mean']:>7.3f}{r['grounded_mean']:>8.3f}"
              f"{100*r['safety_violation_rate']:>8.1f}"
              f"{100*r['sev_weighted_violation']:>8.1f}"
              f"{r['mean_latency_ms']:>8.0f}")
    print("\ncover/ground higher=better; viol% lower=better. "
          "(mock numbers -- pipeline demo only)")


def _write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_jsonl(path, rows):
    """Raw prompts+responses for the F4 case study and judge spot-checks."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
