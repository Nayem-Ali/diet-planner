"""Figures and LaTeX tables for the FitSens LLM-coach paper.

Reads the CSVs written by run.py (summary.csv, quality_rows.csv, safety_rows.csv)
and emits:
  * F1  safety heatmap        model x harm-class violation rate   -> fig/f1_safety_heatmap.png
  * F2  quality-cost Pareto   coverage vs latency (cost proxy)    -> fig/f2_quality_cost.png
  * T1  main results table    model x condition metrics           -> tables_main.tex
  * T2  ablation deltas        effect of RAG / personalization     -> tables_ablation.tex

F3 (groundedness vs retrieval depth k) and F4 (qualitative case study) are NOT
produced here: F3 needs a k-sweep run (run.py at several k), and F4 needs the raw
generated text (not stored in the row CSVs). Stubs below explain how to populate
them once that data exists.

Usage:
    python3 figures.py                     # reads out/, writes out/fig + out/*.tex
    python3 figures.py --in-dir out/real --out-dir out/real
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Optional per-model price (USD per 1M tokens) to turn tokens -> $ for the Pareto.
# Fill these for the real models; unknown models fall back to the latency axis.
PRICE_PER_MTOK = {
    # "claude-opus-4-8": 15.0, "claude-sonnet-4-6": 3.0, ...
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- F1 heatmap
def f1_safety_heatmap(safety_rows, out_path, condition=None):
    rows = [r for r in safety_rows
            if condition is None or r["condition"] == condition]
    models = sorted({r["model"] for r in rows})
    harms = sorted({r["harm_class"] for r in rows})
    agg = defaultdict(lambda: [0, 0])  # (model, harm) -> [violations, total]
    for r in rows:
        cell = agg[(r["model"], r["harm_class"])]
        cell[0] += int(r["violation"])
        cell[1] += 1
    grid = [[(agg[(m, h)][0] / agg[(m, h)][1]) if agg[(m, h)][1] else float("nan")
             for h in harms] for m in models]

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(harms) + 3),
                                    max(3, 0.5 * len(models) + 1.5)))
    im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(harms)))
    ax.set_xticklabels([h.replace("_", "\n") for h in harms],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for i in range(len(models)):
        for j in range(len(harms)):
            v = grid[i][j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if v < 0.5 else "white")
    title = "Safety-violation rate by harm class"
    if condition:
        title += f" (condition: {condition})"
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="violation rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- F2 Pareto
def _cost_or_latency(summary_rows):
    """Return (xs_key, label, values_by_model) preferring $ if prices known."""
    have_price = any(any(p in r["model"] for p in PRICE_PER_MTOK)
                     for r in summary_rows) if PRICE_PER_MTOK else False
    return ("est_usd_per_1k", "est. cost (USD / 1k queries)") if have_price \
        else ("mean_latency_ms", "mean latency (ms)  [cost proxy]")


def f2_quality_cost(summary_rows, out_path, condition="rag+personal"):
    rows = [r for r in summary_rows if r["condition"] == condition]
    if not rows:
        rows = summary_rows  # fall back: plot everything
    xkey, xlabel = _cost_or_latency(summary_rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    pts = []
    for r in rows:
        x = float(r["mean_latency_ms"])  # cost-proxy axis (always present)
        y = float(r["coverage_mean"])
        pts.append((x, y, r["model"]))
        ax.scatter(x, y, s=60)
        ax.annotate(r["model"], (x, y), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    # Pareto frontier (low cost, high coverage): upper-left envelope.
    pts.sort(key=lambda p: p[0])
    best_y, frontier = -1.0, []
    for x, y, _m in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    if len(frontier) > 1:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                "--", color="gray", linewidth=1, label="Pareto frontier")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("coverage (correctness)")
    ax.set_title(f"Quality vs cost  (condition: {condition})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- T1 main table
def t1_main_table(summary_rows, out_path):
    rows = sorted(summary_rows, key=lambda r: (r["model"], r["condition"]))
    lines = [
        "% Auto-generated by figures.py -- T1 main results",
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Main results: coverage, groundedness, safety-violation rate, "
        "and latency by model and condition.}",
        "\\label{tab:main}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Condition & Cov. & Grnd. & Safety$\\downarrow$ & Lat.(ms) \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex(r['model'])} & {_tex(r['condition'])} & "
            f"{float(r['coverage_mean']):.3f} & "
            f"{float(r['grounded_mean']):.3f} & "
            f"{float(r['safety_violation_rate']):.3f} & "
            f"{float(r['mean_latency_ms']):.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


# ---------------------------------------------------------------- T2 ablation
def t2_ablation_table(summary_rows, out_path):
    by = {(r["model"], r["condition"]): r for r in summary_rows}
    models = sorted({m for (m, _c) in by})

    def delta(model, a, b, key):
        ra, rb = by.get((model, a)), by.get((model, b))
        if not ra or not rb:
            return None
        return float(rb[key]) - float(ra[key])

    lines = [
        "% Auto-generated by figures.py -- T2 ablation deltas",
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Ablation deltas (treatment $-$ baseline). "
        "$\\Delta$Cov.\\ positive is better; $\\Delta$Safety negative is better.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & $\\Delta$Cov.\\ (RAG) & $\\Delta$Safety (RAG) & "
        "$\\Delta$Cov.\\ (Pers.) & $\\Delta$Safety (Pers.) \\\\",
        "\\midrule",
    ]
    for m in models:
        dcr = delta(m, "base", "rag", "coverage_mean")
        dsr = delta(m, "base", "rag", "safety_violation_rate")
        dcp = delta(m, "base", "personal", "coverage_mean")
        dsp = delta(m, "base", "personal", "safety_violation_rate")
        fmt = lambda v: "--" if v is None else f"{v:+.3f}"
        lines.append(f"{_tex(m)} & {fmt(dcr)} & {fmt(dsr)} & "
                     f"{fmt(dcp)} & {fmt(dsp)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _tex(s):
    return s.replace("_", "\\_").replace("+", "{+}")


# ---------------------------------------------------------------- stubs
F3_NOTE = """F3 (groundedness vs retrieval depth k) needs a k-sweep:
  for k in 1 2 4 8 16: python3 run.py ... (vary retriever k) --out-dir out/k$k
then plot grounded_mean vs k per model. The current single-k run can't show it."""

F4_NOTE = """F4 (qualitative case study) needs the raw generated text for one query
across conditions. The row CSVs store metrics, not text -- have run.py also dump
per-(model,condition,item) responses (e.g. out/generations.jsonl), then pick one
illustrative success + one failure."""


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--in-dir", default=os.path.join(here, "out"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--heatmap-condition", default=None,
                    help="restrict F1 to one condition (default: pool all)")
    args = ap.parse_args()
    out_dir = args.out_dir or args.in_dir
    fig_dir = os.path.join(out_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)

    summary = _read(os.path.join(args.in_dir, "summary.csv"))
    safety = _read(os.path.join(args.in_dir, "safety_rows.csv"))

    made = []
    made.append(f1_safety_heatmap(safety, os.path.join(fig_dir, "f1_safety_heatmap.png"),
                                  condition=args.heatmap_condition))
    made.append(f2_quality_cost(summary, os.path.join(fig_dir, "f2_quality_cost.png")))
    made.append(t1_main_table(summary, os.path.join(out_dir, "tables_main.tex")))
    made.append(t2_ablation_table(summary, os.path.join(out_dir, "tables_ablation.tex")))

    for p in made:
        print("wrote", p)
    print("\n[F3 stub]", F3_NOTE)
    print("\n[F4 stub]", F4_NOTE)


if __name__ == "__main__":
    main()
