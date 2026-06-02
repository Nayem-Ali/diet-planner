"""Significance testing for condition contrasts (RAG vs no-RAG, etc.).

Reads the per-item CSVs written by run.py (quality_rows.csv, safety_rows.csv)
and, for each model, computes paired contrasts between conditions on:
  * coverage  (quality set) -- key-fact checklist coverage per item
  * violation (safety set)  -- unsafe-output indicator per item

For each contrast it reports the mean delta, a percentile-bootstrap 95% CI on the
paired difference, and a two-sided paired-bootstrap p-value. p-values are then
Holm-corrected across the whole family of contrasts. Pairing is by item id, so
only items present in BOTH conditions for that model are used.

Reuses the estimators in metrics.py (no scipy required). For the camera-ready a
paired Wilcoxon (scipy.stats.wilcoxon) is an acceptable swap-in.

Usage:
    python3 stats.py                       # reads out/, writes out/stats.{md,csv}
    python3 stats.py --in-dir out/real --out-dir out/real
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

from metrics import bootstrap_ci, mean, paired_bootstrap_pvalue

# Contrasts we care about: (baseline, treatment) -> what the delta isolates.
CONTRASTS = [
    ("base", "rag", "effect of retrieval"),
    ("base", "personal", "effect of personalization"),
    ("rag", "rag+personal", "personalization on top of RAG"),
    ("personal", "rag+personal", "RAG on top of personalization"),
    ("rag+personal", "agentic", "effect of tool use"),
]


def _read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _by_model_cond(rows, value_key, cast=float):
    """(model, condition) -> {item: value}."""
    out = defaultdict(dict)
    for r in rows:
        out[(r["model"], r["condition"])][r["item"]] = cast(r[value_key])
    return out


def _paired(table, model, cond_a, cond_b):
    """Aligned (a_values, b_values) over items present in both conditions."""
    a = table.get((model, cond_a), {})
    b = table.get((model, cond_b), {})
    items = sorted(set(a) & set(b))
    return [a[i] for i in items], [b[i] for i in items], items


def holm(pvals):
    """Holm-Bonferroni: return adjusted p-values aligned with input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)  # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def collect_contrasts(in_dir, seed=42):
    q = _read_rows(os.path.join(in_dir, "quality_rows.csv"))
    s = _read_rows(os.path.join(in_dir, "safety_rows.csv"))
    cov = _by_model_cond(q, "coverage", float)
    vio = _by_model_cond(s, "violation", lambda v: float(int(v)))
    models = sorted({m for (m, _c) in cov})

    results = []
    for metric_name, table, sign in (("coverage", cov, +1),
                                     ("safety_violation", vio, -1)):
        for model in models:
            for cond_a, cond_b, label in CONTRASTS:
                a, b, items = _paired(table, model, cond_a, cond_b)
                if not items:
                    continue
                diffs = [y - x for x, y in zip(a, b)]
                delta = mean(diffs)
                lo, hi = bootstrap_ci(diffs, seed=seed)
                p = paired_bootstrap_pvalue(a, b, seed=seed)
                results.append({
                    "metric": metric_name,
                    "model": model,
                    "contrast": f"{cond_a}->{cond_b}",
                    "interpretation": label,
                    "n": len(items),
                    "mean_a": round(mean(a), 4),
                    "mean_b": round(mean(b), 4),
                    "delta": round(delta, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "p_raw": round(p, 4),
                    # For safety, a *negative* delta (fewer violations) is the
                    # desired direction; `improves` flags the good direction.
                    "improves": (delta > 0) if sign > 0 else (delta < 0),
                })

    p_adj = holm([r["p_raw"] for r in results])
    for r, pa in zip(results, p_adj):
        r["p_holm"] = round(pa, 4)
        r["sig_05"] = pa < 0.05
    return results


def write_outputs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["metric", "model", "contrast", "interpretation", "n",
            "mean_a", "mean_b", "delta", "ci_lo", "ci_hi",
            "p_raw", "p_holm", "sig_05", "improves"]
    csv_path = os.path.join(out_dir, "stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    md_path = os.path.join(out_dir, "stats.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Condition contrasts (paired, Holm-corrected)\n\n")
        f.write("Delta = mean(treatment) - mean(baseline), paired by item. "
                "For safety_violation, a negative delta means fewer unsafe "
                "outputs (good). `*` marks Holm-corrected p < 0.05.\n\n")
        for metric in ("coverage", "safety_violation"):
            f.write(f"## {metric}\n\n")
            f.write("| model | contrast | n | base | treat | delta "
                    "| 95% CI | p (Holm) | sig |\n")
            f.write("|---|---|--:|--:|--:|--:|---|--:|:--:|\n")
            for r in results:
                if r["metric"] != metric:
                    continue
                star = "*" if r["sig_05"] else ""
                f.write(f"| {r['model']} | {r['contrast']} | {r['n']} "
                        f"| {r['mean_a']} | {r['mean_b']} | {r['delta']:+} "
                        f"| [{r['ci_lo']:+}, {r['ci_hi']:+}] "
                        f"| {r['p_holm']} | {star} |\n")
            f.write("\n")
    return csv_path, md_path


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--in-dir", default=os.path.join(here, "out"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = args.out_dir or args.in_dir

    results = collect_contrasts(args.in_dir, seed=args.seed)
    if not results:
        raise SystemExit(f"No contrasts computed from {args.in_dir} "
                         "(are quality_rows.csv / safety_rows.csv present?)")
    csv_path, md_path = write_outputs(results, out_dir)
    n_sig = sum(1 for r in results if r["sig_05"])
    print(f"{len(results)} contrasts, {n_sig} significant after Holm.")
    print(f"Wrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
