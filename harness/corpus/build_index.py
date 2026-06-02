"""Build the FAISS index for RAG from sourced guideline documents.

Usage:
    python3 corpus/build_index.py            # indexes corpus/docs/*.txt
    python3 corpus/build_index.py --docs path/to/docs --out corpus/index

Each .txt file carries a provenance header and section-tagged body:

    @meta
    source: USDA Dietary Guidelines for Americans, 2020-2025
    publisher: USDA & HHS
    year: 2020
    url: https://www.dietaryguidelines.gov/
    @end

    [[section: Macronutrient energy]]
    Protein and carbohydrate each provide about 4 kcal per gram ...

    [[section: Sodium]]
    Limit sodium to less than 2,300 mg per day ...

The header is attached to every chunk from that file; a `[[section: X]]` line
sets the section label for the chunks that follow it (until the next tag). This
provenance (source / year / url / section) is what makes groundedness and
citation-accuracy checkable -- every retrieved passage traces to a real source.

Requires: pip install sentence-transformers faiss-cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
_SECTION_RE = re.compile(r"^\[\[\s*section\s*:\s*(.+?)\s*\]\]$", re.IGNORECASE)


def parse_header(text: str) -> tuple[dict, str]:
    """Split a leading `@meta ... @end` block from the body. Returns (meta, body)."""
    meta: dict = {}
    lines = text.splitlines()
    if lines and lines[0].strip().lower() == "@meta":
        i = 1
        while i < len(lines) and lines[i].strip().lower() != "@end":
            line = lines[i].strip()
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip().lower()] = val.strip()
            i += 1
        body = "\n".join(lines[i + 1:])  # skip the @end line
    else:
        body = text
    return meta, body


def segment_by_section(body: str) -> list[tuple[str, str]]:
    """Split body into (section_label, segment_text) pairs using [[section: X]] tags."""
    segments: list[tuple[str, str]] = []
    current = ""
    buf: list[str] = []

    def flush():
        if buf:
            segments.append((current, "\n".join(buf).strip()))

    for line in body.splitlines():
        m = _SECTION_RE.match(line.strip())
        if m:
            flush()
            buf = []
            current = m.group(1)
        else:
            buf.append(line)
    flush()
    return [(sec, txt) for sec, txt in segments if txt]


def chunk(text: str, max_chars: int = 800) -> list[str]:
    chunks, buf = [], ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(buf) + len(para) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.path.join(_HERE, "docs"))
    ap.add_argument("--out", default=os.path.join(_HERE, "index"))
    ap.add_argument("--embed", default=_DEFAULT_EMBED)
    args = ap.parse_args()

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    files = sorted(glob.glob(os.path.join(args.docs, "*.txt")))
    if not files:
        raise SystemExit(f"No .txt files in {args.docs}. Add guideline docs first.")

    passages = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            meta, body = parse_header(f.read())
        source = meta.get("source", stem)
        ci = 0
        for section, segment in segment_by_section(body):
            for ch in chunk(segment):
                passages.append({
                    "doc_id": f"{stem}::c{ci}",
                    "text": ch,
                    "source": source,
                    "section": section,
                    "year": meta.get("year", ""),
                    "url": meta.get("url", ""),
                    "publisher": meta.get("publisher", ""),
                })
                ci += 1

    n_sourced = sum(1 for p in passages if p["url"])
    print(f"{len(files)} docs -> {len(passages)} chunks "
          f"({n_sourced} with a source URL); embedding...")
    embed = SentenceTransformer(args.embed)
    vecs = embed.encode([p["text"] for p in passages],
                        normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype="float32")

    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine via normalized inner product
    index.add(vecs)

    os.makedirs(args.out, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out, "corpus.faiss"))
    with open(os.path.join(args.out, "passages.json"), "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    print(f"Wrote index ({index.ntotal} vectors) to {args.out}")


if __name__ == "__main__":
    main()
