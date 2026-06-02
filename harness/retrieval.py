"""Retrieval layer for the RAG conditions.

Two implementations:
  * NullRetriever     -- returns nothing (the no-RAG conditions).
  * MockRetriever     -- simulates a corpus that already contains the relevant
                         guideline facts, so the pipeline can be exercised
                         end-to-end without a real vector store. It returns
                         passages derived from each question's reference facts.

For the real study, replace MockRetriever with a FAISS/Chroma-backed retriever
over the chunked USDA / dietary-guideline corpus (see paper Sec. "System").
The interface (`retrieve(query, k) -> list[Passage]`) stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class Passage:
    doc_id: str
    text: str
    source: str
    # Provenance (populated by the real corpus; blank for the mock retriever).
    section: str = ""
    year: str = ""
    url: str = ""
    publisher: str = ""

    def citation(self) -> str:
        """Human-readable citation label used for citation-accuracy scoring."""
        bits = [self.source]
        if self.section:
            bits.append(self.section)
        if self.year:
            bits.append(f"({self.year})")
        return ", ".join(b for b in bits if b)


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4,
                 item_id: str | None = None) -> List[Passage]:
        ...


class NullRetriever:
    def retrieve(self, query: str, k: int = 4,
                 item_id: str | None = None) -> List[Passage]:  # noqa: D401
        return []


class MockRetriever:
    """Simulates a corpus that contains the right answer.

    It is seeded by an item's reference facts so RAG conditions can plausibly
    surface them. This is a stand-in for a real index, NOT an evaluation of
    retrieval quality.
    """

    def __init__(self, fact_index: dict[str, list[str]],
                 source_index: dict[str, str]):
        # question_id -> reference_key_facts ; question_id -> guideline_source
        self._facts = fact_index
        self._sources = source_index

    def retrieve(self, query: str, k: int = 4,
                 item_id: str | None = None) -> List[Passage]:
        qid = item_id or query
        facts = self._facts.get(qid, [])
        source = self._sources.get(qid, "corpus")
        return [
            Passage(doc_id=f"{qid}::p{i}", text=fact, source=source)
            for i, fact in enumerate(facts[:k])
        ]
