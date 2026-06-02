"""Load the FitSens LLM-coach benchmark JSON files into typed records.

Pure standard library so the harness runs with no install in mock mode.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, "..", "benchmark"))


@dataclass
class QualityItem:
    id: str
    category: str
    question: str
    reference_key_facts: List[str]
    guideline_source: str
    requires_personalization: bool


@dataclass
class RedTeamItem:
    id: str
    harm_class: str
    prompt: str
    expected_safe_behavior: str  # refuse_and_refer | strong_caveat | refer_professional
    rationale: str
    severity: str                # low | medium | high


def load_quality(path: str | None = None) -> List[QualityItem]:
    path = path or os.path.join(_BENCH, "quality_questions.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [QualityItem(**x) for x in raw]


def load_redteam(path: str | None = None) -> List[RedTeamItem]:
    path = path or os.path.join(_BENCH, "redteam_prompts.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [RedTeamItem(**x) for x in raw]
