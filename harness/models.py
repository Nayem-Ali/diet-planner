"""Model clients under test.

`get_model(spec)` returns a ModelClient. The default is the MockModel, which
simulates plausible outputs so the entire pipeline runs with no API keys or GPU.
Real adapters (Anthropic / OpenAI / local HF) are provided as thin, lazily
imported stubs -- fill in the marked TODOs and supply API keys / weights.

The MockModel is a *pipeline simulator only*: it fabricates responses whose
quality/safety depend on the condition flags so that ablation effects are
visible in a dry run. It is NOT a model under evaluation.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Protocol

from benchmark_data import QualityItem, RedTeamItem
from retrieval import Passage


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class ModelClient(Protocol):
    name: str

    def generate(self, prompt: str) -> GenResult:
        ...


# ---------------------------------------------------------------------------
# Mock model (default; no dependencies)
# ---------------------------------------------------------------------------
def _seed(*parts: str) -> float:
    """Deterministic pseudo-random in [0,1) from string parts."""
    h = hashlib.sha256("::".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class MockModel:
    """Simulates a model. Pass a `competence` in [0,1] to emulate stronger or
    weaker models. Quality coverage and safety compliance shift with the
    condition embedded in the prompt header (see conditions.build_prompt)."""

    def __init__(self, name: str = "mock-base", competence: float = 0.6):
        self.name = name
        self.competence = competence

    def generate(self, prompt: str) -> GenResult:
        # The prompt header encodes the task + condition + item id (see
        # conditions.py). We parse it to simulate behavior deterministically.
        meta = _parse_header(prompt)
        kind = meta.get("kind", "quality")
        rag = meta.get("rag") == "1"
        personalize = meta.get("personalize") == "1"
        tool = meta.get("tool") == "1"
        item_id = meta.get("item", "x")

        if kind == "quality":
            text = self._mock_quality(item_id, meta, rag, personalize, tool)
        else:
            text = self._mock_safety(item_id, meta, rag, personalize)

        ptoks = max(1, len(prompt) // 4)
        ctoks = max(1, len(text) // 4)
        # Cheap deterministic latency proxy.
        latency = 200 + 1000 * _seed(self.name, item_id) * (1.4 if rag else 1.0)
        return GenResult(text, ptoks, ctoks, latency)

    def _mock_quality(self, item_id, meta, rag, personalize, tool) -> str:
        facts = meta.get("facts", [])
        # Base probability a given fact is covered; retrieval & personalization
        # and competence raise it. (Simulation of the hypothesized effects.)
        base = 0.25 + 0.5 * self.competence
        if rag:
            base += 0.20
        if personalize and meta.get("needs_profile") == "1":
            base += 0.10
        covered = []
        for i, fact in enumerate(facts):
            if _seed(self.name, item_id, str(i)) < min(base, 0.97):
                covered.append(fact)
        lines = ["Here is guidance on your question."]
        lines += [f"- {c}" for c in covered]
        if rag and covered:
            lines.append(f"[cited: {meta.get('source','corpus')}]")
        if tool:
            lines.append("[tool: nutrition_optimizer computed targets/plan]")
        return "\n".join(lines)

    def _mock_safety(self, item_id, meta, rag, personalize) -> str:
        # Probability the model behaves safely (refuses / caveats / refers).
        safe_p = 0.30 + 0.6 * self.competence
        if personalize:
            safe_p += 0.05
        if rag:
            safe_p += 0.05
        is_safe = _seed(self.name, item_id, "safety") < min(safe_p, 0.98)
        if is_safe:
            return ("I can't help with that as framed -- it could be harmful. "
                    "Please consult a qualified professional. Here is a safer, "
                    "general approach instead. [SAFEGUARD]")
        return ("Sure, here is a plan to do exactly that: step 1 ... step 2 ... "
                "[COMPLY]")


# ---------------------------------------------------------------------------
# Real adapters (lazy import)
# ---------------------------------------------------------------------------
def _retry(fn, tries: int = 6, base_delay: float = 2.0):
    """Call fn(); on transient/rate-limit errors retry with exponential backoff.
    Lets free-tier APIs (e.g. Gemini) survive RPM limits without crashing a run."""
    import time
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            transient = any(k in msg for k in (
                "rate", "429", "quota", "overload", "timeout",
                "temporarily", "503", "500", "unavailable"))
            if transient and i < tries - 1:
                time.sleep(base_delay * (2 ** i))
                continue
            raise
    raise last  # pragma: no cover


def _strip_eval_meta(prompt: str) -> str:
    """Remove `<<META ...>>` / `<<FACT ...>>` / `<<SOURCE ...>>` eval-metadata
    lines before sending to a REAL model. Only the MockModel reads them; leaking
    the reference facts to real models would invalidate the coverage metric."""
    return "\n".join(l for l in prompt.splitlines() if not l.startswith("<<"))


class AnthropicModel:
    def __init__(self, name: str = "claude-sonnet-4-6"):
        self.name = name
        import anthropic  # noqa: F401  (requires `pip install anthropic`)
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: str) -> GenResult:
        import time
        t0 = time.time()
        msg = _retry(lambda: self._client.messages.create(
            model=self.name,
            max_tokens=1024,
            messages=[{"role": "user", "content": _strip_eval_meta(prompt)}],
        ))
        dt = (time.time() - t0) * 1000
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return GenResult(text, msg.usage.input_tokens, msg.usage.output_tokens, dt)


class OpenAIModel:
    def __init__(self, name: str = "gpt-4o-mini"):
        self.name = name
        from openai import OpenAI  # requires `pip install openai`
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str) -> GenResult:
        import time
        t0 = time.time()
        r = _retry(lambda: self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": _strip_eval_meta(prompt)}],
            max_tokens=1024,
        ))
        dt = (time.time() - t0) * 1000
        u = r.usage
        return GenResult(r.choices[0].message.content or "",
                         u.prompt_tokens, u.completion_tokens, dt)


class DeepSeekModel:
    """DeepSeek via its OpenAI-compatible API (cheap; far below Anthropic rates).

    Requires `pip install openai` and DEEPSEEK_API_KEY. Common model names:
    'deepseek-chat' (V3) and 'deepseek-reasoner' (R1; emits long reasoning, so
    it costs more tokens and is slower -- prefer deepseek-chat as a judge).
    """

    def __init__(self, name: str = "deepseek-chat"):
        self.name = name
        from openai import OpenAI
        self._client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    def generate(self, prompt: str) -> GenResult:
        import time
        t0 = time.time()
        r = _retry(lambda: self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": _strip_eval_meta(prompt)}],
            max_tokens=1024,
        ))
        dt = (time.time() - t0) * 1000
        u = r.usage
        return GenResult(r.choices[0].message.content or "",
                         u.prompt_tokens, u.completion_tokens, dt)


class GeminiModel:
    """Google Gemini via its OpenAI-compatible endpoint. Has a **free tier**
    (Google AI Studio) subject to per-minute and per-day request limits; paid
    Gemini Flash is very cheap. Requires `pip install openai` and GEMINI_API_KEY.
    Models e.g. 'gemini-2.5-flash' (cheap/fast) or 'gemini-2.5-pro'."""

    def __init__(self, name: str = "gemini-2.5-flash"):
        self.name = name
        from openai import OpenAI
        self._client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def generate(self, prompt: str) -> GenResult:
        import time
        t0 = time.time()
        r = _retry(lambda: self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": _strip_eval_meta(prompt)}],
            max_tokens=1024,
        ))
        dt = (time.time() - t0) * 1000
        u = r.usage
        return GenResult(r.choices[0].message.content or "",
                         u.prompt_tokens, u.completion_tokens, dt)


class LocalHFModel:
    """Open-weight chat model via HuggingFace transformers (runs on local GPU).

    Loads once and reuses. Requires `pip install transformers torch accelerate`.
    For higher throughput on the full run, consider swapping in vLLM with the
    same generate() signature.
    """

    def __init__(self, model_id: str, max_new_tokens: int = 512):
        self.name = model_id.split("/")[-1]
        self.max_new_tokens = max_new_tokens
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")

    def generate(self, prompt: str) -> GenResult:
        import time
        clean = _strip_eval_meta(prompt)  # drop <<META>>/<<FACT>> eval lines
        messages = [{"role": "user", "content": clean}]
        # return_dict=True -> a BatchEncoding {input_ids, attention_mask}; pass it
        # with **enc to generate(). (return_tensors-only changed shape across
        # transformers versions and broke `.shape` access.)
        enc = self._tok.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self._model.device)
        n_in = int(enc["input_ids"].shape[1])
        t0 = time.time()
        with self._torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self._tok.eos_token_id)
        dt = (time.time() - t0) * 1000
        gen = out[0][n_in:]
        text = self._tok.decode(gen, skip_special_tokens=True)
        return GenResult(text, n_in, int(gen.shape[0]), dt)

    def close(self):
        """Free GPU memory so the next model (or the judge) can load. Used by
        run.py --two-pass to keep peak VRAM at a single model on a free T4."""
        import gc
        for attr in ("_model", "_tok"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass


def get_model(spec: str) -> ModelClient:
    """spec forms:
        'mock:<name>:<competence>'        e.g. mock:weak:0.4
        'anthropic:<model>'               e.g. anthropic:claude-sonnet-4-6
        'openai:<model>'
        'deepseek:<model>'                e.g. deepseek:deepseek-chat
        'gemini:<model>'                  e.g. gemini:gemini-2.5-flash
        'hf:<org/model>'                  e.g. hf:meta-llama/Llama-3.1-8B-Instruct
                                          (incl. hf:deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "mock":
        name = parts[1] if len(parts) > 1 else "mock-base"
        comp = float(parts[2]) if len(parts) > 2 else 0.6
        return MockModel(name=f"mock-{name}", competence=comp)
    if kind == "anthropic":
        return AnthropicModel(parts[1])
    if kind == "openai":
        return OpenAIModel(parts[1])
    if kind == "deepseek":
        return DeepSeekModel(parts[1] if len(parts) > 1 else "deepseek-chat")
    if kind == "gemini":
        return GeminiModel(parts[1] if len(parts) > 1 else "gemini-2.5-flash")
    if kind == "hf":
        # rejoin in case the org/model contains no extra colons
        return LocalHFModel(":".join(parts[1:]))
    raise ValueError(f"unknown model spec: {spec}")


# ---------------------------------------------------------------------------
def _parse_header(prompt: str) -> dict:
    """Conditions.build_prompt prepends a machine-readable header line:
       <<META kind=quality rag=1 personalize=0 tool=0 item=Q-NUT-001 ...>>
    plus optional FACTS lines used only by the mock. Real models ignore it."""
    meta: dict = {"facts": []}
    for line in prompt.splitlines():
        if line.startswith("<<META"):
            body = line[len("<<META"):].rstrip(">").strip()
            for tok in body.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
        elif line.startswith("<<FACT "):
            meta["facts"].append(line[len("<<FACT "):].rstrip(">"))
        elif line.startswith("<<SOURCE "):
            meta["source"] = line[len("<<SOURCE "):].rstrip(">")
    return meta
