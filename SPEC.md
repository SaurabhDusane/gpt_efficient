# gpt_efficient — Specification

A ChatGPT-style assistant optimized for **maximum answer quality per token spent**. The goal is not just a working chat app but a measurable **efficiency frontier**: quality retained vs. token cost across a set of optimization strategies.

This is a portfolio/research project. The traces and benchmark results are the primary deliverable; the chat interface is secondary.

---

## 1. Goal

Given a user query, return a good answer while spending as few tokens (and as little money/latency) as possible. Measure the tradeoff rigorously.

Research questions:
- How much token cost can be cut before answer quality degrades?
- Learned router vs. heuristic router — is the extra complexity worth it?
- Where does semantic caching help, and where does it silently serve wrong answers?
- What does context compression cost in quality per token saved?

---

## 2. Request pipeline

```
query
  → [semantic cache]     hit (sim ≥ threshold)? → return cached, log hit
      ↓ miss
  → [context compressor] trims / summarizes conversation history
      ↓
  → [router]             classifies difficulty → picks tier (local → mid → frontier)
      ↓
  → [provider adapter]   calls chosen provider+model
      ↓
  → [trace logger]       records tokens in/out, cost, latency, tier, cache status, sim score
      ↓
  response (+ eval hook)
```

Every request emits exactly one **trace row**. The traces are the research output.

---

## 3. Components

### 3.1 Provider abstraction
Single interface; provider SDKs must not leak past the adapter layer.

```python
class LLMProvider(Protocol):
    name: str
    def complete(self, messages: list[Message], max_tokens: int, model: str) -> Completion: ...

# A leading role="system" message carries the system prompt; each adapter maps it
# to its SDK's convention. `model` is passed per call since one adapter serves several models.

# Completion: text, tokens_in, tokens_out, cost_usd, latency_ms, model
# tokens_out includes thinking/reasoning tokens (they are billed as output).

class Embedder(Protocol):
    name: str
    def embed(self, texts: list[str]) -> list[Vector]: ...   # Vector = list[float]
```

Embeddings are a **separate `Embedder` protocol**, not a method on `LLMProvider`: the embedding backend may differ from the completion provider, and cache/router/compressor depend only on `Embedder`. Embedding model, dimensionality and price come from config.

Adapters (current): **Gemini** (`google-genai`; Flash-Lite / Flash / Pro + `gemini-embedding-2`) is the default provider. **Anthropic** (Haiku / Sonnet / Opus) is implemented but unused until keys are available. **OpenAI** and **Ollama** are deferred (see §7). Router selects a `(provider, model)` tier; everything downstream is provider-agnostic.

### 3.2 Tiers
Abstract "tier" decouples routing from providers:
- `LOCAL`  — budget tier. Currently Gemini Flash-Lite; becomes an Ollama small model (near-zero marginal cost) once that adapter lands
- `MID`    — cheap hosted (Gemini Flash; later Haiku / gpt small)
- `FRONTIER` — Gemini Pro (later Sonnet/Opus / gpt large)

Tier→model mapping lives in config so experiments can swap models without code changes.

The **active tier set** is also config: `tier_mode` selects an entry of `[tier_modes]` (`two` = local + mid, both on Gemini's free tier; `three` adds the paid Pro frontier tier). Router and engine read the active set from config and never assume three tiers.

### 3.3 Semantic cache
- Embed incoming query, cosine-match against stored `(embedding, response)` pairs.
- **Tunable similarity threshold** (config). Log every hit's similarity score.
- Store: SQLite + `sqlite-vec` (preferred, zero infra) or Chroma.
- Correctness risk: too-loose threshold serves wrong answers. Threshold is a first-class experiment knob, not a constant.
- Cache key must include anything that changes the answer (system prompt version, tier policy) to avoid stale/incorrect hits.

Implemented (milestone 3):
- Store: `sqlite-vec` `vec0` table per embedding dimension, cosine distance, partitioned by **namespace** = hash of system prompt, `max_tokens`, default tier, active tier→(provider, model) map, embedding model/dim, embed template and a manual `cache.version`. (Add router type/config here when the router lands.)
- Defaults (config `[cache]`): `threshold = 0.95`, `gemini-embedding-2` at 768 dims, embedded text = raw query (`embed_template = "{text}"`). Chosen conservative; calibrate with the harness.
- Hit = nearest neighbour similarity ≥ threshold. Misses also log the nearest similarity in `cache_sim` (null only when the namespace is empty), so hit rate vs. threshold can be analysed offline.
- A hit row records the tier/provider/model that produced the cached answer, `tokens_in = tokens_out = 0`, and costs only the lookup embedding.
- Requests with conversation history **bypass** the cache (`cache_status = bypass`): a standalone answer may be wrong in context.
- Only successful, non-empty answers are stored. An embedding failure fails the request (row logged as `miss` with `error`).

### 3.4 Router
- **Baseline (heuristic):** query length + complexity keywords + presence of code/math → tier. Must exist first so the learned router has something to beat.
- **Learned:** embedding-based classifier over a labeled "needs-frontier-model" set. Outputs tier + confidence.
- Both implement the same `Router` interface; swappable via config.
- Escalation policy (optional, later): if a low tier's answer fails a cheap quality check, retry one tier up. Log escalations.

### 3.5 Context compressor
- Rolling summary of old turns + embedding-retrieval of only relevant prior turns.
- Optional prompt compression (LLMLingua) as a comparison point.
- Config: max history tokens, summary trigger threshold.
- Log tokens saved per request.

### 3.6 Trace logger
One row per request. Pydantic schema, written to SQLite (and/or JSONL).

Fields: `id, ts, query_hash, cache_status, cache_sim, tier, provider, model, tokens_in, tokens_out, cost_usd, latency_ms, compressed (bool), tokens_saved, escalated (bool), response_len, embed_tokens, error`.

- `cache_status`: `hit` | `miss` | `bypass` (cache on but not consulted) | `disabled`.
- `tokens_in` / `tokens_out` are LLM tokens only: `tokens_in` includes any prompt-cache reads/writes, `tokens_out` includes thinking tokens.
- `embed_tokens` is the (estimated, `chars / embedding_chars_per_token`) size of the text embedded for the cache lookup — Gemini's embed API reports no counts.
- `cost_usd` = LLM cost + embedding cost; `latency_ms` is end-to-end (embed + lookup + LLM).
- `error` is null on success; failed requests still emit their row with the exception recorded.

### 3.7 Eval harness — build early, not last
- **Dataset:** queries with quality labels / reference answers, tagged by difficulty.
- **Scorer:** LLM-as-judge with a fixed rubric (+ task-specific accuracy where applicable).
- **Runner:** runs the same dataset through each config, outputs **quality-per-token**.
- **Report:** efficiency-frontier plots (quality vs. tokens, per strategy).

Without this, every optimization is blind. It is a Phase 1 deliverable, not a final step.

---

## 4. Configuration
All experiment knobs in one `config.toml` (or pydantic-settings): default provider, tier→model map, active tier set (`tier_mode`), per-model pricing (incl. long-context rates), embedding model/dim/price, cache threshold, compressor limits, router type, judge model. Changing a config value and re-running the harness = one experiment.

---

## 5. Milestones / build order
Each is a discrete, testable unit. Do not start the next until the current one's test passes.

1. **Scaffold** — repo, `LLMProvider` interface, Anthropic adapter, trace logger. *Test: one query end-to-end, one logged trace row.*
2. **Gemini provider + embeddings** — Gemini adapter behind `LLMProvider`, `Embedder` protocol + Gemini embedder, two/three-tier switch. *Test: mocked Gemini call → correct `Completion` (thinking counted as output, cost from config); mocked embed → right count/dimension of vectors.* (Originally OpenAI + Ollama; revised because only a Gemini key is available — see §7.)
3. **Semantic cache** — embed + vector store, tunable threshold, sim logging. *Test: repeat/paraphrased query → cache hit.*
4. **Heuristic router** — baseline tier selection. *Test: easy vs. hard query pick different tiers.*
5. **Eval harness** — dataset loader, LLM-judge, quality-per-token report. *Test: report generated over a small dataset.*
6. **Context compressor** — rolling summary + retrieval. *Test: long history → fewer tokens, quality held.*
7. **Learned router** — classifier; benchmark vs. heuristic on the harness. *Test: frontier plot compares both routers.*
8. **Results notebook** — efficiency-frontier plots and writeup.

UI: thin CLI (Rich/Textual) from milestone 1; web UI only after the engine is solid.

---

## 6. Stack
- Python 3.12, `uv` for env/deps
- pydantic / pydantic-settings for schemas + config
- SQLite (+ `sqlite-vec`) for traces and cache
- `google-genai` (Gemini) now; Anthropic SDK (adapter ready); OpenAI SDK + Ollama via HTTP later
- Rich/Textual CLI; matplotlib/plotly for plots; Jupyter for the results notebook

---

## 7. Non-goals (for now)
- Auth, multi-user, deployment
- Streaming responses (add after engine works)
- Fine-tuning models (routing/compression only)
- A polished web UI before the benchmark exists

**Deferred, not dropped:** multi-provider (Anthropic + OpenAI + local Ollama). Until those keys are available (~1 month), the whole project — completions, embeddings and the eval judge — runs on Gemini alone. Because everything sits behind `LLMProvider` / `Embedder`, adding those adapters later is additive: a new adapter file plus config entries. Cross-provider comparisons in the results are out of scope until then.
