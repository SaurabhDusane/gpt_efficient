# CLAUDE.md — working instructions for gpt_efficient

Read `SPEC.md` before doing anything. It is the source of truth for architecture and scope.

## What this project is
A chat assistant optimized for **quality per token**, plus a benchmark that measures the efficiency frontier across caching, routing, and context-compression strategies. The benchmark results are the point — treat the eval harness as a core feature, not an afterthought.

## Golden rules
1. **One milestone at a time.** Work the milestones in `SPEC.md` §5 in order. Do not scaffold future milestones early. Stop after each and let me verify.
2. **Write the test first, then the code, then run the test.** Every milestone in the spec has a defined test — satisfy exactly that before moving on.
3. **No provider SDK leaks.** Gemini/Anthropic/OpenAI/Ollama specifics live only inside their adapter (`providers/<name>_provider.py`). Everything else talks to the `LLMProvider` and `Embedder` interfaces.
4. **Every request emits one trace row.** If a code path can answer a query without logging a trace, that's a bug.
5. **Config over constants.** Anything an experiment would vary (cache threshold, tier→model map, active tier set, prices, embedding model, router type, history limits) goes in config, never hardcoded.

## Architecture decisions
- **Provider:** Gemini is the only live provider for now (`default_provider = "gemini"`). Anthropic/OpenAI/Ollama are deferred, not dropped (SPEC §7).
- **Embeddings:** separate `Embedder` protocol (`providers/base.py`), not an `embed()` method on `LLMProvider` — embeddings and completions may come from different providers. Build via `providers.build_embedder(settings)`; never import an SDK for embeddings elsewhere.
- **Tier set:** `tier_mode` in `config.toml` picks a list from `[tier_modes]` (`two` = free-tier Flash-Lite + Flash, `three` adds paid Pro). Read it via `Settings.active_tiers`; never assume the number of tiers.
- **Token accounting:** `tokens_out` includes thinking tokens on every provider; `tokens_in` includes cached-prompt tokens. Embedding tokens go in `embed_tokens` (estimated) and their cost is included in `cost_usd`.
- **Semantic cache:** `cache.py` stores/searches vectors only; the engine embeds and applies the threshold. Anything that changes answers must be part of `cache_namespace()` — when you add a new answer-affecting setting (e.g. router config), add it there. Queries with history bypass the cache.
- **Router:** `router.py`; implement the `Router` protocol and register the type in `build_router()` + the `RouterConfig.type` Literal. Keep scoring in pure functions. `fixed` is the no-router baseline. Router config is in the cache namespace (`cache._router_key`); a new router type must add its params there.
- **Context compressor:** `compressor.py`, config `[compressor]`, runs between cache and router, only on requests with history (which bypass the cache — so compressor config is deliberately not in the cache namespace). Turn = user+assistant exchange. Summarizer = the `summary_tier` model (budget tier). Summaries and exchange embeddings are memoized per Engine so a conversation never re-pays for old turns. Anything that needs an embedder must go through `Settings.needs_embedder`. The eval reports compression cost one-shot and amortized (`compressor.replay_overhead`, turn-by-turn replay); keep both.
- **Eval harness:** `src/gpt_efficient/evals/` (dataset → runner → judge → report), run via `gpte eval`. Experiments are override tables in `evals/experiments.toml`; add an experiment there, not in code. The judge (`gemini-2.5-pro`, temp 0, rubric `RUBRIC_VERSION` in `evals/judge.py`) must stay blind and must never be a model in the answer ladder; bump `RUBRIC_VERSION` whenever the rubric text changes. Judge cost is never charged to a config. Seed data in `evals/seed.jsonl` — every `exact` answer must appear in its own reference (a test checks this). Multi-turn data in `evals/conversations.jsonl` (compression experiments in `evals/experiments_compression.toml`); every history must exceed `compressor.trigger_tokens`, and needles must sit outside the verbatim window (tests check both).
- **Offline fakes:** `gpt_efficient/fakes.py` (provider, embedder, judge) back every `--fake` mode; fake numbers are illustrative only.
- **Savings scripts:** `scripts/<mechanism>_delta.py` compares a mechanism against the baseline (`--fake` runs offline, illustrative only). Shared fakes/run/delta table live in `scripts/_common.py`.

## Workflow expectations
- Before writing code for a milestone, restate the milestone's goal and its test in one line, then list the files you'll create/change. Wait for nothing if it's within the current milestone; just proceed.
- After finishing a milestone, run its test and show me the output (including a sample trace row where relevant).
- After each efficiency mechanism (cache, router, compressor), add/update a small script that prints the token delta vs. the previous baseline so the savings are visible immediately.
- Keep diffs small and reviewable. Prefer many small commits with clear messages over one large one.

## Code standards
- Python 3.12, managed with `uv`. Add deps via `uv add`.
- Type hints everywhere; pydantic for all data schemas and config.
- Prefer pure functions for router/compressor logic so they're unit-testable without live API calls.
- Mock provider calls in tests; never require real API keys to run the unit suite.
- Secrets from environment (`.env`, git-ignored). Never commit keys.

## Trace schema (canonical)
`id, ts, query_hash, cache_status, cache_sim, tier, provider, model, tokens_in, tokens_out, cost_usd, latency_ms, compressed, tokens_saved, escalated, response_len, embed_tokens, summary_tokens, error`. Keep this in sync with `SPEC.md` §3.6 — if you change one, change both.

## When unsure
If a decision affects the research validity (cache threshold defaults, judge model choice, how quality is scored), stop and ask rather than guessing. Implementation details (file layout, helper functions) — just decide and note it.

## Don't
- Don't add auth, deployment, or a web UI before the engine + benchmark exist (§7 non-goals).
- Don't add streaming until the core pipeline is done.
- Don't optimize for tokens in a way that can't be measured by the harness.
