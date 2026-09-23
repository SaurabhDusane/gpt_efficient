# CLAUDE.md — working instructions for gpt_efficient

Read `SPEC.md` before doing anything. It is the source of truth for architecture and scope.

## What this project is
A chat assistant optimized for **quality per token**, plus a benchmark that measures the efficiency frontier across caching, routing, and context-compression strategies. The benchmark results are the point — treat the eval harness as a core feature, not an afterthought.

## Golden rules
1. **One milestone at a time.** Work the milestones in `SPEC.md` §5 in order. Do not scaffold future milestones early. Stop after each and let me verify.
2. **Write the test first, then the code, then run the test.** Every milestone in the spec has a defined test — satisfy exactly that before moving on.
3. **No provider SDK leaks.** Anthropic/OpenAI/Ollama specifics live only inside their adapter. Everything else talks to the `LLMProvider` interface.
4. **Every request emits one trace row.** If a code path can answer a query without logging a trace, that's a bug.
5. **Config over constants.** Anything an experiment would vary (cache threshold, tier→model map, router type, history limits) goes in config, never hardcoded.

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
`id, ts, query_hash, cache_status, cache_sim, tier, provider, model, tokens_in, tokens_out, cost_usd, latency_ms, compressed, tokens_saved, escalated, response_len, error`. Keep this in sync with `SPEC.md` §3.6 — if you change one, change both.

## When unsure
If a decision affects the research validity (cache threshold defaults, judge model choice, how quality is scored), stop and ask rather than guessing. Implementation details (file layout, helper functions) — just decide and note it.

## Don't
- Don't add auth, deployment, or a web UI before the engine + benchmark exist (§7 non-goals).
- Don't add streaming until the core pipeline is done.
- Don't optimize for tokens in a way that can't be measured by the harness.
