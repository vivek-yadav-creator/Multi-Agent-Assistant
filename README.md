🚀 *Multi-Agent Research Assistant with Cost-Aware Model Routing*

A LangGraph-based multi-agent pipeline that decomposes a research question into sub-questions, routes each sub-question to a cheap or strong LLM based on a deterministic complexity heuristic, and synthesizes the results into a single coherent answer — with full token/cost/latency tracking for benchmarking against an always-strong-model baseline.

────────────────────────────

💡 *WHY THIS PROJECT*

Most "agentic" demos show coordination between models but don't quantify whether the coordination actually saves anything. This project's goal was to make the cost story real: every LLM call is measured (tokens in/out, estimated cost, latency), so the final claim — routed execution is cheaper and faster than a single-model baseline — is backed by numbers from a repeatable benchmark, not a one-off anecdote.

────────────────────────────

⚙️ *ARCHITECTURE*

query ──► [ planner ] (strong model — decomposes query into 2-4 sub-questions)
               │
               ▼
        [ research ] (loops sub-questions; each routed to cheap or strong model via a deterministic complexity classifier)
               │
               ▼
         [ writer ] (strong model — synthesizes sub-answers into one integrated final answer)
               │
               ▼
          final_answer

Three graph nodes rather than one node per sub-question — this keeps state management simple while still demonstrating genuine multi-agent coordination (planning, parallelizable research, synthesis as distinct responsibilities).

🛠️ *Models (via Groq)*

• *Cheap Tier:* `openai/gpt-oss-20b` (Used for sub-questions classified as low-complexity)
• *Strong Tier:* `openai/gpt-oss-120b` (Used for planning, synthesis, and sub-questions classified as high-complexity)

📌 *Note:* Pricing in `CONFIG["pricing"]` is used for relative cost comparison between tiers, not as billed Groq pricing. Verify current rates at groq.com/pricing before quoting numbers externally.

────────────────────────────

🔀 *ROUTING DESIGN: WHY DETERMINISTIC, NOT LLM-BASED*

`classify_complexity()` is a keyword/structure heuristic — no LLM call, no extra token cost:

• Reasoning-indicating keywords (`why`, `compare`, `explain`, `trade-off`, `implications`, etc.) ──► strong tier
• Sub-question longer than 18 words ──► strong tier
• Multi-part phrasing (`and`, 2+ commas) ──► strong tier
• Otherwise ──► cheap tier

📌 *Why not just ask an LLM to classify complexity?* Because that call itself costs tokens and latency, partially defeating the point of the optimization. A cheap, inspectable heuristic means every routing decision can be explained in one sentence — a stronger answer in review than "an LLM decided." The known trade-off: it will occasionally misroute an ambiguous sub-question (see Limitations).

────────────────────────────

📊 *COST & TRUNCATION TRACKING*

Every LLM call records:
• `tokens_in` / `tokens_out` (from LangChain's `usage_metadata`)
• `cost_usd` (estimated from the pricing table)
• `latency_sec`
• `truncated` (whether the response hit `max_tokens` instead of finishing naturally, via `finish_reason == "length"`)

This turns "the router saves money" into a number instead of a guess, and turns "did the model finish its answer" into a checked fact instead of an assumption.

────────────────────────────

📈 *BENCHMARK RESULTS*

Five test queries — each blending a factual and a reasoning component — run through both the routed pipeline and an always-strong-model baseline:

• *Query 1:* Python creation / NumPy / gradient descent
  ├─ *Routed:* $0.000579, 4.61s, 2 cheap / 1 strong
  └─ *Baseline:* $0.000748, 5.12s, 3 strong (forced)

• *Query 2:* OpenAI founding / GPT-2 / transformer attention
  ├─ *Routed:* $0.000559, 7.39s, 2 cheap / 1 strong
  └─ *Baseline:* $0.000782, 5.70s, 3 strong (forced)

• *Query 3:* Boiling point / convection vs conduction
  ├─ *Routed:* $0.000653, 9.38s, 3 cheap / 1 strong
  └─ *Baseline:* $0.000960, 27.55s, 4 strong (forced)

• *Query 4:* Eiffel Tower / lattice structure & wind
  ├─ *Routed:* $0.000519, 7.27s, 2 cheap / 1 strong
  └─ *Baseline:* $0.000735, 14.37s, 3 strong (forced)

• *Query 5:* Capital of Japan / Genji / isolationism
  ├─ *Routed:* $0.000463, 4.92s, 3 cheap / 0 strong
  └─ *Baseline:* $0.000781, 13.87s, 3 strong (forced)

📊 *Summary across all 5 queries:*

• *Total Cost:* Routed: $0.0002773 | Baseline: $0.004006 | *30.8% cheaper*
• *Total Latency:* Routed: 33.57s | Baseline: 66.61s | *49.6% faster*

Routing consistently sent purely factual sub-questions to the cheap tier and reserved the strong tier for genuinely comparative/explanatory sub-questions, cutting cost by roughly a third and more than halving total latency across the test set.

────────────────────────────

🛠️ *CHALLENGES OVERCOME*

• *Output truncation on the strong tier:* Initial `max_tokens` limits (300/500) were sized for short factual answers, but the strong model produces multi-section, table-heavy explanations for reasoning questions — every strong-tier sub-answer was hitting the ceiling and cutting off mid-sentence. Fixed by raising `max_tokens` (ultimately 2000 for the strong tier) and adding an explicit `_was_truncated()` check via `finish_reason == "length"` so truncation is detected and flagged rather than silently guessed at.
• *Rate-limit errors from oversized synthesis prompts:* The writer node was feeding full, untruncated sub-answers into the synthesis prompt, pushing requests over Groq's tokens-per-minute ceiling. Fixed by capping each sub-answer's contribution to the writer's context (~600 characters) and wrapping all model calls in a `tenacity`-based retry with exponential backoff (`_invoke_with_retry`), so transient 429s are retried instead of crashing the pipeline.
• *Planner JSON-parsing fallback:* If the planner's response isn't valid JSON, `planner_node` degrades gracefully to treating the original query as a single sub-question rather than crashing, verified against a simulated malformed response before being removed from the codebase.

────────────────────────────

🧪 *TESTING PERFORMED*

• ✅ Retry logic confirmed to retry on a simulated rate-limit error and succeed on a later attempt
• ✅ Planner fallback confirmed to trigger correctly on malformed (non-JSON) LLM output
• ✅ Baseline comparison implemented and run across 5 test queries
• ✅ Full benchmark suite run (routed vs. baseline), results above

────────────────────────────

⚠️ *KNOWN LIMITATIONS*

• *Heuristic misrouting risk:* The complexity classifier is keyword/length-based, so an ambiguous sub-question (e.g., a short question that's actually conceptually deep) can be misrouted to the cheap tier. A v2 improvement would be a tiny, cheap classifier model instead of hand-written rules.
• *Pricing table is illustrative:* Cost figures reflect the relative gap between tiers, not necessarily current, exact Groq billing — rates should be reverified before being quoted as production numbers.
• *No persistence:* Unlike prior projects with session/index persistence, this pipeline is stateless per run — no caching of sub-answers across repeated or overlapping queries.
• *Small benchmark set:* Five test queries is enough to demonstrate the routing effect directionally; a larger, more varied set (pure factual, pure reasoning, and mixed) would produce a tighter estimate of real-world savings.

────────────────────────────

🚀 *SETUP & USAGE*

1️⃣ *Environment Setup:*
`python -m venv venv`
`venv\Scripts\activate`        # Windows
`pip install -r requirements.txt`

2️⃣ *Create a `.env` file with your Groq API key:*
`GROQ_API_KEY=your_key_here`

3️⃣ *Run a single query through the routed pipeline:*
`python backend.py`

4️⃣ *Run the full routed-vs-baseline benchmark:*
`python benchmark.py`

────────────────────────────

📁 *PROJECT STRUCTURE*

code/
├── backend.py          # Core pipeline: planner / research / writer nodes, routing logic
├── baseline.py         # Always-strong-model comparison pipeline
├── benchmark.py        # Runs both pipelines across test queries, reports savings
├── querys.py           # Test query set
├── requirements.txt
└── .env                # GROQ_API_KEY (not committed)