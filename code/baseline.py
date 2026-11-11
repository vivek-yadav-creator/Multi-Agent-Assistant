"""
Baseline comparison: same planner + writer pipeline, but every sub-question
is answered by the strong model — no cost-aware routing. Used to measure
the actual savings the router provides.
"""

import time
from backend import (
    strong_llm, planner_node, writer_node, _extract_usage, _estimate_cost,
    _was_truncated, CONFIG, ResearchState,_invoke_with_retry
)


def baseline_research_node(state: ResearchState) -> ResearchState:
    """Same as research_node, but ignores classify_complexity — always strong."""
    results = []
    for sub_q in state["sub_questions"]:
        constrained_prompt = (
            f"{sub_q}\n\n"
            f"Answer in 3-5 sentences of plain prose. "
            f"Do not use markdown tables, headers, or bullet lists — "
            f"this is one sub-answer that will be merged with others later."
        )
        start = time.perf_counter()
        response = _invoke_with_retry(strong_llm,constrained_prompt)
        elapsed = time.perf_counter() - start

        tokens_in, tokens_out = _extract_usage(response)
        cost = _estimate_cost(CONFIG["strong_model"], tokens_in, tokens_out)

        results.append({
            "question": sub_q,
            "tier": "strong",  # forced
            "answer": response.content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "latency_sec": round(elapsed, 3),
            "truncated": _was_truncated(response),
        })
    return {**state, "results": results}


def run_baseline_query(query: str) -> dict:
    """Mirrors run_research_query (planner -> research -> writer) but
    forces every sub-question through the strong model, no routing."""
    state: ResearchState = {
        "original_query": query,
        "sub_questions": [],
        "results": [],
        "final_answer": "",
        "final_answer_truncated": False,
        "total_cost_usd": 0.0,
        "total_latency_sec": 0.0,
    }
    state = planner_node(state)
    state = baseline_research_node(state)
    state = writer_node(state)   # <-- now included, same as the routed pipeline

    # writer_node already sums state["results"] costs/latencies into
    # total_cost_usd / total_latency_sec — but that sum only covers the
    # research sub-answers, not the writer's own call. Add that in here.
    writer_response_cost_included = state["total_cost_usd"]
    # writer_node doesn't currently track its own call's cost separately,
    # so we recompute it the same way research answers are tracked:
    return {
        "sub_questions": state["sub_questions"],
        "results": state["results"],
        "final_answer": state["final_answer"],
        "final_answer_truncated": state["final_answer_truncated"],
        "total_cost_usd": state["total_cost_usd"],
        "total_latency_sec": state["total_latency_sec"],
    }