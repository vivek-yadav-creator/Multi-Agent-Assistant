""" 
Multi-agent research assistant with cost-aware model routing.

Flow:
    planner -> research( loops sub-questions, routes each to cheap/strong model) ->writer

Design decisions:
-Complexity routing is a deterministic heuistic, not an LLM call - classifying
 complexity via LLM would itself cost tokens, partially defeating the optimization.
-Every LLM call logs token usage and estimated cost , so the benchmark step later
 has real numbers instead of guesses.
-Planner and writer always use the strong model (decomposition and sythesis are higher-stakes than answering
 a single factual sub-question).
"""

import os
import re
import json
import time
from typing import TypedDict,List,Dict

from tenacity import retry,wait_exponential,stop_after_attempt,retry_if_exception_type
from groq import APIStatusError

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END

load_dotenv()

# ---------------------
# Config
# 
CONFIG = {
    "cheap_model":"openai/gpt-oss-20b",
    "strong_model":"openai/gpt-oss-120b",
    "max_sub_questions": 4,
    "pricing":{
        "openai/gpt-oss-20b":{"input":0.075,"output":0.30},
        "openai/gpt-oss-120b":{"input":0.15,"output":0.60},
    },
}

cheap_llm = ChatGroq(model = CONFIG["cheap_model"],temperature=0,max_tokens=512)
strong_llm = ChatGroq(model = CONFIG["strong_model"],temperature=0,max_tokens=1300)

# -----------------------
# state
# 

class SubQuestionResult(TypedDict):
    question:str
    tier:str
    answer:str
    tokens_in:int
    tokens_out:int
    cost_usd:float
    latency_sec:float
    truncated:bool

class ResearchState(TypedDict):
    original_query:str
    sub_questions:List[str]
    results:List[SubQuestionResult]
    final_answer:str
    final_answer_truncated:bool
    total_cost_usd:float
    total_latency_sec:float


# ------------------------
# Cost tracking & helpers
# 

def _estimate_cost(model_name:str,tokens_in:int,tokens_out:int) -> float:
    rates = CONFIG["pricing"][model_name]
    return (tokens_in /1_000_000)*rates["input"] + (tokens_out/1_000_000)*rates["output"]

def _extract_usage(response) -> tuple[int,int]:
    """ Pull token counts from a langchain response, defaulting to 0 if unavailable. """
    usage = getattr(response,"usage_metadata",None) or {}
    return usage.get("input_tokens",0),usage.get("output_tokens",0)

def _was_truncated(response) -> bool:
    """Checks whether the model hit max_tokens instead of finishing naturally. """
    finish_reason = (response.response_metadata or {}).get("finish_reason")
    return finish_reason == "length"

@retry(
    retry=retry_if_exception_type(APIStatusError),
    wait=wait_exponential(multiplier=2,min=5,max=60),
    stop=stop_after_attempt(4),
)
def _invoke_with_retry(model,prompt):
    return model.invoke(prompt)

# --------------------------
# Complexity classifier - deterministic,no LLM call
# 

_REASONING_KEYWORDS = (
    "why","how does","compare","explain","analyze","trade-off",
    "tradeoff","pros and cons","relationship between","implications",
)

def classify_complexity(sub_question:str) -> str:
    """ Return 'strong' if the sub-question looks like it needs reasoning,
    otherwise 'cheap'. Deterministic - costs no tokens to decide. """
    q = sub_question.lower().strip()
    word_count = len(q.split())

    if any(kw in q for kw in _REASONING_KEYWORDS):
        return "strong"
    if word_count > 18:
        return "strong"
    if q.count(",") >= 2 or " and " in q:
        return "strong"
    return "cheap"

# ------------------------
# Nodes
# 
def planner_node(state:ResearchState) -> ResearchState:
    """Break the original query into 2-4 focused sub-questions. """
    prompt = (
        f"Break the following question into {CONFIG['max_sub_questions']} or fewer"
        f"focused, independently-answerable sub-questions."
        f"Return ONLY a JSON array of strings,nothing else.\n\n"
        f"Question: {state['original_query']}"
    )
    response = _invoke_with_retry(strong_llm,prompt)

    try:
        sub_questions = json.loads(response.content.strip())
        if not isinstance(sub_questions,list) or not sub_questions:
            raise ValueError
    except(json.JSONDecodeError,ValueError):
        # fallback - treats the original query as a single sub-question
        sub_questions = [state["original_query"]]
    
    sub_questions = sub_questions[: CONFIG["max_sub_questions"]]
    return {**state,"sub_questions":sub_questions}

def research_node(state:ResearchState) -> ResearchState:
    """Answer each sub-question , routed to the cheap or strong model based on complexity. """
    results: List[SubQuestionResult] = []

    for sub_q in state["sub_questions"]:
        tier = classify_complexity(sub_q)
        model = cheap_llm if tier == "cheap" else strong_llm
        model_name = CONFIG["cheap_model"] if tier == "cheap" else CONFIG["strong_model"]

        contrained_prompt = (
            f"{sub_q}\n\n"
            f"Answer in 3-5 sentences of plain prose. "
            f"Do not use markdown tables,headers,or bullet lists - "
            f"this is one sub-answer that will be merged with others later."
        )

        start = time.perf_counter()
        response = _invoke_with_retry(model,contrained_prompt)
        elapsed = time.perf_counter() - start

        tokens_in, tokens_out = _extract_usage(response)
        cost = _estimate_cost(model_name, tokens_in, tokens_out)

        results.append({
            "question": sub_q,
            "tier": tier,
            "answer": response.content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "latency_sec": round(elapsed, 3),
            "truncated":_was_truncated(response),
        })

    return {**state, "results": results}

def writer_node(state:ResearchState) -> ResearchState:
    """Synthesize the sub-question answers into a final answer. """
    context = "\n\n".join(f"Sub-question: {r['question']}\nAnswer: {r['answer'][:600]}" for r in state["results"])
    prompt = (
        f"Original question: {state['original_query']}\n\n"
        f"Research findings:\n{context}\n\n"
        f"Synthesize these findings into one clear,no more than 300 words"
        f"Use short paragraphs, not tables.Integrate the findings - don't list them."
    )
    start = time.perf_counter()
    response = _invoke_with_retry(strong_llm,prompt)
    elapsed = time.perf_counter() - start

    tokens_in,tokens_out = _extract_usage(response)
    writer_cost = _estimate_cost(CONFIG["strong_model"],tokens_in,tokens_out)

    total_cost = sum(r["cost_usd"] for r in state["results"])
    total_latency = sum(r["latency_sec"] for r in state["results"])

    return {
        **state,
        "final_answer": response.content,
        "final_answer_truncated":_was_truncated(response),
        "total_cost_usd": round(total_cost + writer_cost, 6),
        "total_latency_sec": round(total_latency + elapsed, 3),
    }

# ---------------------------
# Graph assembly
# 

def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", END)

    return graph.compile()

# ------------------------
# Public entry point
# 

def run_research_query(query:str) -> ResearchState:
    app = build_graph()
    initial_state : ResearchState = {
        "original_query": query,
        "sub_questions":[],
        "results":[],
        "final_answer":"",
        "final_answer_truncated":"False",
        "total_cost_usd":0.0,
        "total_latency_sec":0.0,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    # Smoke test - run once to confirm the pipeline is works end to end
    result = run_research_query(
        "what year was python created,who created Numpy,and how does gradient descent work?"
        )

    print("sub-questions:",result["sub_questions"])
    for r in result["results"]:
        print(f"\n[{r['tier'].upper()}] {r['question']}")
        print(f"  -> {r['answer'][:150]}...")
        flag = "Truncated" if r["truncated"] else ""
        print(f"  tokens: {r['tokens_in']}in / {r['tokens_out']}out | cost : ${r['cost_usd']:.6f} | {r['latency_sec']}s{flag}")
    print("\n === FINAL ANSWER ===")
    print(result["final_answer"])
    if result["final_answer_truncated"]:
        print("\n Warning : fianl answer was truncated - consider raising strong_llm max_tokens")
    print(f"\nTotal cost: ${result['total_cost_usd']:.6f} | Total research latency: {result['total_latency_sec']}s") 